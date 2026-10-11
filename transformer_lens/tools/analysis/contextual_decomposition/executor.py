"""Private raw GPT-2 capture and residual-anchor contextual decomposition."""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP, GPT2Attention, GPT2Block
from transformers.pytorch_utils import Conv1D

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources.transformers_driver import TransformersDriver
from transformer_lens.tools.analysis._model_state import require_eval_mode
from transformer_lens.tools.analysis.contextual_decomposition.rules import (
    _affine,
    _attention,
    _CDConstituents,
    _gelu,
    _layer_norm,
    _residual_add,
)


@dataclass(frozen=True)
class _ReconstructionError:
    """Measured native reconstruction error with its reference scale."""

    max_abs: float
    reference_max_abs: float


@dataclass(frozen=True)
class _GPT2Decomposition:
    """Detached native activations, suffix constituents, and local parity errors."""

    logits: _CDConstituents
    native_logits: torch.Tensor
    native_cache: dict[str, torch.Tensor]
    constituents: dict[str, _CDConstituents]
    errors: dict[str, _ReconstructionError]


def _require_gpt2(model: TransformerBridge) -> GPT2Config:
    if not isinstance(model, TransformerBridge) or not isinstance(
        model._driver, TransformersDriver
    ):
        raise ValueError("CD execution requires a local transformers TransformerBridge")
    if model.compatibility_mode or getattr(model, "_weights_processed", False):
        raise ValueError("CD execution requires raw weights")
    require_eval_mode(model, operation="GPT-2 CD execution")
    config = model.original_model.config
    if not isinstance(config, GPT2Config) or config.add_cross_attention:
        raise ValueError("CD execution supports decoder-only GPT-2")
    if (
        getattr(model.cfg, "use_split_qkv_input", False)
        or getattr(model.cfg, "use_attn_in", False)
        or getattr(model.cfg, "total_ut_steps", 1) != 1
    ):
        raise ValueError("CD execution does not cover split-input forks or repeated blocks")
    if config._attn_implementation != "eager":
        raise ValueError("CD execution requires eager attention")
    if config.activation_function not in ("gelu", "gelu_new", "gelu_pytorch_tanh"):
        raise ValueError("CD execution encountered an unsupported activation")
    for parameter in model.parameters():
        if (
            parameter.dtype != torch.float32
            or parameter.device != model.embed.original_component.weight.device
        ):
            raise ValueError("CD execution requires fp32 weights on one device")
    for block in model.blocks:
        if type(block.original_component) is not GPT2Block:
            raise ValueError("CD execution encountered an unsupported block")
        attention = block.attn.original_component
        if (
            type(attention) is not GPT2Attention
            or type(block.mlp.original_component) is not GPT2MLP
        ):
            raise ValueError("CD execution encountered an unsupported attention or MLP")
        if (
            not attention.scale_attn_weights
            or attention.scale_attn_by_inverse_layer_idx
            or attention.reorder_and_upcast_attn
        ):
            raise ValueError("CD execution encountered unsupported attention scaling or upcasting")
        for norm in (block.ln1, block.ln2):
            if type(norm.original_component) is not torch.nn.LayerNorm:
                raise ValueError("CD execution encountered unsupported normalization")
        for linear in (block.attn.q, block.attn.k, block.attn.v):
            if type(linear.original_component) is not torch.nn.Linear:
                raise ValueError("CD execution requires split native Q/K/V linear weights")
        for linear in (block.attn.o, getattr(block.mlp, "in"), block.mlp.out):
            if type(linear.original_component) is not Conv1D:
                raise ValueError("CD execution encountered unsupported projection layout")
    if (
        type(model.ln_final.original_component) is not torch.nn.LayerNorm
        or type(model.unembed.original_component) is not torch.nn.Linear
    ):
        raise ValueError("CD execution encountered unsupported final readout")
    internal_ids = {handle.id for block in model.blocks for handle in block._capture_hook_handles}
    for module in model.modules():
        if module._backward_hooks:
            raise ValueError("CD execution does not support backward hooks")
        if getattr(module, "_relevance_rule_active", False) or getattr(
            module, "_active_relevance_rules", None
        ):
            raise ValueError("CD execution conflicts with active relevance rules")
        external_forward = [
            callback
            for identifier, callback in module._forward_hooks.items()
            if identifier not in internal_ids
            and not (
                getattr(callback, "__module__", "") == "transformers.utils.output_capturing"
                and getattr(callback, "__qualname__", "")
                == "install_output_capuring_hook.<locals>.output_capturing_hook"
            )
        ]
        if not isinstance(module, HookPoint) and (
            external_forward or set(module._forward_pre_hooks).difference(internal_ids)
        ):
            raise ValueError("CD execution does not support hooks on computational modules")
    for embedding in (model.embed, model.pos_embed):
        if type(embedding.original_component) is not torch.nn.Embedding:
            raise ValueError("CD execution encountered unsupported embeddings")
    if len(model.blocks) != config.n_layer or model.cfg.n_heads != config.n_head:
        raise ValueError("CD execution encountered inconsistent GPT-2 layout")
    model.adapter.validate_output_logits_transform()
    return config


@torch.no_grad()
def _capture(
    model: TransformerBridge,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cache: dict[str, torch.Tensor] = {}
    incoming: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    seen: set[int] = set()
    unwired_blocks = [block for block in model.blocks if not block._capture_hooks_wired]
    original_hooks = {id(module): set(module._forward_hooks) for module in model.modules()}
    collector_flags = [
        (
            module,
            "_output_capturing_hooks_installed" in module.__dict__,
            module.__dict__.get("_output_capturing_hooks_installed"),
        )
        for module in model.modules()
    ]
    original_parameters = {
        name: (id(parameter), parameter._version) for name, parameter in model.named_parameters()
    }

    def observe_input(name: str):
        def callback(module: Any, args: Any) -> None:
            incoming[name] = args[0].detach().clone()

        return callback

    def observe_output(name: str):
        def callback(module: Any, args: Any, output: Any) -> None:
            if not isinstance(output, torch.Tensor) or not torch.equal(incoming[name], output):
                raise ValueError(f"CD observation hook at {name} was edited")
            if name in cache:
                raise ValueError(f"CD hook {name} executed more than once")
            cache[name] = output.detach().clone()

        return callback

    try:
        for name, point in model.hook_dict.items():
            if id(point) in seen:
                continue
            seen.add(id(point))
            handles.append(point.register_forward_pre_hook(observe_input(name), prepend=True))
            handles.append(point.register_forward_hook(observe_output(name)))
        logits = model(tokens, attention_mask=mask, position_ids=positions, use_cache=False)
        if not isinstance(logits, torch.Tensor):
            raise ValueError("CD execution requires tensor logits")
        if original_parameters != {
            name: (id(parameter), parameter._version)
            for name, parameter in model.named_parameters()
        }:
            raise ValueError("CD execution observed changed parameters during native capture")
        for index in range(len(model.blocks)):
            for kind in ("q", "k", "v"):
                name = f"blocks.{index}.attn.{kind}.hook_out"
                cache[name] = cache[name].reshape(
                    *tokens.shape, model.cfg.n_heads, model.cfg.d_head
                )
            name = f"blocks.{index}.attn.o.hook_in"
            cache[name] = cache[name].reshape(*tokens.shape, model.cfg.n_heads, model.cfg.d_head)
        return logits.detach().clone(), cache
    finally:
        for handle in handles:
            handle.remove()
        for block in unwired_blocks:
            block._teardown_capture_hooks()
        # HF lazily installs read-only collectors; remove only those created by this run.
        for module in model.modules():
            for identifier, callback in list(module._forward_hooks.items()):
                if (
                    identifier not in original_hooks[id(module)]
                    and getattr(callback, "__module__", "") == "transformers.utils.output_capturing"
                ):
                    del module._forward_hooks[identifier]
        for module, existed, value in collector_flags:
            if existed:
                module.__dict__["_output_capturing_hooks_installed"] = value
            else:
                module.__dict__.pop("_output_capturing_hooks_installed", None)


@torch.no_grad()
def _decompose_gpt2(
    model: TransformerBridge,
    tokens: torch.Tensor,
    *,
    layer: int = 0,
    beta: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> _GPT2Decomposition:
    """Decompose from a block-input residual anchor through the native GPT-2 suffix.

    Beta defaults to the full residual at the anchor. Capture uses observation
    hooks only and rejects edits; each returned constituent total is checked
    against native intermediates with rtol=2e-5 and atol=2e-6. No tensor is
    corrected to conceal a reconstruction discrepancy. Fully masked queries,
    sharding, caching, alternate attention modes, and compatibility weights are
    outside this executor's contract.
    """
    config = _require_gpt2(model)
    device = model.embed.original_component.weight.device
    if (
        tokens.ndim != 2
        or tokens.numel() == 0
        or tokens.dtype != torch.long
        or tokens.device != device
    ):
        raise ValueError("tokens must be nonempty long [batch, sequence] on the model device")
    if (
        bool(((tokens < 0) | (tokens >= config.vocab_size)).any())
        or tokens.shape[1] > config.n_positions
    ):
        raise ValueError("tokens exceed the vocabulary or context length")
    if isinstance(layer, bool) or not 0 <= layer < len(model.blocks):
        raise ValueError("layer must identify a GPT-2 block")
    if beta is not None:
        if (
            beta.shape != (*tokens.shape, config.n_embd)
            or beta.device != device
            or beta.dtype != torch.float32
        ):
            raise ValueError(
                "beta must have shape [batch, sequence, width], fp32, on the model device"
            )
        _CDConstituents(beta, torch.zeros_like(beta))
    if attention_mask is None:
        attention_mask = torch.ones_like(tokens, dtype=torch.bool)
    if (
        attention_mask.shape != tokens.shape
        or attention_mask.dtype != torch.bool
        or attention_mask.device != device
    ):
        raise ValueError("attention_mask must be boolean and aligned with tokens")
    if not bool(attention_mask[:, 0].all()):
        raise ValueError("causal attention contains a fully masked query row")
    if position_ids is None:
        position_ids = torch.arange(tokens.shape[1], device=device).expand_as(tokens)
    if (
        position_ids.shape != tokens.shape
        or position_ids.dtype != torch.long
        or position_ids.device != device
    ):
        raise ValueError("position_ids must be long and aligned with tokens")
    if bool(((position_ids < 0) | (position_ids >= config.n_positions)).any()):
        raise ValueError("position_ids exceed the position embedding table")

    transient_names = (
        "_pattern",
        "_attn_scores",
        "_captured_pre_ln_residual",
        "_captured_ln1",
        "_fired_hook_out",
        "_suppress_next_hook_in",
    )
    transient_state = [
        (module, name, name in module.__dict__, getattr(module, name))
        for module in model.modules()
        for name in transient_names
        if hasattr(module, name)
    ]
    try:
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            native_logits, cache = _capture(model, tokens, attention_mask, position_ids)
    finally:
        for module, name, existed, value in transient_state:
            if existed:
                module.__dict__[name] = value
            else:
                module.__dict__.pop(name, None)
    constituents: dict[str, _CDConstituents] = {}
    errors: dict[str, _ReconstructionError] = {}

    def record(name: str, parts: _CDConstituents) -> _CDConstituents:
        if name not in cache:
            raise ValueError(f"required native hook {name} did not execute")
        total = parts.beta + parts.gamma
        native = cache[name]
        if total.shape != native.shape:
            raise ValueError(f"native reconstruction shape mismatch at {name}")
        error = _ReconstructionError(float((total - native).abs().max()), float(native.abs().max()))
        errors[name] = error
        if not torch.allclose(total, native, rtol=2e-5, atol=2e-6):
            raise ValueError(f"native reconstruction failed at {name}: max_abs={error.max_abs:.6g}")
        constituents[name] = parts
        return parts

    # The unchanged prefix supplies the actual anchor rather than a recomputed basis.
    anchor_name = f"blocks.{layer}.hook_in"
    if anchor_name not in cache:
        raise ValueError(f"required native hook {anchor_name} did not execute")
    native_anchor = cache[anchor_name]
    relevant = native_anchor.clone() if beta is None else beta.detach().clone()
    parts = _CDConstituents(relevant, native_anchor - relevant)
    record(
        "embed.hook_out",
        _CDConstituents(
            torch.zeros_like(cache["embed.hook_out"]),
            torch.nn.functional.embedding(tokens, model.embed.original_component.weight),
        ),
    )
    record(
        "pos_embed.hook_out",
        _CDConstituents(
            torch.zeros_like(cache["pos_embed.hook_out"]),
            torch.nn.functional.embedding(position_ids, model.pos_embed.original_component.weight),
        ),
    )
    for index in range(layer, len(model.blocks)):
        block = model.blocks[index]
        prefix = f"blocks.{index}"
        record(f"{prefix}.hook_in", parts)
        norm = block.ln1.original_component
        normalized = record(
            f"{prefix}.ln1.hook_out",
            _layer_norm(
                parts,
                norm.weight,
                norm.bias,
                eps=norm.eps,
                denominator=cache[f"{prefix}.ln1.hook_scale"],
            ),
        )
        attn = block.attn
        q, k, v = (projection.original_component for projection in (attn.q, attn.k, attn.v))
        o = attn.o.original_component
        attention = _attention(
            normalized,
            q.weight.T,
            k.weight.T,
            v.weight.T,
            o.weight,
            n_heads=config.n_head,
            q_bias=q.bias,
            k_bias=k.bias,
            v_bias=v.bias,
            o_bias=o.bias,
            key_mask=attention_mask,
        )
        for kind, projected in (
            ("q", attention.query),
            ("k", attention.key),
            ("v", attention.value),
        ):
            record(
                f"{prefix}.attn.{kind}.hook_out",
                _CDConstituents(projected.beta.transpose(1, 2), projected.gamma.transpose(1, 2)),
            )
        record(f"{prefix}.attn.hook_pattern", attention.probabilities)
        record(
            f"{prefix}.attn.o.hook_in",
            _CDConstituents(
                attention.context.beta.transpose(1, 2), attention.context.gamma.transpose(1, 2)
            ),
        )
        record(f"{prefix}.attn.hook_out", attention.output)
        parts = _residual_add(parts, attention.output)
        record(f"{prefix}.ln2.hook_in", parts)
        norm = block.ln2.original_component
        normalized = record(
            f"{prefix}.ln2.hook_out",
            _layer_norm(
                parts,
                norm.weight,
                norm.bias,
                eps=norm.eps,
                denominator=cache[f"{prefix}.ln2.hook_scale"],
            ),
        )
        mlp_in = getattr(block.mlp, "in").original_component
        pre = record(f"{prefix}.mlp.in.hook_out", _affine(normalized, mlp_in.weight, mlp_in.bias))
        post = record(f"{prefix}.mlp.out.hook_in", _gelu(pre, variant=config.activation_function))
        mlp_out = block.mlp.out.original_component
        output = record(f"{prefix}.mlp.hook_out", _affine(post, mlp_out.weight, mlp_out.bias))
        parts = record(f"{prefix}.hook_out", _residual_add(parts, output))
    norm = model.ln_final.original_component
    parts = record(
        "ln_final.hook_out",
        _layer_norm(
            parts, norm.weight, norm.bias, eps=norm.eps, denominator=cache["ln_final.hook_scale"]
        ),
    )
    unembed = model.unembed.original_component
    logits = record("unembed.hook_out", _affine(parts, unembed.weight.T, unembed.bias))
    if not torch.allclose(logits.beta + logits.gamma, native_logits, rtol=2e-5, atol=2e-6):
        raise ValueError("native output transform is not covered by GPT-2 CD execution")
    return _GPT2Decomposition(logits, native_logits, cache, constituents, errors)
