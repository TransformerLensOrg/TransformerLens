"""Private raw-target loading and single-token NLA activation extraction."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis._nla_artifacts import NLAPairConfig, _integer
from transformer_lens.tools.analysis._nla_contracts import (
    ActivationProvenance,
    NLAActivation,
    SourceProvenance,
    validate_context_length,
)


def validate_target_structure(model: Any, pair: NLAPairConfig) -> None:
    """Require raw single-stream causal Bridge weights on one fp32 device."""
    if not isinstance(model, TransformerBridge):
        raise TypeError("NLA extraction requires a TransformerBridge")
    if getattr(model, "compatibility_mode", False) or getattr(model, "_weights_processed", False):
        raise ValueError("NLA extraction requires the raw unprocessed residual basis")
    original = model.original_model
    config = original.config
    if getattr(config, "model_type", None) != pair.model_type:
        raise ValueError("target architecture does not match the artifact")
    if model.cfg.d_model != pair.d_model or model.cfg.n_layers != pair.target_layers:
        raise ValueError("target dimensions/depth do not match the artifact")
    if not model.adapter.supports_generation or getattr(config, "is_encoder_decoder", False):
        raise ValueError("NLA extraction requires a causal decoder-only model")
    if getattr(model.cfg, "attention_dir", "causal") != "causal":
        raise ValueError("NLA extraction requires causal attention")
    if getattr(model.cfg, "total_ut_steps", 1) != 1:
        raise ValueError("NLA extraction requires non-recurrent block execution")
    if getattr(original, "is_quantized", False) or getattr(config, "quantization_config", None):
        raise ValueError("quantized targets are not supported")
    if getattr(config, "_attn_implementation", None) != "eager":
        raise ValueError("NLA target extraction requires eager attention")
    if model.training or original.training:
        raise ValueError("NLA target extraction requires eval mode")
    parameters = list(original.parameters())
    if not parameters or any(
        p.is_floating_point() and p.dtype != torch.float32 for p in parameters
    ):
        raise ValueError("NLA target extraction currently requires float32 weights")
    if len({p.device for p in parameters}) != 1 or any(p.device.type == "meta" for p in parameters):
        raise ValueError("NLA target extraction requires materialized single-device weights")
    mapping = model.adapter.get_component_mapping()
    block = mapping.get("blocks")
    if not getattr(block, "hook_out_is_single_residual_stream", False):
        raise ValueError("target blocks must expose a single residual stream")
    if f"blocks.{pair.layer}.hook_out" not in model.hook_dict:
        raise ValueError("artifact block output hook is unavailable")


def target_provenance(
    model: Any, pair: NLAPairConfig, *, allow_unverified_target: bool = False
) -> SourceProvenance:
    """Check available loading identity, without claiming live-weight integrity."""
    if type(allow_unverified_target) is not bool:
        raise ValueError("allow_unverified_target must be bool")
    config = model.original_model.config
    identity = getattr(config, "_name_or_path", None)
    revision = getattr(config, "_commit_hash", None)
    recorded = getattr(model, "_nla_target_artifact", None)
    local_identity = isinstance(identity, str) and (
        Path(identity).is_absolute()
        or identity.startswith(("./", "../"))
        or Path(identity).exists()
    )
    if identity and not local_identity and identity != pair.target.repo_id:
        raise ValueError("target repository identity does not match the pinned artifact")
    if revision and revision != pair.target.revision:
        raise ValueError("target revision does not match the pinned artifact")
    if recorded is not None and recorded != pair.target:
        raise ValueError("recorded target loading identity does not match the artifact")
    if recorded == pair.target or (
        identity == pair.target.repo_id and revision == pair.target.revision
    ):
        return SourceProvenance.PINNED_LOAD
    if not allow_unverified_target:
        raise ValueError("target loading provenance is unknown; explicit attestation is required")
    warnings.warn(
        "Target loading provenance is unverified; results are outside verified artifact support",
        UserWarning,
        stacklevel=2,
    )
    return SourceProvenance.CALLER_ATTESTED


def load_target(pair: NLAPairConfig, *, device: str = "cpu") -> TransformerBridge:
    """Load pinned fp32 raw weights with eager attention and caller ownership."""
    model = TransformerBridge.boot_transformers(
        pair.target.repo_id,
        revision=pair.target.revision,
        device=device,
        dtype=torch.float32,
        trust_remote_code=False,
    )
    model.eval()
    validate_target_structure(model, pair)
    # Immutable loading evidence contains no reference back to the model.
    model.__dict__["_nla_target_artifact"] = pair.target
    return model


def prepare_tokens(
    model: Any, prompt: str | torch.Tensor, *, prepend_bos: bool | None = None
) -> tuple[torch.Tensor, str]:
    """Tokenize exactly once or accept one explicit unpadded integer sequence."""
    if prepend_bos is not None and type(prepend_bos) is not bool:
        raise ValueError("prepend_bos must be bool or None")
    if isinstance(prompt, str):
        if not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        tokens = model.to_tokens(prompt, prepend_bos=prepend_bos, truncate=False)
        effective_bos = model.cfg.default_prepend_bos if prepend_bos is None else prepend_bos
        policy = f"Bridge string tokenization; prepend_bos={effective_bos}; no chat template"
    else:
        if prepend_bos is not None:
            raise ValueError("prepend_bos cannot modify explicit token input")
        if prompt.dtype not in (torch.int32, torch.int64):
            raise ValueError("explicit tokens must be int32 or int64")
        if prompt.ndim == 1:
            tokens = prompt.unsqueeze(0)
        elif prompt.ndim == 2 and prompt.shape[0] == 1:
            tokens = prompt
        else:
            raise ValueError("only one unpadded token sequence is supported")
        policy = "Explicit tokens; BOS supplied by caller"
    validate_context_length(tokens.shape[-1], model.cfg.n_ctx)
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("only one token sequence is supported")
    vocab_size = model.original_model.config.vocab_size
    if torch.any(tokens < 0).item() or torch.any(tokens >= vocab_size).item():
        raise ValueError("token ids must be within the model vocabulary")
    tokenizer = model.tokenizer
    pad = getattr(tokenizer, "pad_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    # When pad aliases EOS, a token id alone cannot distinguish padding from content.
    if pad is not None and pad != eos and torch.any(tokens == pad).item():
        raise ValueError("padding tokens are not supported")
    return tokens.to(device=model.cfg.device, dtype=torch.int64), policy


def extract_activation(
    model: TransformerBridge,
    pair: NLAPairConfig,
    artifact_name: str,
    prompt: str | torch.Tensor,
    *,
    position: int = -1,
    output_device: str = "cpu",
    prepend_bos: bool | None = None,
    allow_unverified_target: bool = False,
) -> NLAActivation:
    """Copy one raw block-output vector, releasing the full temporary cache."""
    validate_target_structure(model, pair)
    source = target_provenance(model, pair, allow_unverified_target=allow_unverified_target)
    tokens, policy = prepare_tokens(model, prompt, prepend_bos=prepend_bos)
    if type(position) is not int:
        raise ValueError("position must be an integer")
    resolved = position if position >= 0 else tokens.shape[1] + position
    if not 0 <= resolved < tokens.shape[1]:
        raise ValueError("position is outside the original prompt")
    _integer(pair.layer, "layer")
    hook = f"blocks.{pair.layer}.hook_out"
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=hook, return_cache_object=False, return_type=None
        )
        residual = cache[hook]
        if residual.shape != (1, tokens.shape[1], pair.d_model):
            raise ValueError("block output is not the expected single residual stream")
        vector = residual[0, resolved].detach().to(output_device).clone()
    del cache, residual
    provenance = ActivationProvenance(
        artifact_name,
        source,
        pair.target if source is SourceProvenance.PINNED_LOAD else None,
        pair.layer,
        resolved,
        policy,
    )
    return NLAActivation(vector, provenance)
