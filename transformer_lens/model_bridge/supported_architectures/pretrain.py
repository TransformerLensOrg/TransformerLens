"""Architecture adapter for a lightweight decoder-only pretraining model.

Maps a decoder-only transformer using RoPE, RMSNorm, gated SwiGLU MLPs, and
optional sparse mixture-of-experts feed-forward layers into
TransformerBridge, by wrapping the source module and delegating to its own
`forward` rather than translating parameters into a second implementation.

Usage: `build_pretrain_bridge(model, cfg)` -- the public entry point.
`PretrainModelContainer` and direct `build_bridge_from_module` use are
internal/advanced details (see `PretrainModelContainer`'s docstring).

Scope: maps a live module into TransformerBridge. Does not load
checkpoints, merge tensor-parallel shards, or depend on a training
framework.

Required module protocol -- "lightweight decoder-only pretraining models"
describes intent, not a generality guarantee. The wrapped model must
expose:

    model.embed                          (embedding lookup)
    model.blocks[i].norm1                (pre-attention norm)
    model.blocks[i].attn                 (called as attn(x, ...))
    model.blocks[i].norm2                (pre-MLP norm)
    model.blocks[i].mlp                  (gate/up/down, or router/experts)
    model.norm_f                         (final norm)
    model.lm_head                        (unembedding)

`gate`/`up`/`down` and `router`/`experts` name the supported protocol.
`DenseOrMoEFeedForwardBridge` checks these structurally -- attribute
presence plus basic type (each is a module, `experts` is a registered
module collection) -- and raises clearly on a mismatch, but that is
structural validation only: it does not and cannot validate that a
module satisfying the shape actually implements matching forward
semantics. Blocks must take more than the bare hidden state (this target
passes `cos`/`sin`) -- see `PretrainModelContainer`.
"""
from __future__ import annotations

from typing import Any

import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    DelegatedAttentionBlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MoEBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

ARCHITECTURE_NAME = "TransformerLensPretrain"

# Reserved bridge kwargs removed before forwarding to the wrapped model.
# Only known bridge-compatibility kwargs are stripped so genuine caller
# mistakes (e.g. `target=` vs `targets=`) still raise naturally, and no
# signature introspection is needed to support an arbitrary forward.
_BRIDGE_COMPAT_KWARGS = frozenset({"output_attentions"})


class DenseOrMoEFeedForwardBridge(GeneralizedComponent):
    """Wraps a dense SwiGLU MLP or sparse MoE layer behind one interface.
    Dispatch is by structural inspection (`router`/`experts` vs
    `gate`/`up`/`down`), not config, so dense/MoE/mixed architectures all
    share the same component mapping.
    """

    def __init__(self, name: str, config: Any):
        super().__init__(name, config=config, submodules={})
        self._delegate: GeneralizedComponent | None = None

    def set_original_component(self, component: torch.nn.Module) -> None:
        super().set_original_component(component)
        # This subclass's own __init__ takes `name: str` (non-optional), so
        # self.name is always a str here -- but the base GeneralizedComponent
        # attribute is typed `str | None`, which is all mypy sees without this
        # narrowing. MoEBridge/GatedMLPBridge both require a plain `str` name.
        assert self.name is not None
        # hasattr can match by attribute-name coincidence, so verify the
        # protocol's shape too: router/gate/up/down must be modules, and
        # experts a *registered* ModuleList/ModuleDict -- a plain list of
        # experts drops out of parameters()/state_dict()/.to()/train()/eval().
        if hasattr(component, "router") and hasattr(component, "experts"):
            if not isinstance(component.router, torch.nn.Module):
                raise TypeError(
                    f"{type(component).__name__}.router must be an nn.Module; "
                    f"got {type(component.router).__name__}."
                )
            if not isinstance(component.experts, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                raise TypeError(
                    f"{type(component).__name__}.experts must be a registered "
                    "module collection (nn.ModuleList or nn.ModuleDict); got "
                    f"{type(component.experts).__name__}."
                )
            delegate: GeneralizedComponent = MoEBridge(
                name=self.name,
                config=self.config,
                submodules={"gate": LinearBridge(name="router")},
            )
        elif hasattr(component, "gate") and hasattr(component, "up") and hasattr(component, "down"):
            for field in ("gate", "up", "down"):
                value = getattr(component, field)
                if not isinstance(value, torch.nn.Module):
                    raise TypeError(
                        f"{type(component).__name__}.{field} must be an "
                        f"nn.Module; got {type(value).__name__}."
                    )
            delegate = GatedMLPBridge(
                name=self.name,
                config=self.config,
                submodules={
                    "gate": LinearBridge(name="gate"),
                    "in": LinearBridge(name="up"),
                    "out": LinearBridge(name="down"),
                },
            )
        else:
            raise ValueError(
                f"Block.mlp is a {type(component).__name__} with neither "
                "an MoE layer's (router, experts) nor a gated MLP's "
                "(gate, up, down) attributes -- this adapter doesn't know "
                "how to wrap it."
            )
        delegate.set_original_component(component)
        # Not `self._delegate = delegate`: normal registration would
        # duplicate hook_in/hook_out under a nested `._delegate.` path,
        # risking a broad hook selector firing twice.
        # `_delegate` is an execution helper, absent from named_modules()
        # -- safe since parameters/state_dict/dtype are read from the raw
        # wrapped model (see PretrainModelContainer), not this tree.
        object.__setattr__(self, "_delegate", delegate)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        assert self._delegate is not None, f"{self.name}: original component not set"
        if args:
            args = (self.hook_in(args[0]),) + args[1:]
        elif "hidden_states" in kwargs:
            kwargs = {**kwargs, "hidden_states": self.hook_in(kwargs["hidden_states"])}
        output = self._delegate(*args, **kwargs)

        if isinstance(output, tuple):
            if len(output) == 0:
                raise TypeError(
                    "DenseOrMoEFeedForwardBridge expected a non-empty tuple "
                    "whose first element is a torch.Tensor"
                )

            first = output[0]

            if not isinstance(first, torch.Tensor):
                raise TypeError(
                    "DenseOrMoEFeedForwardBridge expected the first tuple element "
                    f"to be a torch.Tensor, got {type(first).__name__}"
                )

            hooked_first = self.hook_out(first)

            # Preserve every auxiliary element without sending it through HookPoint.
            return (hooked_first, *output[1:])

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "DenseOrMoEFeedForwardBridge expected a torch.Tensor or a tuple "
                f"whose first element is a torch.Tensor, got {type(output).__name__}"
            )

        return self.hook_out(output)


class _LogitsAttrDict(dict):
    """Makes a plain dict's keys accessible as attributes (`d.logits` reads
    `d["logits"]`), so a source model's plain-dict forward output satisfies
    the `hasattr(output, "logits")` contract `TransformerBridge` expects.
    Behaves as a plain dict everywhere else (indexing, `.get`, `in`, ...).
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e


class PretrainModelContainer(torch.nn.Module):
    """Wraps the source model one level deeper (`container.inner`) so its
    own `embed`/`blocks` attrs don't collide with `TransformerBridge`
    component_mapping keys, normalizes the forward return to the `.logits`
    contract, and strips `_BRIDGE_COMPAT_KWARGS`. Applied automatically by
    `build_pretrain_bridge`.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.inner = model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        filtered = {k: v for k, v in kwargs.items() if k not in _BRIDGE_COMPAT_KWARGS}
        output = self.inner(*args, **filtered)

        # Already-normalized or already-HF-style outputs pass through
        # unchanged, but only after checking .logits is actually a tensor
        # -- an object merely exposing the attribute isn't enough.
        if isinstance(output, _LogitsAttrDict):
            if "logits" not in output:
                raise ValueError(
                    f"{type(self.inner).__name__}.forward returned a "
                    "_LogitsAttrDict without a 'logits' key."
                )
            if not isinstance(output["logits"], torch.Tensor):
                raise TypeError(
                    f"{type(self.inner).__name__}.forward returned a "
                    f"_LogitsAttrDict with a non-tensor 'logits' value: "
                    f"{type(output['logits']).__name__}."
                )
            return output

        # try/except, not hasattr(): hasattr() would evaluate a
        # property-backed .logits once, then a separate read would
        # evaluate it again. This reads it exactly once.
        try:
            logits = output.logits
        except AttributeError:
            pass
        else:
            if not isinstance(logits, torch.Tensor):
                raise TypeError(
                    f"{type(self.inner).__name__}.forward returned a "
                    f"{type(output).__name__} with a non-tensor .logits value: "
                    f"{type(logits).__name__}."
                )
            return output

        if isinstance(output, torch.Tensor):
            return output

        # TransformerBridge extracts output[0] as logits for tuple returns.
        if isinstance(output, tuple):
            if output and isinstance(output[0], torch.Tensor):
                return output
            raise TypeError(
                f"{type(self.inner).__name__}.forward returned a tuple whose "
                f"first element is a {type(output[0]).__name__ if output else 'empty tuple'}, "
                "not a torch.Tensor -- TransformerBridge extracts output[0] as "
                "logits for tuple returns, so it must be a tensor."
            )

        # The primary case: a plain dict, normalized into _LogitsAttrDict.
        if isinstance(output, dict):
            if "logits" not in output:
                raise ValueError(
                    f"{type(self.inner).__name__}.forward returned a dict with keys "
                    # list(), not sorted(): sorted() raises TypeError on
                    # heterogeneous keys, which would mask this error.
                    f"{list(output.keys())}, but PretrainModelContainer requires a "
                    "'logits' key -- without it, TransformerBridge's own "
                    "hasattr(output, 'logits') check would silently fail the same "
                    "way this container exists to prevent."
                )
            if not isinstance(output["logits"], torch.Tensor):
                raise TypeError(
                    f"{type(self.inner).__name__}.forward returned a dict whose "
                    f"'logits' value is a {type(output['logits']).__name__}, not a "
                    "torch.Tensor."
                )
            return _LogitsAttrDict(output)

        raise TypeError(
            f"{type(self.inner).__name__}.forward must return a torch.Tensor, a "
            "tuple whose first element is a tensor, a dict containing a "
            "tensor-valued 'logits' key, or an object with a tensor-valued "
            f".logits attribute; got {type(output).__name__}."
        )


class NativeForwardAttentionBridge(AttentionBridge):
    """Opaque attention bridge that delegates to the source attention.

    This adapter intentionally exposes only input/output attention hooks.
    It has no mapped Q/K/V/O projection components, so the standard
    per-head aliases and weight aliases do not apply.
    """

    hook_aliases = {}
    property_aliases = {}
    supports_split_qkv_fork = False


class PretrainArchitectureAdapter(ArchitectureAdapter):
    """Adapter for a decoder-only transformer using RoPE, RMSNorm, gated
    SwiGLU MLPs, and optional sparse MoE feed-forward layers.

    Uses an opaque `NativeForwardAttentionBridge` (delegates to
    `Attention.forward`) so RoPE runs under the source's adjacent-pair
    convention rather than HF's rotate-half -- at the cost of only
    block-level hooks, no per-head hooks.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False

        self.component_mapping = {
            # "inner." because this adapter expects the source model to
            # arrive wrapped in `PretrainModelContainer` -- see that
            # class's docstring for why.
            "embed": EmbeddingBridge(name="inner.embed"),
            "blocks": DelegatedAttentionBlockBridge(
                name="inner.blocks",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="norm1", config=self.cfg),
                    "attn": NativeForwardAttentionBridge(
                        name="attn",
                        config=self.cfg,
                        submodules={},  # opaque wrap -- see class docstring
                    ),
                    "ln2": RMSNormalizationBridge(name="norm2", config=self.cfg),
                    "mlp": DenseOrMoEFeedForwardBridge(name="mlp", config=self.cfg),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="inner.norm_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="inner.lm_head"),
        }


def build_pretrain_bridge(
    model: torch.nn.Module,
    cfg: TransformerBridgeConfig,
    *,
    device: Any = None,
    dtype: torch.dtype | None = None,
    model_name: str | None = None,
) -> TransformerBridge:
    """Public entry point: wraps `model` in `PretrainModelContainer` and
    builds a `TransformerBridge` around it. Prefer this over calling
    `build_bridge_from_module` directly -- the container is easy to forget.

    `device`/`dtype`/`model_name` forward to `build_bridge_from_module`
    only when explicitly given.

    `bridge.train()`/`.eval()` propagate to `model` via
    `TransformerBridge.train()` itself, which sets mode on
    `original_model` in addition to the registered module tree
    (`original_model` is deliberately not a registered submodule, so
    `nn.Module.train()`'s own recursion never reaches it). This adapter
    needs nothing extra for mode propagation.

    Setting mode on `model` directly still works too and stays in sync.
    """
    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    if model_name is not None:
        kwargs["model_name"] = model_name

    return build_bridge_from_module(
        PretrainModelContainer(model),
        architecture=ARCHITECTURE_NAME,
        tl_config=cfg,
        **kwargs,
    )
