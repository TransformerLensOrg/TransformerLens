"""LLaDA 2.0 MoE architecture adapter.

Ant Group's LLaDA 2.x (``LLaDA2MoeModelLM``, remote code): masked
block-diffusion language models on a DeepSeek-V3-style MoE decoder —
fused ``query_key_value`` attention with full-width query/key layernorms,
per-expert routed MLPs behind a bias-corrected router plus shared
experts, and dense MLPs on the first ``first_k_dense_replace`` layers.

Attention is bidirectional (``is_causal = False``) and generation is
block-diffusion sampling via the model's own ``generate``, so attention
delegates to HF, the bridge's autoregressive generation is disabled, and
phases 1-3 apply (the Dream/bd3lm treatment). The fused QKV ships no
HookedTransformer-format weight conversions, so LN folding is disabled.

The remote forward validates attention_mask strictly: it must be the 4D
block-diffusion form (batch, 1, seq, seq) — full-ones for full
bidirectional visibility — not a 2D padding mask. A forward pre-hook
drops all-ones 2D masks (informationless) and rejects padded ones with a
clear error instead of the remote validator's opaque failure.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.dream import (
    _register_default_rope_init,
)


class _LLaDA2FusedAttentionBridge(AttentionBridge):
    """Fused query_key_value projection: no separate q/k/v submodules to
    alias — expose the fused output instead."""

    hook_aliases = {
        "hook_qkv": "qkv.hook_out",
        "hook_z": "o.hook_in",
    }


class LLaDA2MoeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for LLaDA2MoeModelLM models."""

    applicable_phases: list[int] = [1, 2, 3]
    supports_generation: bool = False
    # Fused query_key_value with no per-projection conversions to fold into.
    supports_fold_ln = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the LLaDA 2.0 MoE architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()

        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.word_embeddings"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    # Bidirectional diffusion attention with fused QKV; the
                    # bridge reimplementation assumes causal masking, so
                    # delegate to HF. The fused projection and dense output
                    # are hookable; q/k layernorms are full-width.
                    "attn": _LLaDA2FusedAttentionBridge(
                        name="attention",
                        config=self.cfg,
                        submodules={
                            "qkv": LinearBridge(name="query_key_value"),
                            "o": LinearBridge(name="dense"),
                            "q_norm": RMSNormalizationBridge(
                                name="query_layernorm", config=self.cfg
                            ),
                            "k_norm": RMSNormalizationBridge(name="key_layernorm", config=self.cfg),
                        },
                        maintain_native_attention=True,
                    ),
                    # Dense on the first first_k_dense_replace layers, routed
                    # MoE elsewhere — gate and shared_experts are optional so
                    # setup skips them on dense layers (deepseek_v3 pattern).
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": GeneralizedComponent(name="gate", optional=True),
                            "shared_experts": self._gated_mlp(name="shared_experts", optional=True),
                            # Dense-layer projections (absent on MoE layers).
                            "dense_gate": LinearBridge(name="gate_proj", optional=True),
                            "dense_in": LinearBridge(name="up_proj", optional=True),
                            "dense_out": LinearBridge(name="down_proj", optional=True),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Restore the v4 'default' rope init the remote code looks up (Dream shim)."""
        _register_default_rope_init()
        super().prepare_loading(model_name, model_kwargs)

    def setup_hook_compatibility(self, bridge: Any) -> None:
        """Guard the remote forward against auto-passed 2D padding masks."""
        model = getattr(bridge, "original_model", None)
        if model is None or getattr(model, "_llada2_mask_guard", False):
            return

        def _mask_guard(module: Any, args: Any, kwargs: Any) -> Any:
            import torch

            mask = kwargs.get("attention_mask")
            if mask is None:
                # Remote forward calls attention_mask.size() unconditionally;
                # synthesize the full-visibility 4D mask from the input shape.
                ids = kwargs.get("input_ids", args[0] if args else None)
                if isinstance(ids, torch.Tensor) and ids.ndim == 2:
                    batch, seq = ids.shape
                    kwargs["attention_mask"] = torch.ones(batch, 1, seq, seq, device=ids.device)
                return args, kwargs
            if isinstance(mask, torch.Tensor) and mask.ndim == 2:
                if not bool(mask.all()):
                    raise NotImplementedError(
                        "LLaDA2's remote forward rejects 2D padding masks; "
                        "batched padded inputs are unsupported — pass "
                        "equal-length sequences."
                    )
                batch, seq = mask.shape
                kwargs["attention_mask"] = torch.ones(batch, 1, seq, seq, device=mask.device)
            return args, kwargs

        model.register_forward_pre_hook(_mask_guard, with_kwargs=True)
        model._llada2_mask_guard = True

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Delegated attention computes rotary inside HF; nothing to wire."""
