"""Nemotron-H hybrid Mamba2-Transformer architecture adapter.

Supports NemotronHForCausalLM (e.g. nvidia/NVIDIA-Nemotron-Nano-9B-v2, Nemotron-3 series).

Architecture overview:
- Heterogeneous layers defined by ``config.layers_block_type`` — each element is
  one of ``"mamba"``, ``"attention"``, ``"moe"``, or ``"mlp"``.
- ~8% of layers are standard GQA attention; the rest are Mamba-2 SSM, dense MLP,
  or sparse MoE. All share a single pre-norm (``block.norm``) and a single residual
  path; there is no ``ln2`` or post-attention norm.
- Each block exposes a single ``.mixer`` attribute whose type varies by layer.
- No model-level rotary embedding module — attention handles RoPE internally via
  ``position_ids`` passed from the outer model loop.
- Stateful generation: uses ``DynamicCache`` (transformers ≥ 5.12) which carries
  both KV-cache entries (attention layers) and SSM conv/recurrent states
  (Mamba layers) in a unified object.

Key adapter decisions:
- ``SSMBlockBridge`` is used as the block container. It delegates the entire
  forward to the HF block, giving ``hook_in`` / ``hook_out`` on the residual
  stream without hardcoding transformer-specific hook positions (hook_resid_mid,
  hook_mlp_in, etc.) that do not exist in this single-norm architecture.
- ``SSM2MixerBridge`` wraps ``.mixer`` for all layer types. Its forward is a
  pure passthrough (``original_component(*args, **kwargs)``) so it works
  correctly for attention, MLP, and MoE mixers as well as Mamba ones.
  Mamba-specific inner submodules (in_proj, conv1d, inner_norm, out_proj) are
  declared ``optional=True`` so setup skips them gracefully on non-Mamba layers.
- MLP layers use ``relu2`` activation (not SwiGLU); ``gated_mlp = False``.
- ``applicable_phases = [1, 2, 3, 4]``: P1 is exact vs raw HF (passthrough mixers);
  P2/P3 skip without a HookedTransformer; P4 is generation.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    GatedRMSNormBridge,
    LinearBridge,
    RMSNormalizationBridge,
    SSM2MixerBridge,
    SSMBlockBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


def _make_optional(component: "GeneralizedComponent") -> "GeneralizedComponent":
    """Mark a GeneralizedComponent submodule as optional.

    Some bridge classes (e.g. GatedRMSNormBridge) do not forward ``optional``
    through their own ``__init__``, even though ``GeneralizedComponent`` supports
    it. Setting the attribute directly is safe because ``component_setup.py``
    reads ``getattr(submodule, 'optional', False)`` at setup time.
    """
    component.optional = True
    return component


class NemotronHArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for NemotronHForCausalLM.

    Hybrid Mamba-2 + Attention + MoE + dense MLP model. All layers share a
    single pre-norm and a single residual connection; the mixer type per layer
    is determined by ``config.layers_block_type[layer_idx]``.
    """

    # White-box forward: P1 is exact vs raw HF (passthrough mixers); P2/P3 skip
    # without a HookedTransformer; P4 is generation.
    applicable_phases: list[int] = [1, 2, 3, 4]

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.uses_rms_norm = True
        # No model-level rotary embedding module — attention handles RoPE
        # internally via position_ids; set to "none" so the bridge does not
        # attempt to wire a rotary_emb component.
        self.cfg.positional_embedding_type = "none"
        # MLP layers use relu2 (up_proj → act → down_proj), not SwiGLU.
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.final_rms = True
        # Mamba layers require per-step SSM state; generation is stateful.
        self.cfg.is_stateful = True

        # Normalize the per-layer type list as cfg.layers_block_type (HF names it
        # `layer_types`) so analysis tools can find the Mamba layers, as on Granite.
        layers_block_type = (
            getattr(cfg, "layers_block_type", None) or getattr(cfg, "layer_types", None) or []
        )
        setattr(self.cfg, "layers_block_type", list(layers_block_type))

        # Mamba-2 dimensional config (mirrors Mamba2ArchitectureAdapter).
        mamba_num_heads = getattr(cfg, "mamba_num_heads", 128)
        mamba_head_dim = getattr(cfg, "mamba_head_dim", 64)
        mamba_intermediate_size = mamba_num_heads * mamba_head_dim
        n_groups = getattr(cfg, "n_groups", 8)
        ssm_state_size = getattr(cfg, "ssm_state_size", 128)
        conv_dim = mamba_intermediate_size + 2 * n_groups * ssm_state_size
        setattr(self.cfg, "mamba_intermediate_size", mamba_intermediate_size)
        setattr(self.cfg, "conv_dim", conv_dim)

        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embeddings"),
            "blocks": SSMBlockBridge(
                name="model.layers",
                submodules={
                    # Single pre-norm shared across all layer types.
                    "norm": RMSNormalizationBridge(name="norm", config=self.cfg),
                    # Single mixer slot — type varies per layer (mamba / attention
                    # / moe / mlp). SSM2MixerBridge.forward() is a pure
                    # passthrough so it works for all four types. Mamba-specific
                    # inner submodules are optional and skipped on other types.
                    "mixer": SSM2MixerBridge(
                        name="mixer",
                        config=self.cfg,
                        submodules={
                            # ── Mamba-only (optional on attention / moe / mlp) ──
                            "in_proj": LinearBridge(name="in_proj", optional=True),
                            "conv1d": DepthwiseConv1DBridge(name="conv1d", optional=True),
                            # HF names this "norm" inside the mixer; TL calls it
                            # "inner_norm" to avoid collision with the block-level norm.
                            # GatedRMSNormBridge.__init__ does not accept optional=, so
                            # we set the attribute directly after construction.
                            "inner_norm": _make_optional(GatedRMSNormBridge(name="norm")),
                            "out_proj": LinearBridge(name="out_proj", optional=True),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def create_stateful_cache(
        self,
        hf_model: Any,
        batch_size: int,
        device: Any,
        dtype: Any,
    ) -> Any:
        """Build the unified DynamicCache for stateful generation.

        Transformers ≥ 5.12 ships a unified ``DynamicCache`` that carries both
        KV-cache entries (attention layers) and SSM conv/recurrent states
        (Mamba layers) in a single object, using ``has_previous_state()`` to
        distinguish which state is available for a given layer index. The
        config is required so the cache knows each layer's type — matching
        NemotronHModel's own initialization.
        """
        from transformers.cache_utils import DynamicCache

        return DynamicCache(config=hf_model.config)
