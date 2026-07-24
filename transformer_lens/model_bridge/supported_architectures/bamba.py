"""Bamba (``BambaForCausalLM``) adapter: Jamba-lineage hybrid alternating Mamba-2
and llama-style GQA attention mixers per ``config.layers_block_type``."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedRMSNormBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    SSM2MixerBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.depthwise_conv1d import (
    DepthwiseConv1DBridge,
)


class BambaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for BambaForCausalLM models.

    Both mixers are mapped optional — each present only on its layer type.
    The Mamba-2 mixer is wired under the canonical ``.mixer`` slot (HF path
    ``.mamba``) so SSM analyses reach it as on GraniteMoeHybrid / NemotronH.
    """

    _testing_hybrid = True
    _testing_eager = "config"

    def __init__(self, cfg: Any) -> None:
        """Initialize the Bamba architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # Mamba layers require per-step SSM state; generation is stateful.
        self.cfg.is_stateful = True

        # Normalize the per-layer mixer-type list as cfg.layers_block_type so
        # analysis tools can find the Mamba layers, as on the hybrid siblings.
        setattr(self.cfg, "layers_block_type", self._canonical_layer_types(cfg))

        # Mixed mamba/attention layers: keep raw HF weight layout.
        self.supports_fold_ln = False
        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        optional=True,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    "mixer": SSM2MixerBridge(
                        name="mamba",
                        config=self.cfg,
                        optional=True,
                        submodules={
                            "in_proj": LinearBridge(name="in_proj"),
                            "conv1d": DepthwiseConv1DBridge(name="conv1d"),
                            "inner_norm": GatedRMSNormBridge(name="norm"),
                            "out_proj": LinearBridge(name="out_proj"),
                        },
                    ),
                    "ln2": RMSNormalizationBridge(name="pre_ff_layernorm", config=self.cfg),
                    "mlp": self._gated_mlp(name="feed_forward"),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.final_layernorm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def create_stateful_cache(
        self,
        hf_model: Any,
        batch_size: int,
        device: Any,
        dtype: Any,
    ) -> Any:
        """Unified DynamicCache carrying KV entries and SSM conv/recurrent state."""
        from transformers.cache_utils import DynamicCache

        return DynamicCache(config=hf_model.config)
