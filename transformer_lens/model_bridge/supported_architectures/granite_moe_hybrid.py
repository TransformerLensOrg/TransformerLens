"""Granite MoE Hybrid architecture adapter.

GraniteMoeHybridForCausalLM is a hybrid Mamba + Attention architecture with
Sparse Mixture of Experts. Layers alternate between Mamba SSM blocks and
standard attention blocks, with a shared MLP and optional sparse MoE on
every layer.

Since self_attn is None on Mamba layers and mamba is None on attention
layers, we only map submodules that exist on ALL layers (norms, shared_mlp,
block_sparse_moe). The HF native forward handles mamba/attention dispatch.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.granite import (
    GraniteArchitectureAdapter,
)


class GraniteMoeHybridArchitectureAdapter(GraniteArchitectureAdapter):
    """Architecture adapter for IBM Granite MoE Hybrid models.

    Hybrid Mamba2 + Attention architecture with Sparse MoE. Most layers are Mamba
    SSM blocks; a few are standard attention (determined by config.layer_types).

    Since self_attn is None on Mamba layers and mamba is None on attention layers,
    we only map submodules present on ALL layers (norms, shared_mlp, MoE). The HF
    native forward handles mamba/attention dispatch internally.

    Hook coverage:
    - Block-level: hook_resid_pre, hook_resid_post on every layer
    - Normalization: ln1 (input_layernorm), ln2 (post_attention_layernorm)
    - MLP: shared_mlp input/output hooks
    - MoE: block_sparse_moe input/output and router_scores hooks
    - Attention/Mamba internals are NOT individually hooked (conditional per layer)
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Granite MoE Hybrid architecture adapter."""
        # Call ArchitectureAdapter.__init__ directly, not GraniteArchitectureAdapter.__init__,
        # because we need to customize the setup sequence
        ArchitectureAdapter.__init__(self, cfg)

        self._setup_common_config(cfg)

        # Hybrid may use "rope" or "nope" (no positional embeddings)
        pos_emb_type = getattr(cfg, "position_embedding_type", "rope")
        if pos_emb_type != "rope":
            self.cfg.positional_embedding_type = "none"

        # No attention weight conversions — attn Q/K/V aren't mapped as submodules
        self.weight_processing_conversions = {}
        self.component_mapping = self._build_component_mapping()

    def _build_component_mapping(self) -> dict:
        """Build component mapping with only universal (all-layer) submodules."""
        block_submodules = {
            "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
            "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
            "shared_mlp": MLPBridge(
                name="shared_mlp",
                config=self.cfg,
                submodules={
                    "in": LinearBridge(name="input_linear"),
                    "out": LinearBridge(name="output_linear"),
                },
            ),
        }

        num_experts = getattr(self.cfg, "num_experts", None) or getattr(
            self.cfg, "num_local_experts", 0
        )
        if num_experts > 0:
            block_submodules["moe"] = MoEBridge(
                name="block_sparse_moe",
                config=self.cfg,
            )

        mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules=block_submodules,
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

        if self.cfg.positional_embedding_type == "rotary":
            mapping["rotary_emb"] = RotaryEmbeddingBridge(
                name="model.rotary_emb", config=self.cfg
            )

        return mapping

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """No-op for hybrid models.

        Hybrid models don't map attention as a submodule (it's conditional per
        layer), so there are no rotary embedding references to set up.
        """
