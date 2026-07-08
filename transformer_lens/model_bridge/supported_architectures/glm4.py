"""GLM-4-0414 architecture adapter.

Z.ai's GLM-4-0414 family (``Glm4ForCausalLM``: GLM-4-32B-0414, GLM-Z1):
the dense GLM decoder (adjacent-pair partial RoPE, joint ``gate_up_proj``
MLP) with Gemma-2-style sandwich norms — extra RMS norms applied to the
attention and MLP outputs before their residual adds.
"""

from typing import Any

from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    JointGateUpMLPBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.supported_architectures.glm import (
    GlmArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.phi3 import (
    Phi3ArchitectureAdapter,
)


class Glm4ArchitectureAdapter(GlmArchitectureAdapter):
    """Architecture adapter for Glm4ForCausalLM models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GLM-4-0414 architecture adapter."""
        super().__init__(cfg)
        # pre-MLP norm; the sandwich norms are post_self_attn_layernorm and
        # post_mlp_layernorm applied before the residual adds.
        self.components["blocks"] = BlockBridge(
            name="model.layers",
            submodules={
                "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                "ln1_post": RMSNormalizationBridge(
                    name="post_self_attn_layernorm", config=self.cfg
                ),
                "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                "ln2_post": RMSNormalizationBridge(name="post_mlp_layernorm", config=self.cfg),
                "attn": PositionEmbeddingsAttentionBridge(
                    name="self_attn",
                    config=self.cfg,
                    submodules={
                        "q": LinearBridge(name="q_proj"),
                        "k": LinearBridge(name="k_proj"),
                        "v": LinearBridge(name="v_proj"),
                        "o": LinearBridge(name="o_proj"),
                    },
                    requires_attention_mask=True,
                    requires_position_embeddings=True,
                ),
                "mlp": JointGateUpMLPBridge(
                    name="mlp",
                    config=self.cfg,
                    split_gate_up_matrix=Phi3ArchitectureAdapter._split_gate_up,
                    submodules={
                        "out": LinearBridge(name="down_proj"),
                    },
                ),
            },
        )
