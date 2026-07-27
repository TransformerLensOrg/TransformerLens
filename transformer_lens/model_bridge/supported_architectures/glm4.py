"""GLM-4-0414 architecture adapter.

Z.ai's GLM-4-0414 family (``Glm4ForCausalLM``: GLM-4-32B-0414, GLM-Z1):
the dense GLM decoder (adjacent-pair partial RoPE, joint ``gate_up_proj``
MLP) with Gemma-2-style sandwich norms — extra RMS norms applied to the
attention and MLP outputs before their residual adds.
"""

from transformer_lens.model_bridge.generalized_components import RMSNormalizationBridge
from transformer_lens.model_bridge.supported_architectures.glm import (
    GlmArchitectureAdapter,
)


class Glm4ArchitectureAdapter(GlmArchitectureAdapter):
    """Architecture adapter for Glm4ForCausalLM models."""

    def _block_extra_norms(self):
        """Sandwich norms applied before the residual adds."""
        return {
            "ln1_post": RMSNormalizationBridge(name="post_self_attn_layernorm", config=self.cfg),
            "ln2_post": RMSNormalizationBridge(name="post_mlp_layernorm", config=self.cfg),
        }
