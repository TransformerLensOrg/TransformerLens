"""GLM-MoE-DSA architecture adapter."""

from transformer_lens.model_bridge.generalized_components.glm_moe_dsa_attention import (
    GlmMoeDsaAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.deepseek_v2 import (
    DeepSeekMLAFamilyArchitectureAdapter,
)


class GlmMoeDsaArchitectureAdapter(DeepSeekMLAFamilyArchitectureAdapter):
    """Architecture adapter for Z.ai GLM-5 / GLM-5.1 DSA models.

    GLM-MoE-DSA combines MLA-style latent attention, a learned sparse-attention
    indexer, dense early MLP layers, and sparse MoE later layers. Checkpoints
    always compress Q, so the family's required-Q default holds.
    """

    attention_cls = GlmMoeDsaAttentionBridge
    eager_attention = True
    prepend_bos = False
