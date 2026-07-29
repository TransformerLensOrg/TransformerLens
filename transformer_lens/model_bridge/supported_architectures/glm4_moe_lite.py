"""GLM-4 MoE Lite architecture adapter.

Supports the GLM-4.7-Flash family (`Glm4MoeLiteForCausalLM`): DeepSeek-style
Multi-head Latent Attention (LoRA-compressed Q and KV, nope/rope split heads,
interleaved partial RoPE) combined with GLM's sparse MoE — sigmoid router with
e_score_correction_bias, batched routed experts, one shared expert — and a
per-layer dense/sparse MLP mix declared in ``config.mlp_layer_types``.
"""

from transformer_lens.model_bridge.generalized_components import MoERouterBridge
from transformer_lens.model_bridge.supported_architectures.deepseek_v2 import (
    DeepSeekMLAFamilyArchitectureAdapter,
)


class Glm4MoeLiteArchitectureAdapter(DeepSeekMLAFamilyArchitectureAdapter):
    """GLM-4.7-Flash (Glm4MoeLiteForCausalLM) adapter: DeepSeek-V2 MLA + GLM-4-MoE
    routing (dense/sparse per mlp_layer_types)."""

    _testing_eager = None

    # Public GLM-4.7 checkpoints set q_lora_rank — two-stage LoRA Q compression;
    # direct q_proj kept optional for hypothetical uncompressed variants.
    q_lora_optional = True
    # Verified against zai-org/GLM-4.7-Flash: tokenizer has no BOS token.
    prepend_bos = False

    def _build_router(self) -> MoERouterBridge:
        """Sigmoid + e_score_correction_bias router; absent on dense layers."""
        return MoERouterBridge(name="gate", optional=True)
