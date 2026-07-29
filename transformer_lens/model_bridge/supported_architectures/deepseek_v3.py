"""DeepSeek V3 architecture adapter.

Supports DeepSeek V3 and DeepSeek-R1 models (both use DeepseekV3ForCausalLM).
"""

from transformer_lens.model_bridge.supported_architectures.deepseek_v2 import (
    DeepSeekMLAFamilyArchitectureAdapter,
)


class DeepSeekV3ArchitectureAdapter(DeepSeekMLAFamilyArchitectureAdapter):
    """Architecture adapter for DeepSeek V3 / R1 models.

    Uses RMSNorm, MLA with compressed Q/KV projections, partial RoPE,
    MoE on most layers (dense MLP on first few), and no biases. V3 checkpoints
    always compress Q (q_lora_rank set), so the family's required-Q default holds.
    """

    # HF defaults to SDPA which handles MLA correctly; HF's eager attention
    # crashes on MLA's asymmetric Q/K dimensions.
    _testing_eager = None
