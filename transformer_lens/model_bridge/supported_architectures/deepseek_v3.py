"""DeepSeek V3 architecture adapter.

Supports DeepSeek V3 and DeepSeek-R1 models (both use DeepseekV3ForCausalLM).
"""

from transformer_lens.model_bridge.generalized_components import RMSNormalizationBridge
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.deepseek_v2 import (
    DeepSeekMLAFamilyArchitectureAdapter,
)


class DeepSeekV3ArchitectureAdapter(DeepSeekMLAFamilyArchitectureAdapter):
    """Architecture adapter for DeepSeek V3 / R1 models.

    Uses RMSNorm, MLA with compressed Q/KV projections, partial RoPE,
    MoE on most layers (dense MLP on first few), and no biases. HF builds the
    two-stage LoRA Q path only when q_lora_rank is set and a single q_proj
    otherwise (ai-sage/GigaChat3-10B-A1.8B), so both Q paths are mapped optional.
    """

    # HF defaults to SDPA which handles MLA correctly; HF's eager attention
    # crashes on MLA's asymmetric Q/K dimensions.
    _testing_eager = None

    q_lora_optional = True

    def _build_q_a_layernorm(self, optional: bool) -> GeneralizedComponent:
        # Nearly every V3 checkpoint compresses Q: keep the norm's hooks and weight
        # processing instead of downgrading to V2-Lite's plain component.
        return RMSNormalizationBridge(name="q_a_layernorm", config=self.cfg, optional=optional)
