"""OLMo 3 architecture adapter."""

from transformer_lens.model_bridge.supported_architectures.olmo2 import (
    Olmo2ArchitectureAdapter,
)


class Olmo3ArchitectureAdapter(Olmo2ArchitectureAdapter):
    """Architecture adapter for OLMo 3 / OLMo 3.1 models.

    OLMo 3 is architecturally identical to OLMo 2 at the weight and component level.
    The only difference is sliding window attention on some layers (configurable via
    layer_types), which is handled by the HF model's forward pass (mask creation)
    and does not affect weight structure or component mapping.
    """

    pass
