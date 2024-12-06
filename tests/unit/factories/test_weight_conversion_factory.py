import pytest
import torch.nn.functional as F

from transformer_lens.factories.weight_conversion_factory import (
    WeightConversionFactory,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.architectures import SUPPORTED_ARCHITECTURES


@pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
def test_select_weight_conversion_config_provides_all(architecture: str):
    config = HookedTransformerConfig.unwrap(
        {"n_layers": 12, "n_ctx": 1024, "d_head": 64, "d_model": 128, "act_fn": "silu"}
    )
    WeightConversionFactory.select_weight_conversion_config(config, architecture)
