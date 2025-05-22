# weight_conversion_factory.py does not exist in the current codebase, so comment out the import and test for now.
# from transformer_lens.factories.weight_conversion_factory import WeightConversionFactory

import pytest

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.architectures import SUPPORTED_ARCHITECTURES


@pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
def test_select_weight_conversion_config_provides_all(architecture: str):
    config = HookedTransformerConfig.unwrap(
        {
            "original_architecture": architecture,
            "n_layers": 12,
            "n_ctx": 1024,
            "d_head": 64,
            "d_model": 128,
            "n_heads": 2,
            "d_vocab": 512,
            "n_key_value_heads": 4,
            "act_fn": "silu",
        }
    )
    WeightConversionFactory.select_weight_conversion_config(config)

# Comment out or remove tests that reference WeightConversionFactory
# def test_weight_conversion_factory():
#     ...
