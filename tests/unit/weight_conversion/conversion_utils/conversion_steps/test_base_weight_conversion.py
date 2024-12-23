import pytest
import torch

from transformer_lens.weight_conversion.conversion_utils.conversion_steps.base_weight_conversion import (
    BaseWeightConversion,
)


def test_base_weight_conversion_convert_throws_error():
    weight_conversion = BaseWeightConversion()
    with pytest.raises(Exception):
        weight_conversion.convert(torch.zeros(1, 4))
