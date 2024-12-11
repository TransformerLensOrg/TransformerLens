from unittest import mock


import torch
from transformer_lens.weight_conversion.conversion_utils.conversion_steps.direct_weight_conversion import DirectWeightConversion


def test_direct_weight_conversion():
    
    conversion = DirectWeightConversion()
    
    expected = torch.rand((1, 1, 1))
    
    result = conversion.convert(expected)
    
    assert result == expected