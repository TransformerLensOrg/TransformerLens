from unittest import mock


import torch
from transformer_lens.weight_conversion.conversion_utils.conversion_steps.rearrange_weight_conversion import RearrangeWeightConversion


def test_rearrange_weight_conversion():
    
    conversion = RearrangeWeightConversion("(n h) m->n m h", n=8)
    
    starting = torch.rand(80, 5)
    
    result = conversion.convert(starting)
    
    assert result.shape == (8, 5, 10)