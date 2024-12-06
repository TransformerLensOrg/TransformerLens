from unittest import mock


import torch
from transformer_lens.pretrained.conversion_utils.conversion_steps.rearrange_weight_conversion import RearrangeWeightConversion


def test_rearrange_weight_conversion():
    
    conversion = RearrangeWeightConversion("transformer.wpe.weight", "(n h) m->n m h", n=8)
    
    starting = torch.rand(80, 5)
    
    transformers_model = mock.Mock()
    transformers_model.transformer = mock.Mock()
    transformers_model.transformer.wpe = mock.Mock()
    transformers_model.transformer.wpe.weight = starting
    
    result = conversion.convert(transformers_model)
    
    assert result.shape == (8, 5, 10)