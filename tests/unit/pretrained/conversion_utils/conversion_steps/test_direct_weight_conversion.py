from unittest import mock


import torch
from transformer_lens.pretrained.conversion_utils.conversion_steps.direct_weight_conversion import DirectWeightConversion


def test_base_weight_conversion_convert_throws_error():
    
    conversion = DirectWeightConversion("transformer.wpe.weight")
    
    expected = torch.rand((1, 1, 1))
    
    transformers_model = mock.Mock()
    transformers_model.transformer = mock.Mock()
    transformers_model.transformer.wpe = mock.Mock()
    transformers_model.transformer.wpe.weight = expected
    
    result = conversion.convert(transformers_model)
    
    assert result == expected