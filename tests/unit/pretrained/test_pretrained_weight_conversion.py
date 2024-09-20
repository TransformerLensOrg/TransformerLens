from unittest import mock

import torch
from transformer_lens.pretrained.conversion_types.direct_conversion import DirectConversion

def test_weight_conversion_with_string_path():
    conversion = DirectConversion("transformer.wpe.weight")
    
    expected = torch.rand((1, 1, 1))
    
    transformers_model = mock.Mock()
    transformers_model.transformer = mock.Mock()
    transformers_model.transformer.wpe = mock.Mock()
    transformers_model.transformer.wpe.weight = expected
    
    result = conversion.convert(transformers_model)
    
    assert result == expected
    