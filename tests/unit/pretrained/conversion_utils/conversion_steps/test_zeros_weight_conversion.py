from unittest import mock

import torch
from transformer_lens.pretrained.conversion_utils.conversion_steps.zeros_weight_conversion import ZerosWeightConversion


def test_base_weight_conversion_convert_throws_error():
    
    conversion = ZerosWeightConversion(1, 2, 8)
    
    expected = torch.zeros(1, 2, 8)
    
    transformers_model = mock.Mock()
    
    result = conversion.convert(transformers_model)
    
    assert torch.equal(expected, result)