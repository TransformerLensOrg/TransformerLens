from unittest import mock

import torch

from transformer_lens.weight_conversion.conversion_utils.conversion_steps.arithmetic_weight_conversion import (
    ArithmeticWeightConversion,
    OperationTypes,
)


def test_arithmatic_weight_conversion_addition():
    conversion = ArithmeticWeightConversion("transformer.wpe.weight", OperationTypes.ADDITION, 2)

    starting = torch.ones((1, 1, 1))

    transformers_model = mock.Mock()
    transformers_model.transformer = mock.Mock()
    transformers_model.transformer.wpe = mock.Mock()
    transformers_model.transformer.wpe.weight = starting

    result = conversion.convert(transformers_model)

    assert result[0][0] == 3
