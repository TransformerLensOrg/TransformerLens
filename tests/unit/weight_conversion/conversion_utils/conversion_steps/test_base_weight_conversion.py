import pytest
import torch

from transformer_lens.weight_conversion.conversion_utils.conversion_steps.base_weight_conversion import (
    BaseWeightConversion,
)


class MockWeightConversion(BaseWeightConversion):
    def handle_conversion(self, weight):
        return weight + 5


def test_process_weight_conversion_applies_conversion():
    weight_conversion = MockWeightConversion()
    weight = torch.zeros(2, 2)

    converted_weight = weight_conversion.process_weight_conversion(weight)

    expected = torch.tensor((2, 2)) + 5

    assert torch.all(converted_weight == expected)


def test_process_weight_conversion_applies_filters():
    def mock_input_filter(weight):
        return weight * 2

    def mock_output_filter(weight):
        return weight - 1

    weight_conversion = MockWeightConversion(
        input_filter=mock_input_filter, output_filter=mock_output_filter
    )

    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    converted_weight = weight_conversion.process_weight_conversion(weight)

    expected_weight = (weight * 2) + 5 - 1
    assert torch.allclose(converted_weight, expected_weight)


def test_base_weight_conversion_convert_throws_error():
    weight_conversion = BaseWeightConversion()
    with pytest.raises(NotImplementedError):
        weight_conversion.convert(torch.zeros(1, 4))
