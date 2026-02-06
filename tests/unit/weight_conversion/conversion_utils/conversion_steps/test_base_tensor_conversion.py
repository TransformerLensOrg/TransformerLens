import pytest
import torch

from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
)


class MockWeightConversion(BaseTensorConversion):
    def handle_conversion(self, weight):
        return weight + 5


def test_base_tensor_conversion_convert_throws_error():
    weight_conversion = BaseTensorConversion()
    with pytest.raises(NotImplementedError):
        weight_conversion.convert(torch.zeros(1, 4))


def test_mock_weight_conversion_adds_five():
    """
    Verify that the mock subclass adds 5 to every element of the tensor.
    """
    weight_conversion = MockWeightConversion()
    input_tensor = torch.zeros((1, 4), dtype=torch.float32)
    output_tensor = weight_conversion.convert(input_tensor)
    expected_tensor = torch.full((1, 4), 5.0, dtype=torch.float32)

    # Option 1: simple equality check
    assert torch.equal(output_tensor, expected_tensor)

    # Option 2: more robust approximate check
    # torch.testing.assert_close(output_tensor, expected_tensor)


@pytest.mark.parametrize("shape", [(1, 4), (2, 2), (3,)])
def test_mock_weight_conversion_various_shapes(shape):
    """
    Test multiple shapes to ensure .convert() works for different dims.
    """
    weight_conversion = MockWeightConversion()
    input_tensor = torch.zeros(shape)
    output_tensor = weight_conversion.convert(input_tensor)
    expected_tensor = torch.full(shape, 5.0)
    assert torch.equal(output_tensor, expected_tensor)


def test_mock_weight_conversion_empty_tensor():
    """
    Ensure code doesn't crash on an empty tensor.
    """
    weight_conversion = MockWeightConversion()
    input_tensor = torch.zeros((0, 4))
    output_tensor = weight_conversion.convert(input_tensor)
    assert output_tensor.shape == (0, 4)
    # Since shape is (0,4), we can just check shape correctness
