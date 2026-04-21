import pytest
import torch

from transformer_lens.conversion_utils.conversion_steps.arithmetic_tensor_conversion import (
    ArithmeticTensorConversion,
    OperationTypes,
)


@pytest.mark.parametrize(
    "operation, value, input_tensor, expected_output",
    [
        (OperationTypes.ADDITION, 2, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 4.0, 5.0])),
        (
            OperationTypes.SUBTRACTION,
            1,
            torch.tensor([5.0, 6.0, 7.0]),
            torch.tensor([4.0, 5.0, 6.0]),
        ),
        (
            OperationTypes.MULTIPLICATION,
            3,
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([3.0, 6.0, 9.0]),
        ),
        (OperationTypes.DIVISION, 2, torch.tensor([4.0, 6.0, 8.0]), torch.tensor([2.0, 3.0, 4.0])),
    ],
)
def test_arithmetic_operations(operation, value, input_tensor, expected_output):
    conversion = ArithmeticTensorConversion(operation, value)
    output = conversion.convert(input_tensor)
    assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"


def test_scalar_operations():
    conversion = ArithmeticTensorConversion(OperationTypes.MULTIPLICATION, 10)
    assert conversion.convert(5) == 50


def test_tensor_operations():
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    conversion = ArithmeticTensorConversion(OperationTypes.ADDITION, torch.tensor([1.0, 1.0, 1.0]))
    expected_output = torch.tensor([2.0, 3.0, 4.0])
    assert torch.allclose(conversion.convert(input_tensor), expected_output)


def test_input_filter():
    def input_filter(x):
        return x * 2  # Double the input before applying the operation

    conversion = ArithmeticTensorConversion(OperationTypes.ADDITION, 3, input_filter=input_filter)
    assert conversion.convert(torch.tensor(2.0)) == 7.0  # (2 * 2) + 3 = 7


def test_output_filter():
    def output_filter(x):
        return x / 2  # Halve the result after applying the operation

    conversion = ArithmeticTensorConversion(OperationTypes.ADDITION, 3, output_filter=output_filter)
    assert conversion.convert(torch.tensor(2.0)) == 2.5  # (2 + 3) / 2 = 2.5
