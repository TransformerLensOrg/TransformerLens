import pytest
import torch

from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import (
    RearrangeTensorConversion,
)


def test_rearrange_tensor_conversion_basic():
    """
    Basic test: rearranging (n h) m->n m h with n=8.
    Verifies shape, element count, and correctness of mapping.
    """
    conversion = RearrangeTensorConversion("(n h) m->n m h", n=8)
    starting = torch.arange(80 * 5, dtype=torch.float32).reshape(80, 5)

    result = conversion.convert(starting)

    # Check shape
    assert result.shape == (8, 5, 10)

    # Check total number of elements is the same
    assert result.numel() == starting.numel()

    # Verify each element's new position
    # pattern = "(n h) m->n m h", with n=8 => h=80/8=10
    for i in range(80):
        for j in range(5):
            # new index: (i//10, j, i%10)
            expected_value = float(i * 5 + j)  # Because starting is arange(80*5)
            new_val = result[i // 10, j, i % 10].item()
            assert new_val == pytest.approx(expected_value), (
                f"Incorrect rearrangement: expected {expected_value}, "
                f"got {new_val} at new index ({i//10}, {j}, {i%10})."
            )


def test_rearrange_tensor_conversion_input_filter():
    """
    Tests that the input_filter is applied correctly before rearrange.
    """

    def input_filter(tensor):
        # E.g., multiply by 2
        return tensor * 2

    conversion = RearrangeTensorConversion("(n h) m->n m h", input_filter=input_filter, n=8)
    starting = torch.arange(80 * 5, dtype=torch.float32).reshape(80, 5)
    result = conversion.convert(starting)

    # After filter, the input to einops is starting*2, so each element is doubled.
    # So the expected value in new index is (original_index_value * 2).
    assert result.shape == (8, 5, 10)
    for i in range(80):
        for j in range(5):
            expected_value = float(i * 5 + j) * 2
            new_val = result[i // 10, j, i % 10].item()
            assert new_val == pytest.approx(expected_value), (
                f"Incorrect rearrangement or input_filter application: "
                f"expected {expected_value}, got {new_val}"
            )


def test_rearrange_tensor_conversion_output_filter():
    """
    Tests that the output_filter is applied to the rearranged output.
    """

    def output_filter(tensor):
        # E.g., add 10
        return tensor + 10

    conversion = RearrangeTensorConversion("(n h) m->n m h", output_filter=output_filter, n=8)
    starting = torch.arange(80 * 5, dtype=torch.float32).reshape(80, 5)
    result = conversion.convert(starting)

    # The rearrangement hasn't changed, but the entire result is +10
    # from the original rearranged values.
    assert result.shape == (8, 5, 10)
    for i in range(80):
        for j in range(5):
            # rearranged index
            rearranged_val = float(i * 5 + j)
            # after output_filter
            expected_value = rearranged_val + 10
            new_val = result[i // 10, j, i % 10].item()
            assert new_val == pytest.approx(expected_value), (
                f"Incorrect rearrangement or output_filter application: "
                f"expected {expected_value}, got {new_val}"
            )


def test_rearrange_tensor_conversion_input_output_filters():
    """
    Tests that both input_filter and output_filter are applied in sequence.
    """

    def input_filter(tensor):
        return tensor * 2  # Double

    def output_filter(tensor):
        return tensor + 3  # Then add 3

    conversion = RearrangeTensorConversion(
        "(n h) m->n m h", input_filter=input_filter, output_filter=output_filter, n=8
    )
    starting = torch.arange(80 * 5, dtype=torch.float32).reshape(80, 5)
    result = conversion.convert(starting)

    # Expect rearranged_val = (original_index_value * 2) + 3
    assert result.shape == (8, 5, 10)
    for i in range(80):
        for j in range(5):
            base_val = float(i * 5 + j)
            expected_value = (base_val * 2) + 3
            new_val = result[i // 10, j, i % 10].item()
            assert new_val == pytest.approx(
                expected_value
            ), f"Incorrect arrangement or filter chain: expected {expected_value}, got {new_val}"
