import torch

from transformer_lens.conversion_utils.conversion_steps.repeat_tensor_conversion import (
    RepeatTensorConversion,
)


def test_repeat_tensor_conversion_basic():
    """
    Tests a basic repeat operation on a single tensor, verifying shape
    and data repetition.

    Example pattern (h w -> h 2 w):
    We repeat along the second dimension.
    """
    # Pattern says: "h w -> h repeat1 w"
    # let's do: pattern="h w -> h 2 w", so we define repeat2=2 for the new axis
    conversion = RepeatTensorConversion("h w -> h 2 w", h=3, w=4)

    # Start with a known shape: [3,4], fill with arange to track values
    input_tensor = torch.arange(12.0).reshape(3, 4)  # shape [3,4]
    result = conversion.convert(input_tensor)

    # The new shape should be [3, 2, 4]
    assert result.shape == (3, 2, 4)

    # Because we're repeating along the second dimension with factor=2,
    # each row is duplicated in that dimension:
    # For row 0, we expect the same data repeated in the second dimension.
    # We'll just check a few positions to ensure it repeated:
    for h_idx in range(3):
        for w_idx in range(4):
            val_1 = result[h_idx, 0, w_idx].item()
            val_2 = result[h_idx, 1, w_idx].item()
            assert val_1 == val_2, f"Expected repeated values, got {val_1} and {val_2}."


def test_repeat_tensor_conversion_axes_lengths():
    """
    Tests the repeat pattern with multiple expansions,
    e.g., "h w -> (2 h) (3 w)", verifying shape and data duplication.
    """
    conversion = RepeatTensorConversion("h w -> (repeat_h h) (repeat_w w)", repeat_h=2, repeat_w=3)

    input_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # shape [2,2], h=2, w=2
    result = conversion.convert(input_tensor)

    # The new shape should be [2*2, 3*2] => [4, 6]
    assert result.shape == (4, 6)

    # Each original element is repeated 2 times in the 'h' dimension
    # and 3 times in the 'w' dimension. We'll do a thorough check:
    for h_idx in range(2):
        for w_idx in range(2):
            original_val = input_tensor[h_idx, w_idx].item()

            for rep_h in range(2):
                for rep_w in range(3):
                    h_key = h_idx + rep_h * 2
                    w_key = w_idx + rep_w * 2
                    new_val = result[h_key, w_key].item()
                    assert new_val == original_val, (
                        f"Mismatch at repeated block for input ({h_idx},{w_idx}). "
                        f"Expected {original_val}, got {new_val}."
                    )


def test_repeat_tensor_conversion_input_filter():
    """
    Verifies input_filter is applied before the repeat operation.
    """

    def multiply_by_5(tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 5

    # We'll keep a simple pattern for clarity: "b -> (2 b)"
    # Repeats the existing dimension 2 times.
    conversion = RepeatTensorConversion("b -> (b rep)", input_filter=multiply_by_5, rep=2)

    # input shape [4]
    input_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    # input_filter => [5,10,15,20]
    # repeated => shape [8], each original index repeated once
    # final => [5,5,10,10,15,15,20,20]
    result = conversion.convert(input_tensor)

    print("result", result)
    assert result.shape == (8,)
    expected = torch.tensor([5, 5, 10, 10, 15, 15, 20, 20], dtype=torch.float32)
    assert torch.allclose(
        result, expected
    ), "Input filter or repeat pattern didn't apply correctly."


def test_repeat_tensor_conversion_output_filter():
    """
    Verifies output_filter is applied after the repeat operation.
    """

    def add_ten(tensor: torch.Tensor) -> torch.Tensor:
        return tensor + 10

    # pattern: "h -> 2 h", doubling the first dimension
    conversion = RepeatTensorConversion("h -> 2 h", h=3, output_filter=add_ten)

    input_tensor = torch.arange(3.0)
    # repeated => shape [2,3], each row is the same as input_tensor
    # then +10
    result = conversion.convert(input_tensor)

    assert result.shape == (2, 3)
    # repeated: [[0,1,2],[0,1,2]] => after +10 => [[10,11,12],[10,11,12]]
    expected = torch.tensor([[10, 11, 12], [10, 11, 12]], dtype=torch.float32)
    assert torch.allclose(
        result, expected
    ), "Output filter or repeat pattern didn't apply correctly."


def test_repeat_tensor_conversion_both_filters():
    """
    Tests the pipeline: input_filter -> repeat -> output_filter.
    """

    def input_filter(tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 2  # double

    def output_filter(tensor: torch.Tensor) -> torch.Tensor:
        return tensor - 3  # subtract 3

    # pattern: "c -> 3 c", triple the dimension
    conversion = RepeatTensorConversion(
        "c -> rep c", c=2, rep=3, input_filter=input_filter, output_filter=output_filter
    )

    input_tensor = torch.tensor([2.0, 3.0])
    # Steps:
    #   1) input_filter => [4.0, 6.0]
    #   2) repeat => shape [3,2]: [[4,6],[4,6],[4,6]]
    #   3) output_filter => subtract 3 => [[1,3],[1,3],[1,3]]
    result = conversion.convert(input_tensor)
    expected = torch.tensor([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])
    assert torch.allclose(
        result, expected
    ), "Both filters or repeat pattern not applied correctly in sequence!"


def test_repeat_tensor_conversion_repr():
    """
    Simple test confirming __repr__ returns info about the repeat operation.
    """
    conversion = RepeatTensorConversion("h -> 2 h")
    rep_str = repr(conversion).lower()
    assert "repeat operation" in rep_str, f"Expected 'repeat operation', got {rep_str}"
    assert "pattern" in rep_str, f"Expected mention of 'pattern', got {rep_str}"
