import torch

from transformer_lens.conversion_utils.conversion_steps.zeros_like_conversion import (
    ZerosLikeConversion,
)


def test_zeros_like_basic():
    """
    Ensures that handle_conversion returns a zero tensor
    with the same shape and dtype as the input.
    """
    conversion = ZerosLikeConversion()
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    output_tensor = conversion.convert(input_tensor)

    # 1) Check shape matches
    assert (
        output_tensor.shape == input_tensor.shape
    ), f"Expected shape {input_tensor.shape}, got {output_tensor.shape}"
    # 2) Check dtype matches
    assert (
        output_tensor.dtype == input_tensor.dtype
    ), f"Expected dtype {input_tensor.dtype}, got {output_tensor.dtype}"
    # 3) Check values are zeros
    assert torch.count_nonzero(output_tensor) == 0, "Expected all zeros, found nonzero elements!"
