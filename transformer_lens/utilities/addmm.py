"""Addmm

Implementations of Addmm functions matching Huggingface implementations.
"""
import torch
from jaxtyping import Float


def vanilla_addmm(
    input: Float[torch.Tensor, "... #o"],  # Must be broadcastable to "m o"
    mat1: Float[torch.Tensor, "m n"],
    mat2: Float[torch.Tensor, "n o"],
) -> Float[torch.Tensor, "m o"]:
    """Typechecked version of torch.addmm.

    Note that both mat1 and mat2 *must* be 2d matrices.
    """
    return torch.addmm(input, mat1, mat2)


def batch_addmm(
    bias: Float[torch.Tensor, "... #d_out"],  # Must be broadcastable to "... d_out"
    weight: Float[torch.Tensor, "d_in d_out"],
    x: Float[torch.Tensor, "... d_in"],
) -> Float[torch.Tensor, "... d_out"]:
    """Fused add-multiply with support for batch dimensions.

    Must match the Huggingface Conv1D implementation exactly.
    https://github.com/huggingface/transformers/blob/9ba9369a2557e53a01378199a9839ec6e82d8bc7/src/transformers/pytorch_utils.py#L102-L106
    """
    n_output_features = weight.shape[-1]
    size_out = x.size()[:-1] + (n_output_features,)
    x = vanilla_addmm(bias, x.view(-1, x.size(-1)), weight)
    x = x.view(size_out)
    return x
