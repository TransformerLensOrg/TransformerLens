import pytest
import torch
from torch.testing import assert_close

from transformer_lens import FactoredMatrix

@pytest.mark.parametrize(
    "scalar", 
    [
        torch.rand(1), 
        float(torch.rand(1).item()), 
        int(torch.randint(1, 10, (1,)).item())
    ]
)
@pytest.mark.parametrize(
    "leading_dim", 
    [
        False, 
        True
    ]
)
@pytest.mark.parametrize(
    "multiply_from_left", 
    [
        False, 
        True
    ]
)
def test_multiply_by_scalar(scalar, leading_dim, multiply_from_left):
    if leading_dim:
        a = torch.rand(6, 2, 3)
        b = torch.rand(6, 3, 4)
    else:
        a = torch.rand(2, 3)
        b = torch.rand(3, 4)

    fm = FactoredMatrix(a, b)

    if multiply_from_left:
        assert_close((fm * scalar).AB, (a @ b) * scalar)
    else:
        assert_close((scalar * fm).AB, (a @ b) * scalar)
