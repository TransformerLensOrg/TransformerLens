import pytest
import torch
from torch.testing import assert_close

from transformer_lens import FactoredMatrix


def test_left_matmul_by_vector_left():
    a = torch.rand(2, 3)
    b = torch.rand(3, 4)

    fm = FactoredMatrix(a, b)
    vector = torch.rand(4)

    assert_close(fm @ vector, (a @ b) @ vector)


def test_left_matmul_by_vector_leading_dim():
    a = torch.rand(6, 2, 3)
    b = torch.rand(6, 3, 4)

    fm = FactoredMatrix(a, b)
    vector = torch.rand(4)

    assert_close(fm @ vector, (a @ b) @ vector)


@pytest.mark.skip(
    """
    This test fails with
    AssertionError: The values for attribute 'shape' do not match: torch.Size([1, 4]) != torch.Size([4]).

    I think this is a bug in __rmatmul__:
    `return ((other.unsqueeze(-2) @ self.A) @ self.B).squeeze(-1)` is squeezing on the wrong dimension.

    We could fix this by changing the argument to squeeze to `-2`. However there would be a small risk of
    breaking consumers that assume the existing behaviour?
    """
)
def test_right_matmul_by_vector():
    a = torch.rand(2, 3)
    b = torch.rand(3, 4)

    fm = FactoredMatrix(a, b)
    vector = torch.rand(2)

    assert_close(vector @ fm, vector @ (a @ b))


@pytest.mark.skip(
    """
    See comment for [test_right_matmul_by_vector].

    This time, error is
    AssertionError: The values for attribute 'shape' do not match: torch.Size([6, 1, 4]) != torch.Size([6, 4]).

    Fix would be the same.
    """
)
def test_right_matmul_by_vector_leading_dim():
    a = torch.rand(6, 2, 3)
    b = torch.rand(6, 3, 4)

    fm = FactoredMatrix(a, b)
    vector = torch.rand(2)

    assert_close(vector @ fm, vector @ (a @ b))
