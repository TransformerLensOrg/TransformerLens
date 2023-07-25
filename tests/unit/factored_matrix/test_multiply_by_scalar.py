import random

import pytest
import torch
from torch.testing import assert_close

from transformer_lens import FactoredMatrix


# This test function is parametrized with different types of scalars, including non-scalar tensors and arrays, to check that the correct errors are raised.
# Considers cases with and without leading dimensions as well as left and right multiplication.
@pytest.mark.parametrize(
    "scalar, error_expected",
    [
        # Test cases with different types of scalar values.
        (torch.rand(1), None),  # 1-element Tensor. No error expected.
        (random.random(), None),  # float. No error expected.
        (random.randint(-100, 100), None),  # int. No error expected.
        # Test cases with non-scalar values that are expected to raise errors.
        (
            torch.rand(2, 2),
            AssertionError,
        ),  # Non-scalar Tensor. AssertionError expected.
        (torch.rand(2), AssertionError),  # Non-scalar Tensor. AssertionError expected.
    ],
)
@pytest.mark.parametrize("leading_dim", [False, True])
@pytest.mark.parametrize("multiply_from_left", [False, True])
def test_multiply(scalar, leading_dim, multiply_from_left, error_expected):
    # Prepare a FactoredMatrix, with or without leading dimensions
    if leading_dim:
        a = torch.rand(6, 2, 3)
        b = torch.rand(6, 3, 4)
    else:
        a = torch.rand(2, 3)
        b = torch.rand(3, 4)

    fm = FactoredMatrix(a, b)

    if error_expected:
        # If an error is expected, check that the correct exception is raised.
        with pytest.raises(error_expected):
            if multiply_from_left:
                _ = fm * scalar
            else:
                _ = scalar * fm
    else:
        # If no error is expected, check that the multiplication results in the correct value.
        # Use FactoredMatrix.AB to calculate the product of the two factor matrices before comparing with the expected value.
        if multiply_from_left:
            assert_close((fm * scalar).AB, (a @ b) * scalar)
        else:
            assert_close((scalar * fm).AB, scalar * (a @ b))
        # This next test is implementation dependant and can be broken and removed at any time!
        # It checks that the multiplication is performed on the A factor matrix.
        if multiply_from_left:
            assert_close((fm * scalar).A, a * scalar)
        else:
            assert_close((scalar * fm).A, scalar * a)
