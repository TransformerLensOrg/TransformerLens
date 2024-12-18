import pytest
import torch
from torch import randn

from transformer_lens import FactoredMatrix, utils


@pytest.fixture(scope="module")
def random_matrices():
    return [
        (randn(3, 2), randn(2, 3)),
        (randn(4, 5), randn(5, 4)),
        (randn(6, 7), randn(7, 6)),
        (randn(6, 7), randn(7, 3)),
    ]


@pytest.fixture(scope="module")
def factored_matrices(random_matrices):
    return [FactoredMatrix(a, b) for a, b in random_matrices]


def test_get_corner(factored_matrices):
    for factored_matrix in factored_matrices:
        k = 3
        result = utils.get_corner(k)
        expected = utils.get_corner(
            factored_matrix.A[..., :k, :] @ factored_matrix.B[..., :, :k], k
        )
        assert torch.allclose(result, expected)
