import pytest
import torch

from transformer_lens import FactoredMatrix, utils


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
