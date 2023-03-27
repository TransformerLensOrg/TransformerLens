import pytest
import torch
from torch.testing import assert_close

from transformer_lens import FactoredMatrix


@pytest.fixture
def sample_factored_matrix():
    A = torch.rand(2, 2, 2, 2, 2)
    B = torch.rand(2, 2, 2, 2, 2)
    return FactoredMatrix(A, B)


def test_getitem_int(sample_factored_matrix):
    result = sample_factored_matrix[0]
    assert_close(result.A, sample_factored_matrix.A[0])
    assert_close(result.B, sample_factored_matrix.B[0])


def test_getitem_tuple(sample_factored_matrix):
    result = sample_factored_matrix[(0, 1)]
    assert_close(result.A, sample_factored_matrix.A[0, 1])
    assert_close(result.B, sample_factored_matrix.B[0, 1])


def test_getitem_slice(sample_factored_matrix):
    result = sample_factored_matrix[:, 1]
    assert_close(result.A, sample_factored_matrix.A[:, 1])
    assert_close(result.B, sample_factored_matrix.B[:, 1])


def test_getitem_error(sample_factored_matrix):
    with pytest.raises(IndexError):
        _ = sample_factored_matrix[(0, 1, 2)]


def test_getitem_multiple_slices(sample_factored_matrix):
    result = sample_factored_matrix[:, :, 1]
    assert_close(result.A, sample_factored_matrix.A[:, :, 1])
    assert_close(result.B, sample_factored_matrix.B[:, :, 1])
