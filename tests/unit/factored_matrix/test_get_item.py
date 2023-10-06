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


def test_index_dimension_get_line(sample_factored_matrix):
    result = sample_factored_matrix[0, 0, 0, 1]
    assert_close(result.AB.squeeze(), sample_factored_matrix.AB[0, 0, 0, 1])


def test_index_dimension_get_element(sample_factored_matrix):
    result = sample_factored_matrix[0, 0, 0, 0, 1]
    assert_close(result.AB.squeeze(), sample_factored_matrix.AB[0, 0, 0, 0, 1])


def test_index_dimension_too_big(sample_factored_matrix):
    with pytest.raises(Exception):
        _ = sample_factored_matrix[1, 1, 1, 1, 1, 1]


def test_getitem_sequences(sample_factored_matrix):
    A_idx = [0, 1]
    B_idx = [0]
    result = sample_factored_matrix[:, :, :, A_idx, B_idx]
    assert_close(result.A, sample_factored_matrix.A[:, :, :, A_idx, :])
    assert_close(result.B, sample_factored_matrix.B[:, :, :, :, B_idx])


def test_getitem_sequences_and_ints(sample_factored_matrix):
    A_idx = [0, 1]
    B_idx = 0
    result = sample_factored_matrix[:, :, :, A_idx, B_idx]
    assert_close(result.A, sample_factored_matrix.A[:, :, :, A_idx, :])
    # we squeeze result.B, because indexing by ints is designed not to delete dimensions
    assert_close(result.B.squeeze(-1), sample_factored_matrix.B[:, :, :, :, B_idx])


def test_getitem_tensors(sample_factored_matrix):
    A_idx = torch.tensor([0, 1])
    B_idx = torch.tensor([0])
    result = sample_factored_matrix[:, :, :, A_idx, B_idx]
    assert_close(result.A, sample_factored_matrix.A[:, :, :, A_idx, :])
    assert_close(result.B, sample_factored_matrix.B[:, :, :, :, B_idx])
