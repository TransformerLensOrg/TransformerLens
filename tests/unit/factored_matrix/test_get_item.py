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


def test_index_dimension_get_line_negative(sample_factored_matrix):
    # Negative index into the row (ldim) of the matrix. `idx == -1` previously
    # produced an empty slice(-1, 0) and returned a (0, ...) tensor.
    result = sample_factored_matrix[0, 0, 0, -1]
    assert_close(result.AB.squeeze(), sample_factored_matrix.AB[0, 0, 0, -1])


def test_index_dimension_get_element_negative(sample_factored_matrix):
    # Negative index into the column (rdim) of the matrix.
    result = sample_factored_matrix[0, 0, 0, 0, -1]
    assert_close(result.AB.squeeze(), sample_factored_matrix.AB[0, 0, 0, 0, -1])


def test_index_dimension_get_element_both_negative(sample_factored_matrix):
    # Negative index into both matrix dimensions at once.
    result = sample_factored_matrix[0, 0, 0, -1, -1]
    assert_close(result.AB.squeeze(), sample_factored_matrix.AB[0, 0, 0, -1, -1])


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


@pytest.mark.parametrize("leading_shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize(
    "index,matrix_index",
    [
        ((Ellipsis,), (slice(None), slice(None))),
        ((Ellipsis, 1), (slice(None), slice(1, 2))),
        ((Ellipsis, -1), (slice(None), slice(-1, None))),
        ((Ellipsis, 1, slice(None)), (slice(1, 2), slice(None))),
        ((Ellipsis, -1, slice(None)), (slice(-1, None), slice(None))),
        ((Ellipsis, slice(1, 4), slice(2, 6)), (slice(1, 4), slice(2, 6))),
        ((None, Ellipsis, slice(1, 4), slice(None)), (slice(1, 4), slice(None))),
        ((Ellipsis, torch.tensor([1, 4])), (slice(None), torch.tensor([1, 4]))),
        ((Ellipsis, torch.tensor([1, 3]), slice(None)), (torch.tensor([1, 3]), slice(None))),
    ],
    ids=[
        "identity",
        "column",
        "last-column",
        "row",
        "last-row",
        "corner",
        "newaxis",
        "column-tensor",
        "row-tensor",
    ],
)
def test_getitem_ellipsis_matches_dense(
    leading_shape: tuple[int, ...], index: tuple, matrix_index: tuple
) -> None:
    # Unequal row, rank, and column sizes expose indexing into the hidden rank axis.
    matrix = FactoredMatrix(
        torch.randn(*leading_shape, 5, 2, dtype=torch.float64),
        torch.randn(*leading_shape, 2, 7, dtype=torch.float64),
    )
    prefix = (slice(None),) * len(leading_shape)
    if index[0] is None:
        prefix = (None,) + prefix
    expected = matrix.AB[prefix + matrix_index]

    result = matrix[index]

    assert isinstance(result, FactoredMatrix)
    assert result.mdim == matrix.mdim
    assert_close(result.AB, expected)


def test_getitem_ellipsis_between_tensor_and_column() -> None:
    matrix = FactoredMatrix(torch.randn(3, 5, 2), torch.randn(3, 2, 7))
    leading_index = torch.tensor([2, 0])
    result = matrix[leading_index, ..., 1]
    assert_close(result.AB, matrix.AB[leading_index, :, 1:2])


def test_getitem_zero_width_ellipsis() -> None:
    matrix = FactoredMatrix(torch.randn(5, 2), torch.randn(2, 7))
    result = matrix[1, ..., 2]
    assert_close(result.AB, matrix.AB[1:2, 2:3])


def test_getitem_multiple_ellipses_raises(sample_factored_matrix) -> None:
    with pytest.raises(IndexError):
        _ = sample_factored_matrix[..., ...]


def test_getitem_ellipsis_too_many_indices_raises() -> None:
    matrix = FactoredMatrix(torch.randn(5, 2), torch.randn(2, 7))
    with pytest.raises(ValueError, match="too long an index"):
        _ = matrix[..., 0, 0, 0]


def test_getitem_ellipsis_does_not_materialize_product(monkeypatch: pytest.MonkeyPatch) -> None:
    matrix = FactoredMatrix(torch.randn(3, 5, 2), torch.randn(3, 2, 7))
    expected = matrix.AB[..., 1:4, 2:6]

    def forbid_product(self):
        raise AssertionError("Indexing must not materialize the product")

    with monkeypatch.context() as context:
        context.setattr(FactoredMatrix, "AB", property(forbid_product))
        result = matrix[..., 1:4, 2:6]
    assert_close(result.AB, expected)


@pytest.mark.parametrize("mask_ndim", [1, 2], ids=["one-axis", "two-axes"])
@pytest.mark.parametrize("mask_format", ["tensor", "numpy", "list", "tuple"])
@pytest.mark.parametrize("selection", ["leading", "column", "row", "explicit-column", "newaxis"])
def test_getitem_bool_mask_matches_dense(mask_ndim: int, mask_format: str, selection: str) -> None:
    matrix = FactoredMatrix(
        torch.randn(2, 3, 5, 2, dtype=torch.float64),
        torch.randn(2, 3, 2, 7, dtype=torch.float64),
    )
    mask = (
        torch.tensor([True, False])
        if mask_ndim == 1
        else torch.tensor([[True, False, True], [False, True, False]])
    )
    selected = matrix.AB[mask]
    if mask_format == "numpy":
        mask = mask.numpy()
    elif mask_format == "list":
        mask = mask.tolist()
    elif mask_format == "tuple":
        values = mask.tolist()
        mask = tuple(tuple(row) for row in values) if mask_ndim == 2 else tuple(values)
    if selection == "leading":
        index = (mask, Ellipsis)
        expected = selected
    elif selection == "column":
        index = (mask, Ellipsis, 1)
        expected = selected[..., 1:2]
    elif selection == "row":
        index = (mask, Ellipsis, 1, slice(None))
        expected = selected[..., 1:2, :]
    elif selection == "explicit-column":
        index = (mask,) + (slice(None),) * (3 - mask_ndim) + (1,)
        expected = selected[..., 1:2]
    else:
        index = (None, mask, Ellipsis, 1)
        expected = selected[None, ..., 1:2]

    result = matrix[index]

    assert result.mdim == matrix.mdim
    assert_close(result.AB, expected)


@pytest.mark.parametrize("numpy_mask", [False, True], ids=["tensor", "numpy"])
def test_getitem_ellipsis_preserves_legacy_byte_mask(numpy_mask: bool) -> None:
    matrix = FactoredMatrix(torch.randn(2, 3, 5, 2), torch.randn(2, 3, 2, 7))
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.uint8)
    expected = matrix.AB[mask.bool()]
    if numpy_mask:
        mask = mask.numpy()
    with pytest.warns(UserWarning, match="uint8"):
        result = matrix[mask, ...]
    assert_close(result.AB, expected)


@pytest.mark.parametrize("nested", [False, True], ids=["one-dimensional", "two-dimensional"])
@pytest.mark.parametrize("as_tuple", [False, True], ids=["list", "tuple"])
def test_getitem_integer_sequence_consumes_one_axis(nested: bool, as_tuple: bool) -> None:
    matrix = FactoredMatrix(torch.randn(2, 3, 5, 2), torch.randn(2, 3, 2, 7))
    index = [[1, 0], [0, 1]] if nested else [1, 0]
    expected = matrix.AB[torch.tensor(index), ..., 1:2]
    if as_tuple:
        index = tuple(tuple(row) for row in index) if nested else tuple(index)
    result = matrix[index, ..., 1]
    assert_close(result.AB, expected)
