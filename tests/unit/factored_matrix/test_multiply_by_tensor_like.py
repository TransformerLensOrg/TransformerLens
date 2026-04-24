"""Tests that FactoredMatrix matmul works with tensor-like objects.

A "tensor-like" object is one that quacks like a torch.Tensor (supports the
operations FactoredMatrix needs — .ndim, .size, .unsqueeze, __matmul__) but
isn't a torch.Tensor subclass. This is useful for things like jaxtyping wrappers
or custom array types.
"""

from torch import randn
from torch.testing import assert_close

from transformer_lens import FactoredMatrix


class TensorLike:
    """A wrapper that exposes the tensor protocol without subclassing torch.Tensor.

    Implements just enough of the protocol that FactoredMatrix can multiply
    with it: matmul, ndim, size, shape, unsqueeze, squeeze, broadcast_to.
    """

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def ndim(self):
        return self._tensor.ndim

    @property
    def shape(self):
        return self._tensor.shape

    def size(self, dim=None):
        return self._tensor.size() if dim is None else self._tensor.size(dim)

    def unsqueeze(self, dim):
        return TensorLike(self._tensor.unsqueeze(dim))

    def squeeze(self, dim):
        return TensorLike(self._tensor.squeeze(dim))

    def broadcast_to(self, shape):
        return TensorLike(self._tensor.broadcast_to(shape))

    def __matmul__(self, other):
        if isinstance(other, FactoredMatrix):
            # Defer to FactoredMatrix.__rmatmul__ so the result is a FactoredMatrix
            return NotImplemented
        if isinstance(other, TensorLike):
            return TensorLike(self._tensor @ other._tensor)
        return TensorLike(self._tensor @ other)

    def __rmatmul__(self, other):
        if isinstance(other, TensorLike):
            return TensorLike(other._tensor @ self._tensor)
        return TensorLike(other @ self._tensor)


def test_left_multiply_factored_matrix_by_tensor_like_matrix():
    """factored_matrix @ tensor_like_matrix should not return None."""
    a = randn(2, 3)
    b = randn(3, 4)
    matrix = randn(4, 5)
    factored_matrix = FactoredMatrix(a, b)

    result = factored_matrix @ TensorLike(matrix)

    assert result is not None, "matmul with tensor-like silently returned None"
    assert isinstance(result, FactoredMatrix)
    expected = (a @ b) @ matrix
    assert isinstance(result.AB, TensorLike)
    assert_close(result.AB._tensor, expected)


def test_right_multiply_factored_matrix_by_tensor_like_matrix():
    """tensor_like_matrix @ factored_matrix should not return None."""
    a = randn(3, 4)
    b = randn(4, 6)
    matrix = randn(5, 3)
    factored_matrix = FactoredMatrix(a, b)

    result = TensorLike(matrix) @ factored_matrix

    assert result is not None, "rmatmul with tensor-like silently returned None"
    assert isinstance(result, FactoredMatrix)
    expected = matrix @ (a @ b)
    assert isinstance(result.AB, TensorLike)
    assert_close(result.AB._tensor, expected)


def test_left_multiply_factored_matrix_by_tensor_like_vector():
    """factored_matrix @ tensor_like_vector should dispatch through the vector path.

    The vector branch of FactoredMatrix.__matmul__ collapses to a single tensor
    via unsqueeze/squeeze rather than wrapping in a new FactoredMatrix. This test
    exercises that path and verifies the TensorLike protocol methods (unsqueeze,
    squeeze, __rmatmul__) are correctly invoked.
    """
    a = randn(2, 3)
    b = randn(3, 4)
    vector = randn(4)
    factored_matrix = FactoredMatrix(a, b)

    result = factored_matrix @ TensorLike(vector)

    # The fix's core guarantee: the dispatch produces a result instead of None
    assert isinstance(result, TensorLike)
    expected = (a @ b) @ vector
    assert_close(result._tensor, expected)
