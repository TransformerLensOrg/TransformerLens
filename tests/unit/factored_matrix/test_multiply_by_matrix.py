from abc import ABC, abstractmethod

import pytest
from einops import repeat
from torch import randn
from torch.testing import assert_close

from transformer_lens import FactoredMatrix


class MultiplyTestBase(ABC):
    @staticmethod
    @abstractmethod
    def _test_multiply(a, b, matrix):
        pass

    def test_left_multiply_by_matrix(self, a, b, matrix):
        self._test_multiply(a, b, matrix)

    def test_left_multiply_when_factored_matrix_has_leading_dim(self, a, b, matrix):
        a_with_leading = repeat(a, "x y -> b x y", b=2)
        b_with_leading = repeat(b, "x y -> b x y", b=2)

        product = self._test_multiply(a_with_leading, b_with_leading, matrix)

        assert product.A.shape[:-2] == (2,)
        assert product.B.shape[:-2] == (2,)

    def test_left_multiply_when_matrix_has_leading_dim(self, a, b, matrix):
        matrix_with_leading = repeat(matrix, "x y -> b1 b2 x y", b1=2, b2=12)

        product = self._test_multiply(a, b, matrix_with_leading)

        assert product.A.shape[:-2] == (2, 12)
        assert product.B.shape[:-2] == (2, 12)

    def test_left_multiply_when_both_have_leading_dim(self, a, b, matrix):
        a_with_leading = repeat(a, "x y -> b x y", b=2)
        b_with_leading = repeat(b, "x y -> b x y", b=2)
        matrix_with_leading = repeat(matrix, "x y -> b x y", b=2)

        product = self._test_multiply(
            a_with_leading, b_with_leading, matrix_with_leading
        )

        assert product.A.shape[:-2] == (2,)
        assert product.B.shape[:-2] == (2,)


@pytest.mark.parametrize(
    ("a", "b", "matrix"),
    [
        pytest.param(randn(2, 3), randn(3, 4), randn(4, 5), id="rdim > mdim"),
        pytest.param(randn(2, 6), randn(6, 4), randn(4, 5), id="rdim < mdim"),
        pytest.param(randn(2, 4), randn(4, 4), randn(4, 5), id="rdim == mdim"),
    ],
)
class TestLeftMultiplyByMatrix(MultiplyTestBase):
    @staticmethod
    def _test_multiply(a, b, matrix):
        fm = FactoredMatrix(a, b)

        product = fm @ matrix
        expected_product = (a @ b) @ matrix
        assert_close(product.AB, expected_product)

        assert product.ldim == fm.ldim
        assert product.mdim == min(fm.mdim, matrix.shape[-2])
        assert product.rdim == matrix.shape[-1]

        return product


@pytest.mark.parametrize(
    ("matrix", "a", "b"),
    [
        pytest.param(randn(6, 3), randn(3, 4), randn(4, 6), id="ldim > mdim"),
        pytest.param(randn(2, 6), randn(6, 4), randn(4, 2), id="ldim < mdim"),
        pytest.param(randn(2, 2), randn(2, 4), randn(4, 2), id="ldim == mdim"),
    ],
)
class TestRightMultiplyByMatrix(MultiplyTestBase):
    @staticmethod
    def _test_multiply(a, b, matrix):
        fm = FactoredMatrix(a, b)

        product = matrix @ fm
        expected_product = matrix @ (a @ b)
        assert_close(product.AB, expected_product)

        assert product.ldim == matrix.shape[-2]
        assert product.mdim == min(fm.mdim, matrix.shape[-1])
        assert product.rdim == fm.rdim

        return product
