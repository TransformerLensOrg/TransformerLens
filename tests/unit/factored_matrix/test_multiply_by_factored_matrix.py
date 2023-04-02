import pytest
from einops import repeat
from torch import randn
from torch.testing import assert_close

from transformer_lens import FactoredMatrix


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(
            (randn(3, 3), randn(3, 3)),
            (randn(3, 3), randn(3, 3)),
            id="both have ldim == mdim == rdim",
        ),
        pytest.param(
            (randn(1, 1), randn(1, 1)),
            (randn(1, 1), randn(1, 1)),
            id="both have ldim == mdim == rdim == 1",
        ),
        pytest.param(
            (randn(1, 2), randn(2, 1)),
            (randn(1, 6), randn(6, 1)),
            id="both have ldim == rdim == 1 != mdim",
        ),
        pytest.param(
            (randn(1, 1), randn(1, 1)),
            (randn(1, 2), randn(2, 5)),
            id="A has ldim == mdim == rdim == 1",
        ),
        pytest.param(
            (randn(5, 2), randn(2, 1)),
            (randn(1, 1), randn(1, 1)),
            id="B has ldim == mdim == rdim == 1",
        ),
        pytest.param(
            (randn(1, 2), randn(2, 3)),
            (randn(3, 2), randn(2, 4)),
            id="A has ldim == 1",
        ),
        pytest.param(
            (randn(3, 2), randn(2, 1)),
            (randn(1, 3), randn(3, 2)),
            id="B has ldim == 1",
        ),
        pytest.param(
            (randn(1, 2), randn(2, 5)),
            (randn(5, 3), randn(3, 1)),
            id="A has ldim == 1 and B has rdim == 1",
        ),
        pytest.param(
            (randn(3, 4), randn(4, 5)),
            (randn(5, 6), randn(6, 7)),
            id="both have rdim > mdim",
        ),
        pytest.param(
            (randn(8, 6), randn(6, 5)),
            (randn(5, 7), randn(7, 3)),
            id="both have rdim < mdim",
        ),
        pytest.param(
            (randn(3, 4), randn(4, 4)),
            (randn(4, 5), randn(5, 5)),
            id="both have rdim == mdim",
        ),
        pytest.param(
            (randn(7, 6), randn(6, 3)),
            (randn(3, 7), randn(7, 8)),
            id="A has rdim < mdim and B has rdim > mdim",
        ),
    ],
)
class TestMultiplyByFactoredMatrix:
    @staticmethod
    def _test_multiply(a_left, a_right, b_left, b_right) -> FactoredMatrix:
        factored_a = FactoredMatrix(a_left, a_right)
        factored_b = FactoredMatrix(b_left, b_right)

        product = factored_a @ factored_b
        expected_product = (a_left @ a_right) @ (b_left @ b_right)
        assert_close(product.AB, expected_product)

        assert product.ldim == factored_a.ldim
        assert product.mdim == min(
            factored_a.mdim, factored_a.rdim, factored_b.ldim, factored_b.mdim
        )
        assert product.rdim == factored_b.rdim

        return product

    def test_multiply_two_factored_matrices(self, a, b):
        self._test_multiply(*a, *b)

    def test_multiply_when_A_has_leading_dim(self, a, b):
        a_left, a_right = a

        a_left_with_leading = repeat(a_left, "x y -> b x y", b=2)
        a_right_with_leading = repeat(a_right, "x y -> b x y", b=2)

        product = self._test_multiply(a_left_with_leading, a_right_with_leading, *b)

        assert product.A.shape[:1] == (2,)
        assert product.B.shape[:1] == (2,)

    def test_multiply_when_B_has_leading_dim(self, a, b):
        b_left, b_right = b

        b_left_with_leading = repeat(b_left, "x y -> b x y", b=2)
        b_right_with_leading = repeat(b_right, "x y -> b x y", b=2)

        product = self._test_multiply(*a, b_left_with_leading, b_right_with_leading)

        assert product.A.shape[:1] == (2,)
        assert product.B.shape[:1] == (2,)

    def test_multiply_when_both_have_leading_dim(self, a, b):
        a_left, a_right = a
        b_left, b_right = b

        a_left_with_leading = repeat(a_left, "x y -> b1 b2 x y", b1=2, b2=9)
        a_right_with_leading = repeat(a_right, "x y -> b1 b2 x y", b1=2, b2=9)
        b_left_with_leading = repeat(b_left, "x y -> b x y", b=9)
        b_right_with_leading = repeat(b_right, "x y -> b x y", b=9)

        product = self._test_multiply(
            a_left_with_leading,
            a_right_with_leading,
            b_left_with_leading,
            b_right_with_leading,
        )

        assert product.A.shape[:2] == (2, 9)
        assert product.B.shape[:2] == (2, 9)


@pytest.mark.parametrize(
    ("a", "b", "error"),
    [
        pytest.param(
            FactoredMatrix(randn(2, 3, 4), randn(2, 4, 6)),
            FactoredMatrix(randn(4, 6, 7), randn(4, 7, 2)),
            RuntimeError,
            id="Leading dim mismatch where each has one leading dim",
        ),
        pytest.param(
            FactoredMatrix(randn(2, 9, 3, 4), randn(2, 9, 4, 6)),
            FactoredMatrix(randn(2, 6, 7), randn(2, 7, 2)),
            RuntimeError,
            id="Leading dim mismatch where A has two leading dims and B has one",
        ),
        pytest.param(
            FactoredMatrix(randn(3, 4), randn(4, 6)),
            FactoredMatrix(randn(5, 6), randn(6, 7)),
            AssertionError,
            id="Inner dimension mismatch",
        ),
        pytest.param(
            FactoredMatrix(randn(3, 4), randn(4, 6)),
            FactoredMatrix(randn(2, 5, 6), randn(2, 6, 7)),
            AssertionError,
            id="Inner dimension mismatch with batch",
        ),
    ],
)
def test_dimension_mismatch(a, b, error):
    with pytest.raises(error):
        _ = a @ b
