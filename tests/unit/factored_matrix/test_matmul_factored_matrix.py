from dataclasses import dataclass
from typing import Tuple
from einops import repeat

from pytest import mark, param, raises
import torch
from torch import randn, Tensor

from transformer_lens import FactoredMatrix


@dataclass
class TestMatmul:
    a: Tuple[Tensor, Tensor]
    b: Tuple[Tensor, Tensor]

    # suppress PytestCollectionWarning
    __test__ = False

    @classmethod
    def test_case(cls, id, a, b):
        return param(TestMatmul(a, b), id=id)


test_data = [
    TestMatmul.test_case(
        id="both have ldim == mdim == rdim",
        a=(randn(3, 3), randn(3, 3)),
        b=(randn(3, 3), randn(3, 3)),
    ),
    TestMatmul.test_case(
        id="both have ldim == mdim == rdim == 1",
        a=(randn(1, 1), randn(1, 1)),
        b=(randn(1, 1), randn(1, 1)),
    ),
    TestMatmul.test_case(
        id="both have ldim == rdim == 1 != mdim",
        a=(randn(1, 2), randn(2, 1)),
        b=(randn(1, 6), randn(6, 1)),
    ),
    TestMatmul.test_case(
        id="A has ldim == mdim == rdim == 1",
        a=(randn(1, 1), randn(1, 1)),
        b=(randn(1, 2), randn(2, 5)),
    ),
    TestMatmul.test_case(
        id="B has ldim == mdim == rdim == 1",
        a=(randn(5, 2), randn(2, 1)),
        b=(randn(1, 1), randn(1, 1)),
    ),
    TestMatmul.test_case(
        id="A has ldim == 1",
        a=(randn(1, 2), randn(2, 3)),
        b=(randn(3, 2), randn(2, 4)),
    ),
    TestMatmul.test_case(
        id="B has ldim == 1",
        a=(randn(3, 2), randn(2, 1)),
        b=(randn(1, 3), randn(3, 2)),
    ),
    TestMatmul.test_case(
        id="A has ldim == 1 and B has rdim == 1",
        a=(randn(1, 2), randn(2, 5)),
        b=(randn(5, 3), randn(3, 1)),
    ),
    TestMatmul.test_case(
        id="both have rdim > mdim",
        a=(randn(3, 4), randn(4, 5)),
        b=(randn(5, 6), randn(6, 7)),
    ),
    TestMatmul.test_case(
        id="both have rdim < mdim",
        a=(randn(8, 6), randn(6, 5)),
        b=(randn(5, 7), randn(7, 3)),
    ),
    TestMatmul.test_case(
        id="both have rdim == mdim",
        a=(randn(3, 4), randn(4, 4)),
        b=(randn(4, 5), randn(5, 5)),
    ),
    TestMatmul.test_case(
        id="A has rdim < mdim and B has rdim > mdim",
        a=(randn(7, 6), randn(6, 3)),
        b=(randn(3, 7), randn(7, 8)),
    ),
]


@mark.parametrize("test_case", test_data)
def test_matmul(test_case: TestMatmul):
    _test_matmul(*test_case.a, *test_case.b)


@mark.parametrize("test_case", test_data)
def test_matmul_a_has_leading_dim(test_case: TestMatmul):
    a_left, a_right = test_case.a

    a_left_with_leading = repeat(a_left, "x y -> b x y", b=2)
    a_right_with_leading = repeat(a_right, "x y -> b x y", b=2)

    product = _test_matmul(a_left_with_leading, a_right_with_leading, *test_case.b)

    assert product.A.shape[:1] == (2,)
    assert product.B.shape[:1] == (2,)


@mark.parametrize("test_case", test_data)
def test_matmul_b_has_leading_dim(test_case: TestMatmul):
    b_left, b_right = test_case.b

    b_left_with_leading = repeat(b_left, "x y -> b x y", b=2)
    b_right_with_leading = repeat(b_right, "x y -> b x y", b=2)

    product = _test_matmul(
        *test_case.a, b_left_with_leading, b_right_with_leading
    )

    assert product.A.shape[:1] == (2,)
    assert product.B.shape[:1] == (2,)


@mark.parametrize("test_case", test_data)
def test_matmul_both_have_leading_dim(test_case: TestMatmul):
    a_left, a_right = test_case.a
    b_left, b_right = test_case.b

    a_left_with_leading = repeat(a_left, "x y -> b1 b2 x y", b1=2, b2=9)
    a_right_with_leading = repeat(a_right, "x y -> b1 b2 x y", b1=2, b2=9)
    b_left_with_leading = repeat(b_left, "x y -> b x y", b=9)
    b_right_with_leading = repeat(b_right, "x y -> b x y", b=9)

    product = _test_matmul(
        a_left_with_leading,
        a_right_with_leading,
        b_left_with_leading,
        b_right_with_leading,
    )

    assert product.A.shape[:2] == (2,9)
    assert product.B.shape[:2] == (2,9)


def _test_matmul(a_left, a_right, b_left, b_right):
    factored_a = FactoredMatrix(a_left, a_right)
    factored_b = FactoredMatrix(b_left, b_right)

    product = factored_a @ factored_b
    expected_product = (a_left @ a_right) @ (b_left @ b_right)
    assert torch.allclose(product.AB, expected_product, atol=1e-5)

    assert product.ldim == factored_a.ldim
    assert product.mdim == min(
        factored_a.mdim, factored_a.rdim, factored_b.ldim, factored_b.mdim
    )
    assert product.rdim == factored_b.rdim

    return product


test_data_error = [
    param(
        torch.randn(3, 4),
        torch.randn(4, 6),
        torch.randn(2, 5, 6),
        torch.randn(2, 6, 7),
        AssertionError,
        "inner dimension",
    ),
]


@mark.parametrize("a_left,a_right,b_left,b_right,exception, message", test_data_error)
def test_matmul_incompatible(a_left, a_right, b_left, b_right, exception, message):
    a = FactoredMatrix(a_left, a_right)
    b = FactoredMatrix(b_left, b_right)

    with raises(exception) as e:
        a @ b

    if message:
        assert message in str(e.value)
