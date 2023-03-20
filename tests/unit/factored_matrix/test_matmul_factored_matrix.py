from dataclasses import dataclass
from typing import NamedTuple, Tuple

from pytest import mark, param, raises
import torch
from torch import randn, Tensor

from transformer_lens import FactoredMatrix


@dataclass
class TestMatmul:
    a: Tuple[Tensor, Tensor]
    b: Tuple[Tensor, Tensor]
    expected_product_a_shape: Tuple[int, int]
    expected_product_b_shape: Tuple[int, int]

    # suppress PytestCollectionWarning
    __test__ = False

    def run(self):
        print(torch.seed)
        print()

        a_left, a_right = self.a
        b_left, b_right = self.b

        factored_a = FactoredMatrix(a_left, a_right)
        factored_b = FactoredMatrix(b_left, b_right)

        product = factored_a @ factored_b
        expected_product = (a_left @ a_right) @ (b_left @ b_right)
        assert torch.allclose(product.AB, expected_product, atol=1e-5)

        assert product.A.shape == self.expected_product_a_shape
        assert product.B.shape == self.expected_product_b_shape

        # These assertions do the same as the previous assertions on shape,
        # except they don't check the leading dimensions. However they show
        # the expected behaviour more explicitly.
        expected_mdim = min(
            factored_a.mdim, factored_a.rdim, factored_b.ldim, factored_b.mdim
        )
        assert product.ldim == factored_a.ldim
        assert product.mdim == expected_mdim
        assert product.rdim == factored_b.rdim

    def test_case(id, a, b, product_a_shape, product_b_shape):
        return param(TestMatmul(a, b, product_a_shape, product_b_shape), id=id)


test_data = [
    TestMatmul.test_case(
        id="both have ldim == mdim == rdim",
        a=(randn(3,3), randn(3,3)),
        b=(randn(3,3), randn(3,3)),
        product_a_shape=(3,3),
        product_b_shape=(3,3)
    ),
    TestMatmul.test_case(
        id="both have ldim == mdim == rdim == 1",
        a=(randn(1,1), randn(1,1)),
        b=(randn(1,1), randn(1,1)),
        product_a_shape=(1,1),
        product_b_shape=(1,1)
    ),
    TestMatmul.test_case(
        id="both have ldim == rdim == 1 != mdim",
        a=(randn(1,2), randn(2,1)),
        b=(randn(1,6), randn(6,1)),
        product_a_shape=(1,1),
        product_b_shape=(1,1)
    ),
    TestMatmul.test_case(
        id="A has ldim == mdim == rdim == 1",
        a=(randn(1,1), randn(1,1)),
        b=(randn(1,2), randn(2,5)),
        product_a_shape=(1,1),
        product_b_shape=(1,5)
    ),
    TestMatmul.test_case(
        id="B has ldim == mdim == rdim == 1",
        a=(randn(5,2), randn(2,1)),
        b=(randn(1,1), randn(1,1)),
        product_a_shape=(5,1),
        product_b_shape=(1,1)
    ),
    TestMatmul.test_case(
        id="A has ldim == 1",
        a=(randn(1,2), randn(2,3)),
        b=(randn(3,2), randn(2,4)),
        product_a_shape=(1,2),
        product_b_shape=(2,4)
    ),
    TestMatmul.test_case(
        id="B has ldim == 1",
        a=(randn(3,2), randn(2,1)),
        b=(randn(1,3), randn(3,2)),
        product_a_shape=(3,1),
        product_b_shape=(1,2)
    ),
    TestMatmul.test_case(
        id="A has ldim == 1 and B has rdim == 1",
        a=(randn(1,2), randn(2,5)),
        b=(randn(5,3), randn(3,1)),
        product_a_shape=(1,2),
        product_b_shape=(2,1)
    ),
    TestMatmul.test_case(
        id="both have rdim > mdim",
        a=(randn(3, 4), randn(4, 5)),
        b=(randn(5, 6), randn(6, 7)),
        product_a_shape=(3, 4),
        product_b_shape=(4, 7),
    ),
    TestMatmul.test_case(
        id="both have rdim < mdim",
        a=(randn(8, 6), randn(6, 5)),
        b=(randn(5, 7), randn(7, 3)),
        product_a_shape=(8, 5),
        product_b_shape=(5, 3),
    ),
    TestMatmul.test_case(
        id="both have rdim == mdim",
        a=(randn(3, 4), randn(4, 4)),
        b=(randn(4, 5), randn(5, 5)),
        product_a_shape=(3, 4),
        product_b_shape=(4, 5),
    ),
    TestMatmul.test_case(
        id="A has rdim < mdim and B has rdim > mdim",
        a=(randn(7, 6), randn(6, 3)),
        b=(randn(3, 7), randn(7, 8)),
        product_a_shape=(7, 3),
        product_b_shape=(3, 8),
    ),
    TestMatmul.test_case(
        id="A has one leading dimension",
        a=(randn(2, 3, 4), randn(2, 4, 5)),
        b=(randn(5, 6), randn(6, 7)),
        product_a_shape=(2, 3, 4),
        product_b_shape=(2, 4, 7),
    ),
    TestMatmul.test_case(
        id="B has one leading dimension",
        a=(randn(3, 4), randn(4, 5)),
        b=(randn(2, 5, 6), randn(2, 6, 7)),
        product_a_shape=(2, 3, 4),
        product_b_shape=(2, 4, 7),
    ),
    TestMatmul.test_case(
        id="A has two leading dimensions and B has one leading dimension",
        a=(randn(2, 9, 3, 4), randn(2, 9, 4, 5)),
        b=(randn(9, 5, 6), randn(9, 6, 7)),
        product_a_shape=(2, 9, 3, 4),
        product_b_shape=(2, 9, 4, 7),
    ),
    # TestMatmul.test_case(
    # id="both_rdim_less_than_mdim"
    #
    # )
    # TODO
    # both leading dims, not broadcastable
    #
]


@mark.parametrize("test_case", test_data)
def test_matmul(test_case: TestMatmul):
    test_case.run()


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
        product = a @ b

    if message:
        assert message in str(e.value)
