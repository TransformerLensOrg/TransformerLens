import torch

from transformer_lens import FactoredMatrix
from pytest import mark, param, raises

test_data = [
    param(
        torch.randn(3, 4),
        torch.randn(4, 5),
        torch.randn(5, 6),
        torch.randn(6, 7),
        (3, 4),
        (4, 7),
        id="simple",
    ),
    param(
        torch.randn(2, 3, 4),
        torch.randn(2, 4, 5),
        torch.randn(5, 6),
        torch.randn(6, 7),
        (2, 3, 4),
        (2, 4, 7),
        id="a_leading_dims",
    ),
    param(
        torch.randn(3, 4),
        torch.randn(4, 5),
        torch.randn(2, 5, 6),
        torch.randn(2, 6, 7),
        (2, 3, 4),
        (2, 4, 7),
        id="b_leading_dims",
    ),
    # TODO
    # both leading dims
    # both leading dims, not broadcastable
    # 
]


@mark.parametrize("a_left,a_right,b_left,b_right,a_shape,b_shape", test_data)
def test_matmul(a_left, a_right, b_left, b_right, a_shape, b_shape):
    a = FactoredMatrix(a_left, a_right)
    b = FactoredMatrix(b_left, b_right)

    product = a @ b
    expected = (a_left @ a_right) @ (b_left @ b_right)

    assert product.A.shape == a_shape
    assert product.B.shape == b_shape
    assert torch.allclose(product.AB, expected, atol=1e-5)

test_data_error = [ 
    param(
        torch.randn(3, 4),
        torch.randn(4, 6),
        torch.randn(2, 5, 6),
        torch.randn(2, 6, 7),
        AssertionError,
        "inner dimension"
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