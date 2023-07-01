import torch
from torch.testing import assert_close

from transformer_lens import FactoredMatrix


def test_left_matmul_by_vector_left():
    a = torch.rand(2, 3)
    b = torch.rand(3, 4)

    fm = FactoredMatrix(a, b)
    vector = torch.rand(4)

    assert_close(fm @ vector, (a @ b) @ vector)


def test_left_matmul_by_vector_leading_dim():
    a = torch.rand(6, 2, 3)
    b = torch.rand(6, 3, 4)

    fm = FactoredMatrix(a, b)
    vector = torch.rand(4)

    assert_close(fm @ vector, (a @ b) @ vector)


def test_right_matmul_by_vector():
    a = torch.rand(2, 3)
    b = torch.rand(3, 4)

    fm = FactoredMatrix(a, b)
    vector = torch.rand(2)

    assert_close(vector @ fm, vector @ (a @ b))


def test_right_matmul_by_vector_leading_dim():
    a = torch.rand(6, 2, 3)
    b = torch.rand(6, 3, 4)

    fm = FactoredMatrix(a, b)
    vector = torch.rand(2)

    assert_close(vector @ fm, vector @ (a @ b))
