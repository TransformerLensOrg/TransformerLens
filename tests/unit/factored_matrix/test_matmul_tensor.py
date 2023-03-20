import torch

from transformer_lens import FactoredMatrix
from pytest import mark, param, raises

def test_matmul_by_vector():
    a = torch.rand(2,3)
    b = torch.rand(3,4)

    fm = FactoredMatrix(a, b)

    vector = torch.rand(4)

    assert torch.equal(fm @ vector, (a @ b) @ vector)
