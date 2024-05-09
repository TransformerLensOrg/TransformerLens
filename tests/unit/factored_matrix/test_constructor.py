import pytest
import torch

from transformer_lens import FactoredMatrix


def test_factored_matrix():
    A = torch.randn(5, 3)
    B = torch.randn(3, 7)
    f = FactoredMatrix(A, B)

    assert torch.equal(f.A, A)
    assert torch.equal(f.B, B)

    assert (f.ldim, f.mdim, f.rdim) == (5, 3, 7)
    assert not f.has_leading_dims
    assert f.shape == (5, 7)


def test_factored_matrix_b_leading_dims():
    A = torch.ones((5, 3))
    B = torch.ones((2, 4, 3, 7))
    f = FactoredMatrix(A, B)

    assert f.A.shape == (2, 4, 5, 3)
    assert torch.equal(f.B, B)

    assert (f.ldim, f.mdim, f.rdim) == (5, 3, 7)
    assert f.has_leading_dims
    assert f.shape == (2, 4, 5, 7)


def test_factored_matrix_a_b_leading_dims():
    A = torch.ones((4, 5, 3))
    B = torch.ones((2, 4, 3, 7))
    f = FactoredMatrix(A, B)

    assert f.A.shape == (2, 4, 5, 3)
    assert torch.equal(f.B, B)

    assert (f.ldim, f.mdim, f.rdim) == (5, 3, 7)
    assert f.has_leading_dims
    assert f.shape == (2, 4, 5, 7)


def test_factored_matrix_broadcast_mismatch():
    A = torch.ones((9, 5, 3))
    B = torch.ones((2, 4, 3, 7))

    with pytest.raises(RuntimeError) as e:
        FactoredMatrix(A, B)

    assert "Shape mismatch" in str(e.value)


@pytest.mark.skip(
    """
    AssertionError will not be reached due to jaxtyping argument consistency
    checks, which are enabled at test time but not run time.

    See https://github.com/TransformerLensOrg/TransformerLens/issues/190
    """
)
def test_factored_matrix_inner_mismatch():
    A = torch.ones((2, 3, 4))
    B = torch.ones((2, 3, 5))
    with pytest.raises(AssertionError) as e:
        FactoredMatrix(A, B)

    assert "inner dimension" in str(e.value)
