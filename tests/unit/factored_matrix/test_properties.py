import pytest
import torch
from torch import randn

from transformer_lens import FactoredMatrix, utils


@pytest.fixture(scope="module")
def random_matrices():
    return [
        (randn(3, 2), randn(2, 3)),
        (randn(4, 5), randn(5, 4)),
        (randn(6, 7), randn(7, 6)),
        (randn(6, 7), randn(7, 3)),
    ]


@pytest.fixture(scope="module")
def factored_matrices(random_matrices):
    return [FactoredMatrix(a, b) for a, b in random_matrices]


@pytest.fixture(scope="module")
def random_matrices_leading_ones():
    return [
        (randn(1, 8, 9), randn(1, 9, 8)),
    ]


@pytest.fixture(scope="module")
def factored_matrices_leading_ones(random_matrices_leading_ones):
    return [FactoredMatrix(a, b) for a, b in random_matrices_leading_ones]


class TestFactoredMatrixProperties:
    def test_AB_property(self, factored_matrices, random_matrices):
        for i, factored_matrix in enumerate(factored_matrices):
            a, b = random_matrices[i]
            expected_AB = a @ b
            assert torch.allclose(factored_matrix.AB, expected_AB, atol=1e-5)

    def test_BA_property(self, factored_matrices, random_matrices):
        for i, factored_matrix in enumerate(factored_matrices):
            a, b = random_matrices[i]
            if a.shape[-2] == b.shape[-1]:
                expected_BA = b @ a
                assert torch.allclose(factored_matrix.BA, expected_BA, atol=1e-5)
            else:
                with pytest.raises(AssertionError):
                    _ = factored_matrix.BA

    def test_transpose_property(self, factored_matrices):
        for factored_matrix in factored_matrices:
            transposed_factored_matrix = factored_matrix.T
            assert torch.allclose(
                transposed_factored_matrix.A,
                factored_matrix.B.transpose(-2, -1),
                atol=1e-5,
            )
            assert torch.allclose(
                transposed_factored_matrix.B,
                factored_matrix.A.transpose(-2, -1),
                atol=1e-5,
            )

    def test_svd_property(self, factored_matrices):
        for factored_matrix in factored_matrices:
            U, S, Vh = factored_matrix.svd()
            assert torch.allclose(factored_matrix.AB, U @ torch.diag_embed(S) @ Vh.T, atol=1e-5)
            # test that U and Vh are unitary
            assert torch.allclose(U.T @ U, torch.eye(U.shape[-1]), atol=1e-5)
            assert torch.allclose(Vh.T @ Vh, torch.eye(Vh.shape[-1]), atol=1e-5)

    def test_svd_property_leading_ones(self, factored_matrices_leading_ones):
        for factored_matrix in factored_matrices_leading_ones:
            U, S, Vh = factored_matrix.svd()
            assert torch.allclose(factored_matrix.AB, U @ torch.diag_embed(S) @ Vh.mT, atol=1e-5)
            # test that U and Vh are unitary
            assert torch.allclose(U.mT @ U, torch.eye(U.shape[-1]), atol=1e-5)
            assert torch.allclose(Vh.mT @ Vh, torch.eye(Vh.shape[-1]), atol=1e-5)

    @pytest.mark.skip(
        """
        Jaxtyping throws a TypeError when this test is run.
        TypeError: type of the return value must be jaxtyping.Float[Tensor, '*leading_dims mdim']; got torch.Tensor instead

        I'm not sure why. The error is not very informative. When debugging the shape was equal to mdim, and *leading_dims should
        match zero or more leading dims according to the [docs](https://github.com/google/jaxtyping/blob/main/API.md).

        Sort of related to https://github.com/TransformerLensOrg/TransformerLens/issues/190 because jaxtyping
        is only enabled at test time and not runtime.
        """
    )
    def test_eigenvalues_property(self, factored_matrices):
        for factored_matrix in factored_matrices:
            if factored_matrix.ldim == factored_matrix.rdim:
                eigenvalues = factored_matrix.eigenvalues
                expected_eigenvalues = torch.linalg.eig(
                    factored_matrix.B @ factored_matrix.A
                ).eigenvalues
                assert torch.allclose(
                    torch.abs(eigenvalues), torch.abs(expected_eigenvalues), atol=1e-5
                )
            else:
                with pytest.raises(AssertionError):
                    _ = factored_matrix.eigenvalues

    def test_ndim_property(self, factored_matrices, random_matrices):
        for i, factored_matrix in enumerate(factored_matrices):
            a, b = random_matrices[i]
            expected_ndim = max(a.ndim, b.ndim)
            assert factored_matrix.ndim == expected_ndim

    def test_pair_property(self, factored_matrices, random_matrices):
        for i, factored_matrix in enumerate(factored_matrices):
            a, b = random_matrices[i]
            assert torch.allclose(factored_matrix.pair[0], a, atol=1e-5)
            assert torch.allclose(factored_matrix.pair[1], b, atol=1e-5)

    def test_norm_property(self, factored_matrices):
        for factored_matrix in factored_matrices:
            assert torch.allclose(factored_matrix.norm(), factored_matrix.AB.norm(), atol=1e-5)

    def test_get_corner(self, factored_matrices):
        for factored_matrix in factored_matrices:
            k = 3
            result = factored_matrix.get_corner(k)
            expected = utils.get_corner(
                factored_matrix.A[..., :k, :] @ factored_matrix.B[..., :, :k], k
            )
            assert torch.allclose(result, expected)

    def test_ndim(self, factored_matrices):
        for factored_matrix in factored_matrices:
            assert factored_matrix.ndim == len(factored_matrix.shape)

    def test_collapse_l(self, factored_matrices):
        for factored_matrix in factored_matrices:
            result = factored_matrix.collapse_l()
            expected = factored_matrix.S[..., :, None] * utils.transpose(factored_matrix.Vh)
            assert torch.allclose(result, expected)

    def test_collapse_r(self, factored_matrices):
        for factored_matrix in factored_matrices:
            result = factored_matrix.collapse_r()
            expected = factored_matrix.U * factored_matrix.S[..., None, :]
            assert torch.allclose(result, expected)

    def test_unsqueeze(self, factored_matrices_leading_ones):
        for factored_matrix in factored_matrices_leading_ones:
            k = 0
            unsqueezed_A = factored_matrix.A.unsqueeze(k)
            unsqueezed_B = factored_matrix.B.unsqueeze(k)
            inner_dim_A = unsqueezed_A.size(-1)
            inner_dim_B = unsqueezed_B.size(-2)

            if inner_dim_A == inner_dim_B:
                result = FactoredMatrix(unsqueezed_A, unsqueezed_B)
                assert isinstance(result, FactoredMatrix)
                assert torch.allclose(result.A, unsqueezed_A)
                assert torch.allclose(result.B, unsqueezed_B)
