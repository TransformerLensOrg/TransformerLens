from __future__ import annotations

from functools import lru_cache
from typing import Tuple, Union

import torch
from jaxtyping import Float
from typeguard import typeguard_ignore

import transformer_lens.utils as utils


class FactoredMatrix:
    """
    Class to represent low rank factored matrices, where the matrix is represented as a product of two matrices. Has utilities for efficient calculation of eigenvalues, norm and SVD.
    """

    def __init__(
        self,
        A: Float[torch.Tensor, "... ldim mdim"],
        B: Float[torch.Tensor, "... mdim rdim"],
    ):
        self.A = A
        self.B = B
        assert self.A.size(-1) == self.B.size(
            -2
        ), f"Factored matrix must match on inner dimension, shapes were a: {self.A.shape}, b:{self.B.shape}"
        self.ldim = self.A.size(-2)
        self.rdim = self.B.size(-1)
        self.mdim = self.B.size(-2)
        self.has_leading_dims = (self.A.ndim > 2) or (self.B.ndim > 2)
        self.shape = torch.broadcast_shapes(self.A.shape[:-2], self.B.shape[:-2]) + (
            self.ldim,
            self.rdim,
        )
        self.A = self.A.broadcast_to(self.shape[:-2] + (self.ldim, self.mdim))
        self.B = self.B.broadcast_to(self.shape[:-2] + (self.mdim, self.rdim))

    def __matmul__(
        self,
        other: Union[
            Float[torch.Tensor, "... rdim new_rdim"],
            Float[torch.Tensor, "rdim"],
            FactoredMatrix,
        ],
    ) -> Union[FactoredMatrix, Float[torch.Tensor, "... ldim"]]:
        if isinstance(other, torch.Tensor):
            if other.ndim < 2:
                # It's a vector, so we collapse the factorisation and just return a vector
                # Squeezing/Unsqueezing is to preserve broadcasting working nicely
                return (self.A @ (self.B @ other.unsqueeze(-1))).squeeze(-1)
            else:
                assert (
                    other.size(-2) == self.rdim
                ), f"Right matrix must match on inner dimension, shapes were self: {self.shape}, other:{other.shape}"
                if self.rdim > self.mdim:
                    return FactoredMatrix(self.A, self.B @ other)
                else:
                    return FactoredMatrix(self.AB, other)
        elif isinstance(other, FactoredMatrix):
            return (self @ other.A) @ other.B

    def __rmatmul__(
        self,
        other: Union[
            Float[torch.Tensor, "... new_rdim ldim"],
            Float[torch.Tensor, "ldim"],
            FactoredMatrix,
        ],
    ) -> Union[FactoredMatrix, Float[torch.Tensor, "... rdim"]]:
        if isinstance(other, torch.Tensor):
            assert (
                other.size(-1) == self.ldim
            ), f"Left matrix must match on inner dimension, shapes were self: {self.shape}, other:{other.shape}"
            if other.ndim < 2:
                # It's a vector, so we collapse the factorisation and just return a vector
                return ((other.unsqueeze(-2) @ self.A) @ self.B).squeeze(-2)
            elif self.ldim > self.mdim:
                return FactoredMatrix(other @ self.A, self.B)
            else:
                return FactoredMatrix(other, self.AB)
        elif isinstance(other, FactoredMatrix):
            return other.A @ (other.B @ self)

    @property
    @typeguard_ignore
    def AB(self) -> Float[torch.Tensor, "*leading_dims ldim rdim"]:
        """The product matrix - expensive to compute, and can consume a lot of GPU memory"""
        return self.A @ self.B

    @property
    @typeguard_ignore
    def BA(self) -> Float[torch.Tensor, "*leading_dims rdim ldim"]:
        """The reverse product. Only makes sense when ldim==rdim"""
        assert (
            self.rdim == self.ldim
        ), f"Can only take ba if ldim==rdim, shapes were self: {self.shape}"
        return self.B @ self.A

    @property
    @typeguard_ignore
    def T(self) -> FactoredMatrix:
        return FactoredMatrix(self.B.transpose(-2, -1), self.A.transpose(-2, -1))

    @lru_cache(maxsize=None)
    def svd(
        self,
    ) -> Tuple[
        Float[torch.Tensor, "*leading_dims ldim mdim"],
        Float[torch.Tensor, "*leading_dims mdim"],
        Float[torch.Tensor, "*leading_dims rdim mdim"],
    ]:
        """
        Efficient algorithm for finding Singular Value Decomposition, a tuple (U, S, Vh) for matrix M st S is a vector and U, Vh are orthogonal matrices, and U @ S.diag() @ Vh.T == M

        (Note that Vh is given as the transpose of the obvious thing)
        """
        Ua, Sa, Vha = torch.svd(self.A)
        Ub, Sb, Vhb = torch.svd(self.B)
        middle = Sa[..., :, None] * utils.transpose(Vha) @ Ub * Sb[..., None, :]
        Um, Sm, Vhm = torch.svd(middle)
        U = Ua @ Um
        Vh = Vhb @ Vhm
        S = Sm
        return U, S, Vh

    @property
    @typeguard_ignore
    def U(self) -> Float[torch.Tensor, "*leading_dims ldim mdim"]:
        return self.svd()[0]

    @property
    @typeguard_ignore
    def S(self) -> Float[torch.Tensor, "*leading_dims mdim"]:
        return self.svd()[1]

    @property
    @typeguard_ignore
    def Vh(self) -> Float[torch.Tensor, "*leading_dims rdim mdim"]:
        return self.svd()[2]

    @property
    @typeguard_ignore
    def eigenvalues(self) -> Float[torch.Tensor, "*leading_dims mdim"]:
        """Eigenvalues of AB are the same as for BA (apart from trailing zeros), because if BAv=kv ABAv = A(BAv)=kAv, so Av is an eigenvector of AB with eigenvalue k."""
        return torch.linalg.eig(self.BA).eigenvalues

    def __getitem__(self, idx: Union[int, Tuple]) -> FactoredMatrix:
        """Indexing - assumed to only apply to the leading dimensions."""
        if not isinstance(idx, tuple):
            idx = (idx,)
        length = len([i for i in idx if i is not None])
        if length <= len(self.shape) - 2:
            return FactoredMatrix(self.A[idx], self.B[idx])
        elif length == len(self.shape) - 1:
            return FactoredMatrix(self.A[idx], self.B[idx[:-1]])
        elif length == len(self.shape):
            return FactoredMatrix(
                self.A[idx[:-1]], self.B[idx[:-2] + (slice(None), idx[-1])]
            )
        else:
            raise ValueError(
                f"{idx} is too long an index for a FactoredMatrix with shape {self.shape}"
            )

    def norm(self) -> Float[torch.Tensor, "*leading_dims"]:
        """
        Frobenius norm is sqrt(sum of squared singular values)
        """
        return self.S.pow(2).sum(-1).sqrt()

    def __repr__(self):
        return f"FactoredMatrix: Shape({self.shape}), Hidden Dim({self.mdim})"

    def make_even(self) -> FactoredMatrix:
        """
        Returns the factored form of (U @ S.sqrt().diag(), S.sqrt().diag() @ Vh) where U, S, Vh are the SVD of the matrix. This is an equivalent factorisation, but more even - each half has half the singular values, and orthogonal rows/cols
        """
        return FactoredMatrix(
            self.U * self.S.sqrt()[..., None, :],
            self.S.sqrt()[..., :, None] * utils.transpose(self.Vh),
        )

    def get_corner(self, k=3):
        return utils.get_corner(self.A[..., :k, :] @ self.B[..., :, :k], k)

    @property
    @typeguard_ignore
    def ndim(self) -> int:
        return len(self.shape)

    def collapse_l(self) -> Float[torch.Tensor, "*leading_dims mdim rdim"]:
        """
        Collapses the left side of the factorization by removing the orthogonal factor (given by self.U). Returns a (..., mdim, rdim) tensor
        """
        return self.S[..., :, None] * utils.transpose(self.Vh)

    def collapse_r(self) -> Float[torch.Tensor, "*leading_dims ldim mdim"]:
        """
        Analogous to collapse_l, returns a (..., ldim, mdim) tensor
        """
        return self.U * self.S[..., None, :]

    def unsqueeze(self, k: int) -> FactoredMatrix:
        return FactoredMatrix(self.A.unsqueeze(k), self.B.unsqueeze(k))

    @property
    @typeguard_ignore
    def pair(
        self,
    ) -> Tuple[
        Float[torch.Tensor, "*leading_dims ldim mdim"],
        Float[torch.Tensor, "*leading_dims mdim rdim"],
    ]:
        return (self.A, self.B)
