"""Unit tests for Projection Kernel subspace geometry."""

import math

import pytest
import torch

from transformer_lens.tools.analysis.projection_kernel import (
    SubspaceBasis,
    orthonormal_subspace,
    projection_kernel,
    random_projection_kernel_moments,
)


class TestOrthonormalSubspace:
    def test_extracts_full_rank_tall_basis(self):
        matrix = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)

        result = orthonormal_subspace(matrix)

        assert result.rank == 2
        assert result.measured_rank == 2
        assert result.ambient_dim == 4
        assert result.input_shape == (4, 2)
        assert result.basis.dtype == torch.float64
        assert torch.allclose(result.basis.T @ result.basis, torch.eye(2, dtype=torch.float64))

    def test_explicit_rank_truncates_but_records_measured_rank(self):
        matrix = torch.diag(torch.tensor([3.0, 2.0, 1.0]))

        result = orthonormal_subspace(matrix, rank=2)

        assert result.rank == 2
        assert result.measured_rank == 3
        assert result.basis.shape == (3, 2)
        assert result.singular_values.tolist() == pytest.approx([3.0, 2.0, 1.0])

    def test_threshold_boundary_is_excluded(self):
        matrix = torch.diag(torch.tensor([1.0, 0.25], dtype=torch.float64))

        result = orthonormal_subspace(matrix, rtol=0.25)

        assert result.measured_rank == 1
        assert result.threshold == pytest.approx(0.25)

    def test_invariant_to_scale_and_invertible_right_transform(self):
        matrix = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]], dtype=torch.float64
        )
        transform = torch.tensor([[2.0, 1.0], [-1.0, 3.0]], dtype=torch.float64)
        original = orthonormal_subspace(matrix)
        scaled = orthonormal_subspace(7.0 * matrix)
        transformed = orthonormal_subspace(matrix @ transform)

        assert projection_kernel(original, scaled).normalized.item() == pytest.approx(1.0)
        assert projection_kernel(original, transformed).normalized.item() == pytest.approx(1.0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_promotes_to_float32(self, dtype):
        result = orthonormal_subspace(torch.eye(3, dtype=dtype))

        assert result.basis.dtype == torch.float32
        assert result.singular_values.dtype == torch.float32

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_preserves_supported_dtype_and_cpu_device(self, dtype):
        result = orthonormal_subspace(torch.eye(3, dtype=dtype, device="cpu"))

        assert result.basis.dtype == dtype
        assert result.singular_values.dtype == dtype
        assert result.basis.device == torch.device("cpu")

    @pytest.mark.parametrize(
        ("matrix", "message"),
        [
            (torch.ones(3), "two-dimensional"),
            (torch.ones(2, 2, dtype=torch.int64), "floating-point"),
            (torch.ones(2, 2, dtype=torch.complex64), "floating-point"),
            (torch.empty(0, 2), "non-empty"),
            (torch.tensor([[1.0, float("nan")]]), "finite"),
            (torch.zeros(2, 2), "numerical rank is zero"),
        ],
    )
    def test_rejects_invalid_matrices(self, matrix, message):
        with pytest.raises(ValueError, match=message):
            orthonormal_subspace(matrix)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"rank": True}, "rank must be an integer"),
            ({"rank": 0}, "rank must be between"),
            ({"rank": 3}, "rank must be between"),
            ({"rtol": -1.0}, "rtol must be"),
            ({"rtol": float("inf")}, "rtol must be"),
        ],
    )
    def test_rejects_invalid_rank_options(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            orthonormal_subspace(torch.eye(2), **kwargs)

    def test_rejects_rank_above_measured_rank(self):
        matrix = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        with pytest.raises(ValueError, match="exceeds measured rank 1"):
            orthonormal_subspace(matrix, rank=2)


class TestProjectionKernel:
    def test_identical_and_orthogonal_spaces(self):
        first = orthonormal_subspace(torch.eye(4)[:, :2])
        same = projection_kernel(first, first)
        orthogonal = projection_kernel(first, orthonormal_subspace(torch.eye(4)[:, 2:]))

        assert same.score.item() == pytest.approx(2.0)
        assert same.normalized.item() == pytest.approx(1.0)
        assert same.cosines.tolist() == pytest.approx([1.0, 1.0])
        assert same.angles.tolist() == pytest.approx([0.0, 0.0])
        assert orthogonal.score.item() == pytest.approx(0.0)
        assert orthogonal.angles.tolist() == pytest.approx([math.pi / 2, math.pi / 2])

    def test_recovers_known_principal_angles(self):
        theta = torch.tensor(math.pi / 3, dtype=torch.float64)
        basis_a = torch.eye(4, dtype=torch.float64)[:, :2]
        basis_b = torch.stack(
            [
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
                torch.stack([torch.tensor(0.0), theta.cos(), theta.sin(), torch.tensor(0.0)]),
            ],
            dim=1,
        )

        result = projection_kernel(orthonormal_subspace(basis_a), orthonormal_subspace(basis_b))

        assert result.cosines.tolist() == pytest.approx([1.0, 0.5])
        assert result.angles.tolist() == pytest.approx([0.0, math.pi / 3])
        assert result.score.item() == pytest.approx(1.25)

    def test_nested_unequal_rank_spaces(self):
        small = orthonormal_subspace(torch.eye(4)[:, :2])
        large = orthonormal_subspace(torch.eye(4)[:, :3])

        result = projection_kernel(small, large)

        assert result.score.item() == pytest.approx(2.0)
        assert result.normalized.item() == pytest.approx(math.sqrt(2 / 3))
        assert result.rank_a == 2
        assert result.rank_b == 3

    def test_equivalent_definitions_and_symmetry(self):
        generator = torch.Generator().manual_seed(3)
        first = orthonormal_subspace(torch.randn(6, 2, generator=generator, dtype=torch.float64))
        second = orthonormal_subspace(torch.randn(6, 3, generator=generator, dtype=torch.float64))

        result = projection_kernel(first, second)
        reverse = projection_kernel(second, first)
        projector_trace = torch.trace(
            (first.basis @ first.basis.T) @ (second.basis @ second.basis.T)
        )

        assert result.score.item() == pytest.approx(result.cosines.square().sum().item())
        assert result.score.item() == pytest.approx(projector_trace.item())
        assert result.score.item() == pytest.approx(reverse.score.item())

    def test_checks_orthonormality(self):
        invalid = SubspaceBasis(
            basis=torch.ones(3, 1) * 2,
            singular_values=torch.ones(1),
            rank=1,
            measured_rank=1,
            rtol=1e-6,
            threshold=1e-6,
            input_shape=(3, 1),
        )

        with pytest.raises(ValueError, match="orthonormal"):
            projection_kernel(invalid, invalid)

    def test_rejects_invalid_singular_value_metadata(self):
        valid = orthonormal_subspace(torch.eye(3))
        invalid = SubspaceBasis(
            basis=valid.basis,
            singular_values=torch.tensor([1.0, float("nan"), 1.0]),
            rank=valid.rank,
            measured_rank=valid.measured_rank,
            rtol=valid.rtol,
            threshold=valid.threshold,
            input_shape=valid.input_shape,
        )

        with pytest.raises(ValueError, match="singular_values must be a finite"):
            projection_kernel(invalid, valid)

    def test_rejects_nonfinite_basis_metadata(self):
        valid = orthonormal_subspace(torch.eye(2))
        nonfinite = SubspaceBasis(
            basis=torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]),
            singular_values=valid.singular_values,
            rank=2,
            measured_rank=2,
            rtol=valid.rtol,
            threshold=valid.threshold,
            input_shape=(2, 2),
        )

        with pytest.raises(ValueError, match="basis must be finite"):
            projection_kernel(nonfinite, valid)

    def test_clamps_tiny_score_overshoot(self):
        valid = orthonormal_subspace(torch.eye(2))
        almost_orthonormal = SubspaceBasis(
            basis=valid.basis * (1 + 5e-7),
            singular_values=valid.singular_values,
            rank=valid.rank,
            measured_rank=valid.measured_rank,
            rtol=valid.rtol,
            threshold=valid.threshold,
            input_shape=valid.input_shape,
        )

        result = projection_kernel(almost_orthonormal, almost_orthonormal)

        assert result.score.item() == 2.0
        assert result.normalized.item() == 1.0

    def test_rejects_large_score_bound_violation(self):
        valid = orthonormal_subspace(torch.eye(2))
        invalid = SubspaceBasis(
            basis=valid.basis * 1.01,
            singular_values=valid.singular_values,
            rank=valid.rank,
            measured_rank=valid.measured_rank,
            rtol=valid.rtol,
            threshold=valid.threshold,
            input_shape=valid.input_shape,
        )

        with pytest.raises(ValueError, match="outside its theoretical bounds"):
            projection_kernel(invalid, invalid, check_orthonormal=False)

    def test_rejects_ambient_dimension_mismatch(self):
        first = orthonormal_subspace(torch.eye(3))
        second = orthonormal_subspace(torch.eye(4))

        with pytest.raises(ValueError, match="ambient dimensions"):
            projection_kernel(first, second)


class TestRandomProjectionKernelMoments:
    def test_formula(self):
        result = random_projection_kernel_moments(8, 2)

        assert result.mean == pytest.approx(0.5)
        assert result.variance == pytest.approx(2 * 4 * 36 / (64 * 7 * 10))

    def test_full_space_is_constant(self):
        result = random_projection_kernel_moments(5, 5)

        assert result.mean == pytest.approx(5.0)
        assert result.variance == pytest.approx(0.0)

    @pytest.mark.parametrize(("ambient_dim", "rank"), [(1, 1), (4, 0), (4, 5), (True, 1)])
    def test_rejects_invalid_dimensions(self, ambient_dim, rank):
        with pytest.raises(ValueError):
            random_projection_kernel_moments(ambient_dim, rank)

    def test_seeded_monte_carlo_matches_mean(self):
        ambient_dim, rank, samples = 8, 2, 1000
        generator = torch.Generator().manual_seed(17)
        first, _ = torch.linalg.qr(
            torch.randn(samples, ambient_dim, rank, generator=generator, dtype=torch.float64)
        )
        second, _ = torch.linalg.qr(
            torch.randn(samples, ambient_dim, rank, generator=generator, dtype=torch.float64)
        )
        overlap = torch.einsum("sdr,sdk->srk", first, second)
        empirical_mean = overlap.square().sum(dim=(-2, -1)).mean().item()
        reference = random_projection_kernel_moments(ambient_dim, rank)
        standard_error = math.sqrt(reference.variance / samples)

        assert empirical_mean == pytest.approx(reference.mean, abs=6 * standard_error)
