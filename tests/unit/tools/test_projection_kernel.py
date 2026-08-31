"""Unit tests for Projection Kernel subspace geometry and head affinity."""

import math
from types import SimpleNamespace

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from transformer_lens.tools.analysis.projection_kernel import (
    SubspaceBasis,
    _pairwise_projection_kernel,
    attention_head_subspace_affinity,
    orthonormal_subspace,
    projection_kernel,
    random_projection_kernel_moments,
)
from tests.typecheck_errors import TYPECHECK_ERRORS


def test_analysis_exports_are_alphabetized():
    from transformer_lens.tools import analysis

    assert analysis.__all__ == sorted(analysis.__all__)


class SyntheticAttention:
    def __init__(self, seed: int, *, d_model: int = 5, d_head: int = 2):
        generator = torch.Generator().manual_seed(seed)
        self.W_Q = torch.randn(2, d_model, d_head, generator=generator, dtype=torch.float64)
        self.W_K = torch.randn(1, d_model, d_head, generator=generator, dtype=torch.float64)
        self.W_V = torch.randn(1, d_model, d_head, generator=generator, dtype=torch.float64)
        self.W_O = torch.randn(2, d_head, d_model, generator=generator, dtype=torch.float64)


class SyntheticBlock:
    def __init__(self, attention):
        self.attn = attention


class SyntheticBridge:
    def __init__(self, layer_indices=(0, 2, 5)):
        self.cfg = SimpleNamespace(device="cpu", original_architecture="SyntheticArchitecture")
        self.attention_blocks = [
            (layer, SyntheticBlock(SyntheticAttention(10 + layer))) for layer in layer_indices
        ]

    def blocks_with(self, submodule):
        return self.attention_blocks if submodule == "attn" else []


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
        assert result.rtol == pytest.approx(torch.finfo(dtype).eps)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_default_detects_rank_deficiency(self, dtype):
        matrix = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=dtype)

        result = orthonormal_subspace(matrix)

        assert result.measured_rank == 1
        assert result.rtol == pytest.approx(torch.finfo(dtype).eps)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_default_handles_realistic_ambient_dimension(self, dtype):
        generator = torch.Generator().manual_seed(7)
        full_rank = torch.randn(4096, 8, generator=generator, dtype=torch.float64)
        rank_deficient = full_rank.clone()
        rank_deficient[:, -1] = 0.3 * full_rank[:, 0] + 0.7 * full_rank[:, 1]

        full_result = orthonormal_subspace(full_rank.to(dtype))
        deficient_result = orthonormal_subspace(rank_deficient.to(dtype))

        assert full_result.measured_rank == 8
        assert deficient_result.measured_rank == 7
        assert full_result.rtol == pytest.approx(torch.finfo(dtype).eps)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_preserves_supported_dtype_and_cpu_device(self, dtype):
        result = orthonormal_subspace(torch.eye(3, dtype=dtype, device="cpu"))

        assert result.basis.dtype == dtype
        assert result.singular_values.dtype == dtype
        assert result.basis.device == torch.device("cpu")

    @pytest.mark.parametrize(
        "matrix",
        [
            [[1.0]],
            torch.ones(3),
            torch.ones(2, 2, dtype=torch.int64),
            torch.ones(2, 2, dtype=torch.complex64),
        ],
    )
    def test_runtime_typecheck_rejects_invalid_matrices(self, matrix):
        with pytest.raises(TYPECHECK_ERRORS):
            orthonormal_subspace(matrix)

    @pytest.mark.parametrize(
        ("matrix", "message"),
        [
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

    def test_rank_deficient_input_has_exact_score_and_angle(self):
        rank_deficient = torch.tensor([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        tilted = torch.tensor([[3.0], [4.0], [0.0]], dtype=torch.float64)

        result = projection_kernel(
            orthonormal_subspace(rank_deficient), orthonormal_subspace(tilted)
        )

        assert result.rank_a == 1
        assert result.score.item() == pytest.approx(0.36)
        assert result.cosines.tolist() == pytest.approx([0.6])
        assert result.angles.tolist() == pytest.approx([math.acos(0.6)])

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
        assert result.cosines.tolist() == [1.0, 1.0]
        assert result.angles.tolist() == [0.0, 0.0]

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

    def test_rejects_large_principal_cosine_bound_violation(self):
        valid = orthonormal_subspace(torch.eye(2))
        invalid = SubspaceBasis(
            basis=torch.diag(torch.tensor([1.01, 0.1])),
            singular_values=valid.singular_values,
            rank=valid.rank,
            measured_rank=valid.measured_rank,
            rtol=valid.rtol,
            threshold=valid.threshold,
            input_shape=valid.input_shape,
        )

        with pytest.raises(ValueError, match="Principal-angle cosine"):
            projection_kernel(valid, invalid, check_orthonormal=False)

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

    def test_seeded_monte_carlo_matches_moments(self):
        ambient_dim, rank, samples = 8, 2, 5000
        generator = torch.Generator().manual_seed(17)
        first, _ = torch.linalg.qr(
            torch.randn(samples, ambient_dim, rank, generator=generator, dtype=torch.float64)
        )
        second, _ = torch.linalg.qr(
            torch.randn(samples, ambient_dim, rank, generator=generator, dtype=torch.float64)
        )
        overlap = torch.einsum("sdr,sdk->srk", first, second)
        scores = overlap.square().sum(dim=(-2, -1))
        empirical_mean = scores.mean().item()
        empirical_variance = scores.var(unbiased=False).item()
        reference = random_projection_kernel_moments(ambient_dim, rank)
        standard_error = math.sqrt(reference.variance / samples)

        assert empirical_mean == pytest.approx(reference.mean, abs=6 * standard_error)
        assert empirical_variance == pytest.approx(reference.variance, rel=0.1)


class TestPairwiseProjectionKernel:
    @pytest.mark.parametrize("max_temp_bytes", [1, 48, 96, 10_000])
    def test_tiled_scores_match_nested_loop_for_unequal_ranks(self, max_temp_bytes):
        generator = torch.Generator().manual_seed(23)
        source, _ = torch.linalg.qr(torch.randn(5, 7, 2, generator=generator, dtype=torch.float64))
        target, _ = torch.linalg.qr(torch.randn(4, 7, 3, generator=generator, dtype=torch.float64))

        actual = _pairwise_projection_kernel(source, target, max_temp_bytes=max_temp_bytes)
        expected = torch.empty(5, 4, dtype=torch.float64)
        for source_index in range(5):
            for target_index in range(4):
                overlap = source[source_index].T @ target[target_index]
                expected[source_index, target_index] = overlap.square().sum()

        assert torch.allclose(actual, expected)


class TestAttentionHeadSubspaceAffinity:
    def test_gqa_roles_have_native_rectangular_axes_and_hybrid_indices(self):
        model = SyntheticBridge()

        oq = attention_head_subspace_affinity(model, target_role="Q")
        ok = attention_head_subspace_affinity(model, target_role="K")
        ov = attention_head_subspace_affinity(model, target_role="V")

        assert oq.scores.shape == (3, 2, 3, 2)
        assert ok.scores.shape == (3, 2, 3, 1)
        assert ov.scores.shape == (3, 2, 3, 1)
        assert oq.source_layer_indices == (0, 2, 5)
        assert oq.target_layer_indices == (0, 2, 5)
        assert oq.source_head_kind == "query"
        assert oq.target_head_kind == "query"
        assert ok.target_head_kind == "kv"
        assert ov.target_head_kind == "kv"
        assert int(oq.valid_mask.sum()) == 12
        assert int(ok.valid_mask.sum()) == 6
        assert int(ov.valid_mask.sum()) == 6

    @pytest.mark.parametrize(("role", "attribute"), [("Q", "W_Q"), ("K", "W_K"), ("V", "W_V")])
    def test_sample_matches_independent_o_to_target_calculation(self, role, attribute):
        model = SyntheticBridge()

        result = attention_head_subspace_affinity(model, target_role=role)
        source_matrix = model.attention_blocks[0][1].attn.W_O[0].T
        target_matrix = getattr(model.attention_blocks[1][1].attn, attribute)[-1]
        expected = projection_kernel(
            orthonormal_subspace(source_matrix), orthonormal_subspace(target_matrix)
        )

        assert result.scores[0, 0, 1, -1].item() == pytest.approx(expected.score.item())
        assert result.normalized[0, 0, 1, -1].item() == pytest.approx(expected.normalized.item())

    def test_all_layer_order_includes_every_pair(self):
        result = attention_head_subspace_affinity(
            SyntheticBridge((1, 4)), target_role="K", layer_order="all"
        )

        assert bool(result.valid_mask.all())
        assert int(result.valid_mask.sum()) == 8

    def test_forward_valid_mask_is_contiguous_and_independently_writable(self):
        result = attention_head_subspace_affinity(SyntheticBridge((0, 3)), target_role="Q")

        assert result.valid_mask.is_contiguous()
        assert result.valid_mask[0, 0, 1, 0]
        assert result.valid_mask[0, 1, 1, 0]

        result.valid_mask[0, 0, 1, 0] = False

        assert result.valid_mask[0, 1, 1, 0]

    def test_invalid_entries_are_zero(self):
        result = attention_head_subspace_affinity(SyntheticBridge(), target_role="Q")

        assert torch.equal(result.scores[~result.valid_mask], torch.zeros(24, dtype=torch.float64))
        assert torch.equal(
            result.normalized[~result.valid_mask], torch.zeros(24, dtype=torch.float64)
        )

    def test_wrapper_does_not_retain_weight_autograd_graph(self):
        model = SyntheticBridge((0, 1))
        for _, block in model.attention_blocks:
            block.attn.W_Q.requires_grad_()
            block.attn.W_O.requires_grad_()

        result = attention_head_subspace_affinity(model, target_role="Q")

        assert not result.scores.requires_grad
        assert not result.normalized.requires_grad

    def test_result_dtype_device_and_rank_metadata(self):
        model = SyntheticBridge((0, 1))
        for _, block in model.attention_blocks:
            block.attn.W_Q = block.attn.W_Q.float()
            block.attn.W_O = block.attn.W_O.float()

        result = attention_head_subspace_affinity(model, target_role="Q")

        assert result.scores.dtype == torch.float32
        assert result.scores.device == torch.device("cpu")
        assert result.source_ranks.dtype == torch.long
        assert result.source_ranks.shape == (2, 2)
        assert result.target_ranks.shape == (2, 2)
        assert result.source_rank == 2
        assert result.target_rank == 2

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_wrapper_uses_least_precise_storage_dtype_for_default_rtol(self, dtype):
        model = SyntheticBridge((0, 1))
        for _, block in model.attention_blocks:
            block.attn.W_Q = block.attn.W_Q.to(dtype=dtype)

        result = attention_head_subspace_affinity(model, target_role="Q")

        assert result.rtol == pytest.approx(torch.finfo(dtype).eps)

    def test_rank_deficiency_names_role_layer_and_head(self):
        model = SyntheticBridge()
        deficient = model.attention_blocks[1][1].attn.W_Q[1]
        deficient[:, 1] = deficient[:, 0]

        with pytest.raises(ValueError, match="role Q at layer 2, head 1"):
            attention_head_subspace_affinity(model, target_role="Q")

    def test_explicit_common_truncation_accepts_rank_one_head(self):
        model = SyntheticBridge()
        deficient = model.attention_blocks[1][1].attn.W_Q[1]
        deficient[:, 1] = deficient[:, 0]

        result = attention_head_subspace_affinity(model, target_role="Q", rank=1)

        assert result.source_rank == 1
        assert result.target_rank == 1
        assert result.target_ranks[1, 1].item() == 1
        assert bool((result.normalized[result.valid_mask] <= 1 + 1e-12).all())

    def test_top_pairs_excludes_masked_entries_and_breaks_ties_lexicographically(self):
        model = SyntheticBridge((0, 3))
        common_q = torch.eye(4, dtype=torch.float64)[:, :2].expand(2, -1, -1).clone()
        common_o = common_q.transpose(-2, -1).clone()
        for _, block in model.attention_blocks:
            block.attn.W_Q = common_q
            block.attn.W_O = common_o

        result = attention_head_subspace_affinity(model, target_role="Q")
        pairs = result.top_pairs(10)

        assert len(pairs) == 4
        assert [
            (pair.source.layer, pair.source.head, pair.target.layer, pair.target.head)
            for pair in pairs
        ] == [(0, 0, 3, 0), (0, 0, 3, 1), (0, 1, 3, 0), (0, 1, 3, 1)]
        assert all(pair.score == pytest.approx(2.0) for pair in pairs)

    @pytest.mark.parametrize("k", [0, -1, True])
    def test_top_pairs_rejects_invalid_k(self, k):
        result = attention_head_subspace_affinity(SyntheticBridge(), target_role="Q")

        with pytest.raises(ValueError, match="k must be"):
            result.top_pairs(k)

    def test_rejects_unsupported_pairing_and_layer_order(self):
        model = SyntheticBridge()

        with pytest.raises(ValueError, match="source_role must be 'O'"):
            attention_head_subspace_affinity(model, source_role="Q", target_role="K")
        with pytest.raises(ValueError, match="target_role must be one of"):
            attention_head_subspace_affinity(model, target_role="O")
        with pytest.raises(ValueError, match="layer_order must be one of"):
            attention_head_subspace_affinity(model, target_role="Q", layer_order="backward")

    def test_rejects_no_attention_and_invalid_weight_shape(self):
        empty = SyntheticBridge(())
        invalid = SyntheticBridge()
        invalid.attention_blocks[0][1].attn.W_K = torch.ones(5, 2)

        with pytest.raises(ValueError, match="No attention layers"):
            attention_head_subspace_affinity(empty, target_role="Q")
        with pytest.raises(ValueError, match="role K at layer 0.*expected a three-dimensional"):
            attention_head_subspace_affinity(invalid, target_role="K")

    def test_rejects_missing_and_nonfinite_weights(self):
        missing = SyntheticBridge()
        nonfinite = SyntheticBridge()
        del missing.attention_blocks[0][1].attn.W_K
        nonfinite.attention_blocks[1][1].attn.W_K[0, 0, 0] = float("nan")

        with pytest.raises(ValueError, match="cannot expose role K at layer 0"):
            attention_head_subspace_affinity(missing, target_role="K")
        with pytest.raises(ValueError, match="role K at layer 2 must contain only finite"):
            attention_head_subspace_affinity(nonfinite, target_role="K")

    def test_preserves_not_implemented_error_from_weight_property(self):
        model = SyntheticBridge()
        original = model.attention_blocks[0][1].attn

        class UnsupportedAttention:
            W_O = original.W_O

            @property
            def W_K(self):
                raise NotImplementedError("K weights are not supported")

        model.attention_blocks[0] = (0, SyntheticBlock(UnsupportedAttention()))

        with pytest.raises(
            NotImplementedError, match="cannot expose role K at layer 0"
        ) as exc_info:
            attention_head_subspace_affinity(model, target_role="K")

        assert isinstance(exc_info.value.__cause__, NotImplementedError)

    @pytest.mark.parametrize("shape", [(2, 5, 2), (1, 6, 2), (1, 5, 3)])
    def test_rejects_inconsistent_role_shapes(self, shape):
        model = SyntheticBridge()
        model.attention_blocks[1][1].attn.W_K = torch.ones(shape, dtype=torch.float64)

        with pytest.raises(ValueError, match="consistent.*shape"):
            attention_head_subspace_affinity(model, target_role="K")

    def test_rejects_empty_head_axis(self):
        model = SyntheticBridge((0,))
        model.attention_blocks[0][1].attn.W_K = torch.empty(0, 5, 2)

        with pytest.raises(ValueError, match="must have non-empty head"):
            attention_head_subspace_affinity(model, target_role="K", layer_order="all")
