"""Model-free tests for Backward Lens gradient-factor contracts."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from beartype.roar import BeartypeCallHintParamViolation

from transformer_lens.tools.analysis.backward_lens import (
    BackwardLensMatrixResult,
    LinearGradientFactors,
    WeightLayout,
    _build_linear_gradient_factors,
    _build_matrix_result,
    _factor_norms_and_normalized_rows,
    _project_residual_factors,
    _rank_vocabulary_logits,
)


class _ReadoutModel:
    def __init__(self, *, affine: bool = True):
        self.cfg = SimpleNamespace(d_model=3)
        self.ln_final = torch.nn.LayerNorm(3, elementwise_affine=affine)
        self.unembed = torch.nn.Linear(3, 5, bias=False)
        with torch.no_grad():
            self.unembed.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10)

    @property
    def W_U(self) -> torch.Tensor:
        return self.unembed.weight.T


def _matrix_result(
    logits: torch.Tensor,
    *,
    normalized: bool = True,
    return_full_logits: bool = True,
    top_k: int = 2,
) -> BackwardLensMatrixResult:
    factors = _build_linear_gradient_factors(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[0.5, -1.0], [2.0, 1.5]]),
        torch.tensor([[6.5, 3.5], [9.0, 4.0]]),
        weight_layout="in_out",
    )
    normalized_logits = logits * 2 if normalized else None
    target_token_id = 2
    return BackwardLensMatrixResult(
        factors=factors,
        projected_factor="forward_inputs",
        factor_norms=torch.tensor([1.0, 2.0]),
        zero_norm_mask=torch.tensor([False, False]),
        vocabulary_size=logits.shape[-1],
        target_token_id=target_token_id,
        top_ranking=_rank_vocabulary_logits(logits, k=top_k, largest=True),
        bottom_ranking=_rank_vocabulary_logits(logits, k=top_k, largest=False),
        target_largest_ranks=(logits > logits[:, target_token_id, None]).sum(
            dim=-1, dtype=torch.int64
        ),
        target_smallest_ranks=(logits < logits[:, target_token_id, None]).sum(
            dim=-1, dtype=torch.int64
        ),
        normalized_top_ranking=(
            _rank_vocabulary_logits(normalized_logits, k=top_k, largest=True)
            if normalized_logits is not None
            else None
        ),
        normalized_bottom_ranking=(
            _rank_vocabulary_logits(normalized_logits, k=top_k, largest=False)
            if normalized_logits is not None
            else None
        ),
        normalized_target_largest_ranks=(
            (normalized_logits > normalized_logits[:, target_token_id, None]).sum(
                dim=-1, dtype=torch.int64
            )
            if normalized_logits is not None
            else None
        ),
        normalized_target_smallest_ranks=(
            (normalized_logits < normalized_logits[:, target_token_id, None]).sum(
                dim=-1, dtype=torch.int64
            )
            if normalized_logits is not None
            else None
        ),
        vocabulary_logits=logits.clone() if return_full_logits else None,
        normalized_vocabulary_logits=(
            normalized_logits.clone()
            if return_full_logits and normalized_logits is not None
            else None
        ),
    )


def test_reconstructs_gpt2_style_in_out_gradient_from_autograd() -> None:
    torch.manual_seed(1)
    inputs = torch.randn(7, 3)
    weight = torch.randn(3, 5, requires_grad=True)
    output_gradients = torch.randn(7, 5)
    outputs = inputs @ weight
    (weight_gradient,) = torch.autograd.grad(outputs, weight, grad_outputs=output_gradients)

    result = _build_linear_gradient_factors(
        inputs, output_gradients, weight_gradient, weight_layout="in_out"
    )

    assert isinstance(result, LinearGradientFactors)
    assert result.reconstructed_gradient.shape == (3, 5)
    assert torch.allclose(result.reconstructed_gradient, weight_gradient)
    assert result.absolute_reconstruction_error < 1e-6
    assert result.relative_reconstruction_error < 1e-6


def test_reconstructs_out_in_gradient_from_autograd() -> None:
    torch.manual_seed(2)
    inputs = torch.randn(6, 3)
    weight = torch.randn(5, 3, requires_grad=True)
    output_gradients = torch.randn(6, 5)
    outputs = F.linear(inputs, weight)
    (weight_gradient,) = torch.autograd.grad(outputs, weight, grad_outputs=output_gradients)

    result = _build_linear_gradient_factors(
        inputs, output_gradients, weight_gradient, weight_layout="out_in"
    )

    assert result.reconstructed_gradient.shape == (5, 3)
    assert torch.allclose(result.reconstructed_gradient, weight_gradient)
    assert result.absolute_reconstruction_error < 1e-6
    assert result.relative_reconstruction_error < 1e-6


def test_asymmetric_dimensions_reject_wrong_orientation() -> None:
    inputs = torch.randn(4, 2)
    output_gradients = torch.randn(4, 5)
    out_in_gradient = output_gradients.T @ inputs

    with pytest.raises(ValueError, match="does not match"):
        _build_linear_gradient_factors(
            inputs, output_gradients, out_in_gradient, weight_layout="in_out"
        )


def test_result_owns_detached_float32_copies_of_all_inputs() -> None:
    inputs = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    output_gradients = torch.randn(3, 4, dtype=torch.float64)
    weight_gradient = inputs.detach().T @ output_gradients
    result = _build_linear_gradient_factors(
        inputs, output_gradients, weight_gradient, weight_layout="in_out"
    )
    saved_inputs = result.forward_inputs.clone()
    saved_output_gradients = result.output_gradients.clone()
    saved_weight_gradient = result.weight_gradient.clone()

    with torch.no_grad():
        inputs.add_(100)
        output_gradients.add_(100)
        weight_gradient.add_(100)

    assert result.forward_inputs.dtype == torch.float32
    assert result.forward_inputs.device.type == "cpu"
    assert result.forward_inputs.grad_fn is None
    assert torch.equal(result.forward_inputs, saved_inputs)
    assert torch.equal(result.output_gradients, saved_output_gradients)
    assert torch.equal(result.weight_gradient, saved_weight_gradient)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "weight_layout", "out_in")


def test_zero_gradient_has_finite_zero_errors() -> None:
    result = _build_linear_gradient_factors(
        torch.zeros(3, 2), torch.zeros(3, 4), torch.zeros(2, 4), weight_layout="in_out"
    )

    assert result.absolute_reconstruction_error == 0.0
    assert result.relative_reconstruction_error == 0.0


def test_zero_reference_with_nonzero_reconstruction_has_finite_relative_error() -> None:
    result = _build_linear_gradient_factors(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.zeros(2, 3),
        weight_layout="in_out",
    )

    assert result.absolute_reconstruction_error == 1.0
    assert result.relative_reconstruction_error == 1.0
    assert torch.isfinite(torch.tensor(result.relative_reconstruction_error))


@pytest.mark.parametrize(
    ("inputs", "gradients", "weight", "message"),
    [
        (torch.randn(2), torch.randn(3, 4), torch.randn(2, 4), "rank 2"),
        (torch.randn(3, 2), torch.randn(4, 4), torch.randn(2, 4), "same number"),
        (torch.empty(0, 2), torch.empty(0, 4), torch.randn(2, 4), "empty"),
        (torch.full((3, 2), float("nan")), torch.ones(3, 4), torch.ones(2, 4), "finite"),
        (torch.ones(3, 2), torch.full((3, 4), float("inf")), torch.ones(2, 4), "finite"),
        (torch.ones(3, 2), torch.ones(3, 4), torch.full((2, 4), float("nan")), "finite"),
    ],
)
def test_factor_validation_errors(inputs, gradients, weight, message) -> None:
    with pytest.raises(ValueError, match=message):
        _build_linear_gradient_factors(inputs, gradients, weight, weight_layout="in_out")


def test_rejects_invalid_layout_and_integer_factors() -> None:
    with pytest.raises((TypeError, BeartypeCallHintParamViolation), match="weight_layout"):
        _build_linear_gradient_factors(
            torch.randn(3, 2),
            torch.randn(3, 4),
            torch.randn(2, 4),
            weight_layout=cast(WeightLayout, "other"),
        )
    with pytest.raises(TypeError, match="floating"):
        _build_linear_gradient_factors(
            torch.ones(3, 2, dtype=torch.long),
            torch.randn(3, 4),
            torch.randn(2, 4),
            weight_layout="in_out",
        )


def test_rejects_non_tensor_factors() -> None:
    with pytest.raises((TypeError, BeartypeCallHintParamViolation), match="torch.Tensor"):
        _build_linear_gradient_factors(
            cast(torch.Tensor, "not a tensor"),
            torch.randn(3, 4),
            torch.randn(2, 4),
            weight_layout="in_out",
        )


def test_rejects_values_that_overflow_during_float32_conversion() -> None:
    with pytest.raises(ValueError, match="remain finite"):
        _build_linear_gradient_factors(
            torch.full((3, 2), 1e300, dtype=torch.float64),
            torch.ones(3, 4, dtype=torch.float64),
            torch.full((2, 4), 3e300, dtype=torch.float64),
            weight_layout="in_out",
        )


def test_rejects_nonfinite_float32_reconstruction() -> None:
    large = torch.full((2, 1), 3e38)
    with pytest.raises(ValueError, match="reconstruction"):
        _build_linear_gradient_factors(
            large,
            large,
            torch.zeros(1, 1),
            weight_layout="in_out",
        )


def test_vocabulary_rankings_match_topk_without_mutation() -> None:
    logits = torch.tensor([[1.0, -3.0, 7.0, 2.0], [8.0, 0.5, -2.0, 4.0]])
    original = logits.clone()

    top = _rank_vocabulary_logits(logits, k=2, largest=True)
    bottom = _rank_vocabulary_logits(logits, k=2, largest=False)

    assert top.indices.tolist() == [[2, 3], [0, 3]]
    assert top.values.tolist() == [[7.0, 2.0], [8.0, 4.0]]
    assert bottom.indices.tolist() == [[1, 0], [2, 1]]
    assert torch.equal(logits, original)


def test_one_dimensional_vocabulary_ranking_is_detached_cpu_copy() -> None:
    logits = torch.tensor([2.0, -4.0, 8.0, 1.0], requires_grad=True)
    bottom = _rank_vocabulary_logits(logits, k=2, largest=False)
    saved_values = bottom.values.clone()

    with torch.no_grad():
        logits.add_(100)

    assert bottom.indices.tolist() == [1, 3]
    assert bottom.values.tolist() == [-4.0, 1.0]
    assert bottom.values.device.type == "cpu"
    assert bottom.values.grad_fn is None
    assert torch.equal(bottom.values, saved_values)


@pytest.mark.parametrize("k", [0, 5, True])
def test_vocabulary_ranking_rejects_invalid_k(k) -> None:
    with pytest.raises(ValueError, match="k must"):
        _rank_vocabulary_logits(torch.randn(2, 4), k=k, largest=True)


def test_vocabulary_ranking_runtime_typechecking_rejects_non_integer_k() -> None:
    with pytest.raises((TypeError, BeartypeCallHintParamViolation), match="parameter.*k"):
        _rank_vocabulary_logits(torch.randn(2, 4), k=cast(int, 1.5), largest=True)


@pytest.mark.parametrize("largest", [1, "yes", None])
def test_vocabulary_ranking_rejects_non_boolean_largest(largest) -> None:
    with pytest.raises((TypeError, BeartypeCallHintParamViolation), match="parameter.*largest"):
        _rank_vocabulary_logits(torch.randn(2, 4), k=1, largest=cast(bool, largest))


@pytest.mark.parametrize(
    "logits",
    [
        torch.randn(2, 3, 4),
        torch.empty(2, 0),
        torch.ones(4, dtype=torch.long),
        torch.tensor([0.0, float("nan")]),
        torch.tensor([0.0, float("inf")]),
    ],
)
def test_vocabulary_ranking_rejects_invalid_logits(logits) -> None:
    with pytest.raises((TypeError, ValueError)):
        _rank_vocabulary_logits(logits, k=1, largest=True)


def test_vocabulary_ranking_rejects_non_tensor_logits() -> None:
    with pytest.raises((TypeError, BeartypeCallHintParamViolation), match="torch.Tensor"):
        _rank_vocabulary_logits(cast(torch.Tensor, [1.0, 2.0]), k=1, largest=True)


def test_projection_matches_fresh_final_norm_and_unembedding() -> None:
    model = _ReadoutModel()
    factors = torch.tensor([[1.0, -2.0, 4.0], [0.25, 0.5, -0.75]])

    actual = _project_residual_factors(model, factors)
    expected = model.unembed(model.ln_final(factors.unsqueeze(0))).squeeze(0)

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    assert actual.grad_fn is None


def test_normalized_rows_preserve_norms_and_handle_zero_without_nan() -> None:
    factors = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])

    norms, zero_mask, normalized = _factor_norms_and_normalized_rows(factors)

    assert norms.tolist() == [5.0, 0.0]
    assert zero_mask.tolist() == [False, True]
    torch.testing.assert_close(normalized[0], torch.tensor([0.6, 0.8, 0.0]))
    assert torch.equal(normalized[1], torch.zeros(3))
    assert torch.isfinite(normalized).all()


def test_normalized_projection_changes_low_norm_readout_but_keeps_zero_finite() -> None:
    model = _ReadoutModel()
    factors = torch.tensor([[1e-6, -2e-6, 1e-6], [0.0, 0.0, 0.0]])
    _, _, normalized = _factor_norms_and_normalized_rows(factors)

    raw_logits = _project_residual_factors(model, factors)
    normalized_logits = _project_residual_factors(model, normalized)

    assert not torch.allclose(raw_logits[0], normalized_logits[0])
    assert torch.isfinite(normalized_logits).all()


def test_projection_sign_symmetry_depends_on_final_norm_bias() -> None:
    factors = torch.tensor([[1.0, -2.0, 4.0]])
    bias_free = _ReadoutModel(affine=False)
    biased = _ReadoutModel(affine=True)
    with torch.no_grad():
        biased.ln_final.bias.fill_(0.25)

    torch.testing.assert_close(
        _project_residual_factors(bias_free, -factors),
        -_project_residual_factors(bias_free, factors),
    )
    assert not torch.allclose(
        _project_residual_factors(biased, -factors),
        -_project_residual_factors(biased, factors),
    )


def test_matrix_result_rankings_decoding_and_target_rank_conventions() -> None:
    logits = torch.tensor([[3.0, 1.0, -2.0, 0.0], [0.0, -1.0, 4.0, 2.0]])
    result = _matrix_result(logits)

    assert result.top(k=2).indices.tolist() == [[0, 1], [2, 3]]
    assert result.bottom(k=2).indices.tolist() == [[2, 3], [1, 0]]
    assert result.target_ranks(2, largest=True).tolist() == [3, 0]
    assert result.gradient_descent_target_ranks(2).tolist() == [0, 3]
    assert result.target_ranks(1, largest=True).tolist() == [1, 3]
    assert torch.equal(result.logits(normalized=True), logits * 2)

    tokenizer = SimpleNamespace(decode=lambda ids: f"token-{ids[0]}")
    assert result.top_tokens(tokenizer, k=1) == [["token-0"], ["token-2"]]
    assert result.bottom_tokens(tokenizer, k=1) == [["token-2"], ["token-1"]]


def test_matrix_result_rejects_missing_normalized_logits_and_invalid_target() -> None:
    logits = torch.randn(2, 4)
    result = _matrix_result(logits)
    result_without_normalized = _matrix_result(logits, normalized=False)

    with pytest.raises(ValueError, match="were not requested"):
        result_without_normalized.logits(normalized=True)
    with pytest.raises(ValueError, match="target_token_id"):
        result.target_ranks(4, largest=True)
    with pytest.raises(TypeError, match="decode"):
        result.top_tokens(object(), k=1)


@pytest.mark.parametrize("k", [0, 3, True])
def test_matrix_result_rejects_invalid_retained_ranking_k(k) -> None:
    result = _matrix_result(torch.randn(2, 4))

    with pytest.raises(ValueError, match="retained top_k=2"):
        result.top(k=cast(int, k))


def test_matrix_result_keeps_rankings_and_analyzed_target_ranks_without_full_logits() -> None:
    result = _matrix_result(
        torch.tensor([[3.0, 1.0, -2.0, 0.0], [0.0, -1.0, 4.0, 2.0]]),
        return_full_logits=False,
    )

    assert result.vocabulary_logits is None
    assert result.normalized_vocabulary_logits is None
    assert result.top(k=2).indices.tolist() == [[0, 1], [2, 3]]
    assert result.bottom(k=2).indices.tolist() == [[2, 3], [1, 0]]
    assert result.gradient_descent_target_ranks(2).tolist() == [0, 3]
    assert result.gradient_descent_target_ranks(2, normalized=True).tolist() == [0, 3]
    with pytest.raises(ValueError, match="return_full_logits=True"):
        result.logits()
    with pytest.raises(ValueError, match="return_full_logits=True"):
        result.target_ranks(1, largest=True)


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("return_full_logits", [False, True])
def test_build_matrix_result_bounds_storage_and_matches_projection(
    normalized: bool, return_full_logits: bool
) -> None:
    model = _ReadoutModel()
    inputs = torch.tensor([[1.0, -2.0, 4.0], [0.25, 0.5, -0.75]])
    output_gradients = torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]])
    factors = _build_linear_gradient_factors(
        inputs,
        output_gradients,
        inputs.T @ output_gradients,
        weight_layout="in_out",
    )
    direct = _project_residual_factors(model, inputs)

    result = _build_matrix_result(
        model,
        factors,
        projected_factor="forward_inputs",
        include_normalized_logits=normalized,
        target_token_id=2,
        top_k=2,
        return_full_logits=return_full_logits,
    )

    direct_top = torch.topk(direct, k=2, dim=-1, largest=True)
    direct_bottom = torch.topk(direct, k=2, dim=-1, largest=False)
    assert torch.equal(result.top_ranking.values, direct_top.values)
    assert torch.equal(result.top_ranking.indices, direct_top.indices)
    assert torch.equal(result.bottom_ranking.values, direct_bottom.values)
    assert torch.equal(result.bottom_ranking.indices, direct_bottom.indices)
    assert (result.vocabulary_logits is not None) is return_full_logits
    assert (result.normalized_top_ranking is not None) is normalized
    assert (result.normalized_bottom_ranking is not None) is normalized
    assert (result.normalized_vocabulary_logits is not None) is (normalized and return_full_logits)


def test_ranking_accessors_return_owned_retained_prefixes() -> None:
    result = _matrix_result(torch.tensor([[3.0, 1.0, -2.0, 0.0]]))
    top = result.top(k=1)
    bottom = result.bottom(k=1, normalized=True)

    top.values.add_(100)
    top.indices.add_(100)
    bottom.values.add_(100)
    bottom.indices.add_(100)

    assert result.top_ranking.values.tolist() == [[3.0, 1.0]]
    assert result.top_ranking.indices.tolist() == [[0, 1]]
    assert result.normalized_bottom_ranking is not None
    assert result.normalized_bottom_ranking.values.tolist() == [[-4.0, 0.0]]
    assert result.normalized_bottom_ranking.indices.tolist() == [[2, 3]]


def test_public_backward_lens_symbols_are_exported() -> None:
    from transformer_lens.tools.analysis import (
        BackwardLens,
        BackwardLensLayerResult,
        BackwardLensMatrixResult,
        BackwardLensResult,
    )

    assert BackwardLens.__name__ == "BackwardLens"
    assert BackwardLensLayerResult.__name__ == "BackwardLensLayerResult"
    assert BackwardLensMatrixResult.__name__ == "BackwardLensMatrixResult"
    assert BackwardLensResult.__name__ == "BackwardLensResult"
