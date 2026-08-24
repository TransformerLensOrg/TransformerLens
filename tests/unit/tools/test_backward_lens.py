"""Model-free tests for Backward Lens gradient-factor contracts."""

from dataclasses import FrozenInstanceError
from typing import cast

import pytest
import torch
import torch.nn.functional as F

from transformer_lens.tools.analysis.backward_lens import (
    LinearGradientFactors,
    WeightLayout,
    _build_linear_gradient_factors,
    _rank_vocabulary_logits,
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
    with pytest.raises(TypeError, match="parameter 'weight_layout'"):
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
    with pytest.raises(TypeError, match="torch.Tensor"):
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
    with pytest.raises(TypeError, match="parameter 'k'"):
        _rank_vocabulary_logits(torch.randn(2, 4), k=cast(int, 1.5), largest=True)


@pytest.mark.parametrize("largest", [1, "yes", None])
def test_vocabulary_ranking_rejects_non_boolean_largest(largest) -> None:
    with pytest.raises(TypeError, match="parameter 'largest'"):
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
    with pytest.raises(TypeError, match="torch.Tensor"):
        _rank_vocabulary_logits(cast(torch.Tensor, [1.0, 2.0]), k=1, largest=True)
