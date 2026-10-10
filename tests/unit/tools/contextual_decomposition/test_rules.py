"""Independent constituent-share oracles for affine and ordered GELU allocation."""

import math
from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn.functional as F
from transformers.activations import NewGELUActivation

from transformer_lens.tools.analysis.contextual_decomposition.rules import (
    _affine,
    _CDConstituents,
    _gelu,
)

DTYPES = (torch.float32, torch.float64)
GELU_VARIANTS = ("gelu", "gelu_new", "gelu_pytorch_tanh")


def _scalar_gelu(value: float, variant: str) -> float:
    if variant == "gelu":
        return value * (1.0 + math.erf(value / math.sqrt(2.0))) / 2.0
    return (
        value * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value**3))) / 2.0
    )


def _affine_oracle(beta, gamma, weight, bias):
    relevant, remaining = [], []
    for column, offset in zip(zip(*weight), bias):
        b = sum(value * coefficient for value, coefficient in zip(beta, column))
        g = sum(value * coefficient for value, coefficient in zip(gamma, column))
        denominator = abs(b) + abs(g)
        share = abs(b) / denominator if denominator else 0.0
        relevant.append(b + share * offset)
        remaining.append(g + (1.0 - share) * offset)
    return relevant, remaining


@pytest.mark.parametrize("dtype", DTYPES)
def test_affine_matches_independent_scalar_shares(dtype):
    beta_rows = [[0.0, 0.0], [1.0, -2.0], [2.0, 1.0], [-2.0, -1.0]]
    gamma_rows = [[0.0, 0.0], [-1.0, 2.0], [-1.0, 3.0], [1.0, 2.0]]
    weights = [[2.0, -1.0, 0.0], [1.0, 3.0, 0.0]]
    offsets = [3.0, -2.0, 5.0]
    expected = [_affine_oracle(b, g, weights, offsets) for b, g in zip(beta_rows, gamma_rows)]
    parts = _CDConstituents(
        torch.tensor(beta_rows, dtype=dtype), torch.tensor(gamma_rows, dtype=dtype)
    )
    weight = torch.tensor(weights, dtype=dtype)
    bias = torch.tensor(offsets, dtype=dtype)
    result = _affine(parts, weight, bias)
    torch.testing.assert_close(result.beta, torch.tensor([row[0] for row in expected], dtype=dtype))
    torch.testing.assert_close(
        result.gamma, torch.tensor([row[1] for row in expected], dtype=dtype)
    )
    torch.testing.assert_close(
        result.beta + result.gamma, (parts.beta + parts.gamma) @ weight + bias
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_projected_constituents_assign_all_bias_to_gamma(dtype):
    parts = _CDConstituents(torch.tensor([1.0, -1.0], dtype=dtype), torch.zeros(2, dtype=dtype))
    result = _affine(parts, torch.ones(2, 1, dtype=dtype), torch.tensor([7.0], dtype=dtype))
    assert torch.equal(result.beta, torch.zeros(1, dtype=dtype))
    assert torch.equal(result.gamma, torch.tensor([7.0], dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_allocation_does_not_lose_tiny_projected_mass(dtype):
    parts = _CDConstituents(torch.tensor([1e-30], dtype=dtype), torch.tensor([3e-30], dtype=dtype))
    result = _affine(parts, torch.ones(1, 1, dtype=dtype), torch.tensor([8.0], dtype=dtype))
    torch.testing.assert_close(result.beta, torch.tensor([2.0], dtype=dtype))
    torch.testing.assert_close(result.gamma, torch.tensor([6.0], dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_ratio_does_not_overflow_for_large_cancelling_magnitudes(dtype):
    maximum = torch.finfo(dtype).max
    parts = _CDConstituents(
        torch.tensor([maximum * 0.75], dtype=dtype),
        torch.tensor([-maximum * 0.5], dtype=dtype),
    )
    result = _affine(
        parts, torch.ones(1, 1, dtype=dtype), torch.tensor([maximum * 0.1], dtype=dtype)
    )
    torch.testing.assert_close(result.beta / maximum, torch.tensor([0.81], dtype=dtype))
    torch.testing.assert_close(result.gamma / maximum, torch.tensor([-0.46], dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_affine_without_bias_handles_noncontiguous_batched_inputs(dtype):
    beta = torch.arange(24, dtype=dtype).reshape(2, 3, 4).transpose(0, 1)
    gamma = -beta / 2.0
    parts = _CDConstituents(beta, gamma)
    weight = torch.arange(8, dtype=dtype).reshape(4, 2)
    result = _affine(parts, weight)
    torch.testing.assert_close(result.beta, beta @ weight)
    torch.testing.assert_close(result.gamma, gamma @ weight)
    assert result.beta.shape == (3, 2, 2)


@pytest.mark.parametrize("variant", GELU_VARIANTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_matches_scalar_ordered_allocation(dtype, variant):
    beta = [0.0, 1.0, -2.0, 4.0, -4.0, 1.5, 10.0, -10.0]
    gamma = [0.0, -1.0, 0.5, -4.0, 4.0, -0.25, 20.0, -20.0]
    parts = _CDConstituents(torch.tensor(beta, dtype=dtype), torch.tensor(gamma, dtype=dtype))
    result = _gelu(parts, variant=variant)
    expected_gamma = [_scalar_gelu(value, variant) for value in gamma]
    expected_beta = [
        _scalar_gelu(b + g, variant) - remaining
        for b, g, remaining in zip(beta, gamma, expected_gamma)
    ]
    torch.testing.assert_close(result.beta, torch.tensor(expected_beta, dtype=dtype))
    torch.testing.assert_close(result.gamma, torch.tensor(expected_gamma, dtype=dtype))
    torch.testing.assert_close(
        result.beta + result.gamma,
        torch.tensor([_scalar_gelu(b + g, variant) for b, g in zip(beta, gamma)], dtype=dtype),
    )


def test_reconstruction_preserving_wrong_share_fails_the_oracle():
    parts = _CDConstituents(
        torch.tensor([1.0], dtype=torch.float64), torch.tensor([-1.0], dtype=torch.float64)
    )
    result = _gelu(parts, variant="gelu")
    wrong_beta = F.gelu(parts.beta)
    wrong_gamma = result.beta + result.gamma - wrong_beta
    torch.testing.assert_close(wrong_beta + wrong_gamma, result.beta + result.gamma)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            wrong_beta, torch.tensor([-_scalar_gelu(-1.0, "gelu")], dtype=torch.float64)
        )


@pytest.mark.parametrize("variant", GELU_VARIANTS)
def test_gelu_zero_relevant_input_preserves_the_remaining_share(variant):
    gamma = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
    result = _gelu(_CDConstituents(torch.zeros_like(gamma), gamma), variant=variant)
    assert torch.equal(result.beta, torch.zeros_like(gamma))


def test_outputs_are_detached_and_do_not_alias_or_mutate_inputs():
    beta = torch.tensor([1.0, -2.0], requires_grad=True)
    gamma = torch.tensor([0.5, 3.0], requires_grad=True)
    weight = torch.eye(2, requires_grad=True)
    bias = torch.ones(2, requires_grad=True)
    originals = [value.detach().clone() for value in (beta, gamma, weight, bias)]
    parts = _CDConstituents(beta, gamma)
    for result in (_affine(parts, weight, bias), _gelu(parts)):
        assert not result.beta.requires_grad and not result.gamma.requires_grad
        result.beta.zero_()
        result.gamma.zero_()
    for value, original in zip((beta, gamma, weight, bias), originals):
        assert torch.equal(value, original)
        assert value.grad is None


def test_policy_identity_is_frozen():
    parts = _CDConstituents(torch.ones(1), torch.zeros(1))
    assert _gelu(parts).policy == _affine(parts, torch.ones(1, 1)).policy == parts.policy
    with pytest.raises(FrozenInstanceError):
        setattr(parts, "policy", "different")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("component", ["beta", "gamma"])
def test_constituents_reject_nonfinite_values(bad, component):
    values = {"beta": torch.ones(1), "gamma": torch.zeros(1)}
    values[component] = torch.tensor([bad])
    with pytest.raises(ValueError, match="finite"):
        _CDConstituents(**values)


@pytest.mark.parametrize(
    "beta,gamma,match",
    [
        (torch.ones(2), torch.zeros(1), "shape"),
        (torch.ones(1), torch.zeros(1, dtype=torch.float64), "dtype"),
        (torch.ones(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64), "float32 or float64"),
        (
            torch.ones(1, dtype=torch.float16),
            torch.zeros(1, dtype=torch.float16),
            "float32 or float64",
        ),
        (torch.tensor(1.0), torch.tensor(0.0), "dimension"),
        (torch.empty(0), torch.empty(0), "empty"),
        (torch.ones(1), torch.empty(1, device="meta"), "device"),
        (torch.empty(1, device="meta"), torch.empty(1, device="meta"), "materialized"),
        (torch.ones(2).to_sparse(), torch.zeros(2).to_sparse(), "strided"),
    ],
)
def test_constituents_reject_malformed_tensors(beta, gamma, match):
    with pytest.raises(ValueError, match=match):
        _CDConstituents(beta, gamma)


@pytest.mark.parametrize("operand", ["weight", "bias"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_affine_rejects_nonfinite_operands(operand, bad):
    args = {"weight": torch.ones(1, 1), "bias": torch.zeros(1)}
    args[operand].fill_(bad)
    with pytest.raises(ValueError, match="finite"):
        _affine(_CDConstituents(torch.ones(1), torch.zeros(1)), **args)


@pytest.mark.parametrize(
    "weight,bias,match",
    [
        (torch.ones(2), None, "weight.*shape"),
        (torch.ones(2, 1), None, "weight.*shape"),
        (torch.ones(1, 0), None, "empty"),
        (torch.ones(1, 1, dtype=torch.float64), None, "dtype"),
        (torch.empty(1, 1, device="meta"), None, "device"),
        (torch.ones(1, 1), torch.ones(1, 1), "bias.*shape"),
        (torch.ones(1, 1), torch.ones(2), "bias.*shape"),
        (torch.ones(1, 1), torch.ones(1, dtype=torch.float64), "dtype"),
    ],
)
def test_affine_rejects_malformed_operands(weight, bias, match):
    with pytest.raises(ValueError, match=match):
        _affine(_CDConstituents(torch.ones(1), torch.zeros(1)), weight, bias)


def test_overflow_is_rejected_instead_of_returning_nonfinite_shares():
    maximum = torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match="finite"):
        _CDConstituents(torch.tensor([maximum]), torch.tensor([maximum]))
    parts = _CDConstituents(torch.tensor([maximum]), torch.zeros(1))
    with pytest.raises(ValueError, match="finite"):
        _affine(parts, torch.tensor([[2.0]]))


@pytest.mark.parametrize("dtype", DTYPES)
def test_small_gamma_bias_share_is_not_rounded_to_zero(dtype):
    parts = _CDConstituents(torch.tensor([1.0], dtype=dtype), torch.tensor([1e-20], dtype=dtype))
    result = _affine(parts, torch.ones(1, 1, dtype=dtype), torch.tensor([1e20], dtype=dtype))
    torch.testing.assert_close(result.gamma, torch.ones(1, dtype=dtype))


@pytest.mark.parametrize("operation", [_gelu, lambda parts: _affine(parts, torch.ones(1, 1))])
def test_mutated_nonfinite_constituents_are_revalidated(operation):
    parts = _CDConstituents(torch.ones(1), torch.zeros(1))
    parts.gamma.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        operation(parts)


@pytest.mark.parametrize("variant", GELU_VARIANTS)
def test_gelu_handles_noncontiguous_leading_dimensions(variant):
    beta = torch.linspace(-2, 2, 24, dtype=torch.float64).reshape(2, 3, 4).transpose(0, 1)
    gamma = beta / 3
    result = _gelu(_CDConstituents(beta, gamma), variant=variant)
    expected_gamma = torch.tensor(
        [_scalar_gelu(value, variant) for value in gamma.flatten().tolist()], dtype=torch.float64
    ).reshape(gamma.shape)
    torch.testing.assert_close(result.gamma, expected_gamma)
    assert result.beta.shape == beta.shape


def test_explicit_gpt2_gelu_preserves_native_operation_order():
    beta = torch.tensor([[-2.0, 0.0, 1.0]], dtype=torch.float32)
    gamma = torch.tensor([[0.25, -1.0, 3.0]], dtype=torch.float32)
    result = _gelu(_CDConstituents(beta, gamma), variant="gelu_new")
    native = NewGELUActivation()
    assert torch.equal(result.gamma, native(gamma))
    assert torch.equal(result.beta, native(beta + gamma) - native(gamma))


def test_gelu_rejects_unsupported_variant():
    with pytest.raises(ValueError, match="variant"):
        _gelu(_CDConstituents(torch.ones(1), torch.zeros(1)), variant="silu")
