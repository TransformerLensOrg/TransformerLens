"""Analytic closed-form tests for the LN-, Identity-, and Half-relevance-rule primitives.

Model-free: every primitive is a plain ``torch.autograd.Function`` exercised on
synthetic tensors, so no model, backward hook, or ``.data`` access is involved. Each
test asserts two properties independently: the forward value is exactly the native
(ordinary-autograd) value, and the backward value matches the rule's closed-form VJP
rather than what ordinary autodiff would produce.
"""

import math
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from transformer_lens.model_bridge._relevance_rules import (
    half_rule,
    identity_rule,
    ln_rule,
)

DTYPES = [torch.float32, torch.float64]


def _leaf(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return values.to(dtype).clone().requires_grad_(True)


def _sample_rows(dtype: torch.dtype) -> torch.Tensor:
    """A batch covering an all-zero, all-negative, mixed-sign, and positive row."""
    values = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -2.0, -0.5, -3.0, -0.25],
            [1.0, -2.0, 3.0, -4.0, 0.5],
            [2.0, 4.0, 6.0, 8.0, 10.0],
        ]
    )
    return values.to(dtype)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TestLNRule:
    """Forward equals ``numerator / denom``; the VJP treats ``denom`` as constant."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_matches_plain_division(self, dtype):
        numerator = _sample_rows(dtype)
        denom = numerator.abs().mean(dim=-1, keepdim=True) + 1e-6
        assert torch.equal(ln_rule(numerator, denom), numerator / denom)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_backward_treats_denom_as_constant(self, dtype):
        eps = 1e-6
        base = _sample_rows(dtype)
        grad_out = torch.ones_like(base)

        x_rule = _leaf(base, dtype)
        denom_rule = x_rule.abs().mean(dim=-1, keepdim=True) + eps
        y_rule = ln_rule(x_rule, denom_rule)
        (grad_rule,) = torch.autograd.grad(y_rule, x_rule, grad_outputs=grad_out)

        # The closed-form VJP of this rule: grad_x = grad_out / denom, with no
        # contribution from d(denom)/dx.
        expected = grad_out / denom_rule.detach()
        torch.testing.assert_close(grad_rule, expected)

        # Plain autodiff through the same expression differentiates the denom too,
        # so it disagrees with the rule everywhere the denom actually depends on x
        # (every row here, since eps alone would zero out that dependency).
        x_plain = _leaf(base, dtype)
        denom_plain = x_plain.abs().mean(dim=-1, keepdim=True) + eps
        y_plain = x_plain / denom_plain
        (grad_plain,) = torch.autograd.grad(y_plain, x_plain, grad_outputs=grad_out)
        assert not torch.allclose(grad_plain, grad_rule)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_backward_zero_row_has_no_division_by_zero(self, dtype):
        eps = 1e-6
        x = _leaf(_sample_rows(dtype)[:1], dtype)
        denom = x.abs().mean(dim=-1, keepdim=True) + eps
        y = ln_rule(x, denom)
        (grad,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x))
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad, torch.full_like(x, 1.0 / eps))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_and_backward_on_non_contiguous_input(self, dtype):
        base = _sample_rows(dtype).t()
        assert not base.is_contiguous()
        numerator = base.clone().requires_grad_(True)
        denom = numerator.abs().mean(dim=-1, keepdim=True) + 1e-6

        y = ln_rule(numerator, denom)
        assert torch.equal(y, numerator.detach() / denom.detach())

        grad_out = torch.ones_like(numerator)
        (grad,) = torch.autograd.grad(y, numerator, grad_outputs=grad_out)
        torch.testing.assert_close(grad, grad_out / denom.detach())

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_batched_leading_dims(self, dtype):
        numerator = _leaf(torch.randn(2, 3, 4), dtype)
        denom = numerator.abs().mean(dim=-1, keepdim=True) + 1e-6
        y = ln_rule(numerator, denom)
        assert y.shape == numerator.shape
        assert torch.equal(y, numerator.detach() / denom.detach())

        (grad,) = torch.autograd.grad(y, numerator, grad_outputs=torch.ones_like(numerator))
        torch.testing.assert_close(grad, torch.ones_like(numerator) / denom.detach())


class TestIdentityRule:
    """Forward equals the native activation; the VJP is ``grad_out * phi(x)``.

    ``phi`` has a closed form for each activation tested here: ``sigmoid`` for SiLU
    and the Gaussian CDF for exact GELU. Both are, by construction, the removable-
    singularity limit of ``f(x) / x`` at ``x == 0``, which is the direct oracle used
    for the tanh-approximate GELU, where no simpler closed form applies.
    """

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_matches_native_silu(self, dtype):
        x = _sample_rows(dtype)
        assert torch.equal(identity_rule(x, F.silu), F.silu(x))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_backward_matches_sigmoid_for_silu(self, dtype):
        x = _leaf(_sample_rows(dtype), dtype)
        y = identity_rule(x, F.silu)
        grad_out = torch.ones_like(x)
        (grad,) = torch.autograd.grad(y, x, grad_outputs=grad_out)
        torch.testing.assert_close(grad, grad_out * torch.sigmoid(x))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_matches_native_gelu_exact(self, dtype):
        x = _sample_rows(dtype)
        act_fn = partial(F.gelu, approximate="none")
        assert torch.equal(identity_rule(x, act_fn), act_fn(x))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_backward_matches_gaussian_cdf_for_exact_gelu(self, dtype):
        x = _leaf(_sample_rows(dtype), dtype)
        act_fn = partial(F.gelu, approximate="none")
        y = identity_rule(x, act_fn)
        grad_out = torch.ones_like(x)
        (grad,) = torch.autograd.grad(y, x, grad_outputs=grad_out)
        torch.testing.assert_close(grad, grad_out * _normal_cdf(x))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_matches_native_gelu_approx(self, dtype):
        x = _sample_rows(dtype)
        act_fn = partial(F.gelu, approximate="tanh")
        assert torch.equal(identity_rule(x, act_fn), act_fn(x))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_backward_matches_ratio_with_zero_limit_for_approx_gelu(self, dtype):
        x = _leaf(_sample_rows(dtype), dtype)
        act_fn = partial(F.gelu, approximate="tanh")
        y = identity_rule(x, act_fn)
        grad_out = torch.ones_like(x)
        (grad,) = torch.autograd.grad(y, x, grad_outputs=grad_out)

        with torch.no_grad():
            safe_x = torch.where(x == 0, torch.ones_like(x), x)
            expected_phi = torch.where(x == 0, torch.full_like(x, 0.5), act_fn(x.detach()) / safe_x)
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad, grad_out * expected_phi)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_and_backward_on_non_contiguous_input(self, dtype):
        base = _sample_rows(dtype).t()
        assert not base.is_contiguous()
        x = base.clone().requires_grad_(True)

        y = identity_rule(x, F.silu)
        assert torch.equal(y, F.silu(x.detach()))

        grad_out = torch.ones_like(x)
        (grad,) = torch.autograd.grad(y, x, grad_outputs=grad_out)
        torch.testing.assert_close(grad, grad_out * torch.sigmoid(x.detach()))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_batched_leading_dims(self, dtype):
        x = _leaf(torch.randn(2, 3, 4), dtype)
        y = identity_rule(x, F.silu)
        assert y.shape == x.shape
        assert torch.equal(y, F.silu(x.detach()))

        (grad,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x))
        torch.testing.assert_close(grad, torch.sigmoid(x.detach()))


class TestHalfRule:
    """Forward equals ``u * v``; the VJP halves each ordinary product-rule term."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_matches_plain_product(self, dtype):
        u = _sample_rows(dtype)
        v = _sample_rows(dtype).flip(0)
        assert torch.equal(half_rule(u, v), u * v)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_backward_halves_each_operand_gradient(self, dtype):
        u = _leaf(_sample_rows(dtype), dtype)
        v = _leaf(_sample_rows(dtype).flip(0), dtype)
        grad_out = torch.ones_like(u)

        y = half_rule(u, v)
        grad_u, grad_v = torch.autograd.grad(y, (u, v), grad_outputs=grad_out)
        torch.testing.assert_close(grad_u, 0.5 * grad_out * v.detach())
        torch.testing.assert_close(grad_v, 0.5 * grad_out * u.detach())

        # Ordinary autodiff of u * v would give the full (unhalved) product-rule
        # terms, so the rule must disagree with it.
        u_plain = _leaf(_sample_rows(dtype), dtype)
        v_plain = _leaf(_sample_rows(dtype).flip(0), dtype)
        y_plain = u_plain * v_plain
        grad_u_plain, grad_v_plain = torch.autograd.grad(
            y_plain, (u_plain, v_plain), grad_outputs=grad_out
        )
        assert not torch.allclose(grad_u_plain, grad_u)
        assert not torch.allclose(grad_v_plain, grad_v)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_and_backward_on_non_contiguous_input(self, dtype):
        u = _sample_rows(dtype).t().clone().requires_grad_(True)
        v = _sample_rows(dtype).flip(0).t().clone().requires_grad_(True)
        assert not u.is_contiguous()
        assert not v.is_contiguous()

        y = half_rule(u, v)
        assert torch.equal(y, u.detach() * v.detach())

        grad_out = torch.ones_like(u)
        grad_u, grad_v = torch.autograd.grad(y, (u, v), grad_outputs=grad_out)
        torch.testing.assert_close(grad_u, 0.5 * grad_out * v.detach())
        torch.testing.assert_close(grad_v, 0.5 * grad_out * u.detach())

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_batched_leading_dims(self, dtype):
        u = _leaf(torch.randn(2, 3, 4), dtype)
        v = _leaf(torch.randn(2, 3, 4), dtype)
        y = half_rule(u, v)
        assert y.shape == u.shape
        assert torch.equal(y, u.detach() * v.detach())

        grad_out = torch.ones_like(u)
        grad_u, grad_v = torch.autograd.grad(y, (u, v), grad_outputs=grad_out)
        torch.testing.assert_close(grad_u, 0.5 * grad_out * v.detach())
        torch.testing.assert_close(grad_v, 0.5 * grad_out * u.detach())
