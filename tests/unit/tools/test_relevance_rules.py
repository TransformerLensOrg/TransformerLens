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
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from transformer_lens.model_bridge._relevance_rules import (
    half_rule,
    identity_rule,
    ln_rule,
)
from transformer_lens.model_bridge.generalized_components.mlp import (
    normalize_mlp_weight,
    weight_layout_in_out,
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


class _Proj:
    """Minimal stand-in for a projection bridge: carries only what
    ``weight_layout_in_out``/``normalize_mlp_weight`` read (``original_component``)."""

    def __init__(self, original_component):
        self.original_component = original_component


class TestGatedMLPWeightOrientation:
    """``MLPBridge``'s ``W_gate``/``W_in``/``W_out`` read the underlying projection
    through ``weight_layout_in_out``/``normalize_mlp_weight``, so an orientation bug in
    those helpers would silently transpose a gate/up/down weight for one backing class.
    Covers both HF module classes: ``nn.Linear`` (weight stored ``[out, in]``,
    transposed to TL orientation) and ``Conv1D`` (weight stored ``[in, out]``, already
    TL-oriented).
    """

    @pytest.fixture(params=["nn.Linear", "Conv1D"])
    def backing_class(self, request):
        return request.param

    def _make_gate_up_down(self, backing_class: str, d_model: int = 3, d_mlp: int = 5):
        torch.manual_seed(0)
        if backing_class == "nn.Linear":
            gate_proj = nn.Linear(d_model, d_mlp)
            up_proj = nn.Linear(d_model, d_mlp)
            down_proj = nn.Linear(d_mlp, d_model)
        else:
            gate_proj = Conv1D(d_mlp, d_model)
            up_proj = Conv1D(d_mlp, d_model)
            down_proj = Conv1D(d_model, d_mlp)
        return gate_proj, up_proj, down_proj

    def _tl_weight(self, proj: torch.nn.Module, pattern: str) -> torch.Tensor:
        wrapper = _Proj(proj)
        layout = weight_layout_in_out(wrapper)
        return normalize_mlp_weight(proj.weight, layout, wrapper, pattern=pattern)

    def test_tl_oriented_matmul_reproduces_native_projection(self, backing_class):
        gate_proj, up_proj, down_proj = self._make_gate_up_down(backing_class)
        x = torch.randn(2, 3)

        w_gate = self._tl_weight(gate_proj, pattern="in")
        w_in = self._tl_weight(up_proj, pattern="in")
        w_out = self._tl_weight(down_proj, pattern="out")

        assert torch.allclose(x @ w_gate + gate_proj.bias, gate_proj(x), atol=1e-6)
        assert torch.allclose(x @ w_in + up_proj.bias, up_proj(x), atol=1e-6)
        hidden = torch.randn(2, 5)
        assert torch.allclose(hidden @ w_out + down_proj.bias, down_proj(hidden), atol=1e-6)


class TestPinnedReferenceParity:
    """Tolerant parity against ``FarnoushRJ/RelP`` pinned at
    ``8219d6dc417c3fd7f318342cf61cd2a0c20b7250``.

    That repository vendors an unrelated pre-Bridge TransformerLens fork, so its
    rule formulas are reimplemented here directly from the pinned commit's
    component diffs rather than imported:

    - LN-rule (``transformer_lens/components/rms_norm.py``): ``x / scale.detach()``.
    - Identity-rule (``transformer_lens/utilities/activation_functions.py``,
      class ``ModifiedAct``): ``zp = stabilize(x); zp * (act_fn(x) / zp).detach()``,
      where ``stabilize(z) = z + ((z == 0) + sign(z)) * 1e-6``
      (``transformer_lens/lrp_utils.py``).
    - Half-rule (``transformer_lens/components/mlps/gated_mlp.py``):
      ``z = u * v; z / 2 + (z / 2).detach()``.

    The LN- and Half-rule reference formulas produce the same VJP as this module's
    primitives to floating-point precision. The Identity-rule reference formula
    does not: its epsilon stabilizer only approximates the paper-defined factor
    away from ``x == 0``, and collapses to exactly zero at ``x == 0`` where the
    paper-defined factor's removable-singularity limit is ``0.5``.
    """

    @staticmethod
    def _reference_stabilize(z: torch.Tensor) -> torch.Tensor:
        return z + ((z == 0).to(z.dtype) + torch.sign(z)) * 1e-6

    @classmethod
    def _reference_ln_rule_grad(cls, x: torch.Tensor, denom_fn) -> torch.Tensor:
        x = x.clone().requires_grad_(True)
        denom = denom_fn(x)
        y = x / denom.detach()
        (grad,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x))
        return grad

    @classmethod
    def _reference_identity_rule_grad(cls, x: torch.Tensor, act_fn) -> torch.Tensor:
        x = x.clone().requires_grad_(True)
        z = act_fn(x)
        zp = cls._reference_stabilize(x)
        y = zp * (z / zp).detach()
        (grad,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x))
        return grad

    @classmethod
    def _reference_half_rule_grad(cls, u: torch.Tensor, v: torch.Tensor):
        u = u.clone().requires_grad_(True)
        v = v.clone().requires_grad_(True)
        z = u * v
        y = z / 2 + (z / 2).detach()
        grad_u, grad_v = torch.autograd.grad(y, (u, v), grad_outputs=torch.ones_like(u))
        return grad_u, grad_v

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_ln_rule_matches_reference_grad(self, dtype):
        def denom_fn(t: torch.Tensor) -> torch.Tensor:
            return (t.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()

        x = _sample_rows(dtype)

        x_rule = _leaf(x, dtype)
        denom_rule = denom_fn(x_rule)
        y_rule = ln_rule(x_rule, denom_rule)
        (grad_rule,) = torch.autograd.grad(y_rule, x_rule, grad_outputs=torch.ones_like(x_rule))

        grad_reference = self._reference_ln_rule_grad(x, denom_fn)
        torch.testing.assert_close(grad_rule, grad_reference)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "act_fn",
        [F.silu, partial(F.gelu, approximate="none"), partial(F.gelu, approximate="tanh")],
        ids=["silu", "gelu_exact", "gelu_tanh"],
    )
    def test_identity_rule_matches_reference_away_from_zero(self, dtype, act_fn):
        x = _sample_rows(dtype)
        nonzero_mask = x != 0

        x_rule = _leaf(x, dtype)
        y_rule = identity_rule(x_rule, act_fn)
        (grad_rule,) = torch.autograd.grad(y_rule, x_rule, grad_outputs=torch.ones_like(x_rule))

        grad_reference = self._reference_identity_rule_grad(x, act_fn)

        torch.testing.assert_close(
            grad_rule[nonzero_mask],
            grad_reference[nonzero_mask],
            rtol=1e-4,
            atol=1e-5,
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "act_fn",
        [F.silu, partial(F.gelu, approximate="none"), partial(F.gelu, approximate="tanh")],
        ids=["silu", "gelu_exact", "gelu_tanh"],
    )
    def test_identity_rule_exact_zero_discrepancy_is_documented(self, dtype, act_fn):
        """At ``x == 0`` this module's Identity-rule uses the paper-defined
        removable-singularity limit ``0.5``, while the pinned reference's epsilon
        stabilizer yields exactly ``0``. Assert both values explicitly, rather than
        letting a tolerance absorb the gap, so a change to either side's zero
        handling is caught instead of silently passing.
        """
        x = torch.zeros(3, dtype=dtype)

        x_rule = _leaf(x, dtype)
        y_rule = identity_rule(x_rule, act_fn)
        (grad_rule,) = torch.autograd.grad(y_rule, x_rule, grad_outputs=torch.ones_like(x_rule))
        torch.testing.assert_close(grad_rule, torch.full_like(x, 0.5))

        grad_reference = self._reference_identity_rule_grad(x, act_fn)
        torch.testing.assert_close(grad_reference, torch.zeros_like(x))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_half_rule_matches_reference_grad(self, dtype):
        u = _sample_rows(dtype)
        v = _sample_rows(dtype).flip(0)

        u_rule = _leaf(u, dtype)
        v_rule = _leaf(v, dtype)
        y_rule = half_rule(u_rule, v_rule)
        grad_u_rule, grad_v_rule = torch.autograd.grad(
            y_rule, (u_rule, v_rule), grad_outputs=torch.ones_like(u_rule)
        )

        grad_u_reference, grad_v_reference = self._reference_half_rule_grad(u, v)
        torch.testing.assert_close(grad_u_rule, grad_u_reference)
        torch.testing.assert_close(grad_v_rule, grad_v_reference)
