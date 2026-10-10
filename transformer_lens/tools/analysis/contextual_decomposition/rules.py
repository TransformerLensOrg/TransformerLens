"""Explicit constituent allocation for affine, GELU, normalization, and attention.

Constituent reconstruction and allocation correctness are distinct properties.
These inference-only operators preserve the declared allocation without epsilon
bias loss or redistribution of cancelling constituents.
"""

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F


def _validate_tensor(name: str, value: torch.Tensor) -> None:
    if value.layout != torch.strided:
        raise ValueError(f"{name} must be a strided tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must have dtype float32 or float64")
    if value.device.type == "meta":
        raise ValueError(f"{name} must have a materialized device")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class _CDConstituents:
    """Two same-shaped, finite constituents with a fixed allocation policy.

    The policy allocates affine bias by projected absolute magnitude (all bias
    to gamma when both projections vanish) and uses ordered activation differences.
    Freezing fields prevents policy reassignment; tensors remain caller-owned.
    """

    beta: torch.Tensor
    gamma: torch.Tensor
    policy: Literal["magnitude-affine-ordered-gelu-v1"] = field(
        default="magnitude-affine-ordered-gelu-v1", init=False
    )

    @torch.no_grad()
    def __post_init__(self) -> None:
        if self.beta.shape != self.gamma.shape:
            raise ValueError("beta and gamma must have the same shape")
        if self.beta.dtype != self.gamma.dtype:
            raise ValueError("beta and gamma must have the same dtype")
        if self.beta.device != self.gamma.device:
            raise ValueError("beta and gamma must be on the same device")
        if self.beta.ndim == 0:
            raise ValueError("constituents must have at least one dimension")
        _validate_tensor("beta", self.beta)
        _validate_tensor("gamma", self.gamma)
        if not bool(torch.isfinite(self.beta + self.gamma).all()):
            raise ValueError("the constituent total must contain only finite values")


def _validate_operand(name: str, value: torch.Tensor, parts: _CDConstituents) -> None:
    if value.device != parts.beta.device:
        raise ValueError(f"{name} must be on the same device as the constituents")
    if value.dtype != parts.beta.dtype:
        raise ValueError(f"{name} must have the same dtype as the constituents")
    _validate_tensor(name, value)


@torch.no_grad()
def _affine(
    parts: _CDConstituents,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> _CDConstituents:
    """Apply a row-vector affine map with magnitude-proportional bias allocation.

    Weight has shape [input_width, output_width], bias [output_width]. Each
    output's beta receives |beta @ W| / (|beta @ W| + |gamma @ W|) of its bias.
    When both projections vanish, gamma receives the entire bias. Operands are
    not cast, mutated, or connected to autograd; outputs own their storage.
    """
    parts.__post_init__()
    _validate_operand("weight", weight, parts)
    if weight.ndim != 2 or weight.shape[0] != parts.beta.shape[-1]:
        raise ValueError("weight must have shape [input_width, output_width]")
    if bias is not None:
        _validate_operand("bias", bias, parts)
        if bias.ndim != 1 or bias.shape[0] != weight.shape[1]:
            raise ValueError("bias must have shape [output_width]")

    beta = parts.beta @ weight
    gamma = parts.gamma @ weight
    _validate_tensor("projected beta", beta)
    _validate_tensor("projected gamma", gamma)
    if bias is not None:
        beta_share, gamma_share = _magnitude_shares(beta, gamma)
        beta = beta + beta_share * bias
        gamma = gamma + gamma_share * bias
    return _CDConstituents(beta, gamma)


def _magnitude_shares(beta: torch.Tensor, gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.maximum(beta.abs(), gamma.abs())
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # Scaling avoids overflow; independent ratios preserve small remaining shares.
    beta_scaled = beta.abs() / safe_scale
    gamma_scaled = gamma.abs() / safe_scale
    denominator = beta_scaled + gamma_scaled
    safe_denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    return beta_scaled / safe_denominator, torch.where(
        denominator == 0, torch.ones_like(denominator), gamma_scaled / safe_denominator
    )


@torch.no_grad()
def _gelu(parts: _CDConstituents, *, variant: str = "gelu_new") -> _CDConstituents:
    """Allocate GELU as gamma=f(gamma), beta=f(beta+gamma)-f(gamma).

    ``gelu`` uses exact PyTorch GELU, ``gelu_new`` uses the explicit GPT-2 tanh
    formula, and ``gelu_pytorch_tanh`` uses PyTorch's fused tanh approximation.
    Mathematically equivalent tanh variants keep their own operation ordering.
    """
    parts.__post_init__()
    if variant not in ("gelu", "gelu_new", "gelu_pytorch_tanh"):
        raise ValueError(f"unsupported GELU variant: {variant!r}")

    def activate(value: torch.Tensor) -> torch.Tensor:
        if variant == "gelu":
            return F.gelu(value, approximate="none")
        if variant == "gelu_pytorch_tanh":
            return F.gelu(value, approximate="tanh")
        return (
            0.5
            * value
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value.pow(3))))
        )

    gamma = activate(parts.gamma)
    beta = activate(parts.beta + parts.gamma) - gamma
    return _CDConstituents(beta, gamma)


@torch.no_grad()
def _layer_norm(
    parts: _CDConstituents,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-5,
    denominator: Optional[torch.Tensor] = None,
) -> _CDConstituents:
    """Center each constituent using the combined input's normalization denominator.

    Gain/bias have shape [width]. Bias shares use uncentered input magnitudes,
    with all bias assigned to gamma at jointly zero coordinates. A supplied
    denominator must match sqrt(var(beta+gamma)+eps) within dtype tolerance.
    """
    parts.__post_init__()
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    for name, operand in (("weight", weight), ("bias", bias)):
        if operand is not None:
            _validate_operand(name, operand, parts)
            if operand.shape != (parts.beta.shape[-1],):
                raise ValueError(f"{name} must have shape [width]")
    total = parts.beta + parts.gamma
    centered = total - total.mean(-1, keepdim=True)
    expected = torch.sqrt(centered.square().mean(-1, keepdim=True) + eps)
    _validate_tensor("full-input denominator", expected)
    if denominator is None:
        denominator = expected
    else:
        _validate_operand("denominator", denominator, parts)
        if denominator.shape != expected.shape:
            raise ValueError("denominator must have shape [..., 1]")
        if not bool((denominator > 0).all()):
            raise ValueError("denominator must be positive")
        tolerance = 32 * torch.finfo(total.dtype).eps
        if not torch.allclose(denominator, expected, rtol=tolerance, atol=0):
            raise ValueError("denominator must match the full input and eps")
    beta = (parts.beta - parts.beta.mean(-1, keepdim=True)) / denominator
    gamma = (parts.gamma - parts.gamma.mean(-1, keepdim=True)) / denominator
    if weight is not None:
        beta = beta * weight
        gamma = gamma * weight
    if bias is not None:
        beta_share, gamma_share = _magnitude_shares(parts.beta, parts.gamma)
        beta = beta + beta_share * bias
        gamma = gamma + gamma_share * bias
    return _CDConstituents(beta, gamma)


@torch.no_grad()
def _masked_softmax(parts: _CDConstituents, allowed: torch.Tensor) -> _CDConstituents:
    """Allocate probabilities as softmax(beta) and softmax(total)-softmax(beta).

    True entries are allowed keys. The boolean mask must broadcast to, but not
    expand, the scores. Fully masked rows are rejected, including padded queries.
    """
    parts.__post_init__()
    if allowed.dtype != torch.bool:
        raise ValueError("attention mask must have bool dtype")
    if allowed.device != parts.beta.device:
        raise ValueError("attention mask must be on the scores' device")
    try:
        mask = torch.broadcast_to(allowed, parts.beta.shape)
    except RuntimeError as error:
        raise ValueError("attention mask shape must broadcast to scores") from error
    if not bool(mask.any(-1).all()):
        raise ValueError("attention contains a fully masked query row")
    beta = torch.softmax(parts.beta.masked_fill(~mask, -torch.inf), dim=-1)
    total = torch.softmax((parts.beta + parts.gamma).masked_fill(~mask, -torch.inf), dim=-1)
    return _CDConstituents(beta, total - beta)


@torch.no_grad()
def _residual_add(left: _CDConstituents, right: _CDConstituents) -> _CDConstituents:
    """Merge aligned residual constituents without broadcasting or aliasing."""
    left.__post_init__()
    right.__post_init__()
    if left.beta.shape != right.beta.shape:
        raise ValueError("residual constituents must have the same shape")
    _validate_operand("residual", right.beta, left)
    return _CDConstituents(left.beta + right.beta, left.gamma + right.gamma)


_Index = Union[int, slice, tuple[Union[int, slice], ...]]


@torch.no_grad()
def _slice(parts: _CDConstituents, index: _Index) -> _CDConstituents:
    """Select matching nonempty constituents with independent output storage."""
    parts.__post_init__()
    return _CDConstituents(parts.beta[index].clone(), parts.gamma[index].clone())


@dataclass(frozen=True)
class _CDAttention:
    """Attention allocation diagnostics and residual writes excluding shared bias."""

    query: _CDConstituents
    key: _CDConstituents
    value: _CDConstituents
    scores: _CDConstituents
    probabilities: _CDConstituents
    context: _CDConstituents
    head_writes: _CDConstituents
    output: _CDConstituents


@torch.no_grad()
def _attention(
    parts: _CDConstituents,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    *,
    n_heads: int,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
    v_bias: Optional[torch.Tensor] = None,
    o_bias: Optional[torch.Tensor] = None,
    causal: bool = True,
    key_mask: Optional[torch.Tensor] = None,
) -> _CDAttention:
    """Decompose full-sequence multi-head attention with nonlinear score allocation.

    Inputs are [batch, sequence, width], Q/K/V weights [width, heads*head_width],
    O weights [heads*head_width, width]. Head writes are [batch, sequence, heads,
    width]; shared output bias is allocated only after summing the head writes.
    Scores, probabilities, and Q/K/V/context use head-major dimensions. No GQA,
    cache, dropout, positional score terms, or alternate scaling is implied.
    """
    if not isinstance(causal, bool):
        raise ValueError("causal must be a bool")
    parts.__post_init__()
    if parts.beta.ndim != 3:
        raise ValueError("attention input must have shape [batch, sequence, width]")
    if isinstance(n_heads, bool) or not isinstance(n_heads, int) or n_heads <= 0:
        raise ValueError("n_heads must be a positive integer")
    width = parts.beta.shape[-1]
    for name, operand in (("q_weight", q_weight), ("k_weight", k_weight), ("v_weight", v_weight)):
        _validate_operand(name, operand, parts)
        if operand.ndim != 2 or operand.shape[0] != width:
            raise ValueError(f"{name} must have shape [width, heads*head_width]")
    projected_width = q_weight.shape[1]
    if (
        projected_width % n_heads
        or k_weight.shape != q_weight.shape
        or v_weight.shape != q_weight.shape
    ):
        raise ValueError("Q/K/V projected widths must match and be divisible by n_heads")
    _validate_operand("o_weight", o_weight, parts)
    if o_weight.shape != (projected_width, width):
        raise ValueError("o_weight must have shape [heads*head_width, width]")
    if o_bias is not None:
        _validate_operand("o_bias", o_bias, parts)
        if o_bias.shape != (width,):
            raise ValueError("o_bias must have shape [width]")

    batch, sequence, _ = parts.beta.shape
    head_width = projected_width // n_heads

    def project(weight: torch.Tensor, bias: Optional[torch.Tensor]) -> _CDConstituents:
        projected = _affine(parts, weight, bias)
        return _CDConstituents(
            projected.beta.reshape(batch, sequence, n_heads, head_width).transpose(1, 2),
            projected.gamma.reshape(batch, sequence, n_heads, head_width).transpose(1, 2),
        )

    query, key, value = (
        project(q_weight, q_bias),
        project(k_weight, k_bias),
        project(v_weight, v_bias),
    )
    beta_scores = query.beta @ key.beta.transpose(-1, -2) / math.sqrt(head_width)
    total_scores = (
        (query.beta + query.gamma)
        @ (key.beta + key.gamma).transpose(-1, -2)
        / math.sqrt(head_width)
    )
    scores = _CDConstituents(beta_scores, total_scores - beta_scores)
    allowed = torch.ones((batch, 1, sequence, sequence), dtype=torch.bool, device=parts.beta.device)
    if causal:
        allowed = (
            allowed
            & torch.ones((sequence, sequence), dtype=torch.bool, device=parts.beta.device).tril()
        )
    if key_mask is not None:
        if key_mask.dtype != torch.bool or key_mask.device != parts.beta.device:
            raise ValueError("key_mask must have bool dtype and match the input device")
        if key_mask.shape != (batch, sequence):
            raise ValueError("key_mask must have shape [batch, sequence]")
        allowed = allowed & key_mask[:, None, None, :]
    probabilities = _masked_softmax(scores, allowed)
    beta_context = probabilities.beta @ value.beta
    total_context = (probabilities.beta + probabilities.gamma) @ (value.beta + value.gamma)
    context = _CDConstituents(beta_context, total_context - beta_context)
    head_o = o_weight.reshape(n_heads, head_width, width)
    head_writes = _CDConstituents(
        torch.einsum("bhqd,hdo->bqho", context.beta, head_o),
        torch.einsum("bhqd,hdo->bqho", context.gamma, head_o),
    )
    beta = head_writes.beta.sum(2)
    gamma = head_writes.gamma.sum(2)
    if o_bias is not None:
        beta_share, gamma_share = _magnitude_shares(beta, gamma)
        beta = beta + beta_share * o_bias
        gamma = gamma + gamma_share * o_bias
    return _CDAttention(
        query, key, value, scores, probabilities, context, head_writes, _CDConstituents(beta, gamma)
    )
