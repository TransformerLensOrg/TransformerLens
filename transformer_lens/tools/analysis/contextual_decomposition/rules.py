"""Affine magnitude allocation and ordered contextual-difference GELU rules.

Constituent reconstruction and allocation correctness are distinct properties.
These inference-only operators preserve the declared allocation without epsilon
bias loss or redistribution of cancelling constituents.
"""

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

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
        beta_magnitude = beta.abs()
        gamma_magnitude = gamma.abs()
        scale = torch.maximum(beta_magnitude, gamma_magnitude)
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        # Scaling keeps the magnitude sum finite without discarding small bias shares.
        beta_scaled = beta_magnitude / safe_scale
        gamma_scaled = gamma_magnitude / safe_scale
        denominator = beta_scaled + gamma_scaled
        safe_denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        beta_share = beta_scaled / safe_denominator
        # Computing both ratios preserves a small gamma share beside a dominant beta.
        gamma_share = torch.where(
            denominator == 0, torch.ones_like(denominator), gamma_scaled / safe_denominator
        )
        beta = beta + beta_share * bias
        gamma = gamma + gamma_share * bias
    return _CDConstituents(beta, gamma)


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
