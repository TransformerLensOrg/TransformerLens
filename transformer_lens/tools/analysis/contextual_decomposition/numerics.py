"""Scale-aware native reconstruction diagnostics for cancelling CD constituents."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _ReconstructionError:
    """Absolute error, native tensor scale, and constituent cancellation scale."""

    max_abs: float
    reference_max_abs: float
    constituent_max_abs: float = 0.0

    @property
    def tolerance(self) -> float:
        """Error budget based on the native tensor, never inflated by cancellation."""
        return 2e-6 + 2e-5 * self.reference_max_abs


@torch.no_grad()
def _reconstruction_error(
    beta: torch.Tensor, gamma: torch.Tensor, native: torch.Tensor
) -> _ReconstructionError:
    if beta.shape != native.shape or gamma.shape != native.shape:
        raise ValueError("native reconstruction shape mismatch")
    if not bool(torch.isfinite(native).all()):
        raise ValueError("native reconstruction reference must be finite")
    return _ReconstructionError(
        float((beta + gamma - native).abs().max()),
        float(native.abs().max()),
        float(torch.maximum(beta.abs(), gamma.abs()).max()),
    )


def _check_reconstruction(error: _ReconstructionError, name: str) -> None:
    if not math.isfinite(error.max_abs) or error.max_abs > error.tolerance:
        raise ValueError(
            f"native reconstruction failed at {name}: max_abs={error.max_abs:.6g}, "
            f"budget={error.tolerance:.6g}, constituent_scale={error.constituent_max_abs:.6g}"
        )
