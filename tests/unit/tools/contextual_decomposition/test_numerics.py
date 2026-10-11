"""Native tensor-scale error budgets do not conceal cancelling constituent errors."""

import pytest
import torch

from transformer_lens.tools.analysis.contextual_decomposition.numerics import (
    _check_reconstruction,
    _reconstruction_error,
)


def test_tensor_scale_budget_handles_small_native_coordinates():
    native = torch.tensor([100.0, 0.0])
    beta = torch.tensor([100.0, 1e-4])
    gamma = torch.zeros(2)
    error = _reconstruction_error(beta, gamma, native)
    assert error.max_abs > 2e-6
    assert error.tolerance == pytest.approx(0.002002)
    _check_reconstruction(error, "test")


def test_cancellation_does_not_inflate_native_error_budget():
    native = torch.ones(2)
    beta = torch.full((2,), 1e5)
    gamma = torch.tensor([-99999.0, -99998.0])
    error = _reconstruction_error(beta, gamma, native)
    assert error.constituent_max_abs == 1e5
    with pytest.raises(ValueError, match="budget=.*constituent_scale"):
        _check_reconstruction(error, "cancelling")


@pytest.mark.parametrize("native", [torch.zeros(3), torch.tensor([float("nan"), 1.0])])
def test_malformed_native_reference_rejected(native):
    with pytest.raises(ValueError, match="shape|finite"):
        _reconstruction_error(torch.ones(2), torch.zeros(2), native)
