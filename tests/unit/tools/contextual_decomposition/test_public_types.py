"""Public baseline/result validation without model loads."""

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from transformer_lens.tools.analysis import (
    CDBaseline,
    CDReconstructionError,
    CDResult,
    CDSource,
    CDTarget,
    CDTensor,
    ContextualDecomposition,
)


def test_public_exports_and_constructors():
    from transformer_lens.tools.analysis import contextual_decomposition as cd

    for name in cd.__all__:
        assert getattr(cd, name) is not None
    assert CDSource.attention_head(1, 0, positions=(2,)).anchor == "blocks.1.ln2.hook_in"
    assert CDSource.residual(1, location="post").anchor == "blocks.1.hook_out"
    assert CDTarget.internal("ln_final.hook_out").kind == "internal"
    assert CDBaseline.zero().description == "zero"
    assert not hasattr(ContextualDecomposition, "fit_reference")
    assert not hasattr(ContextualDecomposition, "score_sources")


@pytest.mark.parametrize(
    "reference",
    [
        torch.ones(8),
        torch.ones(1, 2, 3, dtype=torch.float64),
        torch.full((1, 2, 3), float("nan")),
        torch.empty(0, 2, 3),
    ],
)
def test_invalid_baseline_tensor_rejected(reference):
    with pytest.raises(ValueError):
        CDBaseline.from_tensor(reference)


def test_baseline_is_detached_and_owned():
    reference = torch.ones(1, 2, 3, requires_grad=True)
    baseline = CDBaseline.from_tensor(reference)
    reference.detach().zero_()
    assert torch.equal(baseline._reference, torch.ones(1, 2, 3))
    assert not baseline._reference.requires_grad
    with pytest.raises(FrozenInstanceError):
        setattr(baseline, "description", "changed")
    with pytest.raises(ValueError, match="description"):
        CDBaseline.from_tensor(reference, description=" ")


def test_cd_tensor_validates_and_owns_all_fields():
    beta = torch.ones(2, requires_grad=True)
    gamma = torch.ones(2, requires_grad=True)
    total = torch.full((2,), 2.0)
    value = CDTensor(beta, gamma, total)
    value.beta.zero_()
    assert torch.equal(beta, torch.ones(2)) and value.beta.grad_fn is None
    with pytest.raises(ValueError, match="match"):
        CDTensor(beta, gamma, torch.zeros(1))


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_reconstruction_error_rejected(value):
    with pytest.raises(ValueError, match="magnitudes"):
        CDReconstructionError(value, 1.0)


def test_residual_factory_rejects_undefined_location():
    with pytest.raises(ValueError, match="location"):
        CDSource.residual(0, location="unknown")
    assert CDSource.residual(0, location="mid").anchor == "blocks.0.ln2.hook_in"


def test_result_mapping_and_tensor_ownership():
    original = CDTensor(torch.ones(2), torch.zeros(2), torch.ones(2))
    intermediates = {"ln_final.hook_out": original}
    errors = {"ln_final.hook_out": CDReconstructionError(0.0, 1.0)}
    result = CDResult(
        torch.ones(2),
        torch.zeros(2),
        torch.ones(2),
        CDSource.residual(0),
        CDTarget.logit(token_id=1, position=2),
        "magnitude-affine-ordered-gelu-v1",
        "zero",
        ("ln_final.hook_out",),
        errors,
        intermediates,
    )
    original.beta.zero_()
    errors.clear()
    intermediates.clear()
    assert torch.equal(result.intermediates["ln_final.hook_out"].beta, torch.ones(2))
    assert result.diagnostics
    for overrides in (
        {"policy": "unknown"},
        {"baseline_description": " "},
        {"coverage": ("same", "same")},
    ):
        with pytest.raises(ValueError):
            replace(result, **overrides)
    with pytest.raises(TypeError):
        result.intermediates["another"] = original
