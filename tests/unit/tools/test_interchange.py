"""Unit tests for interchange interventions on linear subspaces."""

import pytest
import torch
from torch import nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedRootModule import HookedRootModule
from transformer_lens.tools.analysis import SubspaceInterchange as exported_interchange
from transformer_lens.tools.analysis.interchange import (
    NormMatchedPlacebo,
    SubspaceInterchange,
    collect_activations,
    mean_difference_basis,
    probe_basis,
    random_basis,
    run_with_intervention,
)

D_MODEL = 6


class _Toy(HookedRootModule):
    """Linear in, hook, linear out; with ``seq``, the output reads only the last position."""

    def __init__(self, seq: bool = False):
        super().__init__()
        torch.manual_seed(0)
        self.inp = nn.Linear(D_MODEL, D_MODEL)
        self.out = nn.Linear(D_MODEL, 2)
        self.hook_mid = HookPoint()
        self.seq = seq
        self.setup()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hook_mid(self.inp(x))
        return self.out(h[:, -1] if self.seq else h)


def _orthonormal(basis: torch.Tensor) -> bool:
    return torch.allclose(basis.T @ basis, torch.eye(basis.shape[1]), atol=1e-5)


def test_public_analysis_exports():
    assert exported_interchange is SubspaceInterchange


def test_interchange_takes_the_source_inside_the_subspace_and_keeps_the_base_outside():
    base, source = torch.randn(4, D_MODEL), torch.randn(4, D_MODEL)
    B = random_basis(D_MODEL, 2, seed=0)
    out = SubspaceInterchange(B)(base, source)
    assert torch.allclose(out @ B, source @ B, atol=1e-5)
    complement = torch.eye(D_MODEL) - B @ B.T
    assert torch.allclose(out @ complement, base @ complement, atol=1e-5)
    assert torch.allclose(SubspaceInterchange(torch.eye(D_MODEL))(base, source), source, atol=1e-5)
    assert torch.allclose(SubspaceInterchange(B, strength=0.0)(base, source), base)


def test_random_basis_is_orthonormal_and_seeded():
    assert _orthonormal(random_basis(D_MODEL, 3, seed=1))
    assert torch.equal(random_basis(D_MODEL, 3, seed=1), random_basis(D_MODEL, 3, seed=1))
    assert not torch.equal(random_basis(D_MODEL, 3, seed=1), random_basis(D_MODEL, 3, seed=2))


def test_mean_difference_basis_spans_the_centred_class_means():
    acts = torch.randn(200, D_MODEL)
    labels = [i % 2 for i in range(200)]
    acts[torch.tensor(labels) == 1, 0] += 5.0
    direction = mean_difference_basis(acts, labels)
    assert direction.shape == (D_MODEL, 1)
    expected = acts[1::2].mean(0) - acts[0::2].mean(0)
    cosine = torch.nn.functional.cosine_similarity(direction[:, 0], expected, dim=0)
    assert cosine.abs() > 1 - 1e-5

    three = mean_difference_basis(torch.randn(90, D_MODEL), [i % 3 for i in range(90)])
    assert three.shape == (D_MODEL, 2) and _orthonormal(three)
    with pytest.raises(ValueError, match="two classes"):
        mean_difference_basis(torch.randn(10, D_MODEL), [0] * 10)


def test_probe_basis_follows_the_weights():
    w = torch.randn(1, D_MODEL)
    one = probe_basis(w)
    assert torch.allclose(one[:, 0].abs(), (w[0] / w[0].norm()).abs(), atol=1e-5)
    rows = torch.randn(3, D_MODEL)
    centred = probe_basis(rows)
    assert centred.shape == (D_MODEL, 2) and _orthonormal(centred)
    # The centred rows lie in the returned span
    residual = (rows - rows.mean(0)).T - centred @ (centred.T @ (rows - rows.mean(0)).T)
    assert residual.abs().max() < 1e-4


def test_placebo_moves_as_far_as_the_reference_inside_its_own_subspace():
    base, source = torch.randn(8, D_MODEL), torch.randn(8, D_MODEL)
    reference = SubspaceInterchange(random_basis(D_MODEL, 2, seed=0))
    placebo_basis = random_basis(D_MODEL, 2, seed=1)
    moved = NormMatchedPlacebo(reference, placebo_basis)(base, source)
    expected = (reference(base, source) - base).norm(dim=-1)
    assert torch.allclose((moved - base).norm(dim=-1), expected, atol=1e-5)
    outside = (moved - base) - (moved - base) @ placebo_basis @ placebo_basis.T
    assert outside.abs().max() < 1e-5


def test_a_full_interchange_reproduces_the_source_output():
    model = _Toy()
    base, source = torch.randn(5, D_MODEL), torch.randn(5, D_MODEL)
    source_acts = collect_activations(model, "hook_mid", source, model)
    out = run_with_intervention(
        model, "hook_mid", SubspaceInterchange(torch.eye(D_MODEL)), base, source_acts, model
    )
    assert torch.allclose(out, model(source), atol=1e-5)


def test_positions_select_one_vector_per_example():
    model = _Toy(seq=True)
    base, source = torch.randn(4, 3, D_MODEL), torch.randn(4, 3, D_MODEL)
    last = torch.full((4,), 2)
    _, cache = model.run_with_cache(source)
    acts = collect_activations(model, "hook_mid", source, model, positions=last, batch_size=3)
    assert torch.allclose(acts, cache["hook_mid"][:, 2])

    full = SubspaceInterchange(torch.eye(D_MODEL))
    at_last = run_with_intervention(model, "hook_mid", full, base, acts, model, positions=last)
    assert torch.allclose(at_last, model(source), atol=1e-5)
    first = collect_activations(
        model, "hook_mid", source, model, positions=torch.zeros(4, dtype=torch.long)
    )
    at_first = run_with_intervention(
        model, "hook_mid", full, base, first, model, positions=torch.zeros(4, dtype=torch.long)
    )
    assert torch.allclose(at_first, model(base), atol=1e-6)  # the output never reads position 0

    with pytest.raises(ValueError, match="one position index per example"):
        collect_activations(model, "hook_mid", source, model)


def test_the_intervention_is_differentiable():
    model = _Toy()
    basis = random_basis(D_MODEL, 2, seed=0).requires_grad_(True)
    base, source = torch.randn(4, D_MODEL), torch.randn(4, D_MODEL)
    source_acts = collect_activations(model, "hook_mid", source, model)
    out = run_with_intervention(
        model, "hook_mid", SubspaceInterchange(basis), base, source_acts, model
    )
    out.sum().backward()
    assert basis.grad is not None and basis.grad.abs().sum() > 0


def test_a_hook_that_fires_twice_is_refused():
    class _Twice(_Toy):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.hook_mid(self.inp(x))
            return self.out(self.hook_mid(h))

    model = _Twice()
    with pytest.raises(RuntimeError, match="fired 2 times"):
        collect_activations(model, "hook_mid", torch.randn(2, D_MODEL), model)
