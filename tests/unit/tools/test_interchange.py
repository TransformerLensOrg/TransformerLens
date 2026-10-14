"""Unit tests for interchange interventions on linear subspaces."""

import pytest
import torch
from torch import nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedRootModule import HookedRootModule
from transformer_lens.tools.analysis import SubspaceInterchange as exported_interchange
from transformer_lens.tools.analysis.interchange import (
    LearnedSubspace,
    NormMatchedPlacebo,
    SubspaceInterchange,
    _model_device,
    collect_activations,
    fit_das,
    intervened_logits,
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
        self.eval()

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


class _SumToy(_Toy):
    """Reads every position, so each row's write shows in its output."""

    def __init__(self):
        super().__init__(seq=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.hook_mid(self.inp(x)).sum(1))


def test_positions_select_one_vector_per_example():
    rows = torch.tensor([0, 2, 1, 2])
    base, source = torch.randn(4, 3, D_MODEL), torch.randn(4, 3, D_MODEL)
    full = SubspaceInterchange(torch.eye(D_MODEL))

    model = _Toy(seq=True)  # reads only the last position
    _, cache = model.run_with_cache(source)
    acts = collect_activations(model, "hook_mid", source, model, positions=rows, batch_size=3)
    assert torch.allclose(acts, cache["hook_mid"][torch.arange(4), rows])
    out = run_with_intervention(model, "hook_mid", full, base, acts, model, positions=rows)
    at_last = rows == 2
    assert torch.allclose(out[at_last], model(source)[at_last], atol=1e-5)
    assert torch.allclose(out[~at_last], model(base)[~at_last], atol=1e-6)

    summed = _SumToy()
    source_acts = collect_activations(summed, "hook_mid", source, summed, positions=rows)
    out = run_with_intervention(summed, "hook_mid", full, base, source_acts, summed, positions=rows)
    with torch.no_grad():
        expected = summed.inp(base)
        expected[torch.arange(4), rows] = summed.inp(source)[torch.arange(4), rows]
        assert torch.allclose(out, summed.out(expected.sum(1)), atol=1e-5)

    with pytest.raises(ValueError, match="one position index per example"):
        collect_activations(model, "hook_mid", source, model)


class _Pattern(HookedRootModule):
    """A 4-D hook, shaped like an attention pattern: [batch, head, query, key]."""

    def __init__(self):
        super().__init__()
        self.hook_pattern = HookPoint()
        self.setup()
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hook_pattern(x).flatten(1)[:, :2]


def test_positions_must_match_the_activation():
    pattern = _Pattern()
    x, at = torch.randn(4, 2, 3, 3), torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError, match="must be \\[batch, pos, d_model\\]"):
        collect_activations(pattern, "hook_pattern", x, pattern, positions=at)
    with pytest.raises(ValueError, match="must be \\[batch, pos, d_model\\]"):
        run_with_intervention(
            pattern,
            "hook_pattern",
            SubspaceInterchange(torch.eye(3)),
            x,
            torch.randn(4, 3),
            pattern,
            positions=at,
        )
    model = _Toy(seq=True)
    with pytest.raises(ValueError, match="one index per example"):
        collect_activations(model, "hook_mid", torch.randn(4, 3, D_MODEL), model, positions=at[:3])


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


def test_a_hook_must_fire_exactly_once():
    class _Twice(_Toy):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.hook_mid(self.inp(x))
            return self.out(self.hook_mid(h))

    class _Unused(_Toy):
        def __init__(self):
            super().__init__()
            self.hook_unused = HookPoint()
            self.setup()
            self.eval()

    x, source, full = (
        torch.randn(2, D_MODEL),
        torch.randn(2, D_MODEL),
        SubspaceInterchange(torch.eye(D_MODEL)),
    )
    for model, hook, times in ((_Twice(), "hook_mid", 2), (_Unused(), "hook_unused", 0)):
        with pytest.raises(RuntimeError, match=f"fired {times} times"):
            collect_activations(model, hook, x, model)
        with pytest.raises(RuntimeError, match=f"fired {times} times"):
            run_with_intervention(model, hook, full, x, source, model)


def test_intervened_logits_matches_one_pass_in_bounded_batches():
    model = _Toy()
    base, source = torch.randn(10, D_MODEL), torch.randn(10, D_MODEL)
    acts = collect_activations(model, "hook_mid", source, model)
    intervention = SubspaceInterchange(random_basis(D_MODEL, 2, seed=0).requires_grad_(True))
    one = run_with_intervention(model, "hook_mid", intervention, base, acts, model)
    batched = intervened_logits(model, "hook_mid", intervention, base, acts, model, batch_size=3)
    assert torch.allclose(batched, one.detach(), atol=1e-6)
    assert not batched.requires_grad and batched.device.type == "cpu"
    with pytest.raises(ValueError, match="one row per base"):
        intervened_logits(model, "hook_mid", intervention, base, acts[:-1], model)


class _Planted(HookedRootModule):
    """The output depends on the hidden state only through one planted direction."""

    def __init__(self, direction: torch.Tensor):
        super().__init__()
        self.register_buffer("direction", direction / direction.norm())
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.hook_mid = HookPoint()
        self.setup()
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.scale * (self.hook_mid(x) @ self.direction)
        return torch.stack([-score, score], dim=-1)


def _planted_pairs(n: int = 256, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(D_MODEL, generator=generator)
    model = _Planted(direction)
    base = torch.randn(n, D_MODEL, generator=generator)
    source = torch.randn(n, D_MODEL, generator=generator)
    targets = (source @ model.direction > 0).long()  # the variable comes from the source
    return model, base, source, targets


def test_learned_subspace_stays_orthonormal_and_checks_its_rank():
    subspace = LearnedSubspace(D_MODEL, 3)
    assert subspace.basis.shape == (D_MODEL, 3) and _orthonormal(subspace.basis.detach())
    with pytest.raises(ValueError, match="rank must be in"):
        LearnedSubspace(D_MODEL, 0)
    with pytest.raises(ValueError, match="rank must be in"):
        LearnedSubspace(D_MODEL, D_MODEL + 1)


def test_das_recovers_a_planted_direction():
    model, base, source, targets = _planted_pairs()
    result = fit_das(model, "hook_mid", base, source, targets, model, rank=1, epochs=30, lr=5e-2)
    cosine = (result.basis[:, 0] @ model.direction).abs()
    assert cosine > 0.95
    assert result.losses[-1] < result.losses[0]

    _, test_base, test_source, _ = _planted_pairs(seed=1)
    source_acts = collect_activations(model, "hook_mid", test_source, model)
    logits = intervened_logits(
        model, "hook_mid", SubspaceInterchange(result.subspace), test_base, source_acts, model
    )
    # The planted direction differs per seed, so evaluate on the training model's direction
    held_out_targets = (test_source @ model.direction > 0).long()
    assert (logits.argmax(-1) == held_out_targets).float().mean() > 0.9


def test_splitting_batches_trains_the_same_way():
    model, base, source, targets = _planted_pairs()
    kwargs = dict(rank=1, epochs=30, lr=5e-2, batch_size=64)
    sizes: dict[str, list[int]] = {"whole": [], "split": []}

    def recording(name):
        def task_logits(x):
            sizes[name].append(len(x))
            return model(x)

        return task_logits

    whole = fit_das(model, "hook_mid", base, source, targets, recording("whole"), **kwargs)
    split = fit_das(
        model, "hook_mid", base, source, targets, recording("split"), micro_batch_size=16, **kwargs
    )
    # Only a split run keeps every forward pass to the micro-batch size
    assert max(sizes["whole"]) == 64 and max(sizes["split"]) == 16
    # Gradients agree up to rounding; Adam's sign-like first steps can amplify that slightly
    assert abs(whole.losses[0] - split.losses[0]) < 1e-2
    for result in (whole, split):
        assert (result.basis[:, 0] @ model.direction).abs() > 0.95


def test_das_leaves_the_callers_random_stream_alone():
    model, base, source, targets = _planted_pairs(n=32)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    first = fit_das(model, "hook_mid", base, source, targets, model, rank=1, epochs=1, seed=5)
    assert torch.equal(torch.rand(3), expected)
    again = fit_das(model, "hook_mid", base, source, targets, model, rank=1, epochs=1, seed=5)
    assert torch.equal(first.basis, again.basis)  # the seed still fixes the result


def test_the_subspace_defaults_to_the_models_device():
    assert _model_device(nn.Linear(2, 2, device="meta")) == torch.device("meta")
    assert _model_device(_Pattern()) == torch.device("cpu")  # no parameters
    model, base, source, targets = _planted_pairs(n=32)
    result = fit_das(model, "hook_mid", base, source, targets, model, rank=1, epochs=1)
    assert result.basis.device == model.scale.device


def test_models_in_training_mode_are_rejected():
    model, base, source, targets = _planted_pairs(n=8)
    model.train()
    acts = torch.randn(8, D_MODEL)
    full = SubspaceInterchange(torch.eye(D_MODEL))
    with pytest.raises(ValueError, match="collect_activations requires .* evaluation mode"):
        collect_activations(model, "hook_mid", source, model)
    with pytest.raises(ValueError, match="intervened_logits requires .* evaluation mode"):
        intervened_logits(model, "hook_mid", full, base, acts, model)
    with pytest.raises(ValueError, match="fit_das requires .* evaluation mode"):
        fit_das(model, "hook_mid", base, source, targets, model, rank=1)


def test_das_leaves_the_model_untouched():
    model, base, source, targets = _planted_pairs(n=32)
    before = model.scale.detach().clone()
    fit_das(model, "hook_mid", base, source, targets, model, rank=1, epochs=2)
    assert model.scale.requires_grad and torch.equal(model.scale.detach(), before)
    with pytest.raises(ValueError, match="same length"):
        fit_das(model, "hook_mid", base, source[:-1], targets, model, rank=1)
