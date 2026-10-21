"""Unit tests for interchange interventions on linear subspaces."""

import numpy as np
import pytest
import torch
from torch import nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedRootModule import HookedRootModule
from transformer_lens.tools.analysis import SubspaceInterchange as exported_interchange
from transformer_lens.tools.analysis import fit_sparse_probe
from transformer_lens.tools.analysis.interchange import (
    LearnedSubspace,
    NormMatchedPlacebo,
    SubspaceInterchange,
    _model_device,
    arbitrary_targets,
    collect_activations,
    fit_das,
    intervened_logits,
    mean_difference_basis,
    neuron_basis,
    probe_basis,
    random_basis,
    random_neuron_basis,
    reinitialized,
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

    labels3 = torch.tensor([0] * 20 + [1] * 30 + [2] * 50)
    generator = torch.Generator().manual_seed(0)
    acts3 = torch.randn(100, D_MODEL, generator=generator)
    acts3 += 3.0 * torch.randn(3, D_MODEL, generator=generator)[labels3]
    three = mean_difference_basis(acts3, labels3.tolist())
    assert three.shape == (D_MODEL, 2) and _orthonormal(three)
    means = torch.stack([acts3[labels3 == c].mean(0) for c in range(3)])
    centred = (means - means.mean(0)).T
    assert (centred - three @ (three.T @ centred)).abs().max() < 1e-4
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


def test_the_epoch_loss_is_the_mean_cross_entropy_per_pair():
    model, base, source, targets = _planted_pairs(n=50)
    # lr=0 keeps the basis at its initialisation; uneven batches of 16 and a last batch of 2
    result = fit_das(
        model, "hook_mid", base, source, targets, model, rank=1, epochs=2, batch_size=16, lr=0.0
    )
    acts = collect_activations(model, "hook_mid", source, model)
    logits = intervened_logits(
        model, "hook_mid", SubspaceInterchange(result.subspace), base, acts, model
    )
    expected = torch.nn.functional.cross_entropy(logits, targets).item()
    assert result.losses == pytest.approx((expected, expected), rel=1e-5)


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


def _forbid_reseeding(monkeypatch):
    def reseeded(*args, **kwargs):
        raise AssertionError("a global or device generator was reseeded")

    for module, name in (
        (torch, "manual_seed"),
        (torch.random, "manual_seed"),
        (torch.cuda, "manual_seed"),
        (torch.cuda, "manual_seed_all"),
        (torch.mps, "manual_seed"),
    ):
        monkeypatch.setattr(module, name, reseeded)


def test_das_never_reseeds_a_device_generator(monkeypatch):
    _forbid_reseeding(monkeypatch)
    model, base, source, targets = _planted_pairs(n=32)
    fit_das(model, "hook_mid", base, source, targets, model, rank=1, epochs=1, seed=5)


def test_the_subspace_defaults_to_the_models_device():
    assert _model_device(nn.Linear(2, 2, device="meta")) == torch.device("meta")
    assert _model_device(_Pattern()) == torch.device("cpu")  # no parameters
    model, base, source, targets = _planted_pairs(n=32)
    result = fit_das(model, "hook_mid", base, source, targets, model, rank=1, epochs=1)
    assert result.basis.device == model.scale.device

    class _OffCpu(HookedRootModule):
        """Stands in for a GPU model: its first parameter, which sets the default, is off the CPU."""

        def __init__(self, direction):
            super().__init__()
            self.offload = nn.Parameter(torch.empty(1, device="meta"))
            self.planted = _Planted(direction)
            self.setup()
            self.eval()

        def forward(self, x):
            return self.planted(x)

    off_cpu = _OffCpu(model.direction)
    assert _model_device(off_cpu) == torch.device("meta")
    # No epochs, as a meta basis cannot train against CPU activations
    result = fit_das(off_cpu, "planted.hook_mid", base, source, targets, off_cpu, rank=1, epochs=0)
    assert result.basis.device == torch.device("meta")


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
    assert model.scale.grad is None  # frozen while training, so no gradient reaches the model
    with pytest.raises(ValueError, match="same length"):
        fit_das(model, "hook_mid", base, source[:-1], targets, model, rank=1)


def test_arbitrary_targets_are_fixed_per_key_and_seed():
    keys = [f"source {i}" for i in range(400)]
    targets = arbitrary_targets(keys, n_classes=3, seed=0)
    assert targets.dtype == torch.long and set(targets.tolist()) == {0, 1, 2}
    assert torch.equal(targets, arbitrary_targets(keys, n_classes=3, seed=0))
    assert not torch.equal(targets, arbitrary_targets(keys, n_classes=3, seed=1))
    assert torch.equal(arbitrary_targets(keys[:5] * 2, n_classes=3, seed=0)[5:], targets[:5])
    assert all(abs(c - 400 / 3) < 50 for c in torch.bincount(targets).tolist())


class _Layers(HookedRootModule):
    """Only standard layers, each with reset_parameters, and a fixed buffer."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(D_MODEL, D_MODEL)
        self.norm = nn.LayerNorm(D_MODEL)
        self.register_buffer("fixed", torch.randn(D_MODEL))
        self.hook_mid = HookPoint()
        self.setup()
        self.eval()


def test_reinitialized_resets_every_parameter_and_keeps_buffers():
    model = _Layers()
    with torch.no_grad():
        model.norm.weight.add_(1.0)  # as training would, away from LayerNorm's ones and zeros
        model.norm.bias.add_(1.0)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    fresh = reinitialized(model, seed=0)
    for name, parameter in fresh.named_parameters():
        assert not torch.equal(parameter, before[name]), name
        assert torch.equal(dict(model.named_parameters())[name], before[name])  # original untouched
    assert torch.equal(fresh.fixed, model.fixed) and not fresh.training
    assert torch.equal(reinitialized(model, seed=0).inp.weight, fresh.inp.weight)


def test_reinitialized_resets_a_root_that_resets_its_own_parameters():
    class _SelfResetting(_Layers):  # the root owns a parameter and knows how to reset it
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.full((D_MODEL,), 3.0))

        def reset_parameters(self):
            nn.init.normal_(self.scale)

    model = _SelfResetting()
    fresh = reinitialized(model, seed=0)
    assert not torch.equal(fresh.scale, model.scale)


def test_reinitialized_refuses_parameters_it_cannot_reset():
    with pytest.raises(ValueError, match="cannot reset scale"):
        reinitialized(_Planted(torch.randn(D_MODEL)), seed=0)  # a parameter on the root itself

    class _RawWeight(nn.Module):  # like HuggingFace's Conv1D: a parameter and no reset_parameters
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(D_MODEL, D_MODEL))

    model = _Layers()
    model.conv = _RawWeight()
    with pytest.raises(ValueError, match="cannot reset conv.weight"):
        reinitialized(model, seed=0)


def test_reinitialized_leaves_every_random_stream_alone(monkeypatch):
    model = _Layers()
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    _forbid_reseeding(monkeypatch)
    reinitialized(model, seed=0)
    assert torch.equal(torch.rand(3), expected)


def _typed_pairs(prototypes: torch.Tensor, n: int, generator: torch.Generator):
    """Random bases; sources drawn from a few recurring types, so the types act as keys."""
    kinds = torch.randint(len(prototypes), (n,), generator=generator)
    sources = prototypes[kinds] + 0.05 * torch.randn(n, D_MODEL, generator=generator)
    return torch.randn(n, D_MODEL, generator=generator), sources, kinds


def _held_out_iia(model, fit, test, fit_targets, test_targets) -> float:
    result = fit_das(
        model, "hook_mid", fit[0], fit[1], fit_targets, model, rank=1, epochs=40, lr=5e-2
    )
    acts = collect_activations(model, "hook_mid", test[1], model)
    logits = intervened_logits(
        model, "hook_mid", SubspaceInterchange(result.subspace), test[0], acts, model
    )
    return (logits.argmax(-1) == test_targets).float().mean().item()


def test_das_beats_the_arbitrary_label_control_on_recurring_keys():
    generator = torch.Generator().manual_seed(0)
    model = _Planted(torch.randn(D_MODEL, generator=generator))
    prototypes = torch.randn(64, D_MODEL, generator=generator)
    fit, test = _typed_pairs(prototypes, 256, generator), _typed_pairs(prototypes, 256, generator)
    # Held-out sources share their keys with training sources, so the control could be learned
    assert set(test[2].tolist()) <= set(fit[2].tolist())
    real = lambda sources: (sources @ model.direction > 0).long()
    labels = lambda kinds: arbitrary_targets(kinds.tolist(), 2, seed=0)
    assert _held_out_iia(model, fit, test, real(fit[1]), real(test[1])) > 0.9
    assert _held_out_iia(model, fit, test, labels(fit[2]), labels(test[2])) < 0.7


def test_the_arbitrary_label_control_can_reject_a_subspace():
    # Types placed so that one direction fits their arbitrary labels: the control then does as
    # well as the real alignment, and DAS does not beat it
    generator = torch.Generator().manual_seed(0)
    model = _Planted(torch.randn(D_MODEL, generator=generator))
    prototypes = torch.randn(8, D_MODEL, generator=generator)
    wanted = arbitrary_targets(list(range(8)), 2, seed=0)
    prototypes[(prototypes @ model.direction > 0).long() != wanted] *= -1
    fit, test = _typed_pairs(prototypes, 256, generator), _typed_pairs(prototypes, 256, generator)
    real = lambda sources: (sources @ model.direction > 0).long()
    labels = lambda kinds: arbitrary_targets(kinds.tolist(), 2, seed=0)
    control = _held_out_iia(model, fit, test, labels(fit[2]), labels(test[2]))
    assert control > 0.9
    assert _held_out_iia(model, fit, test, real(fit[1]), real(test[1])) - control < 0.1


def test_neuron_basis_selects_exactly_the_given_neurons():
    basis = neuron_basis(D_MODEL, [4, 1])
    assert basis.shape == (D_MODEL, 2) and _orthonormal(basis)
    assert basis[4, 0] == 1 and basis[1, 1] == 1 and basis.sum() == 2
    base, source = torch.randn(3, D_MODEL), torch.randn(3, D_MODEL)
    out = SubspaceInterchange(basis)(base, source)
    chosen = torch.tensor([1, 4])
    others = torch.tensor([0, 2, 3, 5])
    assert torch.allclose(out[:, chosen], source[:, chosen])
    assert torch.equal(out[:, others], base[:, others])


def test_neuron_basis_accepts_lists_arrays_and_tensors():
    expected = neuron_basis(D_MODEL, [0, 3])
    for neurons in ([0, 3], np.array([0, 3]), torch.tensor([0, 3], dtype=torch.int32)):
        assert torch.equal(neuron_basis(D_MODEL, neurons), expected)


@pytest.mark.parametrize(
    ("neurons", "message"),
    [
        ([], "at least one neuron"),
        ([D_MODEL], "must be in"),
        ([-1], "must be in"),
        ([2, 2], "must not repeat"),
        (torch.tensor([1.0]), "must be integers"),
        (torch.tensor([True, False]), "must be integers"),
        (torch.tensor([[0], [3]]), "one-dimensional"),
    ],
)
def test_neuron_basis_refuses_invalid_indices(neurons, message):
    with pytest.raises(ValueError, match=message):
        neuron_basis(D_MODEL, neurons)


def test_random_neuron_basis_is_seeded_and_avoids_excluded_neurons():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    first = random_neuron_basis(100, 5, seed=0, exclude=[0, 1, 2])
    assert torch.equal(torch.rand(3), expected)  # the caller's stream is untouched
    assert torch.equal(first, random_neuron_basis(100, 5, seed=0, exclude=[0, 1, 2]))
    assert not torch.equal(first, random_neuron_basis(100, 5, seed=1, exclude=[0, 1, 2]))
    assert _orthonormal(first) and first[:3].sum() == 0
    assert random_neuron_basis(100, 5, seed=0).shape == (100, 5)  # no exclusions
    with pytest.raises(ValueError, match="k must be in"):
        random_neuron_basis(D_MODEL, D_MODEL - 1, seed=0, exclude=[0, 1])


def test_the_placebo_stays_within_random_neurons():
    base, source = torch.randn(8, 100), torch.randn(8, 100)
    reference = SubspaceInterchange(neuron_basis(100, [3, 7]))
    placebo = NormMatchedPlacebo(reference, random_neuron_basis(100, 2, seed=0, exclude=[3, 7]))
    moved = placebo(base, source) - base
    assert torch.allclose(
        moved.norm(dim=-1), (reference(base, source) - base).norm(dim=-1), atol=1e-5
    )
    assert (moved.abs() > 1e-6).sum(-1).max() <= 2 and moved[:, [3, 7]].abs().max() == 0


class _OneNeuron(HookedRootModule):
    """The output reads a single planted neuron of the hooked activation."""

    def __init__(self, neuron: int):
        super().__init__()
        self.neuron = neuron
        self.hook_post = HookPoint()
        self.setup()
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = 4.0 * self.hook_post(x)[:, self.neuron]
        return torch.stack([-score, score], dim=-1)


def test_the_neuron_a_sparse_probe_selects_carries_a_planted_variable():
    d_mlp, planted = 64, 17
    model = _OneNeuron(planted)
    generator = torch.Generator().manual_seed(0)

    def pairs(n):
        base, source = torch.randn(n, d_mlp, generator=generator), torch.randn(
            n, d_mlp, generator=generator
        )
        return base, source, (source[:, planted] > 0).long()

    fit_base, fit_source, fit_targets = pairs(200)
    test_base, test_source, test_targets = pairs(200)
    probe = fit_sparse_probe(
        collect_activations(model, "hook_post", fit_source, model), fit_targets, k=1
    )
    assert probe.selected_features.tolist() == [planted]

    source_acts = collect_activations(model, "hook_post", test_source, model)
    needed = (test_base[:, planted] > 0).long() != test_targets  # the output must change

    def iia(basis):
        logits = intervened_logits(
            model, "hook_post", SubspaceInterchange(basis), test_base, source_acts, model
        )
        return (logits.argmax(-1) == test_targets)[needed].float().mean().item()

    assert iia(neuron_basis(d_mlp, probe.selected_features)) == 1.0
    assert iia(random_neuron_basis(d_mlp, 1, seed=0, exclude=probe.selected_features)) == 0.0
