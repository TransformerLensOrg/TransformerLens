"""Real-Bridge integration checks for interchange interventions."""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import fit_sparse_probe
from transformer_lens.tools.analysis._model_state import require_eval_mode
from transformer_lens.tools.analysis.interchange import (
    NormMatchedPlacebo,
    SubspaceInterchange,
    collect_activations,
    fit_das,
    intervened_logits,
    neuron_basis,
    random_basis,
    random_init_bridge,
    random_neuron_basis,
)

PROMPTS = ["The key near the cabinet", "The keys near the cabinet"]


def _setup(bridge):
    tokens = bridge.to_tokens(PROMPTS)
    answers = [bridge.to_single_token(" is"), bridge.to_single_token(" are")]
    last = torch.full((len(PROMPTS),), tokens.shape[1] - 1)
    return tokens, answers, last, (lambda toks: bridge(toks)[:, -1, answers])


def test_collected_activations_match_the_cache(gpt2_bridge):
    tokens, _, last, task_logits = _setup(gpt2_bridge)
    acts = collect_activations(
        gpt2_bridge, "blocks.5.hook_out", tokens, task_logits, positions=last
    )
    _, cache = gpt2_bridge.run_with_cache(tokens)
    assert torch.allclose(acts, cache["blocks.5.hook_out"][:, -1].cpu(), atol=1e-5)


def test_a_full_interchange_at_the_last_block_reproduces_the_source_logits(gpt2_bridge):
    tokens, _, last, task_logits = _setup(gpt2_bridge)
    hook = f"blocks.{gpt2_bridge.cfg.n_layers - 1}.hook_out"
    swapped = tokens.flip(0)  # each prompt takes the other's activation
    source_acts = collect_activations(gpt2_bridge, hook, swapped, task_logits, positions=last)
    out = intervened_logits(
        gpt2_bridge,
        hook,
        SubspaceInterchange(torch.eye(gpt2_bridge.cfg.d_model)),
        tokens,
        source_acts,
        task_logits,
        positions=last,
        batch_size=1,
    )
    with torch.no_grad():
        expected = task_logits(swapped)
    assert torch.allclose(out, expected, atol=1e-4)
    # Singular and plural prompts prefer different verbs, so the swap flips both predictions
    assert torch.equal(out.argmax(-1), task_logits(tokens).argmax(-1).flip(0))


def test_the_placebo_matches_the_reference_displacement_on_real_activations(gpt2_bridge):
    tokens, _, last, task_logits = _setup(gpt2_bridge)
    acts = collect_activations(
        gpt2_bridge, "blocks.3.hook_out", tokens, task_logits, positions=last
    )
    d_model = gpt2_bridge.cfg.d_model
    reference = SubspaceInterchange(random_basis(d_model, 4, seed=0))
    placebo = NormMatchedPlacebo(reference, random_basis(d_model, 4, seed=1))
    base, source = acts, acts.flip(0)
    moved = (placebo(base, source) - base).norm(dim=-1)
    assert torch.allclose(moved, (reference(base, source) - base).norm(dim=-1), rtol=1e-4)


NOUNS = "key dog cat book car boy girl bird tree house road table door letter song plan".split()


def _number_pairs(bridge, nouns):
    """Base and source differ only in the subject's number; the target is the source's verb."""
    singular = bridge.to_tokens([f"The {n} near the cabinet" for n in nouns])
    plural = bridge.to_tokens([f"The {n}s near the cabinet" for n in nouns])
    # BOS, "The", noun, "near", "the", "cabinet": every noun and its plural is one token
    assert singular.shape[1] == plural.shape[1] == 6
    base = torch.cat([singular, plural])
    source = torch.cat([plural, singular])
    targets = torch.cat([torch.ones(len(nouns)), torch.zeros(len(nouns))]).long()
    return base, source, targets


def test_das_finds_subject_number_on_held_out_nouns(gpt2_bridge):
    answers = [gpt2_bridge.to_single_token(" is"), gpt2_bridge.to_single_token(" are")]
    task_logits = lambda toks: gpt2_bridge(toks)[:, -1, answers]
    hook = "blocks.1.hook_out"
    base, source, targets = _number_pairs(gpt2_bridge, NOUNS[:10])
    subject = torch.full((len(base),), 2)  # after the BOS token and "The"
    result = fit_das(
        gpt2_bridge,
        hook,
        base,
        source,
        targets,
        task_logits,
        rank=1,
        base_positions=subject,
        source_positions=subject,
        epochs=10,
        batch_size=20,
        lr=1e-2,
    )

    base, source, targets = _number_pairs(gpt2_bridge, NOUNS[10:])
    subject = torch.full((len(base),), 2)
    source_acts = collect_activations(gpt2_bridge, hook, source, task_logits, positions=subject)

    def accuracy(basis):
        logits = intervened_logits(
            gpt2_bridge,
            hook,
            SubspaceInterchange(basis),
            base,
            source_acts,
            task_logits,
            positions=subject,
        )
        return (logits.argmax(-1) == targets).float().mean().item()

    assert accuracy(result.subspace) >= 0.9
    assert accuracy(random_basis(gpt2_bridge.cfg.d_model, 1, seed=0)) <= 0.2


@pytest.mark.parametrize("name", ["gpt2", "tiny-stories-1M"])
def test_random_init_bridge_matches_the_trained_architecture_without_the_training(name):
    trained = TransformerBridge.boot_transformers(name, device="cpu")
    before = torch.random.get_rng_state()
    untrained = random_init_bridge(name, seed=0, device="cpu")
    assert torch.equal(torch.random.get_rng_state(), before)  # the caller's stream is untouched
    assert type(untrained.original_model) is type(trained.original_model)
    assert untrained.cfg.n_layers == trained.cfg.n_layers
    assert untrained.cfg.d_model == trained.cfg.d_model
    assert untrained.W_E.dtype == torch.float32
    assert not torch.allclose(untrained.W_E, trained.W_E)
    assert torch.equal(untrained.W_E, random_init_bridge(name, seed=0, device="cpu").W_E)
    require_eval_mode(untrained, operation="the random-init control")  # ready for fit_das
    tokens = untrained.to_tokens(PROMPTS)
    with torch.no_grad():
        assert untrained(tokens).shape == trained(tokens).shape


def test_random_init_bridge_casts_to_a_dtype_other_than_the_configs():
    untrained = random_init_bridge(
        "gpt2", seed=0, dtype=torch.bfloat16, device="cpu"
    )  # config: float32
    floating = [p for p in untrained.original_model.parameters() if p.is_floating_point()]
    assert floating and all(p.dtype == torch.bfloat16 for p in floating)


def test_random_init_bridge_puts_its_weights_on_the_device_it_records(monkeypatch):
    # Stand in for a GPU machine, where boot's default device is not the CPU
    import transformer_lens.model_bridge.sources.transformers.source as source

    monkeypatch.setattr(source, "get_device", lambda: "meta")
    untrained = random_init_bridge("gpt2", seed=0)
    assert str(untrained.cfg.device) == "meta"
    assert all(p.device.type == "meta" for p in untrained.original_model.parameters())
    requested = random_init_bridge("gpt2", seed=0, device="cpu")  # wins over boot's default
    assert str(requested.cfg.device) == "cpu"
    assert all(p.device.type == "cpu" for p in requested.original_model.parameters())


def test_the_neuron_a_sparse_probe_selects_carries_subject_number(gpt2_bridge):
    answers = [gpt2_bridge.to_single_token(" is"), gpt2_bridge.to_single_token(" are")]
    task_logits = lambda toks: gpt2_bridge(toks)[:, -1, answers]
    hook = "blocks.0.mlp.hook_post"

    def pairs(place):
        singular = gpt2_bridge.to_tokens([f"The {n} {place}" for n in NOUNS])
        plural = gpt2_bridge.to_tokens([f"The {n}s {place}" for n in NOUNS])
        assert singular.shape[1] == plural.shape[1] == 6  # every word is one token
        base, source = torch.cat([singular, plural]), torch.cat([plural, singular])
        targets = (torch.arange(len(base)) < len(NOUNS)).long()  # " are" when the source is plural
        return base, source, targets, torch.full((len(base),), 2)

    fit, test = pairs("near the cabinet"), pairs("behind the table")
    fit_acts = collect_activations(gpt2_bridge, hook, fit[1], task_logits, positions=fit[3])
    test_acts = collect_activations(gpt2_bridge, hook, test[1], task_logits, positions=test[3])
    probe = fit_sparse_probe(fit_acts, fit[2], k=1)
    d_mlp = gpt2_bridge.cfg.d_mlp

    def accuracy(basis):
        logits = intervened_logits(
            gpt2_bridge,
            hook,
            SubspaceInterchange(basis),
            test[0],
            test_acts,
            task_logits,
            positions=test[3],
        )
        return (logits.argmax(-1) == test[2]).float().mean().item()

    assert accuracy(neuron_basis(d_mlp, probe.selected_features)) >= 0.9
    placebo = random_neuron_basis(d_mlp, 1, seed=0, exclude=probe.selected_features)
    assert accuracy(placebo) <= 0.2
