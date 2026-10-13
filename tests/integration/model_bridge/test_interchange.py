"""Real-Bridge integration checks for interchange interventions."""

import torch

from transformer_lens.tools.analysis.interchange import (
    NormMatchedPlacebo,
    SubspaceInterchange,
    collect_activations,
    intervened_logits,
    random_basis,
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
