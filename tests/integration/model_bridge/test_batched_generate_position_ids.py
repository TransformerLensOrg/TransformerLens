"""position_ids handling during batched-list generation.

Batched list input is left-padded internally, so each row's real tokens start at
a different offset. The bridge derives position_ids for that, but only models
that neither reject the kwarg nor derive positions themselves may receive them
(#1626).

The cached decoding path needs the same gate as the prompt path. Every branch
there supplies position_ids, including a ``total_len - 1`` fallback that counts
pad slots, so gating only the prompt derivation diverts a refused model into the
fallback instead of leaving it alone.

OPT is the vehicle for the refused case: ``OPTLearnedPositionalEmbedding``
consumes the attention mask and derives its own positions, so the gate declines
it, while its forward would happily accept the kwarg and use it.
"""

from __future__ import annotations

import functools

import pytest
import torch

GREEDY = dict(max_new_tokens=4, do_sample=False, verbose=False)
PROMPTS = ["The capital of France is the city of", "Hi"]


@pytest.fixture(scope="module")
def opt_bridge():
    """A model the gate refuses. Its positional embedding reads the mask."""
    from transformer_lens.model_bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers(
        "hf-internal-testing/tiny-random-OPTForCausalLM", device="cpu", dtype=torch.float32
    )
    bridge.eval()
    return bridge


def _stack_logits(output) -> torch.Tensor:
    logits = output.logits
    return torch.stack(list(logits)) if isinstance(logits, (list, tuple)) else logits


def _position_ids_per_step(bridge, use_past_kv_cache: bool) -> list:
    """The position_ids each forward actually received, one entry per step."""
    seen: list = []
    original = bridge.original_model.forward

    # functools.wraps so the gate still sees the real signature; a bare
    # (*args, **kwargs) spy would look like it accepts position_ids.
    @functools.wraps(original)
    def _spy(*args, **kwargs):
        supplied = kwargs.get("position_ids")
        seen.append(None if supplied is None else supplied.tolist())
        return original(*args, **kwargs)

    bridge.original_model.forward = _spy
    try:
        bridge.generate(list(PROMPTS), use_past_kv_cache=use_past_kv_cache, **GREEDY)
    finally:
        bridge.original_model.forward = original
    return seen


def test_gate_refuses_opt(opt_bridge) -> None:
    """Guards the premise of the tests below: OPT must be the refused case."""
    assert opt_bridge._accepts_derived_position_ids() is False


def test_refused_model_generates_identically_with_and_without_cache(opt_bridge) -> None:
    """Cached decoding must not change the answer.

    Compared on logits rather than decoded text on purpose: greedy argmax
    absorbs the drift and the strings match even when the positions are wrong.
    """
    cached = opt_bridge.generate(
        list(PROMPTS), use_past_kv_cache=True, output_logits=True, **GREEDY
    )
    uncached = opt_bridge.generate(
        list(PROMPTS), use_past_kv_cache=False, output_logits=True, **GREEDY
    )

    torch.testing.assert_close(_stack_logits(cached), _stack_logits(uncached), rtol=0, atol=1e-5)


def test_refused_model_receives_no_position_ids_on_cached_steps(opt_bridge) -> None:
    """The mechanism, not just the symptom.

    The fallback supplies a per-batch constant, so a coarser check can miss it;
    assert the kwarg never reaches a model that derives positions itself.
    """
    assert _position_ids_per_step(opt_bridge, use_past_kv_cache=True) == [None] * (
        GREEDY["max_new_tokens"]
    )


def test_accepted_model_still_receives_per_row_position_ids(distilgpt2_bridge) -> None:
    """The gate must not disarm the models it was never meant to exclude."""
    if distilgpt2_bridge.tokenizer.pad_token_id is None:
        distilgpt2_bridge.tokenizer.pad_token = distilgpt2_bridge.tokenizer.eos_token

    seen = _position_ids_per_step(distilgpt2_bridge, use_past_kv_cache=True)

    assert seen[0] is not None, "prompt step must still receive derived positions"
    cached_steps = [step for step in seen[1:] if step is not None]
    assert len(cached_steps) == len(seen) - 1, "cached steps must still be supplied"
    # Row 1 ("Hi") is left-padded, so its position must be strictly lower than
    # row 0's. A pad-counting fallback would give both rows the same value.
    first_cached = cached_steps[0]
    assert first_cached[1][0] < first_cached[0][0], first_cached


def test_accepted_model_generates_identically_with_and_without_cache(distilgpt2_bridge) -> None:
    """Control for the parity property on a model the gate allows.

    Looser than the OPT case at 1e-3. Cached decoding and full recompute sum in
    different orders, which on distilgpt2's logit scale of ~132 shows as 1.4e-04,
    or 1e-06 relative. The regression this guards moves logits by ~0.3, so the
    margin is still more than two orders of magnitude.
    """
    if distilgpt2_bridge.tokenizer.pad_token_id is None:
        distilgpt2_bridge.tokenizer.pad_token = distilgpt2_bridge.tokenizer.eos_token

    cached = distilgpt2_bridge.generate(
        list(PROMPTS), use_past_kv_cache=True, output_logits=True, **GREEDY
    )
    uncached = distilgpt2_bridge.generate(
        list(PROMPTS), use_past_kv_cache=False, output_logits=True, **GREEDY
    )

    torch.testing.assert_close(_stack_logits(cached), _stack_logits(uncached), rtol=0, atol=1e-3)
