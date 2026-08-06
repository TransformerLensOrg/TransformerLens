"""Generation from an already-padded prompt.

``generate()`` had no way to be told which prompt tokens are padding, so a
pre-padded tensor generated as though its pads were real context: every real
token's position was shifted and the continuation diverged from the same prompt
unpadded. See #1612.

Two routes now work. ``attention_mask`` states the padding explicitly, and
``padding_side`` — which the bridge accepted but never applied to token input —
reads it off the pad token. Only the explicit mask can express an interior gap
or a pad id that also occurs as a real token.
"""

from __future__ import annotations

import pytest
import torch

GREEDY = dict(max_new_tokens=5, do_sample=False, verbose=False)


@pytest.fixture(scope="module")
def prompt(distilgpt2_bridge) -> torch.Tensor:
    return distilgpt2_bridge.to_tokens("The capital of France is")


@pytest.fixture(scope="module")
def unpadded_continuation(distilgpt2_bridge, prompt) -> list[int]:
    return distilgpt2_bridge.generate(prompt, **GREEDY)[0, prompt.shape[1] :].tolist()


def _left_pad(bridge, tokens: torch.Tensor, n_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = bridge.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = bridge.tokenizer.eos_token_id
    padded = torch.cat([torch.full((1, n_pad), pad_id, dtype=tokens.dtype), tokens], dim=1)
    mask = torch.cat(
        [torch.zeros(1, n_pad, dtype=torch.long), torch.ones(1, tokens.shape[1], dtype=torch.long)],
        dim=1,
    )
    return padded, mask


@pytest.mark.parametrize("use_past_kv_cache", [True, False])
@pytest.mark.parametrize("n_pad", [1, 3, 7])
def test_attention_mask_recovers_the_unpadded_continuation(
    distilgpt2_bridge, prompt, unpadded_continuation, n_pad, use_past_kv_cache
) -> None:
    """The whole point: padding a prompt must not change what it generates."""
    padded, mask = _left_pad(distilgpt2_bridge, prompt, n_pad)

    out = distilgpt2_bridge.generate(
        padded, attention_mask=mask, use_past_kv_cache=use_past_kv_cache, **GREEDY
    )

    assert out[0, n_pad + prompt.shape[1] :].tolist() == unpadded_continuation


def test_without_a_mask_the_pads_are_treated_as_context(
    distilgpt2_bridge, prompt, unpadded_continuation
) -> None:
    """The unfixed behaviour, pinned so a regression is visible rather than silent.

    padding_side defaults to "right", so leading pads are not recognised and the
    continuation drifts. This is the case #1612 reported.
    """
    padded, _ = _left_pad(distilgpt2_bridge, prompt, 4)

    out = distilgpt2_bridge.generate(padded, **GREEDY)

    assert out[0, 4 + prompt.shape[1] :].tolist() != unpadded_continuation


def test_padding_side_left_is_applied_to_token_input(
    distilgpt2_bridge, prompt, unpadded_continuation
) -> None:
    """generate() has always documented a padding_side argument, but applied it
    only when tokenizing string or list input. For a token tensor it was inert."""
    padded, _ = _left_pad(distilgpt2_bridge, prompt, 4)

    out = distilgpt2_bridge.generate(padded, padding_side="left", **GREEDY)

    assert out[0, 4 + prompt.shape[1] :].tolist() == unpadded_continuation


def test_padding_side_is_restored_afterwards(distilgpt2_bridge, prompt) -> None:
    """The tokenizer is shared across a session, so the override must not leak."""
    before = distilgpt2_bridge.tokenizer.padding_side
    padded, _ = _left_pad(distilgpt2_bridge, prompt, 3)

    distilgpt2_bridge.generate(padded, padding_side="left", **GREEDY)

    assert distilgpt2_bridge.tokenizer.padding_side == before


def test_explicit_mask_wins_over_the_padding_side_heuristic(
    distilgpt2_bridge, prompt, unpadded_continuation
) -> None:
    """A caller who states the padding must not be second-guessed by the pad-token
    scan, which here would mask nothing because padding_side is "right"."""
    padded, mask = _left_pad(distilgpt2_bridge, prompt, 4)

    out = distilgpt2_bridge.generate(padded, attention_mask=mask, padding_side="right", **GREEDY)

    assert out[0, 4 + prompt.shape[1] :].tolist() == unpadded_continuation


def test_interior_gap_needs_the_explicit_mask(distilgpt2_bridge, prompt) -> None:
    """padding_side can only describe padding at one edge. A masked-out token in
    the middle shifts later positions just the same, and only a mask says so."""
    pad_id = distilgpt2_bridge.tokenizer.eos_token_id
    gapped = prompt.clone()
    gapped[0, 2] = pad_id
    mask = torch.ones_like(prompt)
    mask[0, 2] = 0
    compact = torch.cat([prompt[:, :2], prompt[:, 3:]], dim=1)

    reference = distilgpt2_bridge.generate(compact, **GREEDY)[0, compact.shape[1] :].tolist()
    via_mask = distilgpt2_bridge.generate(gapped, attention_mask=mask, **GREEDY)[
        0, prompt.shape[1] :
    ].tolist()

    assert via_mask == reference


def test_rows_padded_to_different_lengths(distilgpt2_bridge) -> None:
    """Each row must generate what it would alone, whatever its own pad count."""
    long_prompt = distilgpt2_bridge.to_tokens("The capital of France is")
    short_prompt = distilgpt2_bridge.to_tokens("Hello")
    width = max(long_prompt.shape[1], short_prompt.shape[1])

    rows, masks = [], []
    for tokens in (long_prompt, short_prompt):
        padded, mask = _left_pad(distilgpt2_bridge, tokens, width - tokens.shape[1])
        rows.append(padded)
        masks.append(mask)

    out = distilgpt2_bridge.generate(
        torch.cat(rows, dim=0), attention_mask=torch.cat(masks, dim=0), **GREEDY
    )

    for index, tokens in enumerate((long_prompt, short_prompt)):
        solo = distilgpt2_bridge.generate(tokens, **GREEDY)[0, tokens.shape[1] :].tolist()
        assert out[index, width:].tolist() == solo


def test_the_mask_reaches_every_step_not_just_the_first(distilgpt2_bridge, prompt) -> None:
    """Before #1612 an attention_mask kwarg was absorbed into **multimodal_kwargs,
    which are merged into the forward kwargs on step 0 only. That made the first
    token come out right and every later one wrong, which is worse to debug than a
    uniform failure. Each step must see a mask covering the prompt plus the tokens
    generated so far.
    """
    n_pad = 3
    padded, mask = _left_pad(distilgpt2_bridge, prompt, n_pad)
    prompt_width = padded.shape[1]
    seen: list[torch.Tensor | None] = []

    original = distilgpt2_bridge.original_model.forward

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("attention_mask"))
        return original(*args, **kwargs)

    distilgpt2_bridge.original_model.forward = _spy
    try:
        distilgpt2_bridge.generate(padded, attention_mask=mask, **GREEDY)
    finally:
        distilgpt2_bridge.original_model.forward = original

    assert len(seen) == GREEDY["max_new_tokens"]
    for step, observed in enumerate(seen):
        assert observed is not None, f"step {step} received no attention_mask"
        assert observed.shape[1] == prompt_width + step
        # The prompt's padding stays masked however far generation has run.
        assert observed[0, :n_pad].sum() == 0
        assert observed[0, n_pad:].all()


def test_unpadded_generation_is_unchanged(distilgpt2_bridge, prompt, unpadded_continuation) -> None:
    """An all-ones mask is what the model assumes anyway, so supplying one must
    be a no-op rather than a second code path."""
    out = distilgpt2_bridge.generate(prompt, attention_mask=torch.ones_like(prompt), **GREEDY)

    assert out[0, prompt.shape[1] :].tolist() == unpadded_continuation


def test_string_and_list_input_still_work(distilgpt2_bridge) -> None:
    """The list path builds its own mask; neither route may regress."""
    if distilgpt2_bridge.tokenizer.pad_token_id is None:
        distilgpt2_bridge.tokenizer.pad_token = distilgpt2_bridge.tokenizer.eos_token

    solo = distilgpt2_bridge.generate("The capital of France is", **GREEDY)
    batched = distilgpt2_bridge.generate(["The capital of France is", "Hi"], **GREEDY)

    assert isinstance(solo, str) and solo.startswith("The capital of France is")
    assert batched[0] == solo


def test_mask_shape_must_match_the_prompt(distilgpt2_bridge, prompt) -> None:
    """generate() extends the mask itself, so a pre-extended one is a mistake
    worth naming rather than broadcasting into something unintended."""
    with pytest.raises(ValueError, match="does not match the prompt shape"):
        distilgpt2_bridge.generate(
            prompt, attention_mask=torch.ones(1, prompt.shape[1] + 5, dtype=torch.long), **GREEDY
        )


def test_inputs_embeds_forwards_the_mask_untouched(distilgpt2_bridge, prompt) -> None:
    """There are no token positions to correct on the embeds path, but processors
    emit an attention_mask alongside their other outputs and callers pass the lot
    straight through. Before this parameter existed that mask reached the model via
    **multimodal_kwargs, so it must still arrive rather than raise.
    """
    embeds = distilgpt2_bridge.original_model.get_input_embeddings()(prompt)
    mask = torch.ones_like(prompt)
    seen: list[torch.Tensor | None] = []

    original = distilgpt2_bridge.original_model.forward

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("attention_mask"))
        return original(*args, **kwargs)

    distilgpt2_bridge.original_model.forward = _spy
    try:
        distilgpt2_bridge.generate(embeds, attention_mask=mask, **GREEDY)
    finally:
        distilgpt2_bridge.original_model.forward = original

    assert seen and seen[0] is not None
    torch.testing.assert_close(seen[0], mask)
