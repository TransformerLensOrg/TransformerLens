"""padding_side on token input to forward().

forward() applied padding_side only while tokenizing a string list, so a token
tensor that was already left-padded ran unmasked: the pads were attended and every
real token sat at a shifted position. generate() reads the padding off the pad
token when padding_side is passed explicitly (#1617); forward(), and with it
run_with_cache and the loss, now does the same.

The derived mask also has to respect prepend_bos. distilgpt2's pad id is its BOS
id, so the last leading pad is only a real BOS when one was prepended.
"""

from __future__ import annotations

import copy

import pytest
import torch

PROMPTS = ["The capital of France is", "Hello"]


def _real_row(logits: torch.Tensor, row: int, n_real: int) -> torch.Tensor:
    """The last n_real positions of a left-padded row."""
    return logits[row, -n_real:]


def _mask(padded: torch.Tensor, n_real_short: int) -> torch.Tensor:
    """The explicit mask for PROMPTS padded on the left: row 0 is unpadded."""
    short = (torch.arange(padded.shape[1]) >= padded.shape[1] - n_real_short).long()
    return torch.stack([torch.ones_like(short), short])


@pytest.fixture(scope="module")
def padded(distilgpt2_bridge) -> torch.Tensor:
    return distilgpt2_bridge.to_tokens(PROMPTS, padding_side="left")


@pytest.fixture(scope="module")
def short_alone(distilgpt2_bridge) -> torch.Tensor:
    with torch.no_grad():
        return distilgpt2_bridge(distilgpt2_bridge.to_tokens(PROMPTS[1]))[0]


def test_padding_side_left_is_applied_to_token_input(
    distilgpt2_bridge, padded, short_alone
) -> None:
    """The short row must give the logits it gives on its own."""
    with torch.no_grad():
        logits = distilgpt2_bridge(padded, padding_side="left")

    actual = _real_row(logits, 1, short_alone.shape[0])
    torch.testing.assert_close(actual, short_alone, rtol=1e-4, atol=1e-4)


def test_token_input_matches_the_same_strings(distilgpt2_bridge, padded) -> None:
    """Pre-tokenizing must not change the result of a left-padded batch."""
    with torch.no_grad():
        from_strings = distilgpt2_bridge(PROMPTS, padding_side="left")
        from_tokens = distilgpt2_bridge(padded, padding_side="left")

    torch.testing.assert_close(from_tokens, from_strings, rtol=0, atol=0)


def test_run_with_cache_reads_the_padding_too(distilgpt2_bridge, padded) -> None:
    """Activations, not just logits, must be those of the unpadded prompt."""
    hook = "blocks.1.hook_resid_post"
    with torch.no_grad():
        _, cache = distilgpt2_bridge.run_with_cache(padded, padding_side="left")
        _, alone = distilgpt2_bridge.run_with_cache(distilgpt2_bridge.to_tokens(PROMPTS[1]))

    expected = alone[hook][0]
    actual = cache[hook][1, -expected.shape[0] :]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_loss_ignores_the_pads(distilgpt2_bridge, padded) -> None:
    """The derived mask must reach the loss, as an explicit one does."""
    mask = _mask(padded, distilgpt2_bridge.to_tokens(PROMPTS[1]).shape[1])
    with torch.no_grad():
        derived = distilgpt2_bridge(padded, return_type="loss", padding_side="left")
        explicit = distilgpt2_bridge(padded, return_type="loss", attention_mask=mask)

    torch.testing.assert_close(derived, explicit)


def test_without_padding_side_token_input_is_unchanged(distilgpt2_bridge, padded) -> None:
    """Callers who don't pass padding_side keep the tensor as given."""
    with torch.no_grad():
        default = distilgpt2_bridge(padded)
        unmasked = distilgpt2_bridge(padded, attention_mask=torch.ones_like(padded))

    torch.testing.assert_close(default, unmasked, rtol=1e-5, atol=1e-5)


def test_unpadded_tensor_keeps_the_unmasked_path(distilgpt2_bridge) -> None:
    """An all-ones mask is what the model assumes anyway, so none is passed."""
    tokens = distilgpt2_bridge.to_tokens(PROMPTS[0])
    seen: dict[str, object] = {}
    original = distilgpt2_bridge.original_model.forward

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    distilgpt2_bridge.original_model.forward = _spy
    try:
        with torch.no_grad():
            distilgpt2_bridge(tokens, padding_side="left")
    finally:
        distilgpt2_bridge.original_model.forward = original

    assert seen.get("attention_mask") is None
    assert seen.get("position_ids") is None


def test_explicit_mask_wins_over_padding_side(distilgpt2_bridge, padded, short_alone) -> None:
    """A stated mask must not be replaced by the pad-token scan."""
    mask = _mask(padded, short_alone.shape[0])
    with torch.no_grad():
        logits = distilgpt2_bridge(padded, attention_mask=mask, padding_side="right")

    actual = _real_row(logits, 1, short_alone.shape[0])
    torch.testing.assert_close(actual, short_alone, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("as_tokens", [False, True])
def test_prepend_bos_false_keeps_every_pad_masked(distilgpt2_bridge, as_tokens) -> None:
    """Without a prepended BOS, no leading pad may be unmasked as one."""
    bridge = distilgpt2_bridge
    with torch.no_grad():
        alone = bridge(bridge.to_tokens(PROMPTS[1], prepend_bos=False), prepend_bos=False)[0]
        if as_tokens:
            batch = bridge.to_tokens(PROMPTS, prepend_bos=False, padding_side="left")
        else:
            batch = PROMPTS
        logits = bridge(batch, prepend_bos=False, padding_side="left")

    torch.testing.assert_close(_real_row(logits, 1, alone.shape[0]), alone, rtol=1e-4, atol=1e-4)


def test_padding_side_without_a_tokenizer_is_an_error(distilgpt2_bridge, padded) -> None:
    """With no tokenizer the argument can't be honored; ignoring it would bring back
    the wrong logits silently."""
    bridge = copy.copy(distilgpt2_bridge)
    bridge.tokenizer = None

    with pytest.raises(ValueError, match="attention_mask"):
        bridge(padded, padding_side="left")


def test_padding_side_without_a_pad_token_is_an_error(distilgpt2_bridge, padded) -> None:
    tokenizer = distilgpt2_bridge.tokenizer
    original_pad = tokenizer.pad_token_id
    tokenizer.pad_token_id = None
    try:
        with pytest.raises(ValueError, match="pad_token_id"):
            distilgpt2_bridge(padded, padding_side="left")
    finally:
        tokenizer.pad_token_id = original_pad


def test_right_padded_loss_ignores_the_pads(distilgpt2_bridge) -> None:
    """Causal right padding leaves the real-token logits alone, but the loss must
    still skip the transitions into the pads."""
    padded = distilgpt2_bridge.to_tokens(PROMPTS, padding_side="right")
    n_real = distilgpt2_bridge.to_tokens(PROMPTS[1]).shape[1]
    mask = torch.ones_like(padded)
    mask[1, n_real:] = 0
    with torch.no_grad():
        derived = distilgpt2_bridge(padded, return_type="loss", padding_side="right")
        explicit = distilgpt2_bridge(padded, return_type="loss", attention_mask=mask)

    torch.testing.assert_close(derived, explicit)


def test_right_padded_logits_still_run_unmasked(distilgpt2_bridge) -> None:
    """The loss mask must not reach the model: return_type="both" gives the same
    logits as return_type="logits", pad positions included."""
    padded = distilgpt2_bridge.to_tokens(PROMPTS, padding_side="right")
    with torch.no_grad():
        logits = distilgpt2_bridge(padded, padding_side="right")
        both_logits, _ = distilgpt2_bridge(padded, return_type="both", padding_side="right")

    torch.testing.assert_close(both_logits, logits, rtol=0, atol=0)
