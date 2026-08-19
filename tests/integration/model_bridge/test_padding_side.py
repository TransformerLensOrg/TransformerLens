"""Padding-side behavior for ragged TransformerBridge text batches."""

import pytest
import torch

PROMPTS = ["Short", "This is a longer sentence for padding."]


def _real_tokens(bridge, prompt: str) -> int:
    return bridge.to_tokens(prompt).shape[1]


def test_to_tokens_honors_padding_side(distilgpt2_bridge) -> None:
    """The per-call argument must override the tokenizer default."""
    bridge = distilgpt2_bridge
    left = bridge.to_tokens(PROMPTS, padding_side="left")
    right = bridge.to_tokens(PROMPTS, padding_side="right")
    short_length = _real_tokens(bridge, PROMPTS[0])
    pad_length = left.shape[1] - short_length

    assert not torch.equal(left, right)
    torch.testing.assert_close(left[0, pad_length:], right[0, :short_length])
    assert torch.all(left[0, :pad_length] == bridge.tokenizer.pad_token_id)
    assert torch.all(right[0, short_length:] == bridge.tokenizer.pad_token_id)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_ragged_forward_preserves_single_prompt_logits(distilgpt2_bridge, padding_side) -> None:
    """Padding must not change logits at any real-token position."""
    bridge = distilgpt2_bridge
    with torch.inference_mode():
        batch_logits = bridge(PROMPTS, padding_side=padding_side)
        single_logits = [bridge(prompt)[0] for prompt in PROMPTS]

    for row, prompt_logits in enumerate(single_logits):
        length = prompt_logits.shape[0]
        actual = (
            batch_logits[row, -length:] if padding_side == "left" else batch_logits[row, :length]
        )
        torch.testing.assert_close(actual, prompt_logits, rtol=1e-6, atol=1e-5)


def test_ragged_compatibility_logits_match_hooked_transformer(
    distilgpt2_bridge_compat, distilgpt2_hooked_processed
) -> None:
    """Default string-list forwarding must use HookedTransformer's layout."""
    with torch.inference_mode():
        bridge_logits = distilgpt2_bridge_compat(PROMPTS)
        hooked_logits = distilgpt2_hooked_processed(PROMPTS)

    torch.testing.assert_close(bridge_logits, hooked_logits, rtol=0, atol=1e-5)


def test_default_forward_matches_to_tokens_layout(distilgpt2_bridge) -> None:
    """String-list forwarding must use the same default layout as ``to_tokens``."""
    bridge = distilgpt2_bridge
    tokens = bridge.to_tokens(PROMPTS)

    with torch.inference_mode():
        text_logits = bridge(PROMPTS)
        token_logits = bridge(tokens)

    torch.testing.assert_close(text_logits, token_logits, rtol=0, atol=0)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_forward_respects_tokenizer_padding_side(distilgpt2_bridge, padding_side) -> None:
    """The tokenizer setting controls forward when no per-call override is given."""
    bridge = distilgpt2_bridge
    original = bridge.tokenizer.padding_side
    try:
        bridge.tokenizer.padding_side = padding_side
        with torch.inference_mode():
            default_logits = bridge(PROMPTS)
            explicit_logits = bridge(PROMPTS, padding_side=padding_side)
    finally:
        bridge.tokenizer.padding_side = original

    torch.testing.assert_close(default_logits, explicit_logits, rtol=0, atol=0)


def test_batched_generation_still_forces_left_padding(distilgpt2_bridge) -> None:
    """Generation keeps real tokens flush-right even when right padding is requested."""
    bridge = distilgpt2_bridge
    _, input_tokens = bridge.generate(
        PROMPTS,
        max_new_tokens=1,
        do_sample=False,
        padding_side="right",
        return_input_tokens=True,
        verbose=False,
    )

    expected = bridge.to_tokens(PROMPTS, padding_side="left")
    assert torch.equal(input_tokens, expected)
