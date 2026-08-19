"""Padding behavior for ragged string lists passed to TransformerBridge."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from transformer_lens.utilities import get_attention_mask

PROMPTS = ["The quick brown fox", "Hello there, world! This is a longer sentence."]
RESID_PRE = "blocks.0.hook_resid_pre"


@pytest.fixture(scope="module")
def compatibility_bridge():
    """A cached model used for token-layout and real-token numerical checks."""
    from transformer_lens.model_bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu", dtype=torch.float32)
    bridge.enable_compatibility_mode()
    bridge.eval()
    return bridge


def test_to_tokens_honors_explicit_padding_side(compatibility_bridge) -> None:
    """A per-call padding override must win without mutating the shared tokenizer."""
    tokenizer = compatibility_bridge.tokenizer
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        left = compatibility_bridge.to_tokens(PROMPTS, padding_side="left")
        right = compatibility_bridge.to_tokens(PROMPTS, padding_side="right")

        assert not torch.equal(left, right)
        assert left[0, 0].item() == tokenizer.pad_token_id
        assert left[0, -1].item() != tokenizer.pad_token_id
        assert right[0, -1].item() == tokenizer.pad_token_id
        assert tokenizer.padding_side == "right"
    finally:
        tokenizer.padding_side = original_side


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_ragged_forward_matches_public_layout_and_single_sequences(
    compatibility_bridge, monkeypatch, padding_side: str
) -> None:
    """String-list forward uses public token layout and preserves real-token values."""
    tokenizer = compatibility_bridge.tokenizer
    original_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        expected_tokens = compatibility_bridge.to_tokens(PROMPTS)
        attention_mask = get_attention_mask(tokenizer, expected_tokens, prepend_bos=True).bool()
        tokenization_calls: list[torch.Tensor] = []
        original_to_tokens: Callable[..., torch.Tensor] = compatibility_bridge.to_tokens

        def recording_to_tokens(input: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
            tokens = original_to_tokens(input, *args, **kwargs)
            if isinstance(input, list):
                tokenization_calls.append(tokens.detach().clone())
            return tokens

        monkeypatch.setattr(compatibility_bridge, "to_tokens", recording_to_tokens)

        with torch.no_grad():
            batch_logits, batch_cache = compatibility_bridge.run_with_cache(
                PROMPTS, names_filter=[RESID_PRE]
            )
            single_results = [
                compatibility_bridge.run_with_cache(prompt, names_filter=[RESID_PRE])
                for prompt in PROMPTS
            ]

        assert len(tokenization_calls) == 1
        assert torch.equal(tokenization_calls[0], expected_tokens)

        for index, (single_logits, single_cache) in enumerate(single_results):
            real_tokens = attention_mask[index]
            torch.testing.assert_close(
                batch_logits[index, real_tokens], single_logits[0], rtol=1e-5, atol=1e-4
            )
            torch.testing.assert_close(
                batch_cache[RESID_PRE][index, real_tokens],
                single_cache[RESID_PRE][0],
                rtol=1e-5,
                atol=1e-4,
            )
    finally:
        tokenizer.padding_side = original_side


def test_generate_keeps_left_padding_for_ragged_strings(compatibility_bridge) -> None:
    """Generation keeps real tokens flush-right even when the tokenizer defaults right."""
    tokenizer = compatibility_bridge.tokenizer
    original_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        expected_tokens = compatibility_bridge.to_tokens(PROMPTS)
        tokenizer.padding_side = "right"

        _, input_tokens = compatibility_bridge.generate(
            PROMPTS,
            max_new_tokens=1,
            do_sample=False,
            verbose=False,
            return_input_tokens=True,
        )

        assert torch.equal(input_tokens, expected_tokens)
        assert tokenizer.padding_side == "right"
    finally:
        tokenizer.padding_side = original_side
