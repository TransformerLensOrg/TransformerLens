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


def test_default_right_padded_text_matches_pretokenized_logits(compatibility_bridge) -> None:
    """Default text forwarding must match the public right-padded token layout."""
    tokenizer = compatibility_bridge.tokenizer
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        tokens = compatibility_bridge.to_tokens(PROMPTS)
        with torch.no_grad():
            text_logits = compatibility_bridge(PROMPTS)
            token_logits = compatibility_bridge(tokens)

        torch.testing.assert_close(text_logits, token_logits, rtol=0, atol=0)
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
        original_tokenize: Callable[
            ..., tuple[torch.Tensor, torch.Tensor]
        ] = compatibility_bridge._to_tokens_and_mask

        def recording_tokenize(
            input: Any, *args: Any, **kwargs: Any
        ) -> tuple[torch.Tensor, torch.Tensor]:
            tokens, mask = original_tokenize(input, *args, **kwargs)
            if isinstance(input, list):
                tokenization_calls.append(tokens.detach().clone())
            return tokens, mask

        monkeypatch.setattr(compatibility_bridge, "_to_tokens_and_mask", recording_tokenize)

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


def test_generate_leaves_padding_side_intact_when_tokenization_raises(
    compatibility_bridge, monkeypatch
) -> None:
    """A raise mid-tokenization must not pin the shared tokenizer to left-padding.

    The tokenizer is shared with to_tokens/forward/run_with_cache, so a side that
    survives a failed generate() silently pads every later call the wrong way, with
    nothing tying the bad activations back to the earlier failure.
    """
    tokenizer = compatibility_bridge.tokenizer
    original_side = tokenizer.padding_side

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("tokenization failed")

    try:
        tokenizer.padding_side = "right"
        monkeypatch.setattr(compatibility_bridge, "_to_tokens_and_mask", _explode)

        with pytest.raises(RuntimeError, match="tokenization failed"):
            compatibility_bridge.generate(PROMPTS, max_new_tokens=1, do_sample=False, verbose=False)

        assert tokenizer.padding_side == "right"
    finally:
        tokenizer.padding_side = original_side


@pytest.mark.parametrize("prepend_bos", [True, False])
def test_leading_literal_pad_token_matches_single_prompt(
    compatibility_bridge, prepend_bos: bool
) -> None:
    """A real leading token with the pad id must stay attended, as the tokenizer marks it.

    distilgpt2's pad, BOS and EOS share one id, so the token ids alone can't show
    where a left-padded row's padding ends.
    """
    eos = compatibility_bridge.tokenizer.eos_token
    prompts = [eos + " hello", "the cat sat on a mat"]
    with torch.no_grad():
        alone = compatibility_bridge(prompts[0], prepend_bos=prepend_bos)[0, -1]
        logits = compatibility_bridge(prompts, prepend_bos=prepend_bos, padding_side="left")
        cached_logits, _ = compatibility_bridge.run_with_cache(
            prompts, prepend_bos=prepend_bos, padding_side="left", names_filter=[RESID_PRE]
        )

    torch.testing.assert_close(logits[0, -1], alone, rtol=0, atol=1e-4)
    torch.testing.assert_close(cached_logits[0, -1], alone, rtol=0, atol=1e-4)


def test_input_to_embed_honors_padding_side_for_strings(compatibility_bridge) -> None:
    """A left-padded string batch must return the mask that resuming the pass needs."""
    tokenizer = compatibility_bridge.tokenizer
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        with torch.no_grad():
            alone = compatibility_bridge(PROMPTS[0])[0, -1]
            residual, _, _, mask = compatibility_bridge.input_to_embed(PROMPTS, padding_side="left")
            resumed = compatibility_bridge(residual, start_at_layer=0, attention_mask=mask)
    finally:
        tokenizer.padding_side = original_side

    torch.testing.assert_close(resumed[0, -1], alone, rtol=0, atol=1e-4)


LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def llama_bridge():
    """A BOS-prepending tokenizer whose pad id differs from BOS, unlike distilgpt2's."""
    from transformer_lens.model_bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers(LLAMA, device="cpu", dtype=torch.float32)
    assert bridge.cfg.tokenizer_prepends_bos and not bridge.cfg.tokenizer_appends_eos
    return bridge.eval()


@pytest.fixture(scope="module")
def llama_eos_bridge():
    """The same model with an EOS-appending tokenizer (OLMo-style), detected at boot."""
    from transformers import AutoTokenizer

    from transformer_lens.model_bridge import TransformerBridge

    tokenizer = AutoTokenizer.from_pretrained(LLAMA, add_eos_token=True)
    bridge = TransformerBridge.boot_transformers(
        LLAMA, device="cpu", dtype=torch.float32, tokenizer=tokenizer
    )
    assert bridge.cfg.tokenizer_prepends_bos and bridge.cfg.tokenizer_appends_eos
    return bridge.eval()


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("prepend_bos", [False, True])
def test_bos_removal_keeps_every_pad_masked(llama_bridge, prepend_bos, padding_side) -> None:
    """Dropping the tokenizer's BOS shifts each row by one token; the mask must shift with it."""
    bos = llama_bridge.tokenizer.bos_token_id
    _, tokens, _, mask = llama_bridge.input_to_embed(
        PROMPTS, prepend_bos=prepend_bos, padding_side=padding_side
    )
    with torch.no_grad():
        logits = llama_bridge(PROMPTS, prepend_bos=prepend_bos, padding_side=padding_side)
    for row, prompt in enumerate(PROMPTS):
        alone_tokens = llama_bridge.to_tokens(prompt, prepend_bos=prepend_bos)[0]
        assert (alone_tokens == bos).sum().item() == int(prepend_bos)
        keep = mask[row].bool()
        assert torch.equal(tokens[row][keep], alone_tokens)
        with torch.no_grad():
            alone = llama_bridge(prompt, prepend_bos=prepend_bos)[0]
        torch.testing.assert_close(logits[row][keep], alone, rtol=0, atol=1e-5)


@pytest.mark.parametrize("prepend_bos", [False, True])
def test_eos_strip_keeps_every_pad_masked(llama_eos_bridge, prepend_bos) -> None:
    """A left-padded batch ends in EOS on every row, so the strip drops a column from both."""
    eos = llama_eos_bridge.tokenizer.eos_token_id
    _, tokens, _, mask = llama_eos_bridge.input_to_embed(
        PROMPTS, prepend_bos=prepend_bos, padding_side="left"
    )
    assert not (tokens[:, -1] == eos).any()
    for row, prompt in enumerate(PROMPTS):
        alone_tokens = llama_eos_bridge.to_tokens(prompt, prepend_bos=prepend_bos)[0]
        assert alone_tokens[-1].item() != eos
        assert torch.equal(tokens[row][mask[row].bool()], alone_tokens)
