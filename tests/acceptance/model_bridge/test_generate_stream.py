"""Tests for TransformerBridge.generate_stream()."""

import torch


def test_stream_matches_generate(gpt2_bridge):
    """Concatenated stream output should match generate() for the same prompt."""
    prompt = "The future of AI"
    # Get generate() output as string
    expected_text = gpt2_bridge.generate(prompt, max_new_tokens=10, do_sample=False, verbose=False)
    assert isinstance(expected_text, str)

    # Stream as tokens so we can concatenate and compare
    chunks = list(
        gpt2_bridge.generate_stream(
            prompt,
            max_new_tokens=10,
            max_tokens_per_yield=3,
            do_sample=False,
            verbose=False,
            return_type="tokens",
        )
    )
    assert len(chunks) >= 1

    # First chunk = input + first tokens, subsequent = new tokens only.
    all_tokens = chunks[0]
    for chunk in chunks[1:]:
        all_tokens = torch.cat([all_tokens, chunk], dim=-1)

    streamed_text = gpt2_bridge.tokenizer.decode(all_tokens[0], skip_special_tokens=True)
    assert (
        expected_text == streamed_text
    ), f"Stream output mismatch:\n  generate: {expected_text!r}\n  stream: {streamed_text!r}"


def test_stream_yields_progressively(gpt2_bridge):
    """Multiple yields should occur with small max_tokens_per_yield."""
    chunks = list(
        gpt2_bridge.generate_stream(
            "Hello world",
            max_new_tokens=10,
            max_tokens_per_yield=3,
            do_sample=False,
            verbose=False,
            return_type="tokens",
        )
    )
    assert len(chunks) > 1, f"Expected multiple yields, got {len(chunks)}"


def test_stream_single_prompt(gpt2_bridge):
    """Basic single-string streaming should produce output."""
    results = list(
        gpt2_bridge.generate_stream(
            "Test",
            max_new_tokens=5,
            do_sample=False,
            verbose=False,
            return_type="tokens",
        )
    )
    assert len(results) >= 1
    assert results[0].shape[0] == 1  # batch=1
    assert results[0].shape[1] > 1  # has at least input + 1 generated token


def test_stream_stops_at_eos(gpt2_bridge):
    """Streaming should respect stop_at_eos."""
    results = list(
        gpt2_bridge.generate_stream(
            "Test",
            max_new_tokens=200,
            max_tokens_per_yield=5,
            stop_at_eos=True,
            do_sample=False,
            verbose=False,
            return_type="tokens",
        )
    )
    total_tokens = sum(r.shape[1] for r in results)
    assert total_tokens < 210


def test_stream_returns_strings(gpt2_bridge):
    """With return_type='str', yields should be strings."""
    results = list(
        gpt2_bridge.generate_stream(
            "Hello",
            max_new_tokens=5,
            do_sample=False,
            verbose=False,
            return_type="str",
        )
    )
    assert len(results) >= 1
    assert all(isinstance(r, str) for r in results)
