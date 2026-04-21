"""Tests for TransformerBridge.generate_stream()."""

import torch


def test_stream_matches_generate(gpt2_bridge):
    """Concatenated stream output should match generate() for the same prompt."""
    prompt = "The future of AI"
    expected = gpt2_bridge.generate(
        prompt, max_new_tokens=10, do_sample=False, verbose=False
    )

    # Collect all streamed chunks
    chunks = list(
        gpt2_bridge.generate_stream(
            prompt,
            max_new_tokens=10,
            max_tokens_per_yield=3,
            do_sample=False,
            verbose=False,
        )
    )
    assert len(chunks) >= 1

    # Reconstruct: first chunk has input+tokens, subsequent have only new tokens
    full_tokens = torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]
    # The first chunk includes input tokens, so just take the last chunk's end
    # Actually, each chunk is independent — first has input+new, rest have only new
    # So concatenating all gives input + all new tokens (with input repeated).
    # Instead, compare decoded strings.
    expected_text = gpt2_bridge.to_string(expected[0] if isinstance(expected, torch.Tensor) else gpt2_bridge.to_tokens(expected)[0])

    # Decode last chunk which should have the most recent window of tokens
    # Better: decode all chunks and concatenate
    stream_texts = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            stream_texts.append(gpt2_bridge.to_string(chunk[0]))
        else:
            stream_texts.append(gpt2_bridge.to_string(chunk[0]))

    # The first chunk has input+initial tokens, subsequent have only new tokens.
    # The simplest comparison: the final full output should match.
    # Reconstruct by taking the first chunk and appending decoded new tokens.
    # Actually easier: just compare using the full token sequence.
    # First chunk = input + first N tokens, subsequent = next tokens only.
    all_tokens = chunks[0]
    for chunk in chunks[1:]:
        all_tokens = torch.cat([all_tokens, chunk], dim=-1)

    streamed_text = gpt2_bridge.to_string(all_tokens[0])
    assert expected_text == streamed_text, (
        f"Stream output mismatch:\n  generate: {expected_text!r}\n  stream: {streamed_text!r}"
    )


def test_stream_yields_progressively(gpt2_bridge):
    """Multiple yields should occur with small max_tokens_per_yield."""
    chunks = list(
        gpt2_bridge.generate_stream(
            "Hello world",
            max_new_tokens=10,
            max_tokens_per_yield=3,
            do_sample=False,
            verbose=False,
        )
    )
    assert len(chunks) > 1, f"Expected multiple yields, got {len(chunks)}"


def test_stream_single_prompt(gpt2_bridge):
    """Basic single-string streaming should produce output."""
    results = list(
        gpt2_bridge.generate_stream(
            "Test", max_new_tokens=5, do_sample=False, verbose=False
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
        )
    )
    # Count total generated tokens (first chunk has input, rest are new)
    total_tokens = sum(r.shape[1] for r in results)
    # Should have stopped well before 200 new tokens for a short prompt
    assert total_tokens < 210
