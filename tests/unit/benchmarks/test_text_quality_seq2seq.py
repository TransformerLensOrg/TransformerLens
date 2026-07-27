"""Encoder-decoder text-quality scoring must score the full decoder output.

Seq2seq models (Marian/T5/BART) emit a standalone output, not a continuation
of the prompt. Scoring it as a continuation subtracts the prompt length and
trips the "continuation too short (< 2 tokens)" guard for every prompt when the
output is ~ the prompt length (Marian nl->en on an English prompt), scoring 0.
The fix scores the whole generated sequence for encoder-decoder models.
"""

import pytest

pytest.importorskip("transformers")


def test_marian_text_quality_scores_full_output():
    from transformer_lens.benchmarks.text_quality import benchmark_text_quality
    from transformer_lens.model_bridge import TransformerBridge

    try:
        bridge = TransformerBridge.boot_transformers("Helsinki-NLP/opus-mt-nl-en", device="cpu")
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"marian unavailable offline: {exc}")

    assert bridge.original_model.config.is_encoder_decoder  # precondition

    result = benchmark_text_quality(
        bridge, "Natural language processing is", max_new_tokens=20, device="cpu"
    )
    # Pre-fix this returned "Scoring failed for all prompts" (score absent).
    assert result.details is not None, result.message
    assert "score" in result.details, result.message
    assert result.details["score"] > 0
