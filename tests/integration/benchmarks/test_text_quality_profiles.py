"""End-to-end Phase-4 profile scoring against real models and the real judge.

Each test guards a profile path that unit stubs cannot: the Marian test keeps
seq2seq whole-output scoring (pre-profile P4 masked a prompt "continuation"
that seq2seq output does not have and scored 0); the Florence-2 test keeps the
caption path (text-only prompts yield a bare EOS on image-conditioned models);
the judge tests pin the revision and the fluent-vs-corrupted separation the
bake-off measured.
"""

import pytest

pytest.importorskip("transformers")

from transformer_lens.benchmarks.text_quality import benchmark_text_quality


def _boot(model_id, **kwargs):
    from transformer_lens.model_bridge import TransformerBridge

    try:
        return TransformerBridge.boot_transformers(model_id, device="cpu", **kwargs)
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"{model_id} unavailable offline: {exc}")


@pytest.fixture(scope="module")
def judge():
    from transformer_lens.benchmarks.text_quality import load_judge

    try:
        return load_judge()
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"judge unavailable offline: {exc}")


def test_translation_profile_marian(judge):
    """Seq2seq output is standalone (a translation), not a continuation of the
    prompt; the translation profile must score it whole, against the pivot
    reference, in the direction parsed from the model id."""
    bridge = _boot("Helsinki-NLP/opus-mt-nl-en")
    assert bridge.original_model.config.is_encoder_decoder  # precondition

    judge_model, judge_tokenizer = judge
    result = benchmark_text_quality(
        bridge,
        "task:translation@nl-en",
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
    )
    assert result.details is not None, result.message
    assert result.details["prompt_profile"] == "task:translation@nl-en"
    # A working translator of 3 short pivot sentences must land well above the
    # broken floor (score 0 = judge's typical-corruption perplexity ratio).
    assert result.details["score"] > 50.0, result.details


def test_caption_profile_florence2(judge):
    """Florence-2 emits a bare EOS for text-only prompts; P4 must drive real
    image-conditioned captions and score them under the caption profile."""
    pytest.importorskip("PIL")
    bridge = _boot("florence-community/Florence-2-base", trust_remote_code=True)

    judge_model, judge_tokenizer = judge
    result = benchmark_text_quality(
        bridge,
        "continuation",  # deliberately wrong: the caption adjustment must win
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
    )
    assert result.details is not None, result.message
    assert result.details["prompt_profile"] == "caption"
    assert result.details["score"] > 0.0


def test_chat_profile_templates_and_scores(judge):
    """Chat models are scored through their own template (prepend_bos=False —
    the template supplies BOS); output must not be the template markers."""
    bridge = _boot("Qwen/Qwen2.5-0.5B-Instruct")

    judge_model, judge_tokenizer = judge
    result = benchmark_text_quality(
        bridge,
        "chat",
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
    )
    assert result.details is not None, result.message
    assert result.details["prompt_profile"] == "chat"
    assert "<|im_start|>" not in result.details["generated_text"]
    assert result.details["score"] > 50.0, result.details


def test_fluent_vs_shuffled_separation_end_to_end(judge):
    """The full scoring chain must separate real model output from word salad:
    any break (mask slip, ratio inversion, penalty loss) collapses the gap."""
    import random

    from transformer_lens.benchmarks.text_quality import (
        _compute_repetition_penalty,
        _judge_perplexity,
        _ratio_to_score,
    )
    from transformer_lens.benchmarks.text_quality_profiles import CONTINUATION_PROMPTS

    judge_model, judge_tokenizer = judge
    entry = CONTINUATION_PROMPTS["en"][0]
    # A DISTINCT on-topic fluent paraphrase, not the reference itself: ref/ref
    # is identically 1 -> 100 and would pass with the judge deleted.
    # Measured: ppl 8.3 vs ref 5.1 -> score 83.2 (judge conditioned on the
    # relativity prompt correctly rejects off-topic fluent text).
    fluent = (
        " measurements of time and distance depend on the observer's motion,"
        " so no single frame of reference is absolute."
    )
    words = entry.reference.split()
    random.Random(42).shuffle(words)
    shuffled = " ".join(words)

    ref_ppl, err = _judge_perplexity(entry.reference, entry.prompt, judge_tokenizer, judge_model)
    assert err is None
    fluent_ppl, err = _judge_perplexity(fluent, entry.prompt, judge_tokenizer, judge_model)
    assert err is None
    fluent_score = _ratio_to_score(fluent_ppl / ref_ppl) * _compute_repetition_penalty(fluent)
    shuf_ppl, err = _judge_perplexity(shuffled, entry.prompt, judge_tokenizer, judge_model)
    assert err is None
    shuffled_score = _ratio_to_score(shuf_ppl / ref_ppl) * _compute_repetition_penalty(shuffled)

    assert fluent_score >= 60.0, fluent_score
    assert shuffled_score < 50.0, (shuf_ppl, ref_ppl)
    assert fluent_score - shuffled_score >= 30.0


def test_reference_perplexities_match_pinned_values(judge):
    """Judge-revision/reference drift guard: the judge's perplexity on a few
    fixed reference strings must match values measured at bake-off time
    (2026-08-20, Qwen2.5-0.5B@060db649, fp32 CPU). A judge unpin or a silent
    reference edit moves these."""
    judge_model, judge_tokenizer = judge
    from transformer_lens.benchmarks.text_quality import _judge_perplexity
    from transformer_lens.benchmarks.text_quality_profiles import PIVOT_SENTENCES

    pinned = {
        ("en", 0): 30.79,
        ("fr", 0): 81.85,
        ("zh", 0): 53.15,
    }
    for (lang, idx), expected in pinned.items():
        ppl, err = _judge_perplexity(PIVOT_SENTENCES[lang][idx], "", judge_tokenizer, judge_model)
        assert err is None
        assert ppl == pytest.approx(expected, rel=0.15), (lang, idx, ppl)


def test_continuation_references_share_a_scale(judge):
    """Per-language reference PPLs must sit within 3.5x of the language
    median: an outlier reference makes its prompt's bar proportionally looser
    (the old en[3] measured 31.3 vs median 8.9 and handed gpt2 a clamp-100 on
    output worse than its 62-scoring sibling prompt)."""
    judge_model, judge_tokenizer = judge
    from transformer_lens.benchmarks.text_quality import _judge_perplexity
    from transformer_lens.benchmarks.text_quality_profiles import CONTINUATION_PROMPTS

    for lang, prompts in CONTINUATION_PROMPTS.items():
        ppls = []
        for pp in prompts:
            ppl, err = _judge_perplexity(pp.reference, pp.prompt, judge_tokenizer, judge_model)
            assert err is None, (lang, err)
            ppls.append(ppl)
        median = sorted(ppls)[len(ppls) // 2]
        for i, ppl in enumerate(ppls):
            assert ppl <= 3.5 * median, (lang, i, round(ppl, 1), round(median, 1))
