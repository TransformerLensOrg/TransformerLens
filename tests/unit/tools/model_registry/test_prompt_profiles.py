"""Profile resolution: curation must beat unreliable Hub metadata (observed
mis-tags: mt0-base as text-generation, conversational on base models,
unordered Helsinki-NLP language tags), and gaps must fall through safely."""

import pytest

pytest.importorskip("transformers")

from transformer_lens.benchmarks.text_quality_profiles import (
    DEFAULT_PROFILE,
    HFSignals,
    ProfileSpec,
    extract_languages,
    profile_from_hf_signals,
    resolve_profile,
)


class TestPrecedence:
    def test_override_beats_architecture_rule(self, monkeypatch):
        """A real override-vs-arch clash: Pegasus's arch rule says
        summarization; a per-model override must still win."""
        from transformer_lens.benchmarks import text_quality_profiles as tp

        monkeypatch.setitem(tp.MODEL_PROFILE_OVERRIDES, "google/pegasus-xsum", "continuation")
        spec = resolve_profile("google/pegasus-xsum", "PegasusForConditionalGeneration")
        assert spec == ProfileSpec("continuation")

    def test_override_beats_signals(self):
        """long-t5 override must win even when signals disagree."""
        spec = resolve_profile(
            "google/long-t5-tglobal-base",
            "LongT5ForConditionalGeneration",
            signals=HFSignals(pipeline_tag="summarization"),
        )
        assert spec == ProfileSpec("task:denoise")

    def test_architecture_rule_beats_fetched_tag(self):
        """Pegasus is summarization by architecture even if the Hub tag lies."""
        spec = resolve_profile(
            "google/pegasus-xsum",
            "PegasusForConditionalGeneration",
            signals=HFSignals(pipeline_tag="text-generation"),
        )
        assert spec.kind == "task:summarization"

    def test_mt0_mistag_resolves_to_instruction(self):
        """Hub tags mt0-base text-generation; the override must correct it."""
        spec = resolve_profile(
            "bigscience/mt0-base",
            "MT5ForConditionalGeneration",
            signals=HFSignals(pipeline_tag="text-generation"),
        )
        assert spec.kind == "task:instruction"

    def test_fetched_tag_fills_gap(self):
        """BART has no arch rule (checkpoint-dependent); the Hub tag decides."""
        spec = resolve_profile(
            "facebook/bart-large-cnn",
            "BartForConditionalGeneration",
            signals=HFSignals(pipeline_tag="summarization", languages=("en",)),
        )
        assert spec.kind == "task:summarization"

    def test_null_pipeline_tag_falls_through_to_arch_rule(self):
        """m2m100 has pipeline_tag=None on the Hub; the arch rule must hold."""
        spec = resolve_profile(
            "facebook/m2m100_418M",
            "M2M100ForConditionalGeneration",
            signals=HFSignals(pipeline_tag=None),
        )
        assert spec.kind == "task:translation"

    def test_stored_registry_value_used_when_no_signals(self):
        spec = resolve_profile("some/model", "GPT2LMHeadModel", "continuation@fr")
        assert spec == ProfileSpec("continuation", "fr")

    def test_unknown_seq2seq_defaults_to_denoise_not_continuation(self):
        """An unlabelled seq2seq cannot continue text; denoising is its only prompt."""
        spec = resolve_profile("someone/random-t5", "T5GemmaForConditionalGeneration")
        assert spec.kind == "task:denoise"

    def test_unknown_causal_lm_defaults_to_continuation(self):
        assert resolve_profile("someone/random-lm", "LlamaForCausalLM") == DEFAULT_PROFILE


class TestHubSignals:
    def test_conversational_tag_alone_is_not_chat(self):
        """HF adds `conversational` to ANY repo shipping a chat template, base
        models included (observed on Qwen/Qwen2.5-0.5B)."""
        spec = profile_from_hf_signals(
            "Qwen/Qwen2.5-0.5B",
            "Qwen2ForCausalLM",
            HFSignals(pipeline_tag="text-generation", tags=("conversational",)),
        )
        assert spec is not None and spec.kind == "continuation"

    def test_code_tag_maps_to_code_continuation(self):
        spec = profile_from_hf_signals(
            "bigcode/some-model", "GPTBigCodeForCausalLM", HFSignals(tags=("code",))
        )
        assert spec == ProfileSpec("continuation", "code")

    def test_marian_direction_from_model_id_not_tag_order(self):
        """Helsinki-NLP language tags are unordered; only opus-mt-{src}-{tgt}
        carries the direction."""
        spec = resolve_profile(
            "Helsinki-NLP/opus-mt-nl-en",
            "MarianMTModel",
            signals=HFSignals(languages=("en", "nl")),  # tag order is wrong on purpose
        )
        assert (spec.src, spec.lang) == ("nl", "en")

    def test_translation_tag_without_direction_returns_none(self):
        """Tag lists are unordered: guessing a pair risks a reversed or
        identity direction, so signals alone must abstain (resolution then
        falls through to overrides/arch rules — t5-small still lands on
        en-de via its override)."""
        spec = profile_from_hf_signals(
            "google-t5/t5-small",
            "T5ForConditionalGeneration",
            HFSignals(pipeline_tag="translation"),
        )
        assert spec is None
        resolved = resolve_profile(
            "google-t5/t5-small",
            "T5ForConditionalGeneration",
            signals=HFSignals(pipeline_tag="translation"),
        )
        assert (resolved.src, resolved.lang) == ("en", "de")

    def test_translation_tag_with_en_and_target_infers_pair(self):
        spec = profile_from_hf_signals(
            "someone/en-fr-translator",
            "BartForConditionalGeneration",
            HFSignals(pipeline_tag="translation", languages=("en", "fr")),
        )
        assert spec is not None and (spec.src, spec.lang) == ("en", "fr")


class TestLanguageExtraction:
    def test_handles_str_and_list(self):
        assert extract_languages("fr", []) == ("fr",)
        assert extract_languages(["de", "en"], []) == ("de", "en")

    def test_merges_iso_tags_and_drops_noise(self):
        langs = extract_languages(
            None, ["pytorch", "transformers", "nl", "marian", "safetensors", "en"]
        )
        assert langs == ("nl", "en")

    def test_caps_at_eight(self):
        many = ["fr", "es", "de", "it", "nl", "pt", "ru", "ja", "ar", "hi"]
        assert len(extract_languages(many, [])) == 8


class TestProfileSpecGrammar:
    def test_round_trip(self):
        for text in ("continuation", "continuation@code", "chat@fr", "task:translation@en-de"):
            assert str(ProfileSpec.parse(text)) == text

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            ProfileSpec.parse("poetry@en")

    def test_rejects_translation_without_pair(self):
        with pytest.raises(ValueError):
            ProfileSpec.parse("task:translation@de")


class TestChatIdHeuristic:
    """Instruct/chat/-it ids resolve to the chat profile — nothing else can
    (the conversational tag covers base models; no arch distinguishes tuned
    from base). Runtime downgrades template-less models back to continuation."""

    def test_instruct_id_resolves_chat(self):
        spec = resolve_profile("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2ForCausalLM")
        assert spec.kind == "chat"

    def test_it_suffix_resolves_chat(self):
        spec = resolve_profile("google/gemma-2-2b-it", "Gemma2ForCausalLM")
        assert spec.kind == "chat"

    def test_base_id_stays_continuation(self):
        assert resolve_profile("Qwen/Qwen2.5-0.5B", "Qwen2ForCausalLM").kind == "continuation"
        assert resolve_profile("google/gemma-2-2b", "Gemma2ForCausalLM").kind == "continuation"

    def test_override_still_beats_chat_heuristic(self, monkeypatch):
        from transformer_lens.benchmarks import text_quality_profiles as tp

        monkeypatch.setitem(tp.MODEL_PROFILE_OVERRIDES, "someone/model-instruct", "continuation@fr")
        spec = resolve_profile("someone/model-instruct", "LlamaForCausalLM")
        assert spec == ProfileSpec("continuation", "fr")

    def test_arch_rule_beats_chat_heuristic(self):
        """A Blenderbot-style arch keeps its rule even with a chatty id."""
        spec = resolve_profile("someone/blenderbot-chat", "BlenderbotForConditionalGeneration")
        assert spec.kind == "chat"  # via arch rule, not id — same outcome
        spec2 = resolve_profile("someone/opus-mt-nl-en-chat", "MarianMTModel")
        assert spec2.kind == "task:translation"


def test_chat_heuristic_keeps_stored_language():
    """The id heuristic fixes the kind; a stored chat@fr must survive
    resolve->writeback (it was flattened to chat@en and clobbered)."""
    from transformer_lens.benchmarks.text_quality_profiles import resolve_profile

    spec = resolve_profile("org/model-7b-instruct", "LlamaForCausalLM", registry_profile="chat@fr")
    assert str(spec) == "chat@fr"
    # A stored non-chat profile does not hijack the heuristic.
    spec = resolve_profile(
        "org/model-7b-instruct", "LlamaForCausalLM", registry_profile="continuation@fr"
    )
    assert str(spec) == "chat"


def test_non_english_denoise_is_a_coverage_gap():
    """IndicBART's stale score was measured under a broken MBart profile;
    until Indic denoise prompts exist, a non-en denoise profile must SKIP
    (coverage gap), never score against English sentences."""
    from transformer_lens.benchmarks.text_quality_profiles import (
        ProfileSpec,
        prompts_for,
        resolve_profile,
    )

    assert str(resolve_profile("ai4bharat/IndicBART", "MBartForConditionalGeneration")) == (
        "task:denoise@hi"
    )
    assert prompts_for(ProfileSpec("task:denoise", lang="hi")) is None
    assert prompts_for(ProfileSpec("task:denoise", lang="en")) is not None
