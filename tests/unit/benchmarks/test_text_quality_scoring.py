"""Reference-ratio Phase-4 scoring: the score must be a judge-handicap-free
comparison against a reference completion, with penalties for loops and
truncation, generated via token-level slicing (string-prefix slicing breaks
under chat templates because generate() strips special tokens on decode)."""

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from transformer_lens.benchmarks.text_quality import (
    JUDGE_R_FAIL,
    _length_penalty,
    _ratio_to_score,
    benchmark_text_quality,
)
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity


class FakeVocabTokenizer:
    """Whitespace tokenizer with a growable vocab and template-marker specials."""

    chat_template = None

    def __init__(self):
        self._vocab: list[str] = []
        self._special: set[int] = set()
        self.mask_token = None

    def _id(self, word: str, special: bool = False) -> int:
        if word not in self._vocab:
            self._vocab.append(word)
        idx = self._vocab.index(word)
        if special:
            self._special.add(idx)
        return idx

    def encode_words(self, text: str, special: bool = False) -> list[int]:
        return [self._id(w, special) for w in text.split()]

    def __call__(self, text, return_tensors=None):
        # Native recipe used for encoder-decoder inputs.
        ids = self.encode_words(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        words = [self._vocab[i] for i in ids if not (skip_special_tokens and i in self._special)]
        return " ".join(words)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return f"<|im_start|> {messages[0]['content']} <|im_end|>"


class FakeBridge:
    """Decoder-only bridge stub: generate() echoes prompt ids + canned continuation."""

    def __init__(
        self, continuation="the quick brown fox jumps over the lazy dog today", chat_template=None
    ):
        self.tokenizer = FakeVocabTokenizer()
        self.tokenizer.chat_template = chat_template
        self.adapter = SimpleNamespace(supports_generation=True, native_sampler=None)
        self.original_model = SimpleNamespace(
            config=SimpleNamespace(is_encoder_decoder=False, architectures=["FakeLM"])
        )
        self.cfg = SimpleNamespace(device="cpu", is_multimodal=False, model_name="fake")
        self._continuation = continuation
        self.generate_calls: list[dict] = []
        self.to_tokens_calls: list = []

    def to_tokens(self, text, prepend_bos=None, **kwargs):
        self.to_tokens_calls.append(prepend_bos)
        special = text.startswith("<|im_start|>")
        if special:
            # Template markers become special ids that decode drops.
            ids = []
            for word in text.split():
                is_marker = word.startswith("<|")
                ids.append(self.tokenizer._id(word, special=is_marker))
            return torch.tensor([ids])
        return torch.tensor([self.tokenizer.encode_words(text)])

    def generate(self, input, **kwargs):
        self.generate_calls.append(kwargs)
        cont_ids = self.tokenizer.encode_words(self._continuation)
        return torch.cat([input, torch.tensor([cont_ids])], dim=1)


class FakeJudgeTokenizer:
    """Word-level judge tokenizer sharing nothing with the bridge's."""

    def __init__(self):
        self._vocab: list[str] = []

    def __call__(self, text, return_tensors=None):
        ids = []
        for w in text.split():
            if w not in self._vocab:
                self._vocab.append(w)
            ids.append(self._vocab.index(w))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


class FakeJudge:
    """Judge whose loss is a configurable function of the scored token ids.

    Records (masked_context_words, scored_words) per call so tests can assert
    what the judge was conditioned on."""

    def __init__(self, tokenizer: FakeJudgeTokenizer, loss_fn):
        self._tokenizer = tokenizer
        self._loss_fn = loss_fn
        self.calls: list = []

    def __call__(self, input_ids, labels=None):
        pairs = list(zip(input_ids[0].tolist(), labels[0].tolist()))
        scored = [int(t) for t, l in pairs if l != -100]
        masked = [int(t) for t, l in pairs if l == -100]
        words = " ".join(self._tokenizer._vocab[i] for i in scored)
        self.calls.append((" ".join(self._tokenizer._vocab[i] for i in masked), words))
        return SimpleNamespace(loss=torch.tensor(self._loss_fn(words)))


def _run(bridge, profile="continuation", loss_fn=lambda text: 1.0, **kwargs):
    judge_tokenizer = FakeJudgeTokenizer()
    judge = FakeJudge(judge_tokenizer, loss_fn)
    result = benchmark_text_quality(
        bridge, profile, judge_model=judge, judge_tokenizer=judge_tokenizer, **kwargs
    )
    bridge.judge_calls = judge.calls
    return result


class TestRatioMath:
    def test_ratio_one_scores_100(self):
        assert _ratio_to_score(1.0) == 100.0

    def test_ratio_r_fail_scores_zero(self):
        assert _ratio_to_score(JUDGE_R_FAIL) == pytest.approx(0.0, abs=1e-9)

    def test_ratio_sqrt_r_fail_scores_50(self):
        """Registry's phase-4 floor of 50 = geometric midpoint of good and broken."""
        assert _ratio_to_score(math.sqrt(JUDGE_R_FAIL)) == pytest.approx(50.0, abs=1e-9)

    def test_ratio_below_one_clamps_to_100(self):
        """Beating the reference is not extra credit (loops get there trivially)."""
        assert _ratio_to_score(0.2) == 100.0

    def test_ratio_is_handicap_invariant(self):
        """A judge that is k-times worse at some language multiplies BOTH sides'
        perplexity, so the score must not move (the old absolute-perplexity
        mapping drops by 10 ln k)."""
        gen_ppl, ref_ppl, k = 40.0, 25.0, 10.0
        assert _ratio_to_score(gen_ppl / ref_ppl) == pytest.approx(
            _ratio_to_score((k * gen_ppl) / (k * ref_ppl))
        )


class TestLengthPenalty:
    def test_neutral_at_reference_length(self):
        assert _length_penalty(40, 40) == 1.0

    def test_neutral_band_half_to_triple(self):
        """Neutral in [0.5x, 3x] of reference: terse-but-complete answers are
        not punished, and the old 25% floor (which never fired in four
        sweeps — a contentless chat stub scored 93.6) is gone."""
        assert _length_penalty(20, 40) == 1.0
        assert _length_penalty(120, 40) == 1.0

    def test_penalizes_below_half(self):
        assert _length_penalty(10, 40) == pytest.approx(0.5)
        assert _length_penalty(13, 41) == pytest.approx(13 / 20.5)

    def test_penalizes_overlength(self):
        """Rambling output the repetition penalty misses: 6x the reference
        pays half."""
        assert _length_penalty(240, 40) == pytest.approx(0.5)

    def test_zero_reference_is_neutral(self):
        assert _length_penalty(3, 0) == 1.0


class TestBenchmarkPipeline:
    def test_fluent_output_scores_high(self):
        # Continuation long enough to sit in the length-penalty neutral band
        # for every reference; ratio 1 then clamps every prompt to 100.
        fluent = "the quick brown fox jumps over the lazy dog today while the sun sets slowly behind the old hills"
        result = _run(FakeBridge(continuation=fluent), loss_fn=lambda t: 1.0)
        assert result.details is not None
        assert result.details["score"] == 100.0

    def test_looping_output_scores_low_despite_low_ratio(self):
        """A degenerate loop has LOW judge perplexity; only the repetition
        penalty catches it under ratio scoring."""
        loop = "the cat sat the cat sat the cat sat the cat sat"
        result = _run(FakeBridge(continuation=loop), loss_fn=lambda t: 0.1)
        assert result.details is not None
        assert result.details["score"] < 50.0

    def test_one_token_output_scores_zero(self):
        """Florence-style bare-EOS output: one token is not scoreable text."""
        result = _run(FakeBridge(continuation="x"), loss_fn=lambda t: 1.0)
        assert result.details is not None
        assert result.details["score"] == 0.0

    def test_generated_segment_sliced_by_token_count(self):
        """Chat-template prompts are not string prefixes of decoded output
        (specials are stripped); prompt words must still be excluded from the
        judged text."""
        seen: list[str] = []

        def record(text):
            seen.append(text)
            return 1.0

        bridge = FakeBridge(chat_template="{{messages}}")
        result = _run(bridge, profile="chat", loss_fn=record)
        assert result.details is not None
        gen_texts = seen[0::2]  # generated, reference alternate
        assert all("<|im_start|>" not in t for t in seen)
        for text in gen_texts:
            assert text == bridge._continuation

    def test_context_mask_does_not_swallow_first_generated_token(self):
        """Tokenizing prompt+text as one string lets the tokenizer merge
        across the seam, so the context mask swallows the first generated
        token; the pieces must be tokenized separately."""
        seen: list[str] = []

        def record(text):
            seen.append(text)
            return 1.0

        bridge = FakeBridge(continuation="zebra jumps over seven quiet green hills today")
        _run(bridge, loss_fn=record)
        gen_texts = [t for t in seen if "zebra" in t or "jumps" in t]
        assert gen_texts, seen
        assert all(t.startswith("zebra") for t in gen_texts), gen_texts

    def test_empty_generation_is_danger(self):
        result = _run(FakeBridge(continuation=""), loss_fn=lambda t: 1.0)
        assert result.severity == BenchmarkSeverity.DANGER
        assert result.passed is False

    def test_uncovered_profile_skips_with_coverage_instruction(self):
        """A coverage gap must tell the operator to file an issue, never score."""
        result = _run(FakeBridge(), profile="task:translation@en-sw")
        assert result.severity == BenchmarkSeverity.SKIPPED
        assert "task:translation@en-sw" in result.message
        assert "file a TransformerLens issue" in result.message

    def test_chat_without_template_downgrades_to_continuation(self):
        bridge = FakeBridge(chat_template=None)
        result = _run(bridge, profile="chat")
        assert result.details is not None
        assert result.details["prompt_profile"] == "continuation"
        assert "no chat template" in result.details["profile_adjustment"]

    def test_per_prompt_seed_independent_of_order(self):
        """Each prompt's sample stream restarts at the benchmark seed, so a
        prompt's output cannot depend on how much RNG earlier prompts consumed."""

        class RngBridge(FakeBridge):
            def generate(self, input, **kwargs):
                # Fixed RNG consumption, then a draw: identical across prompts
                # only if every prompt's stream restarts at the benchmark seed.
                torch.rand(5)
                draw = int(torch.randint(0, 10_000, (1,)).item())
                cont = self.tokenizer.encode_words(f"gen{draw} token one two")
                return torch.cat([input, torch.tensor([cont])], dim=1)

        seen: list[str] = []

        def record(text):
            seen.append(text)
            return 1.0

        _run(RngBridge(), loss_fn=record)
        gen_texts = [t for t in seen if t.startswith("gen")]
        assert len(gen_texts) >= 2
        assert len(set(gen_texts)) == 1, gen_texts


class TestProfileDataIntegrity:
    def test_every_table_entry_is_scoreable(self):
        from transformer_lens.benchmarks import text_quality_profiles as p
        from transformer_lens.benchmarks.text_quality import _wrong_language

        tables = [
            p.CONTINUATION_PROMPTS,
            p.CHAT_PROMPTS,
            p.SUMMARIZATION_PROMPTS,
            p.INSTRUCTION_PROMPTS,
            p.DENOISE_PROMPTS,
        ]
        for table in tables:
            for lang, entries in table.items():
                for entry in entries:
                    assert entry.prompt.strip()
                    assert entry.reference.strip()
                    # Content, not just shape: a reference that self-flags as
                    # wrong-language hard-zeros its own sample (a fr reference
                    # did; the shape checks missed it).
                    if table in (p.CONTINUATION_PROMPTS, p.CHAT_PROMPTS):
                        assert not _wrong_language(entry.reference, lang), (
                            lang,
                            entry.reference[:50],
                        )

    def test_pivot_sentences_index_aligned(self):
        from transformer_lens.benchmarks.text_quality_profiles import PIVOT_SENTENCES

        lengths = {lang: len(rows) for lang, rows in PIVOT_SENTENCES.items()}
        assert set(lengths.values()) == {3}, lengths

    def test_all_kinds_have_knobs(self):
        from transformer_lens.benchmarks import text_quality_profiles as p

        assert set(p.MAX_NEW_TOKENS_BY_KIND) == set(p.PROFILE_KINDS)


class TestTranslationWiring:
    def test_forced_bos_threaded_for_multilingual_translators(self):
        """M2M100/MBart select target language via the first decoder token;
        dropping the forced_bos_token_id kwarg silently translates into an
        arbitrary language (and the judge would score that fluent text well)."""

        class M2M100Bridge(FakeBridge):
            def __init__(self):
                super().__init__(continuation="ich muss jetzt wirklich schlafen gehen heute abend")
                self.original_model.config.is_encoder_decoder = True
                self.tokenizer.get_lang_id = lambda lang: {"de": 777, "en": 700}.get(lang, 0)
                self.tokenizer.src_lang = "en"

            def generate(self, input, **kwargs):
                # Real enc-dec output shape: [decoder_start] + generated, never
                # the echoed source prompt.
                self.generate_calls.append(kwargs)
                start = torch.tensor([[0]])
                cont_ids = self.tokenizer.encode_words(self._continuation)
                return torch.cat([start, torch.tensor([cont_ids])], dim=1)

        bridge = M2M100Bridge()
        result = _run(bridge, profile="task:translation@en-de")
        assert result.details is not None, result.message
        assert all(call.get("forced_bos_token_id") == 777 for call in bridge.generate_calls)
        assert bridge.tokenizer.src_lang == "en"


class TestReviewGuards:
    """Guards for defects found in adversarial review of the rework."""

    def test_cjk_repetition_penalty_uses_characters(self):
        """Whitespace-split n-grams see zh/ja text as one word and never fire —
        exactly where the judge rewards loops with low perplexity."""
        from transformer_lens.benchmarks.text_quality import _compute_repetition_penalty

        assert _compute_repetition_penalty("的" * 20) < 0.2
        assert _compute_repetition_penalty("のの" * 10) < 0.3
        fluent_zh = "长城是中国古代伟大的防御工程，每年吸引大量游客。"
        assert _compute_repetition_penalty(fluent_zh) > 0.7

    def test_wrong_language_output_scores_zero(self):
        """Ratio scoring measures fluency, not language: fluent English beats a
        short German reference and clamps to 100 unless language is checked."""
        bridge = FakeBridge(continuation="the quick brown fox jumps over the lazy dog and the cat")
        seen = []
        judge_tokenizer = FakeJudgeTokenizer()
        judge = FakeJudge(judge_tokenizer, lambda t: (seen.append(t) or 1.0))
        from transformer_lens.benchmarks.text_quality import benchmark_text_quality

        result = benchmark_text_quality(
            bridge, "continuation@de", judge_model=judge, judge_tokenizer=judge_tokenizer
        )
        assert result.details is not None
        assert result.details["score"] == 0.0
        assert "not in 'de'" in result.details["per_prompt"]

    def test_wrong_language_check_passes_correct_language(self):
        bridge = FakeBridge(
            continuation="der alte Zug ist nicht mit einem neuen Wagen gefahren und die Leute"
        )
        judge_tokenizer = FakeJudgeTokenizer()
        judge = FakeJudge(judge_tokenizer, lambda t: 1.0)
        from transformer_lens.benchmarks.text_quality import benchmark_text_quality

        result = benchmark_text_quality(
            bridge, "continuation@de", judge_model=judge, judge_tokenizer=judge_tokenizer
        )
        assert result.details is not None
        assert result.details["score"] > 0.0

    def test_empty_output_scored_zero_not_dropped(self):
        """An empty generation must drag the average down, not vanish from it."""

        class HalfEmptyBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self._call = 0

            def generate(self, input, **kwargs):
                self._call += 1
                if self._call % 2 == 0:
                    return input  # no new tokens -> empty continuation
                cont = self.tokenizer.encode_words(self._continuation)
                return torch.cat([input, torch.tensor([cont])], dim=1)

        result = _run(HalfEmptyBridge(), loss_fn=lambda t: 1.0)
        assert result.details is not None
        assert result.details["num_prompts"] == 4
        assert 40.0 <= result.details["score"] <= 60.0, result.details

    def test_denoise_t5_fill_spliced_into_sentence(self):
        """Bare span fragments have judge PPL in the thousands, making the
        ratio vacuous; the fill must be judged inside the restored sentence."""
        seen: list[str] = []

        class DenoiseBridge(FakeBridge):
            def __init__(self):
                super().__init__(continuation="played happily")
                self.original_model.config.is_encoder_decoder = True

            def generate(self, input, **kwargs):
                self.generate_calls.append(kwargs)
                start = torch.tensor([[0]])
                cont = self.tokenizer.encode_words(self._continuation)
                return torch.cat([start, torch.tensor([cont])], dim=1)

        bridge = DenoiseBridge()
        judge_tokenizer = FakeJudgeTokenizer()
        judge = FakeJudge(judge_tokenizer, lambda t: (seen.append(t) or 1.0))
        from transformer_lens.benchmarks.text_quality import benchmark_text_quality

        result = benchmark_text_quality(
            bridge, "task:denoise", judge_model=judge, judge_tokenizer=judge_tokenizer
        )
        assert result.details is not None
        # The bare fill must never reach the judge; every judged text is a
        # full restored sentence.
        assert seen and all(len(t.split()) >= 8 for t in seen), seen
        assert "The children played happily in the park until the sun went down." in seen

    def test_chat_prepend_bos_false_threaded_to_tokenizer(self):
        """The chat template supplies its own BOS; to_tokens must receive
        prepend_bos=False or the prompt gets a double BOS."""
        bridge = FakeBridge(chat_template="{{messages}}")
        _run(bridge, profile="chat")
        assert bridge.to_tokens_calls and all(v is False for v in bridge.to_tokens_calls)

    def test_translation_scored_jointly(self):
        """Short pivot sentences have unstable judge PPL; the three samples
        must be concatenated into one judged pair."""

        class MarianBridge(FakeBridge):
            def __init__(self):
                super().__init__(continuation="ik moet nu echt gaan slapen vandaag")
                self.original_model.config.is_encoder_decoder = True

            def generate(self, input, **kwargs):
                self.generate_calls.append(kwargs)
                start = torch.tensor([[0]])
                cont = self.tokenizer.encode_words(self._continuation)
                return torch.cat([start, torch.tensor([cont])], dim=1)

        bridge = MarianBridge()
        judge_tokenizer = FakeJudgeTokenizer()
        judge = FakeJudge(judge_tokenizer, lambda t: 1.0)
        from transformer_lens.benchmarks.text_quality import benchmark_text_quality

        result = benchmark_text_quality(
            bridge, "task:translation@en-nl", judge_model=judge, judge_tokenizer=judge_tokenizer
        )
        assert result.details is not None
        assert result.details["num_prompts"] == 1
        assert len(bridge.generate_calls) == 3  # generation stays per-sentence

    def test_task_kinds_generate_greedily(self):
        """Users run translators deterministically; sampling variance also
        makes a single-sample score unstable. Task kinds must pass
        temperature 0.0 while open-ended kinds keep sampling."""
        cont_bridge = FakeBridge()
        _run(cont_bridge, profile="continuation")
        assert all(c["temperature"] == 0.7 for c in cont_bridge.generate_calls)

        class MarianBridge(FakeBridge):
            def __init__(self):
                super().__init__(continuation="ik moet nu echt gaan slapen vandaag")
                self.original_model.config.is_encoder_decoder = True

            def generate(self, input, **kwargs):
                self.generate_calls.append(kwargs)
                start = torch.tensor([[0]])
                cont = self.tokenizer.encode_words(self._continuation)
                return torch.cat([start, torch.tensor([cont])], dim=1)

        task_bridge = MarianBridge()
        _run(task_bridge, profile="task:translation@en-nl")
        assert all(c["temperature"] == 0.0 for c in task_bridge.generate_calls)

    def test_encdec_prompt_uses_native_tokenizer_recipe(self):
        """Encoder input must follow the tokenizer's own recipe (lang token +
        trailing </s>); to_tokens' BOS policy injects <s> and drops </s>,
        which sent m2m100 into a quote-mark loop."""
        BOS, EOS = 901, 902

        class RecipeTokenizer(FakeVocabTokenizer):
            def __call__(self, text, return_tensors=None):
                ids = self.encode_words(text) + [EOS]
                if return_tensors == "pt":
                    return {"input_ids": torch.tensor([ids])}
                return {"input_ids": ids}

        class RecipeBridge(FakeBridge):
            def __init__(self):
                super().__init__(continuation="ik moet gaan slapen vandaag echt nu")
                self.tokenizer.__class__ = RecipeTokenizer
                self.original_model.config.is_encoder_decoder = True
                self.seen_inputs: list = []

            def to_tokens(self, text, prepend_bos=None, **kwargs):
                self.to_tokens_calls.append(prepend_bos)
                return torch.tensor([[BOS] + self.tokenizer.encode_words(text)])

            def generate(self, input, **kwargs):
                self.generate_calls.append(kwargs)
                self.seen_inputs.append(input[0].tolist())
                start = torch.tensor([[0]])
                cont = self.tokenizer.encode_words(self._continuation)
                return torch.cat([start, torch.tensor([cont])], dim=1)

        bridge = RecipeBridge()
        _run(bridge, profile="task:translation@en-nl")
        assert bridge.seen_inputs, "no generation happened"
        for ids in bridge.seen_inputs:
            assert ids[-1] == EOS, ids
            assert BOS not in ids, ids

    def test_dead_encdec_denoise_scores_zero(self):
        """An empty span fill must not be spliced into the prompt sentence:
        the splice hands a dead enc-dec model the near-reference sentence and
        a free 100 (decoder-only dead models already scored 0)."""

        class DeadT5Bridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self.original_model.config.is_encoder_decoder = True
                self.tokenizer.mask_token = None

            def generate(self, input, **kwargs):
                self.generate_calls.append(kwargs)
                return torch.tensor([[0]])

        result = _run(DeadT5Bridge(), profile="task:denoise")
        assert result.details["score"] == 0.0
        assert result.severity == BenchmarkSeverity.DANGER


class TestRound2ReviewGuards:
    """Guards for the second review round's confirmed findings."""

    def test_curated_strings_never_self_flag(self):
        """The wrong-language detector must accept every curated string in its
        own language — a reference that self-flags hard-zeros its sample (a
        French reference did, via 'de/et' hitting other languages' sets)."""
        from transformer_lens.benchmarks.text_quality import _wrong_language
        from transformer_lens.benchmarks.text_quality_profiles import (
            CHAT_PROMPTS,
            CONTINUATION_PROMPTS,
            PIVOT_SENTENCES,
        )

        offenders = []
        for table in (CONTINUATION_PROMPTS, CHAT_PROMPTS):
            for lang, prompts in table.items():
                for pp in prompts:
                    for text in (pp.prompt, pp.reference):
                        if _wrong_language(text, lang):
                            offenders.append((lang, text[:50]))
        for lang, sents in PIVOT_SENTENCES.items():
            for s in sents:
                if _wrong_language(s, lang):
                    offenders.append((lang, s[:50]))
        assert offenders == []

    def test_cjk_loop_penalized_despite_space(self):
        """A single space in a degenerate CJK loop restored the inert word
        path (penalty 1.0 vs 0.053); char mode must key off CJK content."""
        from transformer_lens.benchmarks.text_quality import _compute_repetition_penalty

        assert _compute_repetition_penalty("的的的的的的的的的 的的的的的的的的的的") <= 0.3
        assert _compute_repetition_penalty("我该去睡觉了，因为明天有一个很重要的会议要参加。") > 0.5

    def test_registry_floor_equals_pass_line(self):
        """[floor, pass) previously got passed=False with a clean note; both
        numbers must come from the same constant."""
        from transformer_lens.benchmarks.text_quality_profiles import p4_pass_threshold
        from transformer_lens.tools.model_registry.verify_models import (
            _MIN_PHASE_SCORES,
        )

        assert _MIN_PHASE_SCORES[4] == p4_pass_threshold()

    def test_judge_cannot_self_score(self):
        """Ratio scoring against the judge's own perplexity is self-grading."""
        from transformer_lens.benchmarks.text_quality import JUDGE_MODEL_ID

        result = _run(FakeBridge(), profile="continuation", model_name=JUDGE_MODEL_ID)
        assert result.severity == BenchmarkSeverity.SKIPPED
        assert result.message.startswith("P4 skipped:")

    def test_chat_judged_with_prompt_context(self):
        """A fluent off-topic stub scores ~98 when chat output is judged
        standalone; conditioning on the user prompt is the relevance signal."""
        from transformer_lens.benchmarks.text_quality_profiles import (
            JUDGE_CONTEXT_KINDS,
        )

        assert "chat" in JUDGE_CONTEXT_KINDS
        assert "task:instruction" in JUDGE_CONTEXT_KINDS
        # Unconditioned summarization scored hallucinated summaries 100 (the
        # judge never saw the article); unconditioned denoise rated broken
        # restorations more fluent than the reference.
        assert "task:summarization" in JUDGE_CONTEXT_KINDS
        assert "task:denoise" in JUDGE_CONTEXT_KINDS
        bridge = FakeBridge()
        bridge.tokenizer.chat_template = "{{messages}}"
        _run(bridge, profile="chat")
        contexts = [c for c, _t in bridge.judge_calls]
        assert any(c for c in contexts), "judge never saw the user prompt as context"

    def test_p1_only_note_labels_skip_as_coverage_gap(self):
        """A skipped P4 is a coverage gap; the stale score must not be
        relabeled 'text quality poor'."""
        from transformer_lens.tools.model_registry.verify_models import (
            _p1_only_core_note,
        )

        skipped = BenchmarkResult(
            name="text_quality",
            severity=BenchmarkSeverity.SKIPPED,
            message="P4 skipped: no prompts for profile 'continuation@xx' — file an issue",
        )
        skipped.phase = 4
        note = _p1_only_core_note(None, [skipped])
        assert "P4 skipped" in note and "poor" not in note
        assert "poor (P4=40.0)" in _p1_only_core_note(40.0, [])
        assert "errored" in _p1_only_core_note(None, [])

    def test_arch_rule_keeps_stored_language(self):
        """The arch rule fixes the kind; a scraped @fr of the same kind must
        survive resolve->writeback or curation can never stick."""
        from transformer_lens.benchmarks.text_quality_profiles import resolve_profile

        spec = resolve_profile(
            "some/pegasus-clone",
            "PegasusForConditionalGeneration",
            registry_profile="task:summarization@fr",
        )
        assert str(spec) == "task:summarization@fr"
        spec = resolve_profile(
            "some/pegasus-clone",
            "PegasusForConditionalGeneration",
            registry_profile="continuation@fr",
        )
        assert str(spec) == "task:summarization"


class TestForcedBosVocabCollision:
    """Bare ISO codes collide with ordinary subwords (T5's 'de' id 221,
    Marian's 'en' id 39) and were injected as forced decoder tokens,
    corrupting every translator without a real lang-code system."""

    def test_plain_vocab_word_is_not_a_lang_code(self):
        from transformer_lens.benchmarks.text_quality import _forced_bos_for_target

        class PlainSeq2SeqTokenizer:
            unk_token_id = 3

            def convert_tokens_to_ids(self, tok):
                return {"de": 221, "en": 39}.get(tok, 3)

        assert _forced_bos_for_target(PlainSeq2SeqTokenizer(), "de") is None

    def test_nllb_style_code_still_resolves(self):
        from transformer_lens.benchmarks.text_quality import _forced_bos_for_target

        class NllbLikeTokenizer:
            unk_token_id = 3

            def convert_tokens_to_ids(self, tok):
                return {"deu_Latn": 256042}.get(tok, 3)

        assert _forced_bos_for_target(NllbLikeTokenizer(), "de") == 256042


class TestAllGenerationsCaptured:
    def test_details_carry_every_prompts_generation(self):
        """Only the first prompt's output was stored; the registry-wide
        review needs every generation inspectable."""
        bridge = FakeBridge()
        result = _run(bridge, profile="continuation")
        texts = result.details["generated_texts"]
        assert len(texts) == result.details["num_prompts"]
        assert all(isinstance(t, str) and t for t in texts)
