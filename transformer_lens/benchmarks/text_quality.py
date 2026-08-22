"""Text quality benchmark for TransformerBridge.

Generates text the way a real user of the model would (its prompt profile:
chat template, translation source, code, own-language continuation — see
``text_quality_profiles``) and scores each output against a known-good
reference completion with one pinned multilingual judge. The score derives
from the perplexity ratio PPL_judge(generated)/PPL_judge(reference), which
cancels the judge's per-language handicap; a repetition penalty catches
degenerate loops (which the ratio alone rewards) and a length penalty
catches truncated output.

Generation is seeded per prompt for reproducibility, and the judge is loaded
once (CPU/fp32 always, so scores do not depend on the verifying machine) and
reused across all prompts.
"""

import gc
import math
from typing import Any, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from transformer_lens.benchmarks.text_quality_profiles import (
    CAPTION_REFERENCES,
    JUDGE_CONTEXT_KINDS,
    JUDGE_R_FAIL,
    LANG_ISO3,
    LANG_NAMES,
    MAX_NEW_TOKENS_BY_KIND,
    NLLB_CODES,
    PREPEND_BOS_BY_KIND,
    T5_PREFIX_ARCHITECTURES,
    TEMPERATURE_BY_KIND,
    ProfilePrompt,
    ProfileSpec,
    p4_pass_threshold,
    prompts_for,
)
from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    deterministic_rng,
)
from transformer_lens.model_bridge import TransformerBridge

# The one judge every model is scored with, pinned by revision so a Hub update
# can never silently move every score. Selection + measurements live in
# scripts/text_quality_judge_bakeoff.py; separation is weakest in de/ru, so
# scores there carry wider error bars.
JUDGE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
JUDGE_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def load_judge() -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the pinned judge on CPU in fp32 (machine-independent scores)."""
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID, revision=JUDGE_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID, revision=JUDGE_REVISION, dtype=torch.float32
    )
    torch.nn.Module.to(model, "cpu")
    model.eval()
    return model, tokenizer


def _judge_perplexity(
    text: str,
    context: str,
    tokenizer: Any,
    judge: Any,
) -> Tuple[float, Optional[str]]:
    """Judge perplexity of ``text``; ``context`` tokens are label-masked so only
    ``text`` is scored. Returns (ppl, error)."""
    try:
        # Tokenize the pieces separately: tokenizing the concatenated string
        # lets BPE merge across the boundary and shifts the label mask into
        # the scored text.
        text_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        context_len = 0
        input_ids = text_ids
        if context:
            context_ids = tokenizer(context, return_tensors="pt")["input_ids"]
            context_len = context_ids.shape[1]
            input_ids = torch.cat([context_ids, text_ids], dim=1)

        if text_ids.shape[1] < 2:
            return float("inf"), "Scored text too short (< 2 judge tokens)"

        labels = input_ids.clone()
        if context_len:
            labels[0, :context_len] = -100

        with torch.no_grad():
            loss = judge(input_ids, labels=labels).loss.item()
        return math.exp(loss), None
    except Exception as e:
        return float("inf"), f"Perplexity computation failed: {str(e)}"


def _compute_repetition_penalty(text: str, ns: Tuple[int, ...] = (2, 3, 4)) -> float:
    """Minimum unique-n-gram ratio in [0, 1]; low values mean looping output.

    Load-bearing under ratio scoring: a degenerate loop has LOW judge
    perplexity, so without this multiplier it would score 100.
    """
    words = text.lower().split()
    # Scriptio continua (zh/ja): word n-grams are inert exactly where the
    # judge rewards loops most, and a single stray space would restore the
    # word path — so char n-grams whenever the text is CJK-dominated.
    compact = "".join(text.split())
    if compact:
        cjk = sum(1 for c in compact if 0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF)
        if cjk / len(compact) >= 0.3 and len(compact) >= 8:
            words = list(compact)
    if len(words) < 2:
        return 1.0

    min_ratio = 1.0
    for n in ns:
        if len(words) < n:
            continue
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if len(ngrams) == 0:
            continue
        unique_ratio = len(set(ngrams)) / len(ngrams)
        min_ratio = min(min_ratio, unique_ratio)

    return min_ratio


def _ratio_to_score(ratio: float) -> float:
    """Map generated/reference perplexity ratio to 0-100.

    score = 100 - 100*ln(ratio)/ln(R_FAIL), clamped: ratio<=1 (as good as the
    reference) scores 100, ratio=R_FAIL scores 0, and score 50 falls at
    sqrt(R_FAIL) — the geometric midpoint between reference quality and
    unambiguously broken output, which keeps the registry's phase-4 floor of
    50 principled.
    """
    if ratio <= 0 or math.isinf(ratio) or math.isnan(ratio):
        return 0.0
    if ratio <= 1.0:
        return 100.0
    return max(0.0, min(100.0, 100.0 - 100.0 * math.log(ratio) / math.log(JUDGE_R_FAIL)))


_SCRIPT_RANGES: dict[str, Tuple[Tuple[int, int], ...]] = {
    "zh": ((0x4E00, 0x9FFF),),
    "ja": ((0x3040, 0x30FF), (0x4E00, 0x9FFF)),
    "ar": ((0x0600, 0x06FF),),
    "ru": ((0x0400, 0x04FF),),
    "hi": ((0x0900, 0x097F),),
}

_LATIN_STOPWORDS: dict[str, frozenset] = {
    "en": frozenset("the and of to is that with for was are it in on".split()),
    "fr": frozenset(
        "le la les des une est que je pas dans de et il elle un en du au pour sur ne ce se".split()
    ),
    "es": frozenset("el los una es que no por con para como de en la y se del las".split()),
    "de": frozenset(
        "der die das und ist nicht ich ein eine mit den von zu im auf f\u00fcr sich".split()
    ),
    "it": frozenset("il la di che non per una sono del gli e in un le si con".split()),
    "nl": frozenset("de het een en is niet ik van dat met op voor aan zijn".split()),
    "pt": frozenset("o os uma de e que n\u00e3o para com por em um as dos da".split()),
}


def _wrong_language(text: str, lang: str) -> bool:
    """Conservatively true only when the text is clearly NOT in ``lang``.

    Ratio scoring alone measures fluency, not language: fluent English output
    beats a short German reference and clamps to 100, so an untranslated echo
    would otherwise score perfectly. Non-Latin targets check script presence;
    Latin targets require zero expected-language stopwords while another
    covered language has several.
    """
    if lang in ("code", ""):
        return False
    ranges = _SCRIPT_RANGES.get(lang)
    if ranges is not None:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        in_script = sum(1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in ranges))
        return in_script / len(letters) < 0.3
    expected = _LATIN_STOPWORDS.get(lang)
    if expected is None:
        return False
    tokens = [w.strip(".,;:!?\"'()") for w in text.lower().split()]
    hits = {code: sum(1 for w in tokens if w in stops) for code, stops in _LATIN_STOPWORDS.items()}
    return hits[lang] == 0 and max(hits.values(), default=0) >= 3


def _length_penalty(gen_tokens: int, ref_tokens: int) -> float:
    """Penalize output far shorter OR far longer than its reference.

    Neutral band [0.5x, 3x] of reference length. The old 25% floor never
    fired once in four validation sweeps — a contentless 13-token chat stub
    against a 41-token reference scored 93.6; at a 0.5x floor it drops below
    the pass line. The 3x cap is the second net for rambling output the
    repetition penalty misses."""
    if ref_tokens <= 0:
        return 1.0
    under = gen_tokens / (0.5 * ref_tokens)
    over = (3.0 * ref_tokens) / max(gen_tokens, 1)
    return max(0.0, min(1.0, under, over))


def _build_caption_test_images(n: int = 3) -> list:
    """A few distinct synthetic images so caption scoring averages over samples
    (content is unimportant — P4 only measures whether generation is grammatical)."""
    from PIL import Image, ImageDraw

    specs = [
        (
            "white",
            [("rectangle", (30, 30, 110, 150), "blue"), ("ellipse", (120, 60, 200, 140), "green")],
        ),
        ("black", [("ellipse", (40, 40, 180, 180), "yellow")]),
        (
            "skyblue",
            [
                ("rectangle", (20, 120, 200, 200), "darkgreen"),
                ("ellipse", (140, 20, 200, 80), "orange"),
            ],
        ),
    ]
    images = []
    for bg, shapes in specs[:n]:
        img = Image.new("RGB", (224, 224), color=bg)
        draw = ImageDraw.Draw(img)
        for kind, box, color in shapes:
            getattr(draw, kind)(box, fill=color)
        images.append(img)
    return images


def _generate_image_conditioned_captions(
    bridge: TransformerBridge, max_new_tokens: int
) -> List[Tuple[int, str]]:
    """Caption synthetic images for image-conditioned seq2seq (Florence-2 emits
    nothing text-only, so text-only P4 is uninformative); [] if no processor/PIL."""
    processor = getattr(bridge, "processor", None)
    if processor is None:
        return []
    try:
        images = _build_caption_test_images()
    except Exception:
        return []

    # Task captioners (Florence-2) map a task token to an internal prompt.
    # <DETAILED_CAPTION> yields a full grammatical sentence; plain <CAPTION> is
    # only 2-3 words (too short to score) and <MORE_DETAILED_CAPTION> tends to
    # loop on out-of-distribution synthetic images.
    is_task_captioner = hasattr(processor, "post_process_generation")
    task = "<DETAILED_CAPTION>" if is_task_captioner else "Describe this image in detail."

    samples: List[Tuple[int, str]] = []
    for i, image in enumerate(images):
        try:
            inputs = processor(text=task, images=image, return_tensors="pt")
            input_ids = inputs["input_ids"].to(bridge.cfg.device)
            extra = {
                k: (v.to(bridge.cfg.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
                if k != "input_ids"
            }
            out = bridge.generate(
                input_ids, max_new_tokens=max_new_tokens, return_type="tokens", **extra
            )
            if isinstance(out, torch.Tensor):
                is_encoder_decoder = bool(
                    getattr(getattr(bridge, "original_model", None), "config", None)
                    and getattr(bridge.original_model.config, "is_encoder_decoder", False)
                )
                # Decoder-only VLM output is prompt + continuation; scoring the
                # fluent prompt as caption text would inflate every sample.
                caption_ids = out[0] if is_encoder_decoder else out[0, input_ids.shape[-1] :]
                text = bridge.tokenizer.decode(caption_ids, skip_special_tokens=True).strip()
                if text:
                    samples.append((i, text))
        except Exception:
            continue
    return samples


def _architecture_id(bridge: Any) -> str:
    """First HF architecture name of the wrapped model, or ''."""
    config = getattr(getattr(bridge, "original_model", None), "config", None)
    architectures = getattr(config, "architectures", None) or []
    return architectures[0] if architectures else ""


def _resolve_lang_code(tokenizer, lang: str) -> Optional[str]:
    """The tokenizer's own code string for ``lang`` ("de" / "de_DE" /
    "deu_Latn"), or None. transformers 5.x NllbTokenizer exposes neither
    get_lang_id nor lang_code_to_id, so candidates are probed through the
    vocab as well."""
    lang = lang.lower()
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_code_to_id, dict):
        if lang in lang_code_to_id:
            return lang
        iso3 = LANG_ISO3.get(lang, "")
        for code in lang_code_to_id:
            code_lower = code.lower()
            if code_lower.startswith(lang + "_") or (iso3 and code_lower.startswith(iso3 + "_")):
                return code
    # Vocab probing is only safe for DISTINCTIVE code forms ("deu_Latn"): a
    # bare ISO code collides with ordinary subwords (T5's "de", Marian's "en")
    # and would be injected as a forced decoder token.
    nllb = NLLB_CODES.get(lang)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if nllb and callable(convert):
        try:
            token_id = convert(nllb)
        except Exception:
            return None
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_id:
            return nllb
    return None


def _forced_bos_for_target(tokenizer, tgt_lang: str) -> Optional[int]:
    """Target-language decoder token for multilingual translators, or None."""
    get_lang_id = getattr(tokenizer, "get_lang_id", None)
    if callable(get_lang_id):
        try:
            return int(get_lang_id(tgt_lang))
        except Exception:
            return None
    code = _resolve_lang_code(tokenizer, tgt_lang)
    if code is None:
        return None
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_code_to_id, dict) and code in lang_code_to_id:
        return int(lang_code_to_id[code])
    try:
        token_id = tokenizer.convert_tokens_to_ids(code)
    except Exception:
        return None
    if (
        isinstance(token_id, int)
        and token_id >= 0
        and token_id != getattr(tokenizer, "unk_token_id", None)
    ):
        return int(token_id)
    return None


def _build_model_input(
    bridge: Any,
    spec: ProfileSpec,
    prompt: ProfilePrompt,
    architecture_id: str,
) -> str:
    """Render one profile prompt into the text this model expects."""
    if spec.kind == "chat":
        return bridge.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    if spec.kind == "task:translation" and architecture_id in T5_PREFIX_ARCHITECTURES:
        src_name = LANG_NAMES.get(spec.src or "en", "English")
        tgt_name = LANG_NAMES.get(spec.lang, "German")
        return f"translate {src_name} to {tgt_name}: {prompt.prompt}"
    if spec.kind == "task:summarization" and architecture_id in T5_PREFIX_ARCHITECTURES:
        return f"summarize: {prompt.prompt}"
    return prompt.prompt


def benchmark_text_quality(
    bridge: Any,
    profile: Union[str, ProfileSpec] = "continuation",
    *,
    max_new_tokens: Optional[int] = None,
    judge_model: Optional[Any] = None,
    judge_tokenizer: Optional[Any] = None,
    model_name: Optional[str] = None,
) -> BenchmarkResult:
    """Benchmark text generation quality with profile prompts and reference-ratio scoring.

    Generates from the model's prompt-profile prompts through the real user
    path (``bridge.generate``), then scores each output against the prompt's
    reference completion via the pinned judge's perplexity ratio, with
    repetition and length penalties.
    """
    if model_name is not None and model_name.lower() == JUDGE_MODEL_ID.lower():
        # Ratio scoring against the judge's own perplexity is self-grading.
        return BenchmarkResult(
            name="text_quality",
            severity=BenchmarkSeverity.SKIPPED,
            message=f"P4 skipped: {model_name} is the pinned judge — cannot self-score",
        )
    _loaded_locally = False
    tokenizer = judge_tokenizer
    try:
        spec = ProfileSpec.parse(profile) if isinstance(profile, str) else profile

        # Diffusion LMs produce text through their native sampler; scoring that
        # text is as meaningful as scoring autoregressive output.
        from transformer_lens.benchmarks.generation import resolve_text_generator

        generator = resolve_text_generator(bridge)
        if generator is None:
            return BenchmarkResult(
                name="text_quality",
                severity=BenchmarkSeverity.INFO,
                message="Skipped: architecture supports no text generation",
            )

        is_encoder_decoder = bool(
            getattr(getattr(bridge, "original_model", None), "config", None)
            and getattr(bridge.original_model.config, "is_encoder_decoder", False)
        )
        is_multimodal = bool(getattr(getattr(bridge, "cfg", None), "is_multimodal", False))

        # Effective-profile adjustments. Image-conditioned seq2seq (Florence-2)
        # emits a bare EOS for text-only prompts — caption real images instead.
        # A chat profile without a chat template downgrades to continuation;
        # never the other direction (base models may ship templates).
        adjustment = ""
        if is_encoder_decoder and is_multimodal:
            spec = ProfileSpec("caption")
        elif spec.kind == "chat":
            if getattr(bridge.tokenizer, "chat_template", None) is None:
                spec = ProfileSpec("continuation", spec.lang)
                adjustment = "chat profile downgraded: tokenizer has no chat template"
            else:
                try:
                    bridge.tokenizer.apply_chat_template(
                        [{"role": "user", "content": "probe"}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                except Exception as template_error:
                    spec = ProfileSpec("continuation", spec.lang)
                    adjustment = f"chat profile downgraded: template raised {template_error!r}"

        denoise_style = "mask" if getattr(bridge.tokenizer, "mask_token", None) else "t5"
        profile_prompts = prompts_for(spec, denoise_style=denoise_style)
        if profile_prompts is None:
            return BenchmarkResult(
                name="text_quality",
                severity=BenchmarkSeverity.SKIPPED,
                message=(
                    f"P4 skipped: no prompts for profile '{spec}' — file a "
                    "TransformerLens issue to add coverage in "
                    "benchmarks/text_quality_profiles.py"
                ),
            )

        if max_new_tokens is None:
            max_new_tokens = MAX_NEW_TOKENS_BY_KIND.get(spec.kind, 50)

        architecture_id = _architecture_id(bridge)
        forced_bos: Optional[int] = None
        if spec.kind == "task:translation":
            src_lang_attr = getattr(bridge.tokenizer, "src_lang", None)
            if src_lang_attr is not None and spec.src:
                src_code = _resolve_lang_code(bridge.tokenizer, spec.src)
                if src_code is not None:
                    try:
                        bridge.tokenizer.src_lang = src_code
                    except Exception:
                        pass
            forced_bos = _forced_bos_for_target(bridge.tokenizer, spec.lang)

        # Generate: (profile_prompt, generated_text) pairs. Token-level slicing —
        # generate() decodes with skip_special_tokens, so the prompt string is
        # not reliably a prefix of the output string (chat templates).
        generations: List[Tuple[ProfilePrompt, str]] = []
        primary_generated = ""
        if spec.kind == "caption":
            with deterministic_rng():
                captions = _generate_image_conditioned_captions(bridge, max_new_tokens)
            if not captions:
                return BenchmarkResult(
                    name="text_quality",
                    severity=BenchmarkSeverity.SKIPPED,
                    message="Skipped: image-conditioned model; image processor/PIL unavailable",
                )
            generations = [
                (ProfilePrompt(prompt="", reference=CAPTION_REFERENCES[i]), text)
                for i, text in captions
                if i < len(CAPTION_REFERENCES)
            ]
            primary_generated = captions[0][1]
        else:
            prepend_bos = PREPEND_BOS_BY_KIND.get(spec.kind)
            # Native diffusion samplers take neither return_type nor forced_bos;
            # bound-method identity can't detect them (new object per access).
            is_autoregressive = getattr(bridge.adapter, "supports_generation", True)
            for prompt in profile_prompts:
                model_input = _build_model_input(bridge, spec, prompt, architecture_id)
                if is_encoder_decoder:
                    # Encoder input follows the tokenizer's own recipe (lang
                    # token + trailing </s>); to_tokens' BOS policy corrupts it
                    # (m2m100 loops on a stray <s>).
                    prompt_ids = bridge.tokenizer(model_input, return_tensors="pt")["input_ids"].to(
                        bridge.cfg.device
                    )
                else:
                    prompt_ids = bridge.to_tokens(model_input, prepend_bos=prepend_bos)
                gen_kwargs: dict = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": TEMPERATURE_BY_KIND.get(spec.kind, 0.7),
                }
                if is_autoregressive:
                    gen_kwargs["return_type"] = "tokens"
                    if forced_bos is not None:
                        gen_kwargs["forced_bos_token_id"] = forced_bos
                # Seeded per prompt so each sample stream is independent of the
                # previous prompt's length.
                with deterministic_rng():
                    out = generator(prompt_ids, **gen_kwargs)
                if not isinstance(out, torch.Tensor):
                    continue
                generated_ids = out[0, 1:] if is_encoder_decoder else out[0, prompt_ids.shape[-1] :]
                generated = bridge.tokenizer.decode(generated_ids, skip_special_tokens=True)
                if spec.kind == "task:denoise" and denoise_style == "t5":
                    # Splice the fill back so both ratio sides are full
                    # sentences (bare fragments judge in the thousands). An
                    # EMPTY fill must stay empty or a dead model inherits the
                    # near-reference sentence and a free 100.
                    if generated.strip():
                        generated = prompt.prompt.replace("<extra_id_0>", generated.strip())
                # Empty output is a scored failure (0), not a dropped sample —
                # dropping it would average only over the prompts that worked.
                generations.append((prompt, generated))
                if not primary_generated:
                    primary_generated = generated

        if len(generations) == 0:
            return BenchmarkResult(
                name="text_quality",
                severity=BenchmarkSeverity.DANGER,
                message="Generation produced no scoreable output for any prompt",
                passed=False,
            )

        if judge_model is None or tokenizer is None:
            judge_model, tokenizer = load_judge()
            _loaded_locally = True

        # Judge context per kind is JUDGE_CONTEXT_KINDS' call. Translation is
        # scored jointly: per-sentence judge perplexity on the short pivots is
        # unstable (measured spread 4.8-3497), so the samples concatenate into
        # one gen/ref pair.
        # Captured pre-merge so translation keeps its per-sentence texts.
        all_generated_texts = [text for _, text in generations]

        if spec.kind == "task:translation" and len(generations) > 1:
            joiner = "" if spec.lang in ("zh", "ja") else " "
            joint = ProfilePrompt(
                prompt="",
                reference=joiner.join(g[0].reference for g in generations),
                lang=spec.lang,
            )
            generations = [(joint, joiner.join(g[1] for g in generations))]

        sample_lang = spec.lang if spec.kind != "caption" else "en"
        per_prompt_scores = []
        per_prompt_ratios = []
        per_prompt_penalties = []
        prompt_details_parts = []

        for prompt, generated in generations:
            context = prompt.prompt if spec.kind in JUDGE_CONTEXT_KINDS else ""

            gen_token_count = len(tokenizer(generated)["input_ids"]) if generated.strip() else 0
            if gen_token_count < 2:
                # Empty or one-token output is a scored failure, not a dropped
                # sample (Florence-style bare EOS, dead generation).
                per_prompt_scores.append(0.0)
                per_prompt_ratios.append(float("inf"))
                per_prompt_penalties.append(0.0)
                prompt_details_parts.append("score=0.0 (output < 2 tokens)")
                continue
            check_lang = prompt.lang if spec.kind != "task:translation" else sample_lang
            if _wrong_language(generated, check_lang):
                # Fluency-only ratio scoring would rate untranslated or
                # wrong-language output above the reference; hard zero.
                per_prompt_scores.append(0.0)
                per_prompt_ratios.append(float("inf"))
                per_prompt_penalties.append(0.0)
                prompt_details_parts.append(f"score=0.0 (output not in '{check_lang}')")
                continue

            gen_ppl, gen_err = _judge_perplexity(generated, context, tokenizer, judge_model)
            ref_ppl, ref_err = _judge_perplexity(prompt.reference, context, tokenizer, judge_model)
            if gen_err is not None:
                # The model's own output was unjudgeable — scored failure.
                per_prompt_scores.append(0.0)
                per_prompt_ratios.append(float("inf"))
                per_prompt_penalties.append(0.0)
                prompt_details_parts.append(f"score=0.0 ({gen_err})")
                continue
            if ref_err is not None:
                # Our reference failed to judge — a data problem, not the
                # model's; exclude the sample and say so.
                prompt_details_parts.append(f"excluded (reference: {ref_err})")
                continue

            ratio = gen_ppl / ref_ppl if ref_ppl > 0 else float("inf")
            rep_penalty = _compute_repetition_penalty(generated)
            ref_token_count = len(tokenizer(prompt.reference)["input_ids"])
            len_penalty = _length_penalty(gen_token_count, ref_token_count)
            adjusted_score = _ratio_to_score(ratio) * rep_penalty * len_penalty

            per_prompt_scores.append(adjusted_score)
            per_prompt_ratios.append(ratio)
            per_prompt_penalties.append(rep_penalty)
            prompt_details_parts.append(
                f"ratio={ratio:.2f} ppl={gen_ppl:.1f} ref_ppl={ref_ppl:.1f} "
                f"rep={rep_penalty:.2f} len={len_penalty:.2f} score={adjusted_score:.1f}"
            )

        if len(per_prompt_scores) == 0:
            return BenchmarkResult(
                name="text_quality",
                severity=BenchmarkSeverity.ERROR,
                message="Scoring failed for all prompts",
                details={"generated_text": primary_generated},
                passed=False,
            )

        avg_score = sum(per_prompt_scores) / len(per_prompt_scores)
        finite_ratios = [r for r in per_prompt_ratios if math.isfinite(r)]
        avg_ratio = sum(finite_ratios) / len(finite_ratios) if finite_ratios else float("inf")
        avg_rep_penalty = sum(per_prompt_penalties) / len(per_prompt_penalties)

        pass_threshold = p4_pass_threshold()
        details = {
            "score": round(avg_score, 1),
            "prompt_profile": str(spec),
            "judge_model": JUDGE_MODEL_ID,
            "judge_revision": JUDGE_REVISION,
            "avg_ratio": round(avg_ratio, 3) if math.isfinite(avg_ratio) else "inf",
            "avg_repetition_penalty": round(avg_rep_penalty, 2),
            "num_prompts": len(per_prompt_scores),
            "per_prompt": " | ".join(prompt_details_parts),
            "max_new_tokens": max_new_tokens,
            "generated_text": primary_generated,
            "generated_texts": all_generated_texts,
        }
        if adjustment:
            details["profile_adjustment"] = adjustment

        if avg_score >= pass_threshold:
            return BenchmarkResult(
                name="text_quality",
                severity=BenchmarkSeverity.INFO,
                message=(
                    f"Text quality score: {avg_score:.1f}/100 "
                    f"(profile {spec}, {len(per_prompt_scores)} prompts)"
                ),
                details=details,
            )
        elif avg_score >= pass_threshold / 2:
            return BenchmarkResult(
                name="text_quality",
                severity=BenchmarkSeverity.WARNING,
                message=(
                    f"Text quality score: {avg_score:.1f}/100 "
                    f"(below {pass_threshold:.0f}, profile {spec})"
                ),
                details=details,
                passed=False,
            )
        else:
            return BenchmarkResult(
                name="text_quality",
                severity=BenchmarkSeverity.DANGER,
                message=(
                    f"Text quality score: {avg_score:.1f}/100 "
                    f"(profile {spec}) — generated text may be incoherent"
                ),
                details=details,
                passed=False,
            )

    except Exception as e:
        return BenchmarkResult(
            name="text_quality",
            severity=BenchmarkSeverity.ERROR,
            message=f"Text quality benchmark failed: {str(e)}",
            passed=False,
        )

    finally:
        if _loaded_locally:
            if judge_model is not None:
                del judge_model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
