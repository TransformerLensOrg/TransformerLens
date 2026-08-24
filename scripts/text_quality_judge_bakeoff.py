"""Bake off perplexity judges for the reworked Phase-4 text-quality benchmark.

Scores two small causal LMs as candidate PPL judges: for each of 9 languages
(8 natural + "code"), build a fluent corpus from ``text_quality_profiles`` and
six deterministic corruptions per fluent string (shuffle/repeat/charnoise x2/
crosslang x2). The judge that best separates fluent from corrupted text by
ROC AUC, worst-language-first, wins; its R_FAIL/R_GOOD thresholds and
per-reference PPLs are then emitted for the real benchmark to consume.

Run:  uv run python scripts/text_quality_judge_bakeoff.py
      uv run python scripts/text_quality_judge_bakeoff.py --languages en,fr --models gpt2
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens.benchmarks.text_quality_profiles import (
    CHAT_PROMPTS,
    CONTINUATION_PROMPTS,
    PIVOT_SENTENCES,
)

# All PIVOT_SENTENCES languages plus code, so no scored language is left
# uncalibrated (it/nl/pt/hi were absent from the original judge selection run).
DEFAULT_LANGUAGES = [
    "en",
    "fr",
    "es",
    "de",
    "it",
    "nl",
    "pt",
    "zh",
    "ar",
    "ru",
    "ja",
    "hi",
    "code",
]
DEFAULT_MODELS = ["bigscience/bloom-560m", "Qwen/Qwen2.5-0.5B"]

NO_SPACE_LANGS = {"zh", "ja"}
CHARNOISE_RATES = (0.15, 0.4)
CROSSLANG_RATES = (0.3, 0.7)

OUT_JSON = Path("judge_reference_ppls.json")  # cwd; override with --out

# ---------------------------------------------------------------------------
# Fluent corpus
# ---------------------------------------------------------------------------


def build_fluent_corpus(languages: list[str]) -> dict[str, list[str]]:
    """Fluent strings per language: pivot sentences + continuation/chat references."""
    corpus: dict[str, list[str]] = {}
    for lang in languages:
        if lang == "code":
            corpus[lang] = [p.reference for p in CONTINUATION_PROMPTS["code"]]
            continue
        texts = list(PIVOT_SENTENCES.get(lang, ()))
        texts += [p.reference for p in CONTINUATION_PROMPTS.get(lang, ())]
        texts += [p.reference for p in CHAT_PROMPTS.get(lang, ())]
        corpus[lang] = texts
    return corpus


# ---------------------------------------------------------------------------
# Corruptions (deterministic given a shared random.Random)
# ---------------------------------------------------------------------------


def tokenize_units(text: str, lang: str) -> list[str]:
    """Words for space-delimited languages, characters for zh/ja."""
    return list(text) if lang in NO_SPACE_LANGS else text.split()


def join_units(units: list[str], lang: str) -> str:
    return "".join(units) if lang in NO_SPACE_LANGS else " ".join(units)


def corrupt_shuffle(text: str, lang: str, rng: random.Random) -> str:
    """Permute word (or char) order."""
    units = tokenize_units(text, lang)
    rng.shuffle(units)
    return join_units(units, lang)


def corrupt_repeat(text: str, lang: str) -> str:
    """Repeat the first 3 words (5 chars for zh/ja) until the original length."""
    units = tokenize_units(text, lang)
    if not units:
        return text
    n = 5 if lang in NO_SPACE_LANGS else 3
    seed = units[:n] or units
    out = [seed[i % len(seed)] for i in range(len(units))]
    return join_units(out, lang)


def corrupt_charnoise(text: str, rate: float, rng: random.Random) -> str:
    """Swap-adjacent-or-delete at `rate` of character positions."""
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        if rng.random() < rate:
            if i + 1 < len(chars) and rng.random() < 0.5:
                out.append(chars[i + 1])
                out.append(chars[i])
                i += 2
                continue
            i += 1  # delete
            continue
        out.append(chars[i])
        i += 1
    return "".join(out)


def corrupt_crosslang(
    text: str, lang: str, rate: float, other_units: list[str], rng: random.Random
) -> str:
    """Replace `rate` of tokens with tokens drawn from another language's fluent text."""
    units = tokenize_units(text, lang)
    if not units or not other_units:
        return text
    n_replace = min(len(units), max(1, round(rate * len(units))))
    idxs = rng.sample(range(len(units)), n_replace)
    out = units[:]
    for idx in idxs:
        out[idx] = rng.choice(other_units)
    return join_units(out, lang)


@dataclass
class CorruptedSample:
    """One corrupted variant of a fluent source string."""

    lang: str
    source_text: str
    kind: str
    severity: Optional[float]
    text: str


def build_corruptions(
    languages: list[str], fluent_corpus: dict[str, list[str]], seed: int = 42
) -> dict[str, list[CorruptedSample]]:
    """6 corrupted variants per fluent string, generated once and shared by both candidates."""
    rng = random.Random(seed)
    other_pool: dict[str, list[str]] = {}
    for i, lang in enumerate(languages):
        next_lang = languages[(i + 1) % len(languages)]
        other_text = " ".join(fluent_corpus.get(next_lang, []))
        other_pool[lang] = tokenize_units(other_text, lang)

    by_lang: dict[str, list[CorruptedSample]] = {lang: [] for lang in languages}
    for lang in languages:
        for text in fluent_corpus[lang]:
            samples = by_lang[lang]
            samples.append(
                CorruptedSample(lang, text, "shuffle", None, corrupt_shuffle(text, lang, rng))
            )
            samples.append(CorruptedSample(lang, text, "repeat", None, corrupt_repeat(text, lang)))
            for rate in CHARNOISE_RATES:
                samples.append(
                    CorruptedSample(
                        lang, text, "charnoise", rate, corrupt_charnoise(text, rate, rng)
                    )
                )
            for rate in CROSSLANG_RATES:
                samples.append(
                    CorruptedSample(
                        lang,
                        text,
                        "crosslang",
                        rate,
                        corrupt_crosslang(text, lang, rate, other_pool[lang], rng),
                    )
                )
    return by_lang


# ---------------------------------------------------------------------------
# PPL scoring
# ---------------------------------------------------------------------------


def _with_retry(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    """One retry after 60s on a 429/rate-limit error."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "429" in msg or "rate limit" in msg.lower():
            print(f"429 hit, retrying in 60s: {msg}", file=sys.stderr)
            time.sleep(60)
            return fn(*args, **kwargs)
        raise


def score_text(text: str, tokenizer, model) -> Optional[dict]:  # type: ignore[no-untyped-def]
    """NLL-based PPL for one string; None if tokenization has <2 tokens."""
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"]
    n_tokens = int(ids.shape[1])
    if n_tokens < 2:
        return None
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    dt = time.perf_counter() - t0
    ppl = math.exp(out.loss.item())
    unk_id = tokenizer.unk_token_id
    unk_count = int((ids == unk_id).sum().item()) if unk_id is not None else None
    return {"ppl": ppl, "n_tokens": n_tokens, "unk_count": unk_count, "dt": dt}


# ---------------------------------------------------------------------------
# AUC (hand-rolled, rank-based Mann-Whitney)
# ---------------------------------------------------------------------------


def auc_score(neg_scores: list[float], pos_scores: list[float]) -> float:
    """P(pos > neg) via rank-sum; positive = corrupted, negative = fluent."""
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    combined = [(s, 0) for s in neg_scores] + [(s, 1) for s in pos_scores]
    combined.sort(key=lambda x: x[0])
    n = len(combined)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-indexed, averaged over the tie block
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_pos = sum(r for r, (_, lbl) in zip(ranks, combined) if lbl == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Per-candidate run
# ---------------------------------------------------------------------------


def run_candidate(
    model_id: str,
    languages: list[str],
    fluent_corpus: dict[str, list[str]],
    corrupted_by_lang: dict[str, list[CorruptedSample]],
) -> dict:
    print(f"Loading {model_id} ...")
    tokenizer = _with_retry(AutoTokenizer.from_pretrained, model_id)
    model = _with_retry(AutoModelForCausalLM.from_pretrained, model_id, dtype=torch.float32)
    model.eval()

    fluent_ppl: dict[str, dict[str, float]] = {}
    lang_stats: dict[str, dict] = {}
    total_dt = 0.0
    total_forwards = 0

    for lang in languages:
        fluent_ppl[lang] = {}
        ln_fluent: list[float] = []
        has_unk = tokenizer.unk_token_id is not None
        unk_count = 0
        unk_total = 0

        for text in fluent_corpus[lang]:
            res = score_text(text, tokenizer, model)
            if res is None:
                continue
            total_forwards += 1
            total_dt += res["dt"]
            fluent_ppl[lang][text] = res["ppl"]
            ln_fluent.append(math.log(res["ppl"]))
            if has_unk:
                unk_count += res["unk_count"]
                unk_total += res["n_tokens"]

        ln_corrupt: list[float] = []
        ln_ratios: list[float] = []
        ratio_raw: list[float] = []
        for sample in corrupted_by_lang[lang]:
            if sample.source_text not in fluent_ppl[lang]:
                continue  # source string itself was skipped (too short)
            res = score_text(sample.text, tokenizer, model)
            if res is None:
                continue
            total_forwards += 1
            total_dt += res["dt"]
            ppl_src = fluent_ppl[lang][sample.source_text]
            ln_corrupt.append(math.log(res["ppl"]))
            ln_ratios.append(math.log(res["ppl"]) - math.log(ppl_src))
            ratio_raw.append(res["ppl"] / ppl_src)
            if has_unk:
                unk_count += res["unk_count"]
                unk_total += res["n_tokens"]

        lang_stats[lang] = {
            "auc": auc_score(ln_fluent, ln_corrupt),
            "median_ln_ratio": float(np.median(ln_ratios)) if ln_ratios else float("nan"),
            "unk_rate": (unk_count / unk_total) if has_unk and unk_total else None,
            "mean_fluent_ppl": (
                float(np.mean(list(fluent_ppl[lang].values())))
                if fluent_ppl[lang]
                else float("nan")
            ),
            "ratio_raw": ratio_raw,
        }

    avg_forward_time = total_dt / total_forwards if total_forwards else float("nan")
    del model
    gc.collect()
    return {
        "fluent_ppl": fluent_ppl,
        "lang_stats": lang_stats,
        "avg_forward_time": avg_forward_time,
    }


# ---------------------------------------------------------------------------
# Winner selection + constants
# ---------------------------------------------------------------------------


def pick_winner(results: dict[str, dict], languages: list[str]) -> tuple[str, str]:
    def min_auc(name: str) -> float:
        return min(results[name]["lang_stats"][l]["auc"] for l in languages)

    def spread(name: str) -> float:
        vals = [results[name]["lang_stats"][l]["median_ln_ratio"] for l in languages]
        return max(vals) - min(vals)

    def speed(name: str) -> float:
        return results[name]["avg_forward_time"]

    names = sorted(results, key=min_auc, reverse=True)
    best = min_auc(names[0])
    tied = [n for n in names if best - min_auc(n) <= 0.01]
    if len(tied) == 1:
        return tied[0], "highest minimum per-language AUC"

    best_spread = min(spread(n) for n in tied)
    tied2 = [n for n in tied if spread(n) == best_spread]
    if len(tied2) == 1:
        return tied2[0], "min-AUC tie -> smaller cross-language ln-ratio spread"

    winner = min(tied2, key=speed)
    return winner, "min-AUC tie -> spread tie -> faster wall-clock per forward"


def compute_r_fail(winner_stats: dict, languages: list[str]) -> float:
    """Geo-mean over languages of the MEDIAN corrupted/fluent ratio.

    A low percentile degenerates below 1 in weak-separation languages (some
    corruptions do not raise perplexity there), which would invert the log
    mapping; the median is the robust "typical broken output" anchor. This is
    the exact derivation of the shipped JUDGE_R_FAIL."""
    per_lang = []
    for lang in languages:
        raw = winner_stats["lang_stats"][lang]["ratio_raw"]
        if raw:
            per_lang.append(float(np.median(raw)))
    return statistics.geometric_mean(per_lang)


def compute_r_good(winner_stats: dict, languages: list[str]) -> float:
    per_lang = []
    for lang in languages:
        vals = list(winner_stats["fluent_ppl"][lang].values())
        ratios = [vals[i] / vals[j] for i in range(len(vals)) for j in range(len(vals)) if i != j]
        if ratios:
            per_lang.append(float(np.percentile(ratios, 90)))
    return statistics.geometric_mean(per_lang)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES))
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--out", type=Path, default=OUT_JSON, help="Where to write reference PPLs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    api = HfApi()
    for model_id in models:
        info = _with_retry(api.model_info, model_id)
        print(f"{model_id} revision sha: {info.sha}")

    fluent_corpus = build_fluent_corpus(languages)
    corrupted_by_lang = build_corruptions(languages, fluent_corpus)

    results: dict[str, dict] = {}
    for model_id in models:
        results[model_id] = run_candidate(model_id, languages, fluent_corpus, corrupted_by_lang)

    header = f"{'candidate':28s} {'lang':6s} {'auc':>7s} {'med_ln_ratio':>13s} {'unk_rate':>9s} {'mean_ppl':>10s}"
    print("\n" + header)
    print("-" * len(header))
    for model_id in models:
        for lang in languages:
            st = results[model_id]["lang_stats"][lang]
            unk_str = f"{st['unk_rate']:.4f}" if st["unk_rate"] is not None else "n/a"
            print(
                f"{model_id:28s} {lang:6s} {st['auc']:7.3f} {st['median_ln_ratio']:13.3f} "
                f"{unk_str:>9s} {st['mean_fluent_ppl']:10.2f}"
            )

    winner, reason = pick_winner(results, languages)
    print(f"\nwinner: {winner}  ({reason})")
    for model_id in models:
        print(
            f"  {model_id}: avg forward wall-clock {results[model_id]['avg_forward_time']*1000:.2f} ms"
        )

    r_fail = compute_r_fail(results[winner], languages)
    r_good = compute_r_good(results[winner], languages)
    print(f"JUDGE_R_FAIL = {r_fail:.1f}")
    print(f"JUDGE_R_GOOD = {r_good:.2f}")
    print(f"pass line score(R_GOOD) = {100 - 100 * math.log(r_good) / math.log(r_fail):.1f}")

    flagged = [
        lang
        for lang in languages
        if all(results[m]["lang_stats"][lang]["auc"] < 0.8 for m in models)
    ]
    print(f"languages with AUC < 0.8 for BOTH candidates: {flagged or 'none'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results[winner]["fluent_ppl"], ensure_ascii=False, indent=2))
    print(f"\nreference PPLs written to {args.out}")


if __name__ == "__main__":
    main()
