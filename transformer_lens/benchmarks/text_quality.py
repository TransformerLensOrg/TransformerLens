"""Text quality benchmark for TransformerBridge.

Generates text with the bridge model from multiple diverse prompts and scores
each continuation's legibility using GPT-2 as a perplexity-based judge.
Only the generated continuation tokens are scored (prompt tokens are masked),
and a repetition penalty is applied to catch degenerate looping output.

Generation is seeded for reproducibility, and the scoring model is loaded once
and reused across all prompts.
"""

import gc
import math
from typing import List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity
from transformer_lens.model_bridge import TransformerBridge

# Diverse prompts used alongside the caller-provided test_text to get a robust
# quality signal across different domains and styles.
_DEFAULT_PROMPTS = [
    "The theory of relativity explains that",
    "In the dense forests of the Amazon,",
    "Modern computing relies heavily on",
]


def _load_scoring_model(
    scoring_model_name: str,
    device: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the scoring model and tokenizer.

    Separated from perplexity computation so the caller can load once and
    reuse across multiple prompts.
    """
    tokenizer = AutoTokenizer.from_pretrained(scoring_model_name)
    model = AutoModelForCausalLM.from_pretrained(scoring_model_name)
    torch.nn.Module.to(model, device)
    model.eval()
    return model, tokenizer


def _compute_continuation_perplexity(
    prompt: str,
    full_text: str,
    tokenizer: PreTrainedTokenizerBase,
    scoring_model: PreTrainedModel,
    device: str,
) -> Tuple[float, Optional[str]]:
    """Compute perplexity of only the continuation tokens (excluding prompt).

    Prompt tokens are masked with -100 in labels so CrossEntropyLoss ignores
    them. This prevents well-formed prompt text from artificially lowering
    the perplexity of generated content.

    Args:
        prompt: The original input prompt.
        full_text: The complete text (prompt + generated continuation).
        tokenizer: Pre-loaded tokenizer.
        scoring_model: Pre-loaded scoring model.
        device: Device string.

    Returns:
        Tuple of (perplexity, error_message). error_message is None on success.
    """
    try:
        encodings = tokenizer(full_text, return_tensors="pt")
        input_ids = encodings["input_ids"].to(device)

        # Tokenize just the prompt to find where continuation starts
        prompt_encodings = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_encodings["input_ids"].shape[1]

        # Build labels: -100 for prompt positions, actual ids for continuation
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100

        continuation_len = input_ids.shape[1] - prompt_len
        if continuation_len < 2:
            return float("inf"), "Generated continuation too short (< 2 tokens)"

        with torch.no_grad():
            outputs = scoring_model(input_ids, labels=labels)
            loss = outputs.loss.item()

        perplexity = math.exp(loss)
        return perplexity, None

    except Exception as e:
        return float("inf"), f"Perplexity computation failed: {str(e)}"


def _compute_repetition_penalty(text: str, ns: Tuple[int, ...] = (2, 3, 4)) -> float:
    """Compute a repetition penalty based on n-gram uniqueness ratio.

    Returns a multiplier in [0.0, 1.0] where 1.0 means no repetition and
    lower values penalize repetitive text. The penalty is the minimum
    unique-n-gram ratio across all checked n-gram sizes.

    Args:
        text: The generated continuation text (prompt excluded).
        ns: Tuple of n-gram sizes to check.

    Returns:
        Penalty multiplier in [0.0, 1.0].
    """
    words = text.lower().split()
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


def _perplexity_to_score(perplexity: float) -> float:
    """Map continuation perplexity to a 0-100 legibility score.

    Uses: score = 135 - 10 * ln(perplexity), capped to [0, 100].
    Calibrated for continuation-only perplexity (higher than full-text).
    A well-functioning model typically gets ppl 40-60 -> score 94-98.
    Default pass threshold of 85 corresponds to approximately ppl 150.

    Args:
        perplexity: The perplexity value from the scoring model.

    Returns:
        Score from 0.0 to 100.0.
    """
    if perplexity <= 0 or math.isinf(perplexity):
        return 0.0
    return max(0.0, min(100.0, 135.0 - 10.0 * math.log(perplexity)))


def benchmark_text_quality(
    bridge: TransformerBridge,
    test_text: str,
    max_new_tokens: int = 50,
    scoring_model_name: str = "gpt2",
    pass_threshold: float = 85.0,
    device: str = "cpu",
    scoring_model: Optional[PreTrainedModel] = None,
    scoring_tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> BenchmarkResult:
    """Benchmark text generation quality using continuation-only perplexity scoring.

    Generates text from multiple diverse prompts, scores each continuation using
    GPT-2 perplexity (prompt tokens masked), applies a repetition penalty,
    and returns the averaged score.

    Args:
        bridge: TransformerBridge model to test.
        test_text: Primary input prompt (additional diverse prompts are also used).
        max_new_tokens: Number of tokens to generate per prompt.
        scoring_model_name: HuggingFace model to use as scorer.
        pass_threshold: Minimum average score to pass (default 95.0).
        device: Device for the scoring model.
        scoring_model: Optional pre-loaded scoring model. When provided alongside
            scoring_tokenizer, skips loading and avoids cleanup (caller owns lifecycle).
        scoring_tokenizer: Optional pre-loaded tokenizer for the scoring model.

    Returns:
        BenchmarkResult with quality score details.
    """
    _loaded_locally = False
    tokenizer = scoring_tokenizer
    try:
        prompts = [test_text] + _DEFAULT_PROMPTS

        # Seed for reproducibility
        torch.manual_seed(42)

        # Generate text for each prompt
        generations: List[Tuple[str, str]] = []  # (prompt, full_text)
        primary_generated = ""
        for i, prompt in enumerate(prompts):
            generated = bridge.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
            )
            if not isinstance(generated, str) or len(generated.strip()) == 0:
                continue
            generations.append((prompt, generated))
            if i == 0:
                primary_generated = generated

        if len(generations) == 0:
            return BenchmarkResult(
                name="text_quality",
                severity=BenchmarkSeverity.DANGER,
                message="Generation produced empty output for all prompts",
                passed=False,
            )

        # Load scoring model if not pre-loaded by caller
        if scoring_model is None or tokenizer is None:
            scoring_model, tokenizer = _load_scoring_model(scoring_model_name, device)
            _loaded_locally = True

        # Score each continuation
        per_prompt_scores = []
        per_prompt_perplexities = []
        per_prompt_penalties = []
        prompt_details_parts = []

        for prompt, full_text in generations:
            perplexity, error = _compute_continuation_perplexity(
                prompt, full_text, tokenizer, scoring_model, device
            )
            if error is not None:
                continue

            raw_score = _perplexity_to_score(perplexity)

            # Repetition penalty on continuation only
            continuation = full_text[len(prompt) :]
            rep_penalty = _compute_repetition_penalty(continuation)
            adjusted_score = raw_score * rep_penalty

            per_prompt_scores.append(adjusted_score)
            per_prompt_perplexities.append(perplexity)
            per_prompt_penalties.append(rep_penalty)
            prompt_details_parts.append(
                f"ppl={perplexity:.1f} score={adjusted_score:.1f} rep={rep_penalty:.2f}"
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
        avg_perplexity = sum(per_prompt_perplexities) / len(per_prompt_perplexities)
        avg_rep_penalty = sum(per_prompt_penalties) / len(per_prompt_penalties)

        details = {
            "score": round(avg_score, 1),
            "avg_perplexity": round(avg_perplexity, 2),
            "avg_repetition_penalty": round(avg_rep_penalty, 2),
            "num_prompts": len(per_prompt_scores),
            "per_prompt": " | ".join(prompt_details_parts),
            "scoring_model": scoring_model_name,
            "max_new_tokens": max_new_tokens,
            "generated_text": primary_generated,
        }

        if avg_score >= pass_threshold:
            return BenchmarkResult(
                name="text_quality",
                severity=BenchmarkSeverity.INFO,
                message=(
                    f"Text quality score: {avg_score:.1f}/100 "
                    f"(avg perplexity: {avg_perplexity:.1f}, "
                    f"{len(per_prompt_scores)} prompts)"
                ),
                details=details,
            )
        elif avg_score >= 80.0:
            return BenchmarkResult(
                name="text_quality",
                severity=BenchmarkSeverity.WARNING,
                message=(
                    f"Text quality score: {avg_score:.1f}/100 "
                    f"(below {pass_threshold}, avg perplexity: {avg_perplexity:.1f})"
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
                    f"(avg perplexity: {avg_perplexity:.1f}) "
                    f"— generated text may be incoherent"
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
            if scoring_model is not None:
                del scoring_model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.synchronize()
                torch.mps.empty_cache()
