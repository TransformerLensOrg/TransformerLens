"""The four defects that kept google-bert/bert-base-cased at FAILED since June.

1. The tokenizer fallback installs bos_token='<|endoftext|>' on BERT (which has
   none) and to_tokens string-prepended it, WordPiece-shredding it into 8 subword
   tokens at the front of every input.
2. The tokenizer_prepends_bos probe compared position 0 against that fake BOS id
   instead of recognizing [CLS], desyncing the bridge from HookedTransformer.
3. The component harness fed float tensors to token_type_embed (an nn.Embedding).
4. Phase 2 loaded the masked LM into HookedTransformer — a causal decoder — and
   graded the bridge against a bidirectional model run under a causal mask.
"""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

MODEL = "google-bert/bert-base-cased"
TEXT = (
    "Natural language processing tasks, such as question answering, "
    "machine translation, reading comprehension, and summarization, "
    "are typically approached with supervised learning."
)


@pytest.fixture(scope="module")
def bert():
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


def test_probe_recognizes_cls_as_prepended_bos(bert) -> None:
    assert bert.cfg.tokenizer_prepends_bos is True


def test_to_tokens_is_clean_wordpiece(bert) -> None:
    tokens = bert.to_tokens(TEXT)
    assert tokens.shape[1] == 32, tokens.shape
    decoded = bert.tokenizer.convert_ids_to_tokens(tokens[0])
    assert decoded[:2] == ["[CLS]", "Natural"], decoded[:6]


def test_prepend_false_strips_cls_matching_hooked_transformer(bert) -> None:
    assert bert.to_tokens(TEXT, prepend_bos=False).shape[1] == 31


def test_non_atomic_bos_is_never_string_prepended(bert) -> None:
    """Defense in depth: even with the flag forced to the manual-prepend path,
    a BOS the tokenizer would shred must not be prepended."""
    original = bert.cfg.tokenizer_prepends_bos
    bert.cfg.tokenizer_prepends_bos = False
    try:
        tokens = bert.to_tokens(TEXT, prepend_bos=True)
    finally:
        bert.cfg.tokenizer_prepends_bos = original
    assert tokens.shape[1] == 32, tokens.shape  # 40 when the shred happens
    decoded = bert.tokenizer.convert_ids_to_tokens(tokens[0])
    assert decoded[1] == "Natural", decoded[:6]


def test_component_harness_feeds_ints_to_embedding_tables(bert) -> None:
    from transformers import AutoModelForMaskedLM

    from transformer_lens.benchmarks.component_benchmark import benchmark_all_components

    hf = AutoModelForMaskedLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    result = benchmark_all_components(bert, hf)
    assert result.passed, result.message
    assert "token_type_embed" not in str(result.details), result.details


def test_phase2_never_grades_against_a_causal_reference() -> None:
    """A masked LM graded against a causal reference is noise: bidirectional
    weights run under a causal mask. Phase 2 grades against the HF model itself,
    so a masked LM must clear it."""
    from transformer_lens.benchmarks.main_benchmark import run_benchmark_suite
    from transformer_lens.benchmarks.utils import BenchmarkSeverity

    results = run_benchmark_suite(
        model_name=MODEL,
        device="cpu",
        phases=[1, 2],
        use_hf_reference=True,
        enable_compatibility_mode=False,
        verbose=False,
        track_memory=False,
    )
    hard_failures = [
        r
        for r in results
        if not r.passed and r.severity not in (BenchmarkSeverity.SKIPPED, BenchmarkSeverity.WARNING)
    ]
    assert not hard_failures, [f"{r.name}: {r.message[:90]}" for r in hard_failures]
