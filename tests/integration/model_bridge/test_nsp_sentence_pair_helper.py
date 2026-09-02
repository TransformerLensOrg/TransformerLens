"""Next-sentence prediction from strings on the bridge.

Mirrors BertNextSentencePrediction's string interface, which cannot be adapted
onto a bridge (its forward reaches for encoder_output/pooler/nsp_head). This
helper is where that ergonomics survives the Hooked* removal.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer, BertForNextSentencePrediction

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "google-bert/bert-base-cased"
SENTENCE_A = "A man walked into a grocery store."
SEQUENTIAL_B = "He bought an apple."
UNRELATED_B = "The Eiffel Tower is in Paris."


@pytest.fixture(scope="module")
def nsp_bridge() -> TransformerBridge:
    bridge = TransformerBridge.boot_transformers(
        MODEL, device="cpu", model_class=BertForNextSentencePrediction
    )
    bridge.enable_compatibility_mode()
    return bridge


@pytest.fixture(scope="module")
def hf_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


def test_pair_tokenization_matches_huggingface(nsp_bridge, hf_tokenizer):
    """[CLS] a [SEP] b [SEP] with segment ids, identical to tokenizer(a, b)."""
    tokens = nsp_bridge.to_sentence_pair_tokens(SENTENCE_A, SEQUENTIAL_B)
    expected = hf_tokenizer(SENTENCE_A, SEQUENTIAL_B, return_tensors="pt")

    assert torch.equal(tokens["input_ids"], expected["input_ids"])
    assert torch.equal(tokens["token_type_ids"], expected["token_type_ids"])
    assert tokens["token_type_ids"].unique().tolist() == [0, 1]


def test_logits_match_a_direct_huggingface_nsp_forward(hf_tokenizer):
    """Reference is a FRESH HF load, not nsp_bridge.original_model — compat mode
    mutates the wrapped model in place, which made the old exact-equality
    assertion self-fulfilling. The reference must load eager attention like the
    bridge does: the residual ~2.4e-6 against a default (sdpa) load is entirely
    the attention kernel, and with matching kernels the match is bit-exact."""
    hf = BertForNextSentencePrediction.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    encodings = hf_tokenizer(SENTENCE_A, SEQUENTIAL_B, return_tensors="pt")
    with torch.no_grad():
        expected = hf(**encodings).logits

    bridge = TransformerBridge.boot_transformers(
        MODEL, device="cpu", model_class=BertForNextSentencePrediction
    )
    actual = bridge.predict_next_sentence(SENTENCE_A, SEQUENTIAL_B, return_type="logits")
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_compat_mode_shifts_logits_by_the_centering_amount(nsp_bridge, hf_tokenizer):
    """Compatibility mode's unembed centering moves NSP logits ~1e-2 relative to
    raw HF (uniformly across both classes, so verdicts hold). Pin the band so a
    silent change in compat processing shows up here."""
    hf = BertForNextSentencePrediction.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    encodings = hf_tokenizer(SENTENCE_A, SEQUENTIAL_B, return_tensors="pt")
    with torch.no_grad():
        raw = hf(**encodings).logits
    compat = nsp_bridge.predict_next_sentence(SENTENCE_A, SEQUENTIAL_B, return_type="logits")
    shift = (compat - raw).abs()
    # centering subtracts the mean of the two logits: both classes move together
    assert torch.allclose(shift[0, 0], shift[0, 1], atol=1e-4)
    assert 1e-3 < float(shift.max()) < 1e-1, float(shift.max())


def test_predictions_distinguish_sequential_from_unrelated(nsp_bridge):
    assert nsp_bridge.predict_next_sentence(SENTENCE_A, SEQUENTIAL_B) == (
        "The sentences are sequential"
    )
    assert nsp_bridge.predict_next_sentence(SENTENCE_A, UNRELATED_B) == (
        "The sentences are NOT sequential"
    )


def test_segment_ids_are_load_bearing(nsp_bridge):
    """Dropping token_type_ids collapses the NSP margin — why the helper owns them."""
    tokens = nsp_bridge.to_sentence_pair_tokens(SENTENCE_A, SEQUENTIAL_B)
    with_segments = nsp_bridge.predict_next_sentence(SENTENCE_A, SEQUENTIAL_B, return_type="logits")
    without_segments = nsp_bridge(
        tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        return_type="logits",
    )

    margin = lambda logits: float((logits[0, 0] - logits[0, 1]).abs())
    assert margin(with_segments) > 2 * margin(without_segments)


def test_helper_rejects_a_model_without_an_nsp_head():
    """An MLM-headed bridge has no 2-class output; say so instead of decoding noise."""
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu")
    with pytest.raises(ValueError, match="next-sentence-prediction head"):
        bridge.predict_next_sentence(SENTENCE_A, SEQUENTIAL_B)
