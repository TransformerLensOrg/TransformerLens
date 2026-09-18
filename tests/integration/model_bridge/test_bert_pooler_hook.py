"""The BERT [CLS] pooler is observable through a named bridge hook.

Mirrors HookedEncoder's BertPooler, whose hook_pooler_out carries the
post-tanh pooled [CLS]. Deletion evidence for that component: without a named
bridge hook the pooled [CLS] is only reachable coincidentally, via the NSP
head's unembed.hook_in.
"""

from __future__ import annotations

import pytest
import torch
from transformers import BertForMaskedLM, BertForNextSentencePrediction

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "google-bert/bert-base-cased"


@pytest.fixture(scope="module")
def nsp_bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(
        MODEL, device="cpu", model_class=BertForNextSentencePrediction
    )


def _tokens(bridge: TransformerBridge) -> torch.Tensor:
    return bridge.tokenizer("Hello there my friend.", return_tensors="pt")["input_ids"]


def test_pooler_hook_matches_huggingfaces_own_pooler(nsp_bridge):
    """hook_out is the pooled [CLS], checked against HF's pooler directly."""
    tokens = _tokens(nsp_bridge)
    _, cache = nsp_bridge.run_with_cache(tokens)

    hf = nsp_bridge.original_model
    with torch.no_grad():
        expected = hf.bert.pooler(hf.bert(tokens).last_hidden_state)

    torch.testing.assert_close(cache["pooler.hook_out"], expected, atol=0.0, rtol=0.0)


def test_pooler_hook_is_post_activation(nsp_bridge):
    """HookedEncoder fires hook_pooler_out after tanh; the projection is separate."""
    tokens = _tokens(nsp_bridge)
    _, cache = nsp_bridge.run_with_cache(tokens)

    pre_activation = cache["pooler.dense.hook_out"]
    pooled = cache["pooler.hook_out"]
    assert not torch.allclose(pre_activation, pooled)
    torch.testing.assert_close(torch.tanh(pre_activation), pooled)


def test_hooked_encoder_hook_name_is_aliased(nsp_bridge):
    """Code migrated from HookedEncoder asks for hook_pooler_out."""
    tokens = _tokens(nsp_bridge)
    _, cache = nsp_bridge.run_with_cache(tokens)

    assert torch.equal(cache["pooler.hook_pooler_out"], cache["pooler.hook_out"])


def test_masked_lm_checkpoint_without_a_pooler_still_boots():
    """BertForMaskedLM leaves bert.pooler as None; the mapping must skip it."""
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", model_class=BertForMaskedLM)
    tokens = _tokens(bridge)
    logits, cache = bridge.run_with_cache(tokens)

    assert logits.shape[0] == 1
    assert not [name for name in cache if "pooler" in name]
