"""Tests for HookedTransformer.to_str_tokens."""

import pytest
from transformers import AutoTokenizer

from transformer_lens import HookedTransformer, HookedTransformerConfig


@pytest.fixture(scope="module")
def gemma_named_model():
    """Tiny random model whose tokenizer reports a Gemma ``name_or_path``.

    ``to_str_tokens`` used to special-case tokenizers with "gemma" in ``name_or_path``. A GPT-2
    tokenizer is renamed after model construction so the test exercises that code path without
    downloading a (gated) Gemma tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=8,
        d_head=4,
        n_heads=2,
        n_ctx=32,
        d_vocab=tokenizer.vocab_size,
        act_fn="gelu",
        attn_only=True,
    )
    model = HookedTransformer(cfg, tokenizer=tokenizer)
    assert model.tokenizer is not None
    model.tokenizer.name_or_path = "google/gemma-2-2b"
    return model


def test_to_str_tokens_string_input_gemma_tokenizer(gemma_named_model):
    """String input must decode per token, matching the tensor-input path."""
    model = gemma_named_model
    prompt = "The capital of France is"

    str_tokens = model.to_str_tokens(prompt)

    assert str_tokens == model.to_str_tokens(model.to_tokens(prompt))
    assert len(str_tokens) == model.to_tokens(prompt).shape[-1]
    assert "".join(str_tokens[1:]) == prompt  # first token is BOS


def test_to_str_tokens_list_input_gemma_tokenizer(gemma_named_model):
    """A list of strings is handled element-wise via the string path."""
    model = gemma_named_model
    prompts = ["Hello world", "Hi"]

    result = model.to_str_tokens(prompts)

    assert result == [model.to_str_tokens(model.to_tokens(p)) for p in prompts]
