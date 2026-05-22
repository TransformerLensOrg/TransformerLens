"""Tests for TransformerBridge.generate() and generate_stream() when no tokenizer is set.

Bridge counterpart to tests/unit/test_generate_no_tokenizer.py — regression
coverage for https://github.com/TransformerLensOrg/TransformerLens/issues/483.

The bridge can be constructed via boot_transformers() with a tokenizer loaded
from HF; tests then clear ``bridge.tokenizer`` to exercise the tokenizer-free
generation path (algorithmic/custom-tokenized use cases).
"""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

_PROMPT_TOKENS = torch.tensor([[15496, 11, 314, 1101, 257]], dtype=torch.long)


@pytest.fixture(scope="module")
def tokenizer_free_bridge():
    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    bridge.tokenizer = None
    return bridge


def test_generate_without_tokenizer_stop_at_eos_false_kv_cache(tokenizer_free_bridge):
    """generate() with no tokenizer, stop_at_eos=False, use_past_kv_cache=True."""
    bridge = tokenizer_free_bridge
    assert bridge.tokenizer is None

    tokens = _PROMPT_TOKENS.clone()

    # === TEMP DEBUG: localize where NaN originates on CI ===
    import sys

    def _diag(label: str, t: torch.Tensor) -> None:
        print(
            f"[DIAG] {label}: nan={torch.isnan(t).any().item()} "
            f"inf={torch.isinf(t).any().item()} shape={tuple(t.shape)}",
            file=sys.stderr,
            flush=True,
        )

    with torch.no_grad():
        o0 = bridge.original_model(tokens, use_cache=True)
    _diag("step0_logits", o0.logits)
    cache = o0.past_key_values
    print(
        f"[DIAG] cache_type={type(cache).__name__} "
        f"seq_len={cache.get_seq_length() if hasattr(cache, 'get_seq_length') else 'n/a'} "
        f"layers={len(cache.layers) if hasattr(cache, 'layers') else 'n/a'}",
        file=sys.stderr,
        flush=True,
    )
    if hasattr(cache, "layers"):
        for li, layer in enumerate(cache.layers):
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is not None and v is not None:
                print(
                    f"[DIAG] cache_layer_{li}: K_nan={torch.isnan(k).any().item()} "
                    f"V_nan={torch.isnan(v).any().item()} K_shape={tuple(k.shape)}",
                    file=sys.stderr,
                    flush=True,
                )
                break  # one layer is enough to spot corruption

    next_id = o0.logits[:, -1, :].argmax(-1, keepdim=True)
    attn_mask = torch.ones((1, tokens.shape[1] + 1), dtype=torch.long)
    pos_ids = torch.tensor([[tokens.shape[1]]], dtype=torch.long)

    # Variant A: bridge-fix kwargs (mask + position_ids + cache)
    with torch.no_grad():
        oA = bridge.original_model(
            next_id,
            past_key_values=o0.past_key_values,
            use_cache=True,
            attention_mask=attn_mask,
            position_ids=pos_ids,
        )
    _diag("step1_with_mask_and_pos", oA.logits)

    # Variant B: no cache — feed full 6-token sequence fresh
    full_tokens = torch.cat([tokens, next_id], dim=1)
    with torch.no_grad():
        oB = bridge.original_model(full_tokens)
    _diag("step1_full_no_cache", oB.logits)
    # === END TEMP DEBUG ===

    output = bridge.generate(
        tokens,
        max_new_tokens=3,
        stop_at_eos=False,
        use_past_kv_cache=True,
        return_type="tokens",
        verbose=False,
    )
    assert output.shape == (1, 8), f"Expected shape (1, 8), got {output.shape}"


def test_generate_without_tokenizer_stop_at_eos_false_no_kv_cache(tokenizer_free_bridge):
    """generate() with no tokenizer, stop_at_eos=False, use_past_kv_cache=False."""
    bridge = tokenizer_free_bridge
    assert bridge.tokenizer is None

    tokens = _PROMPT_TOKENS.clone()
    output = bridge.generate(
        tokens,
        max_new_tokens=3,
        stop_at_eos=False,
        use_past_kv_cache=False,
        return_type="tokens",
        verbose=False,
    )
    assert output.shape == (1, 8), f"Expected shape (1, 8), got {output.shape}"


def test_generate_without_tokenizer_explicit_eos_kv_cache(tokenizer_free_bridge):
    """generate() with no tokenizer, explicit eos_token_id, use_past_kv_cache=True.

    Uses a high-valued eos_token_id unlikely to be sampled from zero input so
    generation runs to ``max_new_tokens`` and we can assert exact output shape.
    """
    bridge = tokenizer_free_bridge
    assert bridge.tokenizer is None

    tokens = _PROMPT_TOKENS.clone()
    output = bridge.generate(
        tokens,
        max_new_tokens=3,
        stop_at_eos=True,
        eos_token_id=50256,
        do_sample=False,
        use_past_kv_cache=True,
        return_type="tokens",
        verbose=False,
    )
    assert output.shape == (1, 8), f"Expected shape (1, 8), got {output.shape}"


def test_generate_without_tokenizer_explicit_eos_no_kv_cache(tokenizer_free_bridge):
    """generate() with no tokenizer, explicit eos_token_id, use_past_kv_cache=False."""
    bridge = tokenizer_free_bridge
    assert bridge.tokenizer is None

    tokens = _PROMPT_TOKENS.clone()
    output = bridge.generate(
        tokens,
        max_new_tokens=3,
        stop_at_eos=True,
        eos_token_id=50256,
        do_sample=False,
        use_past_kv_cache=False,
        return_type="tokens",
        verbose=False,
    )
    assert output.shape == (1, 8), f"Expected shape (1, 8), got {output.shape}"


def test_generate_without_tokenizer_stop_at_eos_requires_eos_id(tokenizer_free_bridge):
    """generate() must still error when stop_at_eos=True, no eos_token_id, no tokenizer."""
    bridge = tokenizer_free_bridge
    assert bridge.tokenizer is None

    tokens = _PROMPT_TOKENS.clone()
    with pytest.raises(AssertionError, match="eos_token_id"):
        bridge.generate(
            tokens, max_new_tokens=3, stop_at_eos=True, return_type="tokens", verbose=False
        )


def test_generate_string_input_without_tokenizer_errors(tokenizer_free_bridge):
    """generate() must still error when string input is used without a tokenizer."""
    bridge = tokenizer_free_bridge
    assert bridge.tokenizer is None

    with pytest.raises(AssertionError, match="to_tokens without a tokenizer"):
        bridge.generate("hello", max_new_tokens=3, verbose=False)


def test_generate_return_type_str_without_tokenizer_errors(tokenizer_free_bridge):
    """generate(return_type='str') must error when no tokenizer is set.

    Generation itself succeeds (tensor input, stop_at_eos=False); the assert
    fires only at the decode step, proving the str-decode path is guarded.
    """
    bridge = tokenizer_free_bridge
    assert bridge.tokenizer is None

    tokens = _PROMPT_TOKENS.clone()
    with pytest.raises(AssertionError):
        bridge.generate(
            tokens,
            max_new_tokens=3,
            stop_at_eos=False,
            return_type="str",
            verbose=False,
        )


def test_generate_stream_without_tokenizer_explicit_eos(tokenizer_free_bridge):
    """generate_stream() with no tokenizer; verify all max_new_tokens land in the chunks.

    First chunk contains input + at least one generated token; later chunks contain
    only new tokens. Greedy + high eos_token_id keeps generation from halting early
    so total yielded length is exactly input + max_new_tokens.
    """
    bridge = tokenizer_free_bridge
    assert bridge.tokenizer is None

    input_len = 5
    max_new = 4
    tokens = torch.zeros((1, input_len), dtype=torch.long)
    chunks = list(
        bridge.generate_stream(
            tokens,
            max_new_tokens=max_new,
            max_tokens_per_yield=2,
            stop_at_eos=True,
            eos_token_id=50256,
            do_sample=False,
            use_past_kv_cache=True,
            return_type="tokens",
            verbose=False,
        )
    )
    total_yielded = sum(chunk.shape[-1] for chunk in chunks)
    assert (
        total_yielded == input_len + max_new
    ), f"Expected {input_len + max_new} tokens across chunks, got {total_yielded}"


def test_generate_stream_without_tokenizer_stop_at_eos_requires_eos_id(tokenizer_free_bridge):
    """generate_stream() must error when stop_at_eos=True with no eos_token_id and no tokenizer."""
    bridge = tokenizer_free_bridge
    assert bridge.tokenizer is None

    tokens = _PROMPT_TOKENS.clone()
    with pytest.raises(AssertionError, match="eos_token_id"):
        # Generator is lazy — must consume to trigger the assert.
        list(
            bridge.generate_stream(
                tokens,
                max_new_tokens=3,
                stop_at_eos=True,
                return_type="tokens",
                verbose=False,
            )
        )
