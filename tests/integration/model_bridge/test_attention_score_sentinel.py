"""Compatibility-mode attention-score sentinel regression coverage."""

import torch

SCORES = "blocks.0.attn.hook_attn_scores"
PATTERN = "blocks.0.attn.hook_pattern"


def test_gpt2_compatibility_scores_use_negative_infinity(
    gpt2_bridge_compat, gpt2_goldens_processed
) -> None:
    """GPT-2's direct HF mask is normalized before the compatibility hook.

    Anchored on the frozen HookedTransformer goldens rather than a live
    HookedTransformer, matching the rest of the compatibility suite.
    """
    golden = gpt2_goldens_processed
    tokens = golden.scalars["short_prompt"]
    _, bridge_cache = gpt2_bridge_compat.run_with_cache(tokens, names_filter=[SCORES])
    hooked_cache = golden.tensors("activations")

    bridge_scores, hooked_scores = bridge_cache[SCORES], hooked_cache[SCORES]
    causal_mask = torch.isneginf(hooked_scores)
    assert causal_mask.any()
    assert torch.isneginf(bridge_scores[causal_mask]).all()
    # The goldens were captured on different hardware, so the unmasked scores
    # agree to fp32 accumulation noise rather than bit-exactly. Same tolerance
    # the sibling golden comparison uses for this hook.
    torch.testing.assert_close(
        bridge_scores[~causal_mask], hooked_scores[~causal_mask], rtol=1e-4, atol=1e-4
    )


def test_gpt2_left_padding_uses_negative_infinity_and_finite_patterns(
    gpt2_bridge_compat,
) -> None:
    """Fully masked pad queries are zeroed after softmax in compatibility mode."""
    long = gpt2_bridge_compat.to_tokens("The capital of France is")
    short = gpt2_bridge_compat.to_tokens("Paris")
    n_pad = long.shape[1] - short.shape[1]
    padded_short = torch.cat([torch.zeros_like(long[:, :n_pad]), short], dim=1)
    tokens = torch.cat([long, padded_short], dim=0)
    attention_mask = torch.cat(
        [
            torch.ones_like(long),
            torch.cat([torch.zeros_like(long[:, :n_pad]), torch.ones_like(short)], dim=1),
        ],
        dim=0,
    )

    _, cache = gpt2_bridge_compat.run_with_cache(
        tokens, attention_mask=attention_mask, names_filter=[SCORES, PATTERN]
    )
    scores, pattern = cache[SCORES], cache[PATTERN]
    key_padding = ~attention_mask.bool()[:, None, None, :]
    causal = torch.triu(torch.ones(long.shape[1], long.shape[1], dtype=torch.bool), diagonal=1)[
        None, None
    ]
    masked = (key_padding | causal).expand_as(scores)

    assert torch.isneginf(scores[masked]).all()
    assert torch.isfinite(pattern).all()


def test_gpt2_mixed_dtype_mask_is_normalized_before_addition(gpt2_bridge_compat) -> None:
    """A lower-precision HF mask sentinel must survive score upcasting."""
    scores = torch.zeros(1, 1, 2, 2, dtype=torch.float32)
    attention_mask = torch.zeros_like(scores, dtype=torch.float16)
    attention_mask[..., 0, 1] = torch.finfo(torch.float16).min

    actual = gpt2_bridge_compat.blocks[0].attn._apply_reconstruct_attention_mask(
        scores, attention_mask, seq_len=2
    )

    assert torch.isneginf(actual[..., 0, 1]).all()
