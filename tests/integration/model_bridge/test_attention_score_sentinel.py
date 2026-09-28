"""Compatibility-mode attention-score sentinel regression coverage."""

import pytest
import torch

SCORES = "blocks.0.attn.hook_attn_scores"
PATTERN = "blocks.0.attn.hook_pattern"
RESID = "blocks.0.hook_in"


def _left_padded_batch(bridge) -> tuple[torch.Tensor, torch.Tensor, int]:
    """An unpadded long prompt beside a left-padded short one, with its mask and pad count."""
    long = bridge.to_tokens("The capital of France is")
    short = bridge.to_tokens("Paris")
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
    return tokens, attention_mask, n_pad


def _resid_grad(
    bridge, tokens: torch.Tensor, attention_mask: torch.Tensor, keep: torch.Tensor
) -> torch.Tensor:
    """Gradient of the kept positions' logsumexp w.r.t. the first block's input.

    Not a plain logit sum: processed weights center the unembed, which leaves
    summed logits nearly independent of the residual stream.
    """
    captured: dict[str, torch.Tensor] = {}

    def capture(tensor: torch.Tensor, hook) -> torch.Tensor:
        # A fresh leaf keeps the shared fixture's parameter grads untouched.
        captured["resid"] = tensor.detach().requires_grad_(True)
        return captured["resid"]

    logits = bridge.run_with_hooks(
        tokens, attention_mask=attention_mask, fwd_hooks=[(RESID, capture)]
    )
    (grad,) = torch.autograd.grad(logits[keep].logsumexp(dim=-1).sum(), captured["resid"])
    return grad


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
    tokens, attention_mask, _ = _left_padded_batch(gpt2_bridge_compat)
    seq_len = tokens.shape[1]

    _, cache = gpt2_bridge_compat.run_with_cache(
        tokens, attention_mask=attention_mask, names_filter=[SCORES, PATTERN]
    )
    scores, pattern = cache[SCORES], cache[PATTERN]
    key_padding = ~attention_mask.bool()[:, None, None, :]
    causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)[None, None]
    masked = (key_padding | causal).expand_as(scores)

    assert torch.isneginf(scores[masked]).all()
    assert torch.isfinite(pattern).all()


@pytest.mark.parametrize("fixture", ["gpt2_bridge_compat", "gpt2_bridge_compat_no_processing"])
def test_gpt2_left_padding_gradients_are_finite_and_padding_invariant(request, fixture) -> None:
    """Fully masked pad queries must not leak NaN through softmax's backward.

    Zeroing the NaN pattern row only fixes the forward; the backward used to
    poison every position of the padded row, real tokens included.
    """
    bridge = request.getfixturevalue(fixture)
    tokens, attention_mask, n_pad = _left_padded_batch(bridge)

    everything = torch.ones_like(attention_mask, dtype=torch.bool)
    assert torch.isfinite(_resid_grad(bridge, tokens, attention_mask, everything)).all()

    # Real tokens never attend to pads, so their gradients must match the unpadded prompt.
    real = attention_mask.bool()
    padded_grad = _resid_grad(bridge, tokens, attention_mask, real)
    short = tokens[1:, n_pad:]
    unpadded_grad = _resid_grad(
        bridge, short, torch.ones_like(short), torch.ones_like(short, dtype=torch.bool)
    )
    assert (padded_grad[1, :n_pad] == 0).all()
    # Different sequence lengths reduce in a different order; the non-compat HF
    # path drifts by the same fp32 accumulation noise.
    torch.testing.assert_close(padded_grad[1:, n_pad:], unpadded_grad, rtol=1e-4, atol=1e-4)


def test_gpt2_mixed_dtype_mask_is_normalized_before_addition(gpt2_bridge_compat) -> None:
    """A lower-precision HF mask sentinel must survive score upcasting."""
    scores = torch.zeros(1, 1, 2, 2, dtype=torch.float32)
    attention_mask = torch.zeros_like(scores, dtype=torch.float16)
    attention_mask[..., 0, 1] = torch.finfo(torch.float16).min

    actual = gpt2_bridge_compat.blocks[0].attn._apply_reconstruct_attention_mask(
        scores, attention_mask, seq_len=2
    )

    assert torch.isneginf(actual[..., 0, 1]).all()
