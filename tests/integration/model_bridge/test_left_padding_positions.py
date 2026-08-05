"""Regression tests for left-padding position handling in TransformerBridge.

A causal LM's logits at a sequence's real token positions must not depend on how
that sequence is padded, provided the caller supplies the matching attention_mask.
Left padding shifts every real token's absolute position, so position_ids have to
be derived from the mask; without that the bridge silently returns wrong logits
and a wrong loss. See #1609.

Right padding is included as a control: causality already protects it, so it was
never affected and must stay that way.
"""

from __future__ import annotations

import pytest
import torch

from transformer_lens import utilities as utils

PAD_ID = 0


def _pad(tokens: torch.Tensor, n_pad: int, side: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad `tokens` on `side`, returning (padded_tokens, attention_mask)."""
    pads = torch.full((tokens.shape[0], n_pad), PAD_ID, dtype=tokens.dtype)
    ones = torch.ones(tokens.shape, dtype=torch.long)
    zeros = torch.zeros((tokens.shape[0], n_pad), dtype=torch.long)
    if side == "left":
        return torch.cat([pads, tokens], dim=1), torch.cat([zeros, ones], dim=1)
    return torch.cat([tokens, pads], dim=1), torch.cat([ones, zeros], dim=1)


@pytest.fixture(scope="module")
def tokens(distilgpt2_bridge) -> torch.Tensor:
    return distilgpt2_bridge.to_tokens("The capital of France is")


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("n_pad", [1, 3, 5])
def test_logits_are_invariant_to_padding(distilgpt2_bridge, tokens, side, n_pad) -> None:
    """Padding must not change the logits at a sequence's real positions."""
    padded, mask = _pad(tokens, n_pad, side)
    real = slice(n_pad, None) if side == "left" else slice(None, tokens.shape[1])

    with torch.no_grad():
        baseline = distilgpt2_bridge(tokens, return_type="logits")
        actual = distilgpt2_bridge(padded, attention_mask=mask, return_type="logits")[:, real]

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, baseline, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("side", ["left", "right"])
def test_compat_mode_logits_are_invariant_to_padding(distilgpt2_bridge_compat, side: str) -> None:
    """enable_compatibility_mode() promises HookedTransformer-equivalent numerics,
    which this property is part of."""
    tokens = distilgpt2_bridge_compat.to_tokens("The capital of France is")
    n_pad = 3
    padded, mask = _pad(tokens, n_pad, side)
    real = slice(n_pad, None) if side == "left" else slice(None, tokens.shape[1])

    with torch.no_grad():
        baseline = distilgpt2_bridge_compat(tokens, return_type="logits")
        actual = distilgpt2_bridge_compat(padded, attention_mask=mask, return_type="logits")[
            :, real
        ]

    torch.testing.assert_close(actual, baseline, rtol=1e-3, atol=1e-3)


def test_derived_position_ids_match_hooked_transformer(distilgpt2_bridge, tokens) -> None:
    """The derived positions must be the ones HookedTransformer would use, i.e. the
    shared get_offset_position_ids helper rather than a parallel derivation."""
    n_pad = 3
    padded, mask = _pad(tokens, n_pad, "left")
    expected = utils.get_offset_position_ids(0, mask)

    with torch.no_grad():
        derived = distilgpt2_bridge(padded, attention_mask=mask, return_type="logits")
        supplied = distilgpt2_bridge(
            padded, attention_mask=mask, position_ids=expected, return_type="logits"
        )

    torch.testing.assert_close(derived, supplied, rtol=1e-5, atol=1e-5)


def test_explicit_position_ids_take_precedence(distilgpt2_bridge, tokens) -> None:
    """A caller-supplied position_ids must not be overwritten by the derivation."""
    n_pad = 3
    padded, mask = _pad(tokens, n_pad, "left")
    derived_positions = utils.get_offset_position_ids(0, mask)
    shifted = derived_positions + 1  # deliberately different, but still in range

    with torch.no_grad():
        default = distilgpt2_bridge(padded, attention_mask=mask, return_type="logits")
        overridden = distilgpt2_bridge(
            padded, attention_mask=mask, position_ids=shifted, return_type="logits"
        )

    assert not torch.allclose(default, overridden, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("gap", [slice(3, 5), slice(1, 2)])
def test_interior_mask_gap_uses_derived_positions(distilgpt2_bridge, tokens, gap) -> None:
    """A mask gap that is not leading padding still shifts later positions, so the
    derivation must fire for any mask, not only ones starting with a pad."""
    mask = torch.ones(tokens.shape, dtype=torch.long)
    mask[0, gap] = 0
    gapped = tokens.clone()
    gapped[0, gap] = PAD_ID
    expected = utils.get_offset_position_ids(0, mask)

    with torch.no_grad():
        derived = distilgpt2_bridge(gapped, attention_mask=mask, return_type="logits")
        supplied = distilgpt2_bridge(
            gapped, attention_mask=mask, position_ids=expected, return_type="logits"
        )

    torch.testing.assert_close(derived, supplied, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mask_kind", ["all_ones", "right_padded"])
def test_no_position_ids_injected_when_unnecessary(distilgpt2_bridge, tokens, mask_kind) -> None:
    """Masks whose attended tokens already sit at their default positions must not
    get position_ids injected: it is a no-op at best, and models whose forward does
    not accept position_ids would raise.
    """
    if mask_kind == "all_ones":
        passed, mask = tokens, torch.ones(tokens.shape, dtype=torch.long)
    else:
        passed, mask = _pad(tokens, 3, "right")

    seen: dict[str, object] = {}
    original = distilgpt2_bridge.original_model.forward

    def _spy(*args, **kwargs):
        seen["position_ids"] = kwargs.get("position_ids")
        return original(*args, **kwargs)

    distilgpt2_bridge.original_model.forward = _spy
    try:
        with torch.no_grad():
            distilgpt2_bridge(passed, attention_mask=mask, return_type="logits")
    finally:
        distilgpt2_bridge.original_model.forward = original

    assert seen["position_ids"] is None


def test_cached_step_with_left_padding(distilgpt2_bridge, tokens) -> None:
    """With a KV cache the mask spans past+new while input_ids is only the new
    token, so the derivation must be offset back to the tokens being passed."""
    n_pad = 3
    padded, mask = _pad(tokens, n_pad, "left")

    with torch.no_grad():
        prefill = distilgpt2_bridge.original_model(padded, attention_mask=mask, use_cache=True)
        new_token = tokens[:, -1:].clone()
        extended = torch.cat([mask, torch.ones(1, 1, dtype=torch.long)], dim=1)
        step = distilgpt2_bridge(
            new_token,
            attention_mask=extended,
            past_key_values=prefill.past_key_values,
            return_type="logits",
        )

    assert step.shape[:2] == (1, 1)
    assert torch.isfinite(step).all()
