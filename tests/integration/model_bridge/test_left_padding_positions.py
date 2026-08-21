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


def _mixed_batch(
    tokens: torch.Tensor, n_pad: int
) -> tuple[torch.Tensor, torch.Tensor, tuple, tuple]:
    """A batch of one right-padded, one left-padded and one unpadded row."""
    width = tokens.shape[1] + n_pad
    right, m_right = _pad(tokens, n_pad, "right")
    left, m_left = _pad(tokens, n_pad, "left")
    plain = torch.arange(20, 20 + width, dtype=tokens.dtype).unsqueeze(0)
    m_plain = torch.ones(1, width, dtype=torch.long)
    batch = torch.cat([right, left, plain], dim=0)
    mask = torch.cat([m_right, m_left, m_plain], dim=0)
    return batch, mask, (right, m_right), (plain, m_plain)


def _spy_on_position_ids(bridge, tokens_in: torch.Tensor, mask: torch.Tensor):
    """Run a forward, returning the position_ids the wrapped model actually saw."""
    seen: dict[str, object] = {}
    original = bridge.original_model.forward

    def _spy(*args, **kwargs):
        seen["position_ids"] = kwargs.get("position_ids")
        return original(*args, **kwargs)

    bridge.original_model.forward = _spy
    try:
        with torch.no_grad():
            bridge(tokens_in, attention_mask=mask, return_type="logits")
    finally:
        bridge.original_model.forward = original
    return seen["position_ids"]


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

    assert _spy_on_position_ids(distilgpt2_bridge, passed, mask) is None


def test_unshifted_rows_keep_default_positions(distilgpt2_bridge, tokens) -> None:
    """The decision is per row, not per batch: only rows whose mask moves an
    attended token get derived positions, so the others stay on plain arange."""
    n_pad = 3
    batch, mask, _, _ = _mixed_batch(tokens, n_pad)
    seen = _spy_on_position_ids(distilgpt2_bridge, batch, mask)

    assert seen is not None
    arange = torch.arange(batch.shape[1])
    torch.testing.assert_close(seen[0], arange)  # right-padded
    torch.testing.assert_close(seen[2], arange)  # unpadded
    torch.testing.assert_close(seen[1], utils.get_offset_position_ids(0, mask)[1])  # left-padded


def test_one_left_padded_row_does_not_perturb_its_neighbours(distilgpt2_bridge, tokens) -> None:
    """Derived positions for one row must not change its unshifted neighbours."""
    n_pad = 3
    batch, mask, (right, m_right), (plain, m_plain) = _mixed_batch(tokens, n_pad)
    control_batch = torch.cat([right, right, plain], dim=0)
    control_mask = torch.cat([m_right, m_right, m_plain], dim=0)

    with torch.no_grad():
        mixed = distilgpt2_bridge(batch, attention_mask=mask, return_type="logits")
        control = distilgpt2_bridge(
            control_batch, attention_mask=control_mask, return_type="logits"
        )
        unpadded = distilgpt2_bridge(tokens, return_type="logits")

    # Matching batch shapes isolate derived-position handling from BLAS kernel
    # changes caused by comparing batched and single-row matrix multiplications.
    torch.testing.assert_close(mixed[0:1], control[0:1], rtol=0, atol=1e-6)
    torch.testing.assert_close(mixed[2:3], control[2:3], rtol=0, atol=1e-6)
    # ...while the row that did need correcting still gets it.
    torch.testing.assert_close(mixed[1:2, n_pad:], unpadded, rtol=1e-3, atol=1e-3)


def test_float_attention_mask_is_accepted(distilgpt2_bridge, tokens) -> None:
    """Derived positions index an embedding table, so a float 0/1 mask must not
    produce float position_ids."""
    n_pad = 3
    padded, mask = _pad(tokens, n_pad, "left")

    with torch.no_grad():
        baseline = distilgpt2_bridge(tokens, return_type="logits")
        actual = distilgpt2_bridge(padded, attention_mask=mask.float(), return_type="logits")

    torch.testing.assert_close(actual[:, n_pad:], baseline, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "mask_row",
    [
        pytest.param([0, 0, 0, 0, 0, 0], id="all_masked"),
        pytest.param([0, 0, 0, 0, 0, 1], id="single_real_token"),
        pytest.param([1, 0, 1, 0, 1, 1], id="two_interior_gaps"),
        pytest.param([0, 1, 1, 1, 1, 0], id="padded_both_ends"),
    ],
)
def test_degenerate_masks_do_not_crash(distilgpt2_bridge, tokens, mask_row) -> None:
    """Shapes the happy path never reaches. An all-masked row in particular must
    not inject anything: every position is a pad, so nothing is displaced."""
    mask = torch.tensor([mask_row[: tokens.shape[1]]], dtype=torch.long)
    positions = _spy_on_position_ids(distilgpt2_bridge, tokens, mask)

    if mask.sum() == 0:
        assert positions is None
    else:
        expected = utils.get_offset_position_ids(0, mask)
        torch.testing.assert_close(positions, expected)


def test_four_dimensional_mask_is_left_alone(distilgpt2_bridge, tokens) -> None:
    """A 4-D mask is an additive attention bias, not a 0/1 padding mask, so the
    cumsum derivation is meaningless on it."""
    seq = tokens.shape[1]
    mask = torch.ones(1, 1, seq, seq)
    assert _spy_on_position_ids(distilgpt2_bridge, tokens, mask) is None


def test_inputs_embeds_are_left_alone(distilgpt2_bridge, tokens) -> None:
    """Float input is pre-computed embeddings; there are no token positions to
    derive and the batch/seq layout is not guaranteed to match the mask."""
    embeds = distilgpt2_bridge.original_model.get_input_embeddings()(tokens)
    _, mask = _pad(tokens[:, :-2], 2, "left")
    assert _spy_on_position_ids(distilgpt2_bridge, embeds, mask) is None


def test_gradients_flow_through_a_left_padded_forward(distilgpt2_bridge, tokens) -> None:
    """The derivation must not detach the graph or poison the loss with pads."""
    n_pad = 3
    padded, mask = _pad(tokens, n_pad, "left")

    loss = distilgpt2_bridge(padded, attention_mask=mask, return_type="loss")
    loss.backward()
    grad = distilgpt2_bridge.original_model.get_input_embeddings().weight.grad
    try:
        assert torch.isfinite(loss)
        assert grad is not None and torch.isfinite(grad).all() and (grad != 0).any()
    finally:
        distilgpt2_bridge.zero_grad(set_to_none=True)


def test_run_with_cache_matches_forward_under_left_padding(distilgpt2_bridge, tokens) -> None:
    """run_with_cache routes through a different kwarg-filtering path, so the
    injection has to survive it identically."""
    n_pad = 3
    padded, mask = _pad(tokens, n_pad, "left")

    with torch.no_grad():
        direct = distilgpt2_bridge(padded, attention_mask=mask, return_type="logits")
        cached, activations = distilgpt2_bridge.run_with_cache(padded, attention_mask=mask)

    torch.testing.assert_close(direct, cached, rtol=0, atol=1e-6)
    assert len(activations) > 0


@pytest.fixture(scope="module")
def opt_bridge():
    """OPT is the one supported architecture whose positional embedding consumes
    the attention mask, so it must be left to derive positions for itself."""
    from transformer_lens.model_bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers(
        "hf-internal-testing/tiny-random-OPTForCausalLM", device="cpu", dtype=torch.float32
    )
    bridge.eval()
    return bridge


def test_self_deriving_model_is_left_alone(opt_bridge) -> None:
    """OPTLearnedPositionalEmbedding derives from the mask already, and uses its
    own convention (-1) for padded slots. Overriding it buys no correctness and
    silently changes the padded slots, so the bridge must stay out of the way.
    """
    ids = torch.arange(20, 26).unsqueeze(0)
    n_pad = 3
    padded, mask = _pad(ids, n_pad, "left")

    assert opt_bridge._accepts_derived_position_ids() is False
    assert _spy_on_position_ids(opt_bridge, padded, mask) is None

    with torch.no_grad():
        bridge_out = opt_bridge(padded, attention_mask=mask, return_type="logits")
        hf_out = opt_bridge.original_model(input_ids=padded, attention_mask=mask).logits
        unpadded = opt_bridge(ids, attention_mask=torch.ones_like(ids), return_type="logits")

    # Deferring to OPT keeps the bridge exactly on HF, and OPT's own derivation
    # already delivers the padding-invariance this module is about.
    torch.testing.assert_close(bridge_out, hf_out, rtol=0, atol=1e-6)
    torch.testing.assert_close(bridge_out[:, n_pad:], unpadded, rtol=1e-4, atol=1e-4)


def test_cached_step_with_left_padding(distilgpt2_bridge, tokens) -> None:
    """With a KV cache the mask spans past+new while input_ids is only the new
    token, so the derivation must be offset back to the tokens being passed.

    Prefill goes through the bridge (return_type="logits_and_cache") so the
    cached keys and values are built under the same position convention the step
    uses. Stepping off a cache prefilled by raw HF mixes two conventions and is
    not equivalent to anything.
    """
    n_pad = 3
    padded, mask = _pad(tokens, n_pad, "left")
    new_token = torch.tensor([[318]])
    extended = torch.cat([mask, torch.ones(1, 1, dtype=torch.long)], dim=1)

    with torch.no_grad():
        _, cache = distilgpt2_bridge(padded, attention_mask=mask, return_type="logits_and_cache")
        step = distilgpt2_bridge(
            new_token, attention_mask=extended, past_key_values=cache, return_type="logits"
        )
        # Ground truth: the same prompt and token with no padding at all.
        unpadded = distilgpt2_bridge(torch.cat([tokens, new_token], dim=1), return_type="logits")

    assert step.shape[:2] == (1, 1)
    torch.testing.assert_close(step, unpadded[:, -1:], rtol=1e-3, atol=1e-3)


def test_cached_decoding_with_left_padding_matches_full_recompute(
    distilgpt2_bridge, tokens
) -> None:
    """Several cached steps in a row: each must land on what recomputing the
    whole left-padded sequence would give, or the offset drifts with the cache."""
    n_pad = 3
    padded, mask = _pad(tokens, n_pad, "left")

    with torch.no_grad():
        _, cache = distilgpt2_bridge(padded, attention_mask=mask, return_type="logits_and_cache")
        sequence, grown = padded, mask
        for _ in range(4):
            logits = distilgpt2_bridge(sequence, attention_mask=grown, return_type="logits")
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            sequence = torch.cat([sequence, next_token], dim=1)
            grown = torch.cat([grown, torch.ones(1, 1, dtype=torch.long)], dim=1)

            stepped = distilgpt2_bridge(
                next_token, attention_mask=grown, past_key_values=cache, return_type="logits"
            )
            full = distilgpt2_bridge(sequence, attention_mask=grown, return_type="logits")
            torch.testing.assert_close(stepped, full[:, -1:], rtol=1e-3, atol=1e-3)
