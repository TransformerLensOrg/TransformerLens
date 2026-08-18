"""Regression coverage for `TransformerBridge.run_with_cache(pos_slice=...)`.

`HookedRootModule.run_with_cache` slices the position axis of every cached tensor; the
bridge used to drop the kwarg into the HuggingFace forward, which discards it, so callers
got full-length activations back. The position axis sits two from the end for head-split
tensors ([batch, pos, head, d_head]) and one from the end for everything else — including
the bridge-native `attn.q.hook_out` family, which is head-split even outside compat mode.
"""

import pytest
import torch

PROMPT = "The capital of France is Paris."
HT_NAMES = ["blocks.0.attn.hook_pattern", "blocks.1.attn.hook_z", "blocks.2.hook_resid_post"]
NATIVE_NAMES = ["blocks.1.attn.q.hook_out", "blocks.1.mlp.hook_out"]


@pytest.fixture()
def bridge(distilgpt2_bridge):
    """Alias the session fixture for concise test signatures."""
    return distilgpt2_bridge


@pytest.mark.parametrize(
    "pos_slice,positions", [(3, [3]), ((1, 4), [1, 2, 3]), ([0, 2, 5], [0, 2, 5])]
)
def test_pos_slice_trims_the_position_axis(bridge, pos_slice, positions):
    """Cached tensors keep exactly the selected positions, taken from the right axis.

    Compares against an independent `index_select` of the unsliced run rather than just the
    resulting shape, so an off-by-one or a slice of a same-length neighbouring axis fails.
    """
    tokens = bridge.to_tokens(PROMPT)
    _, full = bridge.run_with_cache(
        tokens, return_type="loss", names_filter=HT_NAMES + NATIVE_NAMES
    )
    _, sliced = bridge.run_with_cache(
        tokens, return_type="loss", names_filter=HT_NAMES + NATIVE_NAMES, pos_slice=pos_slice
    )

    # [batch, pos, head, d_head] for head-split tensors, [batch, head, dest, src] for patterns.
    expected_axis = {
        "blocks.0.attn.hook_pattern": -2,
        "blocks.1.attn.hook_z": -3,
        "blocks.2.hook_resid_post": -2,
        "blocks.1.attn.q.hook_out": -3,
        "blocks.1.mlp.hook_out": -2,
    }
    index = torch.tensor(positions)
    for name, axis in expected_axis.items():
        expected = full[name].index_select(axis, index)
        assert sliced[name].shape == expected.shape, f"{name} sliced on the wrong axis"
        torch.testing.assert_close(sliced[name], expected)


def test_pos_slice_matches_hooked_transformer(
    distilgpt2_bridge_compat, distilgpt2_hooked_processed
):
    """Slicing agrees with HookedRootModule's — activations and gradients, values included.

    HookedRootModule is the reference implementation for both `pos_slice` and `incl_bwd`, so
    comparing values here is what pins the cache to the *right* tensor: shape and finiteness
    checks would still pass if a hook cached a neighbouring layer's activation, or grad_input
    where grad_output was meant.
    """
    tokens = distilgpt2_hooked_processed.to_tokens(PROMPT)

    _, bridge_cache = distilgpt2_bridge_compat.run_with_cache(
        tokens, return_type="loss", names_filter=HT_NAMES, pos_slice=(1, 4), incl_bwd=True
    )
    _, hooked_cache = distilgpt2_hooked_processed.run_with_cache(
        tokens, return_type="loss", names_filter=HT_NAMES, pos_slice=(1, 4), incl_bwd=True
    )

    for name in HT_NAMES + [f"{n}_grad" for n in HT_NAMES]:
        assert bridge_cache[name].shape == hooked_cache[name].shape, name
        torch.testing.assert_close(bridge_cache[name], hooked_cache[name], rtol=1e-3, atol=1e-5)


def test_pos_slice_applies_to_gradients(distilgpt2_bridge_compat, distilgpt2_hooked_processed):
    """Sliced `_grad` entries hold the gradient of the tensor they are named after.

    Covers the bridge-native head-split keys, which have no HookedTransformer counterpart;
    their gradients are checked against the HookedTransformer names that alias them. Runs on
    the compat-mode bridge, whose numerics match the processed HookedTransformer.
    """
    tokens = distilgpt2_bridge_compat.to_tokens(PROMPT)
    _, cache = distilgpt2_bridge_compat.run_with_cache(
        tokens,
        return_type="loss",
        names_filter=HT_NAMES + NATIVE_NAMES,
        pos_slice=(1, 4),
        incl_bwd=True,
    )
    _, hooked_cache = distilgpt2_hooked_processed.run_with_cache(
        tokens,
        return_type="loss",
        names_filter=["blocks.1.attn.hook_q", "blocks.1.hook_mlp_out"],
        pos_slice=(1, 4),
        incl_bwd=True,
    )

    for name in HT_NAMES + NATIVE_NAMES:
        assert cache[f"{name}_grad"].shape == cache[name].shape, name

    # blocks.1.attn.q.hook_out is HookedTransformer's blocks.1.attn.hook_q; mlp.hook_out is
    # its blocks.1.hook_mlp_out.
    for native_name, hooked_name in (
        ("blocks.1.attn.q.hook_out", "blocks.1.attn.hook_q"),
        ("blocks.1.mlp.hook_out", "blocks.1.hook_mlp_out"),
    ):
        torch.testing.assert_close(
            cache[f"{native_name}_grad"],
            hooked_cache[f"{hooked_name}_grad"],
            rtol=1e-3,
            atol=1e-5,
        )


def test_pos_slice_on_two_dimensional_activations_keeps_batch(bridge):
    """Token ids are [batch, pos]; the slice must land on pos, not on batch."""
    tokens = bridge.to_tokens(PROMPT)
    names = ["embed.hook_in"]
    _, full = bridge.run_with_cache(tokens, return_type="loss", names_filter=names)
    _, sliced = bridge.run_with_cache(
        tokens, return_type="loss", names_filter=names, pos_slice=(1, 4)
    )

    assert full["embed.hook_in"].shape == tokens.shape
    assert sliced["embed.hook_in"].shape == (tokens.shape[0], 3)
    torch.testing.assert_close(sliced["embed.hook_in"], tokens[:, 1:4])
