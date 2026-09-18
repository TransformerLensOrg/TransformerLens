"""Regression coverage for `TransformerBridge.run_with_cache(pos_slice=...)`.

`HookedRootModule.run_with_cache` trims the position axis of every cached tensor; the
bridge used to drop the kwarg into the HuggingFace forward, which discards it, so callers
got full-length activations back. The position axis is dim 1 for everything the bridge
caches — including the head-split `attn.hook_z` / `attn.q.hook_out` family, which is
[batch, pos, head, d_head] even outside compatibility mode — except attention maps
([batch, head, dest, src]), where the query position sits at -2.
"""

import pytest
import torch

PROMPT = "The capital of France is Paris."

# HookedTransformer alias names and bridge-native names, both resolvable on the
# non-compat bridge. Mixes 3-D and head-split 4-D tensors plus an attention map.
NAMES = [
    "blocks.0.attn.hook_pattern",
    "blocks.1.attn.hook_z",
    "blocks.2.hook_resid_post",
    "blocks.1.attn.q.hook_out",
    "blocks.1.mlp.hook_out",
    "blocks.1.hook_out",
]


@pytest.fixture()
def bridge(distilgpt2_bridge):
    """Alias the session fixture for concise test signatures."""
    return distilgpt2_bridge


def _pos_dim(name: str) -> int:
    """Axis `pos_slice` trims for `name`: the query position for attention maps, else dim 1."""
    return -2 if name.endswith(("hook_pattern", "hook_attn_scores")) else 1


def _index_select(tensor: torch.Tensor, name: str, positions: list[int]) -> torch.Tensor:
    dim = _pos_dim(name)
    return tensor.index_select(dim, torch.tensor(positions))


@pytest.mark.parametrize(
    "pos_slice,positions", [(3, [3]), ((1, 4), [1, 2, 3]), ([0, 2, 5], [0, 2, 5])]
)
def test_pos_slice_trims_the_position_axis(bridge, pos_slice, positions):
    """Cached tensors keep exactly the selected positions, taken from the right axis.

    Compares against an independent `index_select` of the unsliced run rather than just the
    resulting shape, so an off-by-one or a slice of a same-length neighbouring axis fails.
    """
    tokens = bridge.to_tokens(PROMPT)
    _, full = bridge.run_with_cache(tokens, return_type="loss", names_filter=NAMES)
    _, sliced = bridge.run_with_cache(
        tokens, return_type="loss", names_filter=NAMES, pos_slice=pos_slice
    )

    for name in NAMES:
        expected = _index_select(full[name], name, positions)
        assert sliced[name].shape == expected.shape, f"{name} sliced on the wrong axis"
        torch.testing.assert_close(sliced[name], expected)


def test_int_slice_matches_full_cache_for_every_activation(bridge):
    """Sweep the whole cache, not a curated name list, so no activation is left unsliced.

    An int slice keeps the position axis at size 1. Entries whose candidate axis is not the
    sequence length have no position axis to trim and are out of scope.
    """
    tokens = bridge.to_tokens(PROMPT)
    n_pos = tokens.shape[1]
    _, full = bridge.run_with_cache(tokens)

    for i in (0, n_pos // 2, n_pos - 1, -1):
        _, sliced = bridge.run_with_cache(tokens, pos_slice=i)
        for name, tensor in full.items():
            if not isinstance(tensor, torch.Tensor) or tensor.dim() < 2:
                continue
            dim = _pos_dim(name)
            abs_dim = dim if dim >= 0 else tensor.dim() + dim
            if tensor.shape[abs_dim] != n_pos:
                continue
            expected = _index_select(tensor, name, [i % n_pos])
            assert sliced[name].shape == expected.shape, name
            torch.testing.assert_close(sliced[name], expected, rtol=1e-4, atol=1e-5)


def test_per_head_and_pattern_shapes(bridge):
    """Pin the absolute rank and axis order an int slice leaves behind."""
    tokens = bridge.to_tokens(PROMPT)
    n_pos = tokens.shape[1]
    _, sliced = bridge.run_with_cache(tokens, pos_slice=2)
    n_heads, d_head = bridge.cfg.n_heads, bridge.cfg.d_head

    # per-head projection: [batch, pos, head, d_head] -> pos trimmed to 1
    assert sliced["blocks.0.attn.q.hook_out"].shape == (1, 1, n_heads, d_head)
    # attention pattern: [batch, head, dest, src] -> dest trimmed to 1, src untouched
    assert sliced["blocks.0.attn.hook_pattern"].shape == (1, n_heads, 1, n_pos)


def test_pos_slice_trims_gradients_on_the_same_axis(bridge):
    """`incl_bwd` gradients are sliced like the activations they are named after.

    The ground truth is the unsliced backward run: shape-only checks would still pass if a
    `_grad` entry held a neighbouring layer's gradient, or grad_input where grad_output was
    meant. `pos_slice` trims at cache time, so the two backward passes are the same graph.
    """
    tokens = bridge.to_tokens(PROMPT)
    positions = [1, 2, 3]
    _, full = bridge.run_with_cache(tokens, return_type="loss", names_filter=NAMES, incl_bwd=True)
    _, sliced = bridge.run_with_cache(
        tokens, return_type="loss", names_filter=NAMES, pos_slice=(1, 4), incl_bwd=True
    )

    for name in NAMES:
        grad_name = f"{name}_grad"
        assert grad_name in sliced, f"missing {grad_name}; cached: {sorted(sliced.keys())}"
        assert sliced[grad_name].shape == sliced[name].shape, grad_name
        expected = _index_select(full[grad_name], name, positions)
        assert sliced[grad_name].shape == expected.shape, f"{grad_name} sliced on the wrong axis"
        torch.testing.assert_close(sliced[grad_name], expected, rtol=1e-3, atol=1e-5)


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


def test_none_pos_slice_is_unchanged(bridge):
    """`pos_slice=None` is the default path — nothing is trimmed."""
    tokens = bridge.to_tokens(PROMPT)
    _, full = bridge.run_with_cache(tokens, return_type="loss", names_filter=NAMES)
    _, no_slice = bridge.run_with_cache(
        tokens, return_type="loss", names_filter=NAMES, pos_slice=None
    )

    for name in NAMES:
        assert no_slice[name].shape == full[name].shape, name
        torch.testing.assert_close(no_slice[name], full[name])
