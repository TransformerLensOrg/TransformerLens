"""run_with_cache(pos_slice=...) on TransformerBridge.

Mirrors HookedTransformer's pos_slice: each cached activation is sliced along
its position dimension (dim 1 for resid/per-head/token-id activations; the
query position -2 for attention patterns/scores). An int slice keeps the
position dim at size 1, matching HookedTransformer.
"""

import pytest
import torch


@pytest.fixture()
def bridge(distilgpt2_bridge):
    return distilgpt2_bridge


def _pos_dim(name: str) -> int:
    return -2 if name.endswith(("hook_pattern", "hook_attn_scores")) else 1


def test_int_slice_keeps_pos_dim_and_matches_full_cache(bridge):
    tokens = torch.randint(0, 100, (1, 8))
    _, full = bridge.run_with_cache(tokens)

    for i in (0, 4, 7, -1):
        _, sliced = bridge.run_with_cache(tokens, pos_slice=i)
        for name, tensor in full.items():
            if not isinstance(tensor, torch.Tensor) or tensor.dim() < 2:
                continue
            dim = _pos_dim(name)
            abs_dim = dim if dim >= 0 else tensor.dim() + dim
            expected = tensor.index_select(abs_dim, torch.tensor([i % tensor.shape[abs_dim]]))
            assert sliced[name].shape == expected.shape, name
            assert torch.allclose(sliced[name], expected, atol=1e-5), name


def test_per_head_and_pattern_shapes(bridge):
    tokens = torch.randint(0, 100, (1, 8))
    _, sliced = bridge.run_with_cache(tokens, pos_slice=2)
    n_heads, d_head = bridge.cfg.n_heads, bridge.cfg.d_head

    # per-head projection: [batch, pos, head, d_head] -> pos sliced to 1
    per_head = sliced["blocks.0.attn.q.hook_out"]
    assert per_head.shape == (1, 1, n_heads, d_head)

    # attention pattern: [batch, head, q_pos, k_pos] -> query pos sliced to 1
    pattern = sliced["blocks.0.attn.hook_pattern"]
    assert pattern.shape == (1, n_heads, 1, 8)


def test_tuple_range_slice(bridge):
    tokens = torch.randint(0, 100, (1, 8))
    _, full = bridge.run_with_cache(tokens)
    _, sliced = bridge.run_with_cache(tokens, pos_slice=(1, 4))

    resid = sliced["blocks.1.hook_out"]
    assert resid.shape == (1, 3, bridge.cfg.d_model)
    assert torch.allclose(resid, full["blocks.1.hook_out"][:, 1:4], atol=1e-5)


def test_none_pos_slice_is_unchanged(bridge):
    tokens = torch.randint(0, 100, (1, 8))
    _, full = bridge.run_with_cache(tokens)
    _, no_slice = bridge.run_with_cache(tokens, pos_slice=None)
    assert full["blocks.0.hook_out"].shape == no_slice["blocks.0.hook_out"].shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
