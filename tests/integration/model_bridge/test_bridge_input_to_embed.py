"""input_to_embed on TransformerBridge — the residual entering block 0.

Bridge analog of HookedTransformer.input_to_embed. The correctness anchor is
that feeding its residual to forward(..., start_at_layer=0) reproduces the full
forward exactly.
"""

import pytest
import torch


@pytest.fixture()
def bridge(distilgpt2_bridge):
    return distilgpt2_bridge


def test_returns_residual_tokens_and_none_shortformer(bridge):
    tokens = torch.randint(0, 100, (2, 7))
    residual, out_tokens, shortformer_pos_embed, attention_mask = bridge.input_to_embed(tokens)

    assert residual.shape == (2, 7, bridge.cfg.d_model)
    assert torch.equal(out_tokens, tokens)
    assert shortformer_pos_embed is None


def test_round_trips_through_start_at_layer_0(bridge):
    tokens = torch.randint(0, 100, (2, 7))
    full = bridge.forward(tokens)

    residual, _, _, _ = bridge.input_to_embed(tokens)
    resumed = bridge.forward(residual, start_at_layer=0)

    assert torch.allclose(resumed, full, atol=1e-4)


def test_residual_matches_resid_pre_of_block_0(bridge):
    tokens = torch.randint(0, 100, (2, 7))
    _, cache = bridge.run_with_cache(tokens)
    residual, _, _, _ = bridge.input_to_embed(tokens)
    assert torch.allclose(residual, cache["blocks.0.hook_in"], atol=1e-4)


def test_string_input(bridge):
    residual, tokens, _, _ = bridge.input_to_embed("Hello, world!")
    assert residual.ndim == 3
    assert residual.shape[0] == 1
    assert residual.shape[-1] == bridge.cfg.d_model
    assert tokens.shape[1] == residual.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
