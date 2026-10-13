"""Embedding contribution hooks preserve the outer model's runtime scale."""

import pytest
import torch

from transformer_lens.model_bridge.generalized_components.embedding import (
    EmbeddingBridge,
)


@pytest.mark.parametrize("scale", [0.0, 0.22, 1.0, 12.0])
def test_scaled_embedding_observation_preserves_raw_output(scale):
    embedding = torch.nn.Embedding(8, 4)
    bridge = EmbeddingBridge(name="embed", output_contribution_scale=scale)
    bridge.set_original_component(embedding)
    tokens = torch.tensor([[1, 2]])
    expected = embedding(tokens)
    torch.testing.assert_close(bridge(tokens), expected, rtol=0, atol=0)
    observed = []
    bridge.hook_out.add_hook(lambda value, hook: observed.append(value.detach().clone()))
    torch.testing.assert_close(bridge(tokens), expected, rtol=0, atol=0)
    torch.testing.assert_close(observed[0], expected * scale, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0.22, 1.0, 12.0])
@pytest.mark.parametrize("in_place", [False, True])
def test_scaled_embedding_write_is_not_scaled_twice(scale, in_place):
    bridge = EmbeddingBridge(name="embed", output_contribution_scale=scale)
    bridge.set_original_component(torch.nn.Embedding(8, 4))
    replacement = torch.randn(1, 2, 4)

    def write(value, hook):
        return value.copy_(replacement) if in_place else replacement

    bridge.hook_out.add_hook(write)
    actual = bridge(torch.tensor([[1, 2]])) * scale
    torch.testing.assert_close(actual, replacement)


def test_zero_scale_rejects_unrealizable_edit():
    bridge = EmbeddingBridge(name="embed", output_contribution_scale=0.0)
    bridge.set_original_component(torch.nn.Embedding(8, 4))
    bridge.hook_out.add_hook(lambda value, hook: torch.ones_like(value))
    with pytest.raises(ValueError, match="zero runtime scale"):
        bridge(torch.tensor([[1, 2]]))
