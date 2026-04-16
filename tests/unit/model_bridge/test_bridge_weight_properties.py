"""Tests for TransformerBridge._stack_block_params() and weight properties."""

import torch

from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)


class TestReshapeBias:
    """Tests for AttentionBridge._reshape_bias()."""

    def _make_bridge(self, n_heads, n_key_value_heads=None):
        bridge = AttentionBridge.__new__(AttentionBridge)

        class Cfg:
            pass

        cfg = Cfg()
        cfg.n_heads = n_heads
        if n_key_value_heads is not None:
            cfg.n_key_value_heads = n_key_value_heads
        bridge.config = cfg
        return bridge

    def test_reshapes_1d_bias_to_heads(self):
        bridge = self._make_bridge(n_heads=2)
        bias = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = bridge._reshape_bias(bias)
        assert result.shape == (2, 3)
        assert torch.equal(result[0], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(result[1], torch.tensor([4.0, 5.0, 6.0]))

    def test_none_bias_returns_none(self):
        bridge = self._make_bridge(n_heads=4)
        result = bridge._reshape_bias(None)
        assert result is None

    def test_use_kv_uses_kv_heads(self):
        bridge = self._make_bridge(n_heads=8, n_key_value_heads=2)
        bias = torch.randn(32)  # 2 * 16
        result = bridge._reshape_bias(bias, use_kv=True)
        assert result.shape[0] == 2  # n_kv_heads, not 8

    def test_already_2d_bias_returned_as_is(self):
        bridge = self._make_bridge(n_heads=4)
        bias = torch.randn(4, 16)  # already 2D
        result = bridge._reshape_bias(bias)
        # ndim != 1, so no reshape — returns as-is
        assert result is bias
