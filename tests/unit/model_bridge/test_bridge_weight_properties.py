"""Tests for TransformerBridge._stack_block_params() and weight properties."""

import pytest
import torch

from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge


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


class TestStackBlockParams:
    """Tests for TransformerBridge._stack_block_params() via a duck-typed self."""

    class _Attn(torch.nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.b_Q = bias

    class _Block(torch.nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.attn = TestStackBlockParams._Attn(bias)

    class _FakeBridge:
        def __init__(self, biases):
            self.blocks = [TestStackBlockParams._Block(b) for b in biases]

            class Cfg:
                n_devices = 1
                device = None

            self.cfg = Cfg()

    def test_stacks_present_params(self):
        fake = self._FakeBridge([torch.ones(2, 3), torch.zeros(2, 3)])
        stacked = TransformerBridge._stack_block_params(fake, "attn.b_Q")
        assert stacked.shape == (2, 2, 3)
        assert torch.equal(stacked[0], torch.ones(2, 3))

    def test_none_param_raises_actionable_error(self):
        """Bias-free checkpoints must get a clear error, not a raw stack TypeError."""
        fake = self._FakeBridge([torch.ones(2, 3), None])
        with pytest.raises(AttributeError, match=r"blocks\[1\]\.attn\.b_Q.*bias-free"):
            TransformerBridge._stack_block_params(fake, "attn.b_Q")
