"""Unit tests for KV cache support in attention bridge components.

Tests cover:
- Rectangular causal mask for cached attention (q_seq_len != kv_seq_len)
- _update_kv_cache warning when layer_idx is missing
- layer_idx type coercion (tensor -> int)
- Bloom alibi shape handling with cache
"""

import logging
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
    JointQKVAttentionBridge,
)


class TestConfig:
    n_heads = 2
    d_model = 4
    d_head = 2


class MockOriginalAttention(torch.nn.Module):
    def __init__(self, layer_idx=None):
        super().__init__()
        self.attn_dropout = torch.nn.Identity()
        if layer_idx is not None:
            self.layer_idx = layer_idx

    def forward(self, x):
        return x


def _make_bridge():
    """Create a JointQKVAttentionBridge with mock original component."""
    bridge = JointQKVAttentionBridge(name="qkv", config=TestConfig())
    bridge.add_module("_original_component", MockOriginalAttention())
    return bridge


class TestRectangularCausalMask:
    """Test that causal mask is correctly shaped when q_seq_len != kv_seq_len."""

    def test_square_mask_when_no_cache(self):
        """Without cache, mask is square (q_seq_len == kv_seq_len)."""
        bridge = _make_bridge()
        seq_len = 4
        attn_scores = torch.zeros(1, 2, seq_len, seq_len)

        result = bridge._apply_reconstruct_attention_mask(
            attn_scores=attn_scores,
            attention_mask=None,
            seq_len=seq_len,
        )

        min_val = torch.finfo(result.dtype).min
        # Upper triangle (excluding diagonal) should be masked
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert result[0, 0, i, j] == min_val
                else:
                    assert result[0, 0, i, j] == 0.0

    def test_rectangular_mask_with_cache(self):
        """With cache, mask should be q_seq_len x kv_seq_len (rectangular)."""
        bridge = _make_bridge()
        q_seq_len = 1  # Single new token
        kv_seq_len = 5  # 4 cached + 1 new

        attn_scores = torch.zeros(1, 2, q_seq_len, kv_seq_len)

        result = bridge._apply_reconstruct_attention_mask(
            attn_scores=attn_scores,
            attention_mask=None,
            seq_len=kv_seq_len,
            q_seq_len=q_seq_len,
        )

        # Single query token should attend to ALL kv positions (causal allows it)
        assert result.shape == (1, 2, 1, 5)
        # All positions should be unmasked (query at position 4 can see 0..4)
        assert (
            result == 0.0
        ).all(), "Single query at the last position should attend to all KV positions"

    def test_rectangular_mask_multi_query(self):
        """Multiple new query tokens with cached KV positions."""
        bridge = _make_bridge()
        q_seq_len = 3  # 3 new tokens
        kv_seq_len = 7  # 4 cached + 3 new

        attn_scores = torch.zeros(1, 2, q_seq_len, kv_seq_len)

        result = bridge._apply_reconstruct_attention_mask(
            attn_scores=attn_scores,
            attention_mask=None,
            seq_len=kv_seq_len,
            q_seq_len=q_seq_len,
        )

        min_val = torch.finfo(result.dtype).min
        # Query 0 (position 4 in full seq) can attend to kv[0..4], masked at [5,6]
        assert result[0, 0, 0, 4] == 0.0
        assert result[0, 0, 0, 5] == min_val
        assert result[0, 0, 0, 6] == min_val

        # Query 1 (position 5) can attend to kv[0..5], masked at [6]
        assert result[0, 0, 1, 5] == 0.0
        assert result[0, 0, 1, 6] == min_val

        # Query 2 (position 6) can attend to all kv[0..6]
        assert (result[0, 0, 2, :] == 0.0).all()

    def test_4d_mask_bypasses_causal(self):
        """4D HF masks are treated as authoritative — no extra causal mask."""
        bridge = _make_bridge()
        q_seq_len = 1
        kv_seq_len = 5
        attn_scores = torch.zeros(1, 2, q_seq_len, kv_seq_len)

        # 4D mask that allows attending everywhere
        mask_4d = torch.zeros(1, 1, q_seq_len, kv_seq_len)

        result = bridge._apply_reconstruct_attention_mask(
            attn_scores=attn_scores,
            attention_mask=mask_4d,
            seq_len=kv_seq_len,
            q_seq_len=q_seq_len,
        )

        # 4D mask is additive (0 = attend), so result should be all zeros
        assert (result == 0.0).all()

    def test_backward_compat_no_q_seq_len(self):
        """Without q_seq_len, defaults to seq_len (square mask)."""
        bridge = _make_bridge()
        seq_len = 3
        attn_scores = torch.zeros(1, 2, seq_len, seq_len)

        result = bridge._apply_reconstruct_attention_mask(
            attn_scores=attn_scores,
            attention_mask=None,
            seq_len=seq_len,
            # q_seq_len omitted
        )

        min_val = torch.finfo(result.dtype).min
        # Should be a standard lower-triangular causal mask
        assert result[0, 0, 0, 1] == min_val
        assert result[0, 0, 0, 0] == 0.0
        assert result[0, 0, 2, 2] == 0.0

    def test_reconstruct_attention_with_cache(self):
        """Full _reconstruct_attention produces correct output shape with cached KV.

        _reconstruct_attention reshapes Q/K/V internally, so we pass 4D tensors
        in [batch, seq, heads, head_dim] format (pre-transpose).
        When q_seq_len < kv_seq_len (simulating cache), the cache update is
        bypassed (no past_key_values kwarg), so we pre-expand K/V ourselves.
        """
        bridge = _make_bridge()

        batch, q_len, kv_len, heads, head_dim = 1, 1, 5, 2, 2

        # Pass Q with q_len=1 in [batch, seq, heads, head_dim] format.
        # K/V with kv_len=5 to simulate already-extended cache.
        q = torch.randn(batch, q_len, heads, head_dim)
        k = torch.randn(batch, kv_len, heads, head_dim)
        v = torch.randn(batch, kv_len, heads, head_dim)

        output, pattern = bridge._reconstruct_attention(q, k, v)

        assert output.shape == (batch, q_len, heads * head_dim)
        assert pattern.shape == (batch, heads, q_len, kv_len)
        # All pattern weights should sum to ~1 along kv dim
        sums = pattern.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestUpdateKVCache:
    """Test _update_kv_cache behavior with missing layer_idx and cache interactions."""

    def test_no_cache_returns_unchanged(self):
        """When no past_key_values, K/V returned unchanged."""
        bridge = _make_bridge()
        k = torch.randn(1, 2, 4, 2)
        v = torch.randn(1, 2, 4, 2)

        k_out, v_out = bridge._update_kv_cache(k, v)
        assert k_out is k
        assert v_out is v

    def test_missing_layer_idx_warns(self, caplog):
        """When past_key_values present but _layer_idx is None, emit warning."""
        bridge = _make_bridge()
        bridge._layer_idx = None
        k = torch.randn(1, 2, 4, 2)
        v = torch.randn(1, 2, 4, 2)
        mock_cache = MagicMock()

        with caplog.at_level(logging.WARNING):
            k_out, v_out = bridge._update_kv_cache(k, v, past_key_values=mock_cache)

        assert "layer_idx is None" in caplog.text
        # K/V returned unchanged (no crash)
        assert k_out is k
        assert v_out is v
        # Cache.update was NOT called
        mock_cache.update.assert_not_called()

    def test_cache_update_called_with_layer_idx(self):
        """When layer_idx and cache both present, cache.update is called."""
        bridge = _make_bridge()
        bridge._layer_idx = 3
        k = torch.randn(1, 2, 4, 2)
        v = torch.randn(1, 2, 4, 2)

        mock_cache = MagicMock()
        mock_cache.update.return_value = (k, v)

        k_out, v_out = bridge._update_kv_cache(k, v, past_key_values=mock_cache)

        mock_cache.update.assert_called_once_with(k, v, 3)


class TestLayerIdxCoercion:
    """Test that set_original_component correctly coerces layer_idx types.

    Uses AttentionBridge.set_original_component directly to avoid
    JointQKVAttentionBridge's QKV split machinery.
    """

    @staticmethod
    def _make_attention_bridge():
        return AttentionBridge(name="attn", config=TestConfig())

    def test_int_layer_idx(self):
        """Standard int layer_idx is captured."""
        bridge = self._make_attention_bridge()
        mock_component = MockOriginalAttention(layer_idx=5)

        bridge.set_original_component(mock_component)
        assert bridge._layer_idx == 5
        assert isinstance(bridge._layer_idx, int)

    def test_tensor_layer_idx(self):
        """Tensor scalar layer_idx is coerced to int."""
        bridge = self._make_attention_bridge()
        mock_component = MockOriginalAttention(layer_idx=torch.tensor(7))

        bridge.set_original_component(mock_component)
        assert bridge._layer_idx == 7
        assert isinstance(bridge._layer_idx, int)

    def test_none_layer_idx(self):
        """Missing layer_idx stays None."""
        bridge = self._make_attention_bridge()
        mock_component = MockOriginalAttention()  # no layer_idx attr

        bridge.set_original_component(mock_component)
        assert bridge._layer_idx is None

    def test_numpy_int_layer_idx(self):
        """numpy integer layer_idx is coerced to int."""
        pytest.importorskip("numpy")
        import numpy as np

        bridge = self._make_attention_bridge()
        mock_component = MockOriginalAttention(layer_idx=np.int64(3))

        bridge.set_original_component(mock_component)
        assert bridge._layer_idx == 3
        assert isinstance(bridge._layer_idx, int)


class TestBloomAlibiWithCache:
    """Test Bloom alibi tensor handling when KV cache extends the sequence."""

    def test_alibi_extended_for_longer_kv(self):
        """Alibi is extended when kv_seq_len > original alibi length.

        Bloom's _reconstruct_attention reshapes Q/K/V with Q's seq_len for
        the 3D path, so we pass 4D [batch, seq, heads, head_dim] tensors
        with different Q vs K/V lengths to simulate post-cache state.
        """
        from transformer_lens.model_bridge.generalized_components.bloom_attention import (
            BloomAttentionBridge,
        )

        class BloomConfig:
            n_heads = 2
            d_model = 4
            d_head = 2

        bridge = BloomAttentionBridge(name="bloom_attn", config=BloomConfig())
        bridge.add_module("_original_component", MockOriginalAttention())
        bridge._layer_idx = 0

        batch, heads, head_dim = 1, 2, 2
        q_seq_len = 1
        kv_seq_len = 5

        # 4D: [batch, seq, heads, head_dim] — gets transposed internally
        q = torch.randn(batch, q_seq_len, heads, head_dim)
        k = torch.randn(batch, kv_seq_len, heads, head_dim)
        v = torch.randn(batch, kv_seq_len, heads, head_dim)

        # Build alibi for original seq_len=3 (smaller than kv_seq_len=5)
        original_seq_len = 3
        slopes = torch.tensor([-0.5, -0.25]).view(2, 1, 1)
        positions = torch.arange(original_seq_len).float().view(1, 1, -1)
        alibi = (slopes * positions).reshape(batch * heads, 1, original_seq_len)

        # Should not crash — alibi will be extended internally
        output, pattern = bridge._reconstruct_attention(q, k, v, alibi=alibi)

        assert output.shape == (batch, q_seq_len, heads * head_dim)
        assert pattern.shape == (batch, heads, q_seq_len, kv_seq_len)

    def test_alibi_correct_size_unchanged(self):
        """Alibi that already matches kv_seq_len is used as-is."""
        from transformer_lens.model_bridge.generalized_components.bloom_attention import (
            BloomAttentionBridge,
        )

        class BloomConfig:
            n_heads = 2
            d_model = 4
            d_head = 2

        bridge = BloomAttentionBridge(name="bloom_attn", config=BloomConfig())
        bridge.add_module("_original_component", MockOriginalAttention())

        batch, heads, head_dim = 1, 2, 2
        seq_len = 4

        q = torch.randn(batch, seq_len, heads, head_dim)
        k = torch.randn(batch, seq_len, heads, head_dim)
        v = torch.randn(batch, seq_len, heads, head_dim)

        # Alibi matches seq_len exactly
        slopes = torch.tensor([-0.5, -0.25]).view(2, 1, 1)
        positions = torch.arange(seq_len).float().view(1, 1, -1)
        alibi = (slopes * positions).reshape(batch * heads, 1, seq_len)

        output, pattern = bridge._reconstruct_attention(q, k, v, alibi=alibi)

        assert output.shape == (batch, seq_len, heads * head_dim)
        assert pattern.shape == (batch, heads, seq_len, seq_len)
