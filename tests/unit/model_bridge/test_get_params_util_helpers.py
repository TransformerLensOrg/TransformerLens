"""Tests for get_params_util helper functions."""
import torch

from transformer_lens.model_bridge.get_params_util import (
    _get_n_kv_heads,
    _get_or_create_bias,
    _reshape_kv_weight,
)


class _FakeCfg:
    """Minimal config stub for testing."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestGetNKVHeads:
    def test_prefers_kv_heads_over_n_heads(self):
        cfg = _FakeCfg(n_heads=12, n_key_value_heads=4)
        assert _get_n_kv_heads(cfg) == 4
        assert _get_n_kv_heads(cfg) != cfg.n_heads

    def test_fallback_to_n_heads_when_missing(self):
        cfg = _FakeCfg(n_heads=12)
        assert _get_n_kv_heads(cfg) == 12
        assert not hasattr(cfg, "n_key_value_heads")

    def test_none_kv_heads_falls_back(self):
        # n_key_value_heads exists but is None — should fall back
        cfg = _FakeCfg(n_heads=12, n_key_value_heads=None)
        assert _get_n_kv_heads(cfg) == 12


class TestReshapeKVWeight:
    def test_full_size_preserves_data(self):
        cfg = _FakeCfg(d_model=64, n_heads=4, d_head=16)
        weight = torch.randn(64, 64)
        result = _reshape_kv_weight(weight, cfg, "cpu", torch.float32)
        assert result.shape == (4, 64, 16)
        # Total elements must be preserved
        assert result.numel() == weight.numel()
        # Data must be the same (just reshaped)
        assert torch.equal(result.reshape(-1), weight.reshape(-1))

    def test_mqa_weight_expands_heads(self):
        cfg = _FakeCfg(d_model=64, n_heads=4, d_head=16)
        # MQA: single head (d_head, d_model)
        weight = torch.randn(16, 64)
        result = _reshape_kv_weight(weight, cfg, "cpu", torch.float32)
        assert result.shape == (4, 64, 16)
        # All 4 heads should be identical copies of the single head
        for i in range(1, 4):
            assert torch.equal(result[i], result[0])

    def test_numel_match_uses_view(self):
        cfg = _FakeCfg(d_model=64, n_heads=4, d_head=16)
        # Non-standard shape but total elements match
        weight = torch.randn(4 * 64 * 16).reshape(32, 128)
        result = _reshape_kv_weight(weight, cfg, "cpu", torch.float32)
        assert result.shape == (4, 64, 16)
        assert result.numel() == weight.numel()

    def test_incompatible_shape_returns_zeros(self):
        cfg = _FakeCfg(d_model=64, n_heads=4, d_head=16)
        weight = torch.randn(7, 13)  # impossible to reshape
        result = _reshape_kv_weight(weight, cfg, "cpu", torch.float32)
        assert result.shape == (4, 64, 16)
        assert torch.all(result == 0)
        # Verify it's actually zeros, not just small values
        assert result.sum().item() == 0.0


class TestGetOrCreateBias:
    def test_reshapes_existing_bias(self):
        bias = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = _get_or_create_bias(bias, n_heads=2, d_head=4, device="cpu", dtype=torch.float32)
        assert result.shape == (2, 4)
        # Verify the reshape is correct — first head gets [1,2,3,4]
        assert torch.equal(result[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert torch.equal(result[1], torch.tensor([5.0, 6.0, 7.0, 8.0]))

    def test_none_creates_zeros(self):
        result = _get_or_create_bias(None, n_heads=4, d_head=16, device="cpu", dtype=torch.float32)
        assert result.shape == (4, 16)
        assert result.sum().item() == 0.0

    def test_none_respects_dtype(self):
        result = _get_or_create_bias(None, n_heads=2, d_head=8, device="cpu", dtype=torch.float16)
        assert result.dtype == torch.float16
