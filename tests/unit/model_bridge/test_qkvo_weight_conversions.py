"""Tests for ArchitectureAdapter._qkvo_weight_conversions() factory method."""

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter


class _FakeCfg:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_adapter(n_heads=8, n_key_value_heads=None):
    """Create an ArchitectureAdapter with minimal config for testing."""
    cfg = _FakeCfg(n_heads=n_heads)
    if n_key_value_heads is not None:
        cfg.n_key_value_heads = n_key_value_heads
    adapter = ArchitectureAdapter.__new__(ArchitectureAdapter)
    adapter.cfg = cfg
    return adapter


class TestQKVOWeightConversions:
    def test_returns_all_four_keys(self):
        adapter = _make_adapter(n_heads=8)
        conversions = adapter._qkvo_weight_conversions()
        expected_keys = {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }
        assert set(conversions.keys()) == expected_keys

    def test_q_rearrange_produces_correct_shape(self):
        adapter = _make_adapter(n_heads=4)
        conversions = adapter._qkvo_weight_conversions()
        q_conversion = conversions["blocks.{i}.attn.q.weight"]
        # Simulate: [n_heads*d_head, d_model] = [64, 32] → [n_heads, d_model, d_head] = [4, 32, 16]
        weight = torch.randn(64, 32)
        result = q_conversion.tensor_conversion.handle_conversion(weight)
        assert result.shape == (4, 32, 16)

    def test_o_rearrange_produces_correct_shape(self):
        adapter = _make_adapter(n_heads=4)
        conversions = adapter._qkvo_weight_conversions()
        o_conversion = conversions["blocks.{i}.attn.o.weight"]
        # O pattern: [d_model, n_heads*d_head] = [32, 64] → [n_heads, d_head, d_model] = [4, 16, 32]
        weight = torch.randn(32, 64)
        result = o_conversion.tensor_conversion.handle_conversion(weight)
        assert result.shape == (4, 16, 32)

    def test_kv_use_n_kv_heads_for_gqa(self):
        adapter = _make_adapter(n_heads=8, n_key_value_heads=2)
        conversions = adapter._qkvo_weight_conversions()
        # K/V should use n_kv_heads=2, not n_heads=8
        k_conversion = conversions["blocks.{i}.attn.k.weight"]
        # [n_kv_heads*d_head, d_model] = [32, 128] with n_kv_heads=2 → [2, 128, 16]
        weight = torch.randn(32, 128)
        result = k_conversion.tensor_conversion.handle_conversion(weight)
        assert result.shape[0] == 2  # n_kv_heads, not n_heads

    def test_q_and_o_always_use_n_heads(self):
        adapter = _make_adapter(n_heads=8, n_key_value_heads=2)
        conversions = adapter._qkvo_weight_conversions()
        q_conversion = conversions["blocks.{i}.attn.q.weight"]
        weight = torch.randn(128, 64)
        result = q_conversion.tensor_conversion.handle_conversion(weight)
        assert result.shape[0] == 8  # n_heads, not n_kv_heads

    def test_explicit_n_kv_heads_overrides_config(self):
        adapter = _make_adapter(n_heads=8, n_key_value_heads=4)
        conversions = adapter._qkvo_weight_conversions(n_kv_heads=2)
        k_conversion = conversions["blocks.{i}.attn.k.weight"]
        weight = torch.randn(32, 128)
        result = k_conversion.tensor_conversion.handle_conversion(weight)
        assert result.shape[0] == 2  # explicit n_kv_heads=2, not cfg's 4
