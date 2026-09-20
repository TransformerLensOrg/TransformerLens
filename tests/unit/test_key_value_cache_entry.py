"""Tests that KeyValueCacheEntry.init_cache_entry buffers follow cfg.dtype, not torch's global default."""

import torch

from transformer_lens.cache.key_value_cache_entry import (
    TransformerLensKeyValueCacheEntry,
)
from transformer_lens.config.transformer_lens_config import TransformerLensConfig


def _make_cfg(dtype: torch.dtype, n_heads: int = 4, d_head: int = 8, n_key_value_heads=None):
    return TransformerLensConfig(
        d_model=n_heads * d_head,
        d_head=d_head,
        n_layers=1,
        n_ctx=32,
        n_heads=n_heads,
        n_key_value_heads=n_key_value_heads,
        dtype=dtype,
    )


def test_init_cache_entry_uses_cfg_dtype_float32():
    """Baseline: cfg.dtype=float32 produces fp32 buffers."""
    cfg = _make_cfg(dtype=torch.float32)
    entry = TransformerLensKeyValueCacheEntry.init_cache_entry(cfg, device="cpu")
    assert entry.past_keys.dtype == torch.float32
    assert entry.past_values.dtype == torch.float32


def test_init_cache_entry_uses_cfg_dtype_float16():
    cfg = _make_cfg(dtype=torch.float16)
    entry = TransformerLensKeyValueCacheEntry.init_cache_entry(cfg, device="cpu")
    assert entry.past_keys.dtype == torch.float16
    assert entry.past_values.dtype == torch.float16


def test_init_cache_entry_uses_cfg_dtype_bfloat16():
    cfg = _make_cfg(dtype=torch.bfloat16)
    entry = TransformerLensKeyValueCacheEntry.init_cache_entry(cfg, device="cpu")
    assert entry.past_keys.dtype == torch.bfloat16
    assert entry.past_values.dtype == torch.bfloat16


def test_init_cache_entry_dtype_independent_of_global_default():
    """Regression guard: cache dtype follows cfg.dtype, not the global default."""
    cfg = _make_cfg(dtype=torch.float16)
    original_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        entry = TransformerLensKeyValueCacheEntry.init_cache_entry(cfg, device="cpu")
        assert entry.past_keys.dtype == torch.float16
        assert entry.past_values.dtype == torch.float16
    finally:
        torch.set_default_dtype(original_default)


def test_append_preserves_cfg_dtype():
    """After append, past_keys stays in cfg.dtype — no float promotion."""
    cfg = _make_cfg(dtype=torch.float16)
    entry = TransformerLensKeyValueCacheEntry.init_cache_entry(cfg, device="cpu")
    new_keys = torch.randn(1, 3, cfg.n_heads, cfg.d_head, dtype=torch.float16)
    new_values = torch.randn(1, 3, cfg.n_heads, cfg.d_head, dtype=torch.float16)
    updated_keys, updated_values = entry.append(new_keys, new_values)
    assert updated_keys.dtype == torch.float16
    assert updated_values.dtype == torch.float16
    assert entry.past_keys.dtype == torch.float16
    assert entry.past_values.dtype == torch.float16


def test_init_cache_entry_handles_grouped_query_attention():
    """When n_key_value_heads is set (GQA), it should be used instead of n_heads."""
    cfg = _make_cfg(dtype=torch.float16, n_heads=32, d_head=128, n_key_value_heads=8)
    entry = TransformerLensKeyValueCacheEntry.init_cache_entry(cfg, device="cpu", batch_size=2)
    assert entry.past_keys.shape == (2, 0, 8, 128)
    assert entry.past_values.shape == (2, 0, 8, 128)
    assert entry.past_keys.dtype == torch.float16
