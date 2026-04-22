"""Tests for TransformerLensKeyValueCacheEntry.init_cache_entry dtype behaviour.

The buggy pre-fix code used ``torch.get_default_dtype()`` to initialise
``past_keys`` and ``past_values``. PyTorch's default is ``torch.float32``,
so the bug silently produced the correct dtype for fp32 models but the
wrong dtype (fp32 instead of fp16/bf16) for reduced-precision ones. Of
the tests below, ``test_init_cache_entry_uses_cfg_dtype_float32`` is
therefore a baseline sanity check that passes against both the buggy
and fixed code — it verifies the common case works, not that the bug is
absent. The real regression guards are
``test_init_cache_entry_uses_cfg_dtype_float16``,
``..._bfloat16``, ``..._dtype_independent_of_global_default``, and
``test_append_preserves_cfg_dtype``, which all fail against the buggy
code (the fp16 cache was getting promoted to fp32 by the bug, breaking
the downstream attention-score matmul).
"""

import torch

from transformer_lens.cache.key_value_cache_entry import (
    TransformerLensKeyValueCacheEntry,
)
from transformer_lens.config.TransformerLensConfig import TransformerLensConfig


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
    """Baseline: cfg.dtype=float32 produces fp32 buffers.

    Note: this test passes against both the buggy and fixed implementations
    because torch's default dtype is also float32. It is a sanity check
    that the common case works, not a regression guard for the specific
    bug this module was added to prevent. See module docstring and
    ``test_init_cache_entry_dtype_independent_of_global_default`` for the
    tests that discriminate fix vs bug.
    """
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
    """Regression guard: cache dtype follows cfg.dtype, not the global default.

    Also covers the fp32 case indirectly: if someone reintroduces the old
    ``torch.get_default_dtype()`` behaviour, this test plus the fp16 /
    bfloat16 / append / GQA tests catch it; the fp32-only baseline above
    would not, since fp32 happens to be torch's global default.
    """
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
