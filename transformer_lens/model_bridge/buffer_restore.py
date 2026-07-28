"""Repair non-persistent buffers destroyed by meta-device loading.

transformers 5.x replaces every non-persistent buffer with ``torch.empty_like``
(``modeling_utils._move_missing_keys_from_meta_to_device``) and only restores
rotary tables for modules exposing ``original_inv_freq``. Remote code predating
that attribute silently keeps uninitialized memory — internlm2 loads a table of
zeros, which collapses RoPE to the identity transform.

The defect is invisible to the phase benchmarks: they compare the bridge against
an HF reference loaded the same way, so both sides are wrong together and agree.
Shared by adapters (``prepare_model``) and the benchmark's HF reference so the
two can never drift apart.
"""

from typing import Any

import torch


def _is_valid_inv_freq(inv_freq: torch.Tensor) -> bool:
    """True when the table looks like real rotary frequencies.

    Valid tables are finite, strictly decreasing, and lie in (0, 1] — including
    the scaled variants (linear / NTK / llama3), whose tables keep that shape.
    Uninitialized memory (zeros, denormals, huge magnitudes) does not.
    """
    if inv_freq.numel() == 0 or not bool(torch.isfinite(inv_freq).all()):
        return False
    values = inv_freq.float()
    if not bool((values > 0).all()) or float(values.max()) > 1.0 + 1e-6:
        return False
    return bool((values[:-1] > values[1:]).all()) if values.numel() > 1 else True


def restore_rotary_inv_freq(hf_model: Any) -> int:
    """Recompute invalid rotary ``inv_freq`` tables in place; returns the count.

    Only tables failing :func:`_is_valid_inv_freq` are touched, so legitimately
    scaled tables are never clobbered.
    """
    restored = 0
    for module in hf_model.modules():
        if "RotaryEmbedding" not in type(module).__name__:
            continue
        inv_freq = getattr(module, "inv_freq", None)
        dim = getattr(module, "dim", None)
        base = getattr(module, "base", None)
        if not isinstance(inv_freq, torch.Tensor) or dim is None or base is None:
            continue
        if _is_valid_inv_freq(inv_freq):
            continue
        recomputed = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        module.inv_freq = recomputed.to(device=inv_freq.device, dtype=inv_freq.dtype)
        # Older remote code caches cos/sin from inv_freq at init and indexes the
        # cache directly (`self.cos_cached[:seq_len]`), so it must be rebuilt,
        # never cleared — setting it to None raises on the next forward.
        rebuild = getattr(module, "_set_cos_sin_cache", None)
        if callable(rebuild) and hasattr(module, "cos_cached"):
            try:
                rebuild(
                    seq_len=getattr(module, "max_seq_len_cached", None)
                    or getattr(module, "max_position_embeddings", 2048),
                    device=inv_freq.device,
                    dtype=module.cos_cached.dtype,
                )
            except Exception:
                # Signature varies across remote forks; force a rebuild on the
                # next forward instead, which every such implementation guards
                # with `if seq_len > self.max_seq_len_cached`.
                module.max_seq_len_cached = 0
        if hasattr(module, "original_inv_freq"):
            module.original_inv_freq = module.inv_freq
        restored += 1
    return restored
