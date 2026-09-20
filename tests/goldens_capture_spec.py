"""Pure, model-free spec for the frozen golden fixtures.

The runnable capture script (``scripts/capture_ht_goldens.py``) was removed with
HookedTransformer at 4.0 — it booted the class to produce the goldens. Its
deterministic, model-free helpers (checksum, seeded sampling, the config matrix,
directory naming) live on here: the committed goldens were built with these, and
``tests/goldens.py`` plus the golden round-trip tests verify against them.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

SCHEMA_VERSION = 1
SAMPLE_COUNT = 1024
SAMPLE_SEED = 20260803

SHORT_PROMPT = "Natural language processing"

# Processing configs the goldens are captured under (data-only; the removed
# script mapped these to HookedTransformer load flags). Preserved verbatim so
# the golden cell names and the fixtures stay consistent.
CONFIGS: dict[str, dict[str, Any]] = {
    "no_processing": {
        "kwargs": {},
        "no_processing": True,
        "full_state_dict": True,
    },
    "full_defaults": {
        "kwargs": {
            "fold_ln": True,
            "center_writing_weights": True,
            "center_unembed": True,
            "fold_value_biases": True,
        },
        "no_processing": False,
        "full_state_dict": True,
    },
    "fold_ln_only": {
        "kwargs": {
            "fold_ln": True,
            "center_writing_weights": False,
            "center_unembed": False,
            "fold_value_biases": False,
        },
        "no_processing": False,
        "full_state_dict": False,
    },
    "fold_ln_center_writing": {
        "kwargs": {
            "fold_ln": True,
            "center_writing_weights": True,
            "center_unembed": False,
            "fold_value_biases": False,
        },
        "no_processing": False,
        "full_state_dict": False,
    },
    # gpt2 only: the sole model whose refactor test exists today.
    "refactor_factored": {
        "kwargs": {
            "fold_ln": True,
            "center_writing_weights": True,
            "center_unembed": True,
            "fold_value_biases": True,
            "refactor_factored_attn_matrices": True,
        },
        "no_processing": False,
        "full_state_dict": False,
        "models": ["gpt2"],
    },
}


def _model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def _tensor_checksum(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()


def _seeded_sample(t: torch.Tensor, seed: int) -> torch.Tensor:
    """Fixed random-index sample of a tensor, deterministic across runs."""
    flat = t.detach().cpu().contiguous().float().flatten()
    if flat.numel() <= SAMPLE_COUNT:
        return flat.clone()
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(flat.numel(), generator=gen)[:SAMPLE_COUNT]
    return flat[idx.sort().values]
