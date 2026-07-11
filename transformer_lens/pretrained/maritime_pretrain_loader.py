"""Load a from-scratch ``pretrain.py`` checkpoint as a HookedTransformer.

``pretrain.py`` (the maritime-intent-probe foundation-model framework) writes
one weight shard per tensor-parallel rank plus a ``config.json`` describing the
architecture. This helper reassembles the (optionally TP-sharded) weights into a
single state dict, builds the matching :class:`HookedTransformerConfig`, and
returns a ready ``HookedTransformer`` whose residual stream can be probed with
the standard hook API.

Example
-------
.. code-block:: python

    from transformer_lens.pretrained.maritime_pretrain_loader import (
        load_maritime_pretrain,
    )

    model = load_maritime_pretrain("runs/base", tag="best")
    logits, cache = model.run_with_cache(tokens)
    resid = cache["resid_post", 6]  # hand off to the linear probes in probe.py
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from transformer_lens import HookedTransformer
from transformer_lens.config.hooked_transformer_config import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions.maritime_pretrain import (
    convert_maritime_pretrain_weights,
)


def _shard_dim(key: str) -> int | None:
    """Which axis a pretrain.py tensor-parallel weight was split along -- or
    None if the weight is replicated on every rank (router, norms, embedding,
    lm_head), in which case any shard's copy is authoritative."""
    if key.endswith(("attn.qkv.weight", "gate.weight", "up.weight")):
        return 0  # column-parallel: split along output features
    if key.endswith(("attn.proj.weight", "down.weight")):
        return 1  # row-parallel: split along input features
    return None


def _merge_tp_shards(run_dir: Path, tag: str) -> dict[str, torch.Tensor]:
    """Reassemble the full weights from per-rank shards."""
    shards = sorted(run_dir.joinpath(tag).glob("model_tp*.pt"))
    if not shards:
        raise FileNotFoundError(f"no model_tp*.pt under {run_dir / tag}")
    states = [torch.load(s, map_location="cpu", weights_only=False)["model"] for s in shards]
    if len(states) == 1:
        return states[0]
    return {
        key: (
            states[0][key]
            if (dim := _shard_dim(key)) is None
            else torch.cat([st[key] for st in states], dim=dim)
        )
        for key in states[0]
    }


def build_config(
    arch: Mapping[str, Any], dtype: torch.dtype = torch.float32
) -> HookedTransformerConfig:
    """Translate pretrain.py's ArchConfig dict into a HookedTransformerConfig."""
    d_model, n_heads = arch["d_model"], arch["n_heads"]
    is_moe = arch.get("moe", False)
    return HookedTransformerConfig(
        n_layers=arch["n_layers"],
        d_model=d_model,
        n_ctx=arch["max_seq_len"],
        d_head=d_model // n_heads,
        n_heads=n_heads,
        d_mlp=arch["d_ff"],
        d_vocab=arch["vocab_size"],
        act_fn="silu",
        normalization_type="RMS",  # RMSNorm, no bias, no mean-subtraction
        eps=arch.get("rmsnorm_eps", 1e-5),
        gated_mlp=True,  # SwiGLU
        positional_embedding_type="rotary",
        rotary_dim=d_model // n_heads,
        rotary_base=arch.get("rope_theta", 10000.0),
        rotary_adjacent_pairs=True,  # pretrain.py rotates adjacent pairs
        final_rms=True,
        num_experts=(arch["n_experts"] if is_moe else None),
        experts_per_token=(arch["top_k"] if is_moe else None),
        # pretrain.py renormalises the top-k routing weights (top_p / top_p.sum).
        # Harmless when dense (no MoE layer consults it).
        norm_topk_prob=bool(is_moe),
        dtype=dtype,
    )


def load_maritime_pretrain(
    run_dir: str | Path,
    tag: str = "best",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    fold_ln: bool = False,
) -> HookedTransformer:
    """Return a HookedTransformer with the pretrain.py checkpoint loaded.

    ``fold_ln`` defaults to False because RMSNorm folding changes the residual
    stream basis probes may depend on; enable it only if you specifically want
    the folded form.
    """
    run_dir = Path(run_dir)
    arch = json.loads((run_dir / "config.json").read_text())
    cfg = build_config(arch, dtype=dtype)

    raw = _merge_tp_shards(run_dir, tag)
    state_dict = convert_maritime_pretrain_weights(raw, cfg)

    model = HookedTransformer(cfg)
    model.load_and_process_state_dict(
        state_dict,
        fold_ln=fold_ln,
        center_writing_weights=False,  # RMSNorm has no bias to center
        center_unembed=False,
        fold_value_biases=False,
    )
    return model.to(device)
