"""Freeze legacy HookedTransformer checkpoints for the converter tests.

The legacy TL property-format is frozen by definition — historical checkpoints
(OthelloGPT, grokking, ARENA) never change — so the converter's test inputs are
captured once from a live HookedTransformer and committed, letting the tests
survive HookedTransformer's 4.0 deletion. Rerun only to ADD variants.

    uv run python scripts/capture_tl_checkpoint_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from transformer_lens import HookedTransformer
from transformer_lens.config import HookedTransformerConfig

OUT = Path(__file__).parents[1] / "tests" / "fixtures" / "tl_checkpoints"

BASE = dict(
    d_model=32,
    d_head=16,
    n_heads=2,
    n_layers=2,
    n_ctx=8,
    d_vocab=16,
    d_mlp=64,
    act_fn="gelu",
    normalization_type="LN",
    seed=0,
)

VARIANTS: dict[str, dict] = {
    "default": {},
    "gqa": dict(n_heads=4, d_head=8, n_key_value_heads=2),
    "lnpre": dict(normalization_type="LNPre"),
    "attn_only": dict(attn_only=True),
    "gated_rms": dict(gated_mlp=True, normalization_type="RMS", act_fn="silu"),
}


def main() -> None:
    torch.manual_seed(0)
    for name, overrides in VARIANTS.items():
        kwargs = {**BASE, **overrides}
        ht = HookedTransformer(HookedTransformerConfig(**kwargs))
        tokens = torch.randint(0, kwargs["d_vocab"], (1, 4))
        with torch.no_grad():
            logits = ht(tokens)
        out_dir = OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ht.state_dict(), out_dir / "checkpoint.pt")
        reference = {"tokens": tokens, "logits": logits}
        if name == "default":
            # stacked per-head views for the head-slot placement test
            for attr in ("W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V", "b_O"):
                reference[attr] = getattr(ht, attr)
        torch.save(reference, out_dir / "reference.pt")
        (out_dir / "meta.json").write_text(json.dumps(kwargs, indent=1) + "\n")
        print(f"[done] {name}: {sum(v.numel() for v in ht.state_dict().values())} params")


if __name__ == "__main__":
    main()
