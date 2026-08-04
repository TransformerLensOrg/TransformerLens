"""Capture golden fixtures from HookedTransformer before its removal.

The bridge's ``enable_compatibility_mode()`` promises HookedTransformer-equivalent
numerics. Once ``HookedTransformer`` is deleted, frozen goldens captured by this
script are the only way to certify that promise. Run it while HT still exists;
upload the output directory to the goldens dataset repo (see
``tests/goldens.py`` for how tests consume it).

Usage (run serially — never load two models at once):
    uv run python scripts/capture_ht_goldens.py --output-dir tl-compat-goldens
    uv run python scripts/capture_ht_goldens.py --models gpt2 --configs no_processing
    uv run python scripts/capture_ht_goldens.py --include-olmo2

Captured per (model, processing-config):
    state_dict.safetensors          full processed TL-key state dict (2 of 5 configs)
    state_dict.checksums.json       per-tensor sha256 + shape/dtype (all configs)
    state_dict.samples.safetensors  seeded 1024-element samples per tensor (all configs)
    views.safetensors               W_E / W_U / b_U (+ QK/OV factors on GQA models)
    activations.safetensors         full run_with_cache snapshot for the short prompt
    logits_short.safetensors        full logits for the short prompt
    logits_long_final.safetensors   final-position logits for the Main-Demo text
    hook_manifest.json              every hook_dict name + fired shape (null if unfired)
    scalars.json                    CE losses + the L0/H8 hook_v ablation anchors
    provenance.json                 versions, commit, platform, exact load kwargs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1

# Short prompt: used across today's HT-vs-bridge tests. Long text: the Main Demo
# anchor pinned by tests/integration/model_bridge/test_weight_processing.py.
SHORT_PROMPT = "Natural language processing"
MAIN_DEMO_TEXT = (
    "Natural language processing tasks, such as question answering, machine "
    "translation, reading comprehension, and summarization, are typically "
    "approached with supervised learning on taskspecific datasets."
)
MAIN_DEMO_LAYER = 0
MAIN_DEMO_HEAD = 8

SAMPLE_COUNT = 1024
SAMPLE_SEED = 20260803

# Models chosen to cover the processing branches (see ws9-oracle-reanchoring-plan.md):
# canonical GPT-2, the flag-matrix workhorse, parallel-block/rotary, GQA, SoLU.
DEFAULT_MODELS = [
    "gpt2",
    "distilgpt2",
    "EleutherAI/pythia-14m",
    "Qwen/Qwen2-0.5B",
    "solu-1l",
]
# Post-norm carve-out model (fold_ln refused); opt-in via --include-olmo2.
OLMO2_MODEL = "allenai/OLMo-2-0425-1B"

# GQA models where QK/OV factor expansion is the contract under test.
GQA_MODELS = {"Qwen/Qwen2-0.5B"}

# Processing configs. `full_state_dict` marks the two configs whose complete
# processed state dict is stored; the rest store checksums + samples only.
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


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return None


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


def _save_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    # .clone() breaks storage sharing: HT's cache aliases tensors (e.g.
    # blocks.N.hook_resid_post IS blocks.N+1.hook_resid_pre), which safetensors
    # rejects. Each entry must own its memory.
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}, str(path))


def _run_ablation(model: Any, text: str, layer: int, head: int) -> tuple[float, float]:
    """Mirror of tests/integration/model_bridge/test_weight_processing.py::_run_ablation."""
    from transformer_lens import utilities as utils

    tokens = model.to_tokens(text)

    def ablation_hook(value: torch.Tensor, hook: Any) -> torch.Tensor:
        value[:, :, head, :] = 0.0
        return value

    hook_name = utils.get_act_name("v", layer)
    orig = model(tokens, return_type="loss").item()
    ablated = model.run_with_hooks(
        tokens, return_type="loss", fwd_hooks=[(hook_name, ablation_hook)]
    ).item()
    return orig, ablated


def capture_one(model_name: str, config_name: str, out_root: Path, skip_existing: bool) -> None:
    """Capture the full golden set for one (model, processing-config) cell."""
    from transformer_lens import HookedTransformer

    cfg = CONFIGS[config_name]
    out_dir = out_root / _model_dir_name(model_name) / config_name
    if skip_existing and (out_dir / "provenance.json").exists():
        print(f"[skip] {model_name} / {config_name} (exists)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[capture] {model_name} / {config_name}")

    if cfg["no_processing"]:
        model = HookedTransformer.from_pretrained_no_processing(
            model_name, device="cpu", dtype=torch.float32
        )
        load_kwargs: dict[str, Any] = {"no_processing": True}
    else:
        load_kwargs = dict(cfg["kwargs"])
        model = HookedTransformer.from_pretrained(
            model_name, device="cpu", dtype=torch.float32, **load_kwargs
        )
    model.eval()

    with torch.no_grad():
        state = model.state_dict()

        # --- state dict: checksums + samples always; full dict for the 2 pinned configs
        checksums = {
            k: {
                "sha256": _tensor_checksum(v),
                "shape": list(v.shape),
                "dtype": str(v.dtype),
            }
            for k, v in state.items()
        }
        (out_dir / "state_dict.checksums.json").write_text(
            json.dumps(checksums, indent=1, sort_keys=True)
        )
        _save_safetensors(
            out_dir / "state_dict.samples.safetensors",
            {k: _seeded_sample(v, SAMPLE_SEED) for k, v in state.items()},
        )
        if cfg["full_state_dict"]:
            _save_safetensors(out_dir / "state_dict.safetensors", dict(state))

        # --- derived weight views asserted by today's tests
        views: dict[str, torch.Tensor] = {
            "W_E": model.W_E,
            "W_U": model.W_U,
            "b_U": model.b_U,
        }
        if model_name in GQA_MODELS:
            views.update(
                {
                    "QK.A": model.QK.A,
                    "QK.B": model.QK.B,
                    "OV.A": model.OV.A,
                    "OV.B": model.OV.B,
                }
            )
        _save_safetensors(out_dir / "views.safetensors", views)

        # --- short-prompt logits + full activation snapshot + hook manifest
        logits, cache = model.run_with_cache(SHORT_PROMPT)
        _save_safetensors(out_dir / "logits_short.safetensors", {"logits": logits})
        _save_safetensors(
            out_dir / "activations.safetensors",
            {k: v for k, v in cache.items() if isinstance(v, torch.Tensor)},
        )
        manifest = {
            name: (list(cache[name].shape) if name in cache else None)
            for name in sorted(model.hook_dict.keys())
        }
        (out_dir / "hook_manifest.json").write_text(json.dumps(manifest, indent=1))

        # --- long-text loss + final-position logits; Main-Demo ablation anchors
        long_tokens = model.to_tokens(MAIN_DEMO_TEXT)
        long_logits = model(long_tokens, return_type="logits")
        long_loss = model(long_tokens, return_type="loss").item()
        _save_safetensors(
            out_dir / "logits_long_final.safetensors",
            {"final_logits": long_logits[:, -1, :]},
        )

    orig_loss, ablated_loss = _run_ablation(model, MAIN_DEMO_TEXT, MAIN_DEMO_LAYER, MAIN_DEMO_HEAD)

    scalars = {
        "short_prompt": SHORT_PROMPT,
        "long_text_ce_loss": long_loss,
        "ablation": {
            "text": MAIN_DEMO_TEXT,
            "layer": MAIN_DEMO_LAYER,
            "head": MAIN_DEMO_HEAD,
            "hook": f"blocks.{MAIN_DEMO_LAYER}.attn.hook_v",
            "orig_loss": orig_loss,
            "ablated_loss": ablated_loss,
        },
    }
    (out_dir / "scalars.json").write_text(json.dumps(scalars, indent=1))

    import transformers

    import transformer_lens

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "config": config_name,
        "load_kwargs": load_kwargs,
        "device": "cpu",
        "dtype": "float32",
        "transformer_lens_version": getattr(transformer_lens, "__version__", None),
        "transformer_lens_commit": _git_commit(),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "sample_seed": SAMPLE_SEED,
        "sample_count": SAMPLE_COUNT,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=1))

    del model, state, cache, logits, long_logits
    print(f"[done] {model_name} / {config_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", default="tl-compat-goldens", type=Path)
    parser.add_argument("--models", nargs="*", default=None, help="subset of models")
    parser.add_argument("--configs", nargs="*", default=None, help="subset of config names")
    parser.add_argument(
        "--include-olmo2", action="store_true", help="also capture the post-norm carve-out model"
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the capture matrix and exit")
    args = parser.parse_args()

    models = list(args.models) if args.models else list(DEFAULT_MODELS)
    if args.include_olmo2 and OLMO2_MODEL not in models:
        models.append(OLMO2_MODEL)
    config_names = list(args.configs) if args.configs else list(CONFIGS.keys())
    unknown = [c for c in config_names if c not in CONFIGS]
    if unknown:
        parser.error(f"unknown configs: {unknown}; valid: {list(CONFIGS)}")

    cells = [(m, c) for m in models for c in config_names if m in CONFIGS[c].get("models", models)]
    if args.dry_run:
        for m, c in cells:
            print(f"{m} / {c}")
        print(f"{len(cells)} cells -> {args.output_dir}")
        return 0

    # Serial on purpose: never hold two models in memory.
    for m, c in cells:
        capture_one(m, c, args.output_dir, args.skip_existing)
    print(f"All {len(cells)} cells captured to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
