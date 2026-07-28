"""Empirical parity report: boot_transformers multi-device vs single-device, per architecture.

For each model this boots the single-device cuda:0 reference and a split placement
(``n_devices=2`` by default) in fp32, then diffs every cached hook, the full logits,
the argmax sequence, and a greedy generation. Both sides run identical HF kernels on
identical GPUs, so parity should be near-exact — a miss means the placement moved or
dropped values, not kernel noise.

GPU-ONLY: requires >= 2 CUDA devices. SKIPs cleanly otherwise. Not a CI job.

Run:  uv run python scripts/bridge_multi_device_parity.py
Env:  TL_BRIDGE_PARITY_MODELS="id1,id2,..."   overrides the model list.
      TL_BRIDGE_ATOL / TL_BRIDGE_RTOL         tolerance (default 1e-4).
      TL_BRIDGE_DEVMAP=balanced|auto|sequential
                                               use device_map=<strategy> instead of
                                               n_devices=2 for the split boot.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import torch

# Architecture spread: GPT-2 (learned pos embeds, tied unembed), Llama-family
# (SmolLM2, RoPE + RMSNorm), Qwen2 (QKV biases), gated Llama-3.2 and Gemma-2
# (soft-capping) when HF_TOKEN grants access. Bad/gated ids -> SKIP.
DEFAULT_MODELS = [
    "gpt2",
    "HuggingFaceTB/SmolLM2-135M",
    "Qwen/Qwen2.5-0.5B",
    "meta-llama/Llama-3.2-1B",
    "google/gemma-2-2b",
]

PROMPT = "The quick brown fox jumps over"
ATOL = float(os.environ.get("TL_BRIDGE_ATOL", "1e-4"))
RTOL = float(os.environ.get("TL_BRIDGE_RTOL", "1e-4"))
DEVMAP = os.environ.get("TL_BRIDGE_DEVMAP")  # None -> n_devices=2

_ROW_MARKER = "##TL_BRIDGE_PARITY_ROW## "


def _split_kwargs() -> dict:
    return {"device_map": DEVMAP} if DEVMAP else {"n_devices": 2}


def verify(model_id: str) -> dict:
    from transformer_lens.model_bridge import TransformerBridge

    result: dict = {"model": model_id, "status": "", "detail": ""}
    single = multi = None
    try:
        single = TransformerBridge.boot_transformers(model_id, device="cuda:0", dtype=torch.float32)
        multi = TransformerBridge.boot_transformers(
            model_id, dtype=torch.float32, **_split_kwargs()
        )
        toks = single.to_tokens(PROMPT)

        logits1, cache1 = single.run_with_cache(toks)
        logits2, cache2 = multi.run_with_cache(toks.to(multi.cfg.device))
        logits1 = logits1.detach().float().cpu()
        logits2 = logits2.detach().float().cpu()

        worst = 0.0
        mism: list[str] = []

        if set(cache1.keys()) != set(cache2.keys()):
            only1 = sorted(set(cache1.keys()) - set(cache2.keys()))[:5]
            only2 = sorted(set(cache2.keys()) - set(cache1.keys()))[:5]
            mism.append(f"cache key sets differ (single-only {only1}, split-only {only2})")
        else:
            for name in cache1:
                t1 = cache1[name].detach().float().cpu()
                t2 = cache2[name].detach().float().cpu()
                if t1.shape != t2.shape:
                    mism.append(f"{name} shape {tuple(t1.shape)} vs {tuple(t2.shape)}")
                    continue
                d = (t1 - t2).abs().max().item()
                worst = max(worst, d)
                if not torch.allclose(t1, t2, atol=ATOL, rtol=RTOL):
                    mism.append(f"{name} maxdiff={d:.3e}")

        d = (logits1 - logits2).abs().max().item()
        worst = max(worst, d)
        if not torch.allclose(logits1, logits2, atol=ATOL, rtol=RTOL):
            mism.append(f"logits maxdiff={d:.3e}")
        if not torch.equal(logits1.argmax(dim=-1), logits2.argmax(dim=-1)):
            mism.append("argmax sequence differs")

        gen1 = single.generate(PROMPT, max_new_tokens=5, do_sample=False, verbose=False)
        gen2 = multi.generate(PROMPT, max_new_tokens=5, do_sample=False, verbose=False)
        if gen1 != gen2:
            mism.append(f"greedy generate differs: {gen1!r} vs {gen2!r}")

        n_cuda = {
            p.device.index for p in multi.original_model.parameters() if p.device.type == "cuda"
        }
        placement = f"devices={sorted(n_cuda)} n_devices={multi.cfg.n_devices}"

        if mism:
            result["status"] = "FAIL"
            result["detail"] = f"{placement}; " + "; ".join(mism[:6])
        else:
            result["status"] = "PASS"
            result["detail"] = f"{placement}; {len(cache1)} hooks, worst diff {worst:.2e}"
    except Exception as exc:  # gated repo, OOM, unsupported arch -> SKIP with reason
        result["status"] = "SKIP"
        result["detail"] = f"{type(exc).__name__}: {exc}"
    finally:
        del single, multi
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _verify_in_subprocess(model_id: str) -> dict:
    """One model pair per process: sequential fp32 dual-boots fragment GPU memory;
    a fresh process guarantees each model its full budget. Child stdout streams
    through; the marker line carries the structured row."""
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--one", model_id],
        capture_output=True,
        text=True,
    )
    row = None
    for line in proc.stdout.splitlines():
        if line.startswith(_ROW_MARKER):
            row = json.loads(line[len(_ROW_MARKER) :])
        else:
            print(line, flush=True)
    if proc.stderr:
        tail = proc.stderr.strip().splitlines()[-3:]
        for line in tail:
            print(f"    [stderr] {line}", flush=True)
    return row or {
        "model": model_id,
        "status": "SKIP",
        "detail": f"child exited {proc.returncode} without a result row",
    }


def _preflight() -> str | None:
    if not torch.cuda.is_available():
        return "no CUDA device"
    if torch.cuda.device_count() < 2:
        return f"needs >= 2 CUDA devices, found {torch.cuda.device_count()}"
    return None


def main() -> None:
    skip_reason = _preflight()
    if skip_reason:
        print(f"SKIP ALL: {skip_reason}")
        return

    models = [
        m.strip()
        for m in os.environ.get("TL_BRIDGE_PARITY_MODELS", ",".join(DEFAULT_MODELS)).split(",")
        if m.strip()
    ]
    split_desc = f"device_map={DEVMAP}" if DEVMAP else "n_devices=2"
    rows = []
    for m in models:
        print(f"==> {m}", flush=True)
        rows.append(_verify_in_subprocess(m))

    print(f"\n========== BRIDGE MULTI-DEVICE PARITY ({split_desc}, atol={ATOL:g}) ==========")
    for r in rows:
        print(f"[{r['status']:4}] {r['model']}: {r['detail']}")
    if any(r["status"] == "FAIL" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        print(_ROW_MARKER + json.dumps(verify(sys.argv[2])), flush=True)
    else:
        main()
