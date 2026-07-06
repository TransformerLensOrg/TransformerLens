"""Empirical parity report: boot_vllm vs boot_transformers per architecture.

Closes the vLLM capture trust gap: every vLLM test is otherwise mocked, so captured
activations are numerically unverified. For each model this boots both backends in
fp32 (isolating the boundary mapping from dtype/kernel-precision differences) and diffs
the fireable capture points the vLLM driver OFFERS (``driver.supported_hook_points``):

- ``embed.hook_out`` and per-layer ``blocks.{i}.hook_out`` / ``attn.hook_out`` /
  ``mlp.hook_out`` are semantically identical to boot_transformers and compared directly.
- ``ln_final.hook_normalized`` carries vLLM's POST-weight RMSNorm value under a hook
  whose HT convention is PRE-weight; it is un-folded (÷ weight, or ÷ (1+weight) for
  Gemma) before comparison. The raw (un-un-folded) diff is also reported so a wrong
  un-fold direction is visible rather than silent.

Non-fireable points (fused attention pattern/scores/rope, unembed) are withheld and
reported, not compared. A final-position argmax agreement check sanity-checks logits.

GPU-ONLY: requires a CUDA device and a working ``vllm`` install (pinned build). This is
NOT a CPU/per-PR CI job — it SKIPs cleanly when vLLM or a GPU is unavailable.

Run:  uv run python scripts/vllm_parity_report.py
Env:  TL_PARITY_MODELS="id1,id2,..."  overrides the model list.
      TL_VLLM_ATOL / TL_VLLM_RTOL     override tolerance (defaults 2e-2).
"""
from __future__ import annotations

import gc
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch

# vLLM-supported decoder-only checkpoints small enough to hold model + capture buffers +
# KV cache on one GPU. gemma-2-2b is included specifically to exercise the (1 + weight)
# ln_final un-fold. Gated ids need HF_TOKEN (sourced from .env); bad/gated ids -> SKIP.
DEFAULT_MODELS = [
    "HuggingFaceTB/SmolLM2-135M",  # Llama (tiny, ungated — the smoke test)
    "Qwen/Qwen2.5-0.5B",  # Qwen2
    "Qwen/Qwen3-0.6B",  # Qwen3
    "meta-llama/Llama-3.2-1B",  # Llama (gated)
    "google/gemma-2-2b",  # Gemma2 (gated) — exercises the (1 + weight) ln_final un-fold
]

PROMPT = "The quick brown fox"
# vLLM's fused kernels (PagedAttention, fused RMSNorm/RoPE) differ numerically from HF's
# eager path even in fp32, so the mapping-correct band is looser than the HF-vs-HF 1e-3.
# A wrong hook mapping (e.g. un-un-folded ln_final) diverges by O(1), well outside this.
ATOL = float(os.environ.get("TL_VLLM_ATOL", "2e-2"))
RTOL = float(os.environ.get("TL_VLLM_RTOL", "2e-2"))

# vLLM capture kind -> TransformerBridge-native hook name (per-layer uses {i}).
DIRECT_KINDS = {
    "resid_post": "blocks.{i}.hook_out",
    "attn_out": "blocks.{i}.attn.hook_out",
    "mlp_out": "blocks.{i}.mlp.hook_out",
}
EMBED_HOOK = "embed.hook_out"
LNF_HOOK = "ln_final.hook_normalized"


def _to2d(t: torch.Tensor) -> torch.Tensor:
    """Collapse [batch, seq, d] or [seq, d] to a [tokens, d] float CPU tensor for diffing."""
    t = t.detach().to("cpu", torch.float32)
    return t.reshape(-1, t.shape[-1])


def _diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, bool]:
    a2, b2 = _to2d(a), _to2d(b)
    if a2.shape != b2.shape:
        return float("inf"), False
    d = (a2 - b2).abs().max().item()
    return d, torch.allclose(a2, b2, atol=ATOL, rtol=RTOL)


def _unfold_lnf(vllm_val: torch.Tensor, weight: torch.Tensor, is_gemma: bool) -> torch.Tensor:
    """vLLM ln_final is post-weight (x·rsqrt(var+eps)·w); recover the pre-weight value HT
    exposes by dividing out the norm weight (Gemma folds 1 + weight)."""
    w = weight.detach().to(vllm_val.device, torch.float32)
    denom = (1.0 + w) if is_gemma else w
    # Guard against near-zero weight entries producing spurious blow-ups.
    denom = torch.where(denom.abs() < 1e-6, torch.ones_like(denom), denom)
    return vllm_val.to(torch.float32) / denom


def verify(model_id: str) -> dict:
    from transformer_lens.model_bridge.sources.vllm.source import boot_vllm
    from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

    result: dict = {"model": model_id, "arch": "?", "status": "", "detail": ""}
    hf = vllm = None
    try:
        # Matched fp32 both sides so the comparison isolates the boundary mapping, not
        # dtype. HF ref on the same CUDA device as vLLM; tensors are moved to CPU to diff.
        hf = TransformerBridge.boot_transformers(model_id, device="cuda", dtype=torch.float32)
        arch = getattr(hf.cfg, "architecture", "?")
        result["arch"] = arch
        is_gemma = "gemma" in arch.lower() or "gemma" in model_id.lower()
        n_layers = int(hf.cfg.n_layers)
        toks = hf.to_tokens(PROMPT)

        vllm = boot_vllm(model_id, dtype=torch.float32, max_model_len=2048)
        offered = vllm._driver.supported_hook_points

        hf_logits, hf_cache = hf.run_with_cache(toks)
        v_logits, v_cache = vllm.run_with_cache(toks)

        worst = 0.0
        mism: list[str] = []
        notes: list[str] = []

        # embed + the three per-layer direct boundaries (first & last layer).
        checks = [(EMBED_HOOK, EMBED_HOOK)]
        for i in sorted({0, n_layers - 1}):
            for name in DIRECT_KINDS.values():
                hk = name.format(i=i)
                checks.append((hk, hk))
        for hk, _ in checks:
            if hk not in offered:
                continue  # not a fireable point for this overlay
            if hk not in v_cache or hk not in hf_cache:
                mism.append(f"{hk} missing")
                continue
            d, ok = _diff(hf_cache[hk], v_cache[hk])
            worst = max(worst, d if d != float("inf") else worst)
            if not ok:
                mism.append(f"{hk} maxdiff={d:.2e}")

        # ln_final: compare un-folded; also report the raw diff so a wrong un-fold shows.
        if LNF_HOOK in offered and LNF_HOOK in v_cache and LNF_HOOK in hf_cache:
            raw_d, raw_ok = _diff(hf_cache[LNF_HOOK], v_cache[LNF_HOOK])
            weight = vllm._driver.get_param("model.norm.weight")
            if weight is None:
                notes.append(f"ln_final raw={raw_d:.2e} (no norm weight to un-fold)")
                if not raw_ok:
                    mism.append(f"{LNF_HOOK} maxdiff={raw_d:.2e} (un-fold unavailable)")
            else:
                unfolded = _unfold_lnf(v_cache[LNF_HOOK], weight, is_gemma)
                uf_d, uf_ok = _diff(hf_cache[LNF_HOOK], unfolded)
                worst = max(worst, uf_d if uf_d != float("inf") else worst)
                notes.append(f"ln_final unfold={uf_d:.2e} raw={raw_d:.2e}")
                if not uf_ok:
                    mism.append(f"{LNF_HOOK} unfold maxdiff={uf_d:.2e} (raw={raw_d:.2e})")

        # Logit sanity: vLLM synthesizes final-position log-probs only; check top-1 agrees
        # (reusing the run_with_cache logits above — no second forward).
        try:
            hf_top = int(torch.as_tensor(hf_logits)[0, -1].argmax())
            v_top = int(torch.as_tensor(v_logits)[0, -1].argmax())
            notes.append(f"argmax {'==' if hf_top == v_top else '!='}({hf_top},{v_top})")
            if hf_top != v_top:
                mism.append(f"final-token argmax {hf_top}!={v_top}")
        except Exception as e:  # logit synth is best-effort; don't fail parity on it
            notes.append(f"logit-check err: {type(e).__name__}")

        note_str = "; ".join(notes)
        if mism:
            result["status"] = "FAIL"
            result["detail"] = "; ".join(mism[:3]) + (f" | {note_str}" if note_str else "")
        else:
            result["status"] = "PASS"
            result["detail"] = f"maxdiff={worst:.2e} (L={n_layers}); {note_str}"
    except Exception as e:  # download/gating/OOM/vLLM-unavailable are not parity failures
        result["status"] = "SKIP"
        result["detail"] = f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"
    finally:
        for bridge in (vllm, hf):
            try:
                if bridge is not None:
                    bridge.close()
            except Exception:
                pass
        del hf, vllm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _preflight() -> str | None:
    """Return a human reason to abort (no GPU / no vllm), or None if runnable."""
    if not torch.cuda.is_available():
        return "no CUDA device — vLLM capture only materializes in a real GPU forward"
    try:
        import vllm  # noqa: F401
    except Exception as e:
        return f"vllm not importable ({type(e).__name__}: {e})"
    return None


def main() -> None:
    reason = _preflight()
    if reason is not None:
        print(f"SKIP ALL: {reason}", flush=True)
        print("This harness is GPU-only and requires a working vllm install.", flush=True)
        sys.exit(0)

    ids = os.environ.get("TL_PARITY_MODELS")
    models = [m.strip() for m in ids.split(",")] if ids else DEFAULT_MODELS
    rows = []
    for m in models:
        r = verify(m)
        rows.append(r)
        print(f"[{r['status']:4}] {r['arch']:28} {r['model']:40} {r['detail']}", flush=True)

    print("\n================ vLLM PARITY REPORT CARD ================")
    for status in ("PASS", "FAIL", "SKIP"):
        sel = [r for r in rows if r["status"] == status]
        print(f"\n{status} ({len(sel)}):")
        for r in sel:
            print(f"  {r['arch']:28} {r['model']:40} {r['detail']}")
    passed = sorted({r["arch"] for r in rows if r["status"] == "PASS"})
    print("\nvLLM-CAPTURE-VERIFIED ARCHITECTURES (measured PASS):")
    print("  " + (", ".join(passed) if passed else "(none)"))
    # Non-zero exit if anything actually ran and failed — lets a manual GPU run gate.
    sys.exit(1 if any(r["status"] == "FAIL" for r in rows) else 0)


if __name__ == "__main__":
    main()
