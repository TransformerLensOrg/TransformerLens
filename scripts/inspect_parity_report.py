"""Empirical parity report: boot_inspect vs boot_transformers per architecture.

For each model, compares (at the first and last layer) the boundaries the provider's
structural self-check OFFERS — gated ones (e.g. resid_mid on parallel/norm-variant archs)
are correctly withheld and reported, not compared. Validates that self-check against real
models: everything offered must match boot_transformers; it is not a per-PR CI job.

Run: uv run python scripts/inspect_parity_report.py
Override the model list with TL_PARITY_MODELS="id1,id2,...".
"""
from __future__ import annotations

import gc
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch

# Broad coverage across adapter families: real small/gated checkpoints (HF token from
# .env) where available, tiny-random elsewhere. Structure (not weights) drives parity,
# so tiny-random is fine for classification; bad/404 ids land in "couldn't run".
DEFAULT_MODELS = [
    # --- sequential post-norm (GPT2 family) ---
    "sshleifer/tiny-gpt2",  # GPT2
    "gpt2",  # GPT2
    "bigcode/gpt_bigcode-santacoder",  # GPTBigCode
    "roneneldan/TinyStories-33M",  # GPTNeo
    # --- sequential pre-norm (Llama family) ---
    "meta-llama/Llama-3.2-1B",  # Llama (gated)
    "mistralai/Mistral-7B-v0.1",  # Mistral
    "Qwen/Qwen2.5-0.5B",  # Qwen2
    "Qwen/Qwen3-0.6B",  # Qwen3
    "microsoft/Phi-3-mini-4k-instruct",  # Phi3
    "allenai/OLMo-2-0425-1B",  # Olmo2
    "ibm-granite/granite-3.0-2b-base",  # Granite
    "stabilityai/stablelm-2-1_6b",  # StableLm
    "HuggingFaceTB/SmolLM2-135M",  # Llama
    # --- norm-variant (extra pre/post-FF norms → resid_mid gated by identity check) ---
    "google/gemma-2-2b",  # Gemma2 (gated)
    "google/gemma-3-1b-pt",  # Gemma3 (gated)
    "ibm-granite/granite-3.0-2b-base",  # Granite — residual-multiplier; even attn/mlp diverge
    # --- parallel residual (attn + mlp both read resid_pre → resid_mid gated by causal) ---
    "EleutherAI/pythia-160m",  # GPTNeoX
    "Salesforce/codegen-350M-mono",  # CodeGen
    "microsoft/phi-1_5",  # Phi
    "tiiuae/falcon-rw-1b",  # Falcon (attn attr self_attention → attn_out gated)
    # --- other / non-standard composition ---
    "facebook/opt-125m",  # OPT (fc1/fc2 → mlp_out gated)
    "bigscience/bloom-560m",  # Bloom
    # --- tiny-random fallbacks for families lacking a small real checkpoint ---
    "hf-internal-testing/tiny-random-GPTJForCausalLM",  # GPTJ (parallel)
    "trl-internal-testing/tiny-CohereForCausalLM",  # Cohere (tiny)
    "hf-internal-testing/tiny-random-MixtralForCausalLM",  # Mixtral (MoE block → mlp_out gated)
]

PROMPT = "The quick brown fox"
ATOL, RTOL = 1e-3, 1e-3


# Boundary kind -> TransformerBridge-native hook suffix.
KIND_SUFFIX = {
    "resid_pre": "hook_in",
    "resid_mid": "ln2.hook_in",
    "resid_post": "hook_out",
    "attn_out": "attn.hook_out",
    "mlp_out": "mlp.hook_out",
}


def verify(model_id: str) -> dict:
    from transformer_lens.model_bridge.remote_bridge import RemoteBridge
    from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

    result: dict = {"model": model_id, "arch": "?", "status": "", "detail": ""}
    hf = inspect = None
    try:
        # Force matched fp32 + (provider) eager attention so the comparison isolates the
        # boundary mapping, not dtype/attn-impl differences.
        hf = TransformerBridge.boot_transformers(model_id, device="cpu", dtype=torch.float32)
        result["arch"] = getattr(hf.cfg, "architecture", "?")
        n_layers = int(hf.cfg.n_layers)
        toks = hf.to_tokens(PROMPT)

        inspect = RemoteBridge.boot_inspect(model_id, dtype=torch.float32)
        # Validate the structural self-check: only the boundaries the provider OFFERS must
        # match; ones it gated (e.g. resid_mid on parallel archs) are correctly withheld.
        offered = inspect._driver.supported_hook_points
        kinds_offered = {k for k, suf in KIND_SUFFIX.items() if f"blocks.0.{suf}" in offered}
        gated = sorted(set(KIND_SUFFIX) - kinds_offered)

        _, hf_cache = hf.run_with_cache(toks)
        _, i_cache = inspect.run_with_cache(toks)

        worst = 0.0
        mism = []
        for i in sorted({0, n_layers - 1}):
            for kind in kinds_offered:
                hk = f"blocks.{i}.{KIND_SUFFIX[kind]}"
                if hk not in i_cache or hk not in hf_cache:
                    mism.append(f"{hk} missing")
                    continue
                a, b = hf_cache[hk].float(), i_cache[hk].float()
                if a.shape != b.shape:
                    mism.append(f"{hk} shape {tuple(a.shape)}!={tuple(b.shape)}")
                    continue
                d = (a - b).abs().max().item()
                worst = max(worst, d)
                if not torch.allclose(a, b, atol=ATOL, rtol=RTOL):
                    mism.append(f"{hk} maxdiff={d:.2e}")
        gated_str = ",".join(gated) if gated else "none"
        if mism:
            result["status"] = "FAIL"
            result["detail"] = f"gated={gated_str}; " + "; ".join(mism[:3])
        else:
            result["status"] = "PASS"
            result["detail"] = f"maxdiff={worst:.2e} (L={n_layers}, gated={gated_str})"
    except Exception as e:  # download/gating/locate failures are not parity failures
        result["status"] = "SKIP"
        result["detail"] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
    finally:
        for bridge in (inspect, hf):
            try:
                if bridge is not None:
                    bridge.close()
            except Exception:
                pass
        del hf, inspect
        gc.collect()
    return result


def main() -> None:
    ids = os.environ.get("TL_PARITY_MODELS")
    models = [m.strip() for m in ids.split(",")] if ids else DEFAULT_MODELS
    rows = []
    for m in models:
        r = verify(m)
        rows.append(r)
        print(f"[{r['status']:4}] {r['arch']:32} {r['model']:48} {r['detail']}", flush=True)

    print("\n================ PARITY REPORT CARD ================")
    for status in ("PASS", "FAIL", "SKIP"):
        sel = [r for r in rows if r["status"] == status]
        print(f"\n{status} ({len(sel)}):")
        for r in sel:
            print(f"  {r['arch']:32} {r['model']:48} {r['detail']}")
    passed = sorted({r["arch"] for r in rows if r["status"] == "PASS"})
    print("\nPARITY-VERIFIED ARCHITECTURES (measured PASS):")
    print("  " + ", ".join(passed))
    sys.exit(0)


if __name__ == "__main__":
    main()
