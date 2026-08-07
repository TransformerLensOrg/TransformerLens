#!/usr/bin/env python
"""GPU-box verification for the #1618/#1619 (gpt-oss-20b) and #1620 (Olmo-3-1025-7B) fixes.

Run from the repo root on the `fix-gpt-oss-olmo3-parity` branch so the patched
transformer_lens is imported:

    uv run python verify_gpt_oss_olmo3_fixes.py            # all three, sequentially
    uv run python verify_gpt_oss_olmo3_fixes.py bridge     # #1618: bridge vs HF parity
    uv run python verify_gpt_oss_olmo3_fixes.py ht-mxfp4   # #1619: HT loads packed MXFP4
    uv run python verify_gpt_oss_olmo3_fixes.py olmo3      # #1620: Olmo-3 K/V loaded

Box requirements: ~80GB GPU (A100/H100). The MXFP4 test additionally needs
~90GB system RAM while the dequantized 20b state dict is converted. Tests run
sequentially and free each model before the next — do not run them in parallel.

The gpt-oss tests use bf16 (fp32 of the 20b does not fit); thresholds are set
at bf16 noise scale, generous relative to the catastrophic pre-fix numbers
(cos 0.04 mid-stack). Olmo-3 runs in fp32. Reference activations are captured
with forward hooks on the decoder layers — never via output_hidden_states,
whose final entry is post-final-norm and never matches the last resid_post.
"""

import gc
import logging
import sys

import torch
from transformers import AutoModelForCausalLM

# Same prompt ids as the issue reports, so numbers are directly comparable.
GPT_OSS_ID = "openai/gpt-oss-20b"
GPT_OSS_IDS = [[15496, 11, 616, 1438, 318, 1757, 13, 314, 1101, 257]]
GPT_OSS_LAYERS = (0, 12, 23)
OLMO3_ID = "allenai/Olmo-3-1025-7B"
OLMO3_IDS = [[100, 200, 300, 400]]
OLMO3_LAYERS = (0, 16, 31)

BF16_COS_MIN = 0.999
BF16_REL_MAX = 2e-2
FP32_COS_MIN = 0.9999
FP32_REL_MAX = 1e-3


def _free(*objs):
    for obj in objs:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


def _cos_rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a, b = a.double().flatten(), b.double().flatten()
    cos = torch.dot(a, b) / (a.norm() * b.norm())
    return cos.item(), ((a - b).norm() / a.norm()).item()


def _capture_layer_outputs(hf_model, layers, ids):
    """Forward-hook the decoder layers; returns ({layer: output}, logits)."""
    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook(_module, _inputs, output):
            captured[idx] = (output[0] if isinstance(output, tuple) else output).float().cpu()

        return hook

    handles = [hf_model.model.layers[i].register_forward_hook(make_hook(i)) for i in layers]
    try:
        with torch.inference_mode():
            logits = hf_model(ids).logits.float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return captured, logits


def test_bridge_gpt_oss() -> bool:
    """#1618: TransformerBridge resid_post/logits must match the HF forward."""
    print(f"\n=== #1618: bridge vs HF eager parity on {GPT_OSS_ID} (bf16) ===")
    ids = torch.tensor(GPT_OSS_IDS).cuda()

    hf = AutoModelForCausalLM.from_pretrained(
        GPT_OSS_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    ).eval()
    ref, ref_logits = _capture_layer_outputs(hf, GPT_OSS_LAYERS, ids)
    _free(hf)

    from transformer_lens.model_bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers(GPT_OSS_ID, device="cuda", dtype=torch.bfloat16)
    wanted = {f"blocks.{i}.hook_resid_post" for i in GPT_OSS_LAYERS} | {
        "blocks.0.attn.hook_pattern"
    }
    with torch.inference_mode():
        bridge_logits, cache = bridge.run_with_cache(ids, names_filter=lambda n: n in wanted)

    ok = True
    for layer in GPT_OSS_LAYERS:
        cos, rel = _cos_rel(ref[layer], cache[f"blocks.{layer}.hook_resid_post"].float().cpu())
        passed = cos > BF16_COS_MIN and rel < BF16_REL_MAX
        ok &= passed
        print(f"  resid_post.{layer}: cos={cos:.4f} rel={rel:.4f}  {'PASS' if passed else 'FAIL'}")

    cos, rel = _cos_rel(ref_logits, bridge_logits.float().cpu())
    top1 = (ref_logits.argmax(-1) == bridge_logits.float().cpu().argmax(-1)).float().mean().item()
    passed = cos > BF16_COS_MIN and top1 == 1.0
    ok &= passed
    print(
        f"  logits: cos={cos:.4f} rel={rel:.4f} top1-agree={top1:.2f}  {'PASS' if passed else 'FAIL'}"
    )

    # Sink spot-check: pattern rows must sum to < 1 (the sink keeps its share).
    row_sums = cache["blocks.0.attn.hook_pattern"].float().sum(dim=-1)
    sink_active = bool((row_sums < 1.0).all())
    ok &= sink_active
    print(f"  pattern row-sums < 1 (sink active): {'PASS' if sink_active else 'FAIL'}")

    _free(bridge, cache)
    return ok


def test_ht_mxfp4() -> bool:
    """#1619: HookedTransformer must load the packed-MXFP4 checkpoint (auto-dequantize)."""
    print(f"\n=== #1619: HookedTransformer load of {GPT_OSS_ID} (auto-dequantized bf16) ===")
    ids = torch.tensor(GPT_OSS_IDS).cuda()

    hf = AutoModelForCausalLM.from_pretrained(
        GPT_OSS_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    ).eval()
    with torch.inference_mode():
        ref_logits = hf(ids).logits.float().cpu()
    _free(hf)

    from transformer_lens import HookedTransformer

    # Pre-fix this raised: TypeError: 'Tensor' object is not subscriptable
    model = HookedTransformer.from_pretrained_no_processing(
        GPT_OSS_ID, device="cuda", dtype=torch.bfloat16
    )
    with torch.inference_mode():
        logits = model(ids, return_type="logits").float().cpu()

    finite = bool(torch.isfinite(logits).all())
    cos, rel = _cos_rel(ref_logits, logits)
    top1 = (ref_logits.argmax(-1) == logits.argmax(-1)).float().mean().item()
    print(f"  loaded and ran forward: {'PASS' if finite else 'FAIL (non-finite logits)'}")
    # Hard gate since the HT-side sink/sliding-window/yarn fixes: HT must now
    # reproduce the HF forward, not merely load.
    parity = cos > BF16_COS_MIN and top1 == 1.0
    print(
        f"  logits vs HF: cos={cos:.4f} rel={rel:.4f} top1-agree={top1:.2f}  {'PASS' if parity else 'FAIL'}"
    )

    _free(model)
    return finite and parity


def test_olmo3() -> bool:
    """#1620: Olmo-3 K/V must load (no zero-fill) and match the HF forward."""
    print(f"\n=== #1620: HookedTransformer vs HF on {OLMO3_ID} (fp32) ===")
    ids = torch.tensor(OLMO3_IDS).cuda()

    hf = AutoModelForCausalLM.from_pretrained(
        OLMO3_ID, dtype=torch.float32, device_map="cuda", attn_implementation="eager"
    ).eval()
    ref, ref_logits = _capture_layer_outputs(hf, OLMO3_LAYERS, ids)
    _free(hf)

    from transformer_lens import HookedTransformer

    # Pre-fix this printed 64 "Missing key ... _W_K/_W_V" warnings and zero-filled K/V.
    missing_key_records: list[str] = []

    class _Catcher(logging.Handler):
        def emit(self, record):
            if "Missing key" in record.getMessage():
                missing_key_records.append(record.getMessage())

    catcher = _Catcher()
    logging.getLogger().addHandler(catcher)
    try:
        model = HookedTransformer.from_pretrained_no_processing(
            OLMO3_ID, device="cuda", dtype=torch.float32
        )
    finally:
        logging.getLogger().removeHandler(catcher)

    ok = True
    no_missing = not missing_key_records
    ok &= no_missing
    print(f"  no 'Missing key' warnings: {'PASS' if no_missing else 'FAIL'}")
    for msg in missing_key_records[:4]:
        print(f"    {msg}")

    zeroed_k = sum(1 for b in model.blocks if b.attn.W_K.abs().max() == 0)
    zeroed_v = sum(1 for b in model.blocks if b.attn.W_V.abs().max() == 0)
    nonzero = zeroed_k == 0 and zeroed_v == 0
    ok &= nonzero
    print(
        f"  zero-filled W_K: {zeroed_k}/{len(model.blocks)}, "
        f"W_V: {zeroed_v}/{len(model.blocks)}  {'PASS' if nonzero else 'FAIL'}"
    )

    wanted = {f"blocks.{i}.hook_resid_post" for i in OLMO3_LAYERS}
    with torch.inference_mode():
        logits, cache = model.run_with_cache(ids, names_filter=lambda n: n in wanted)

    for layer in OLMO3_LAYERS:
        cos, rel = _cos_rel(ref[layer], cache[f"blocks.{layer}.hook_resid_post"].float().cpu())
        passed = cos > FP32_COS_MIN and rel < FP32_REL_MAX
        ok &= passed
        print(f"  resid_post.{layer}: cos={cos:.4f} rel={rel:.2e}  {'PASS' if passed else 'FAIL'}")

    cos, rel = _cos_rel(ref_logits, logits.float().cpu())
    top1 = (ref_logits.argmax(-1) == logits.float().cpu().argmax(-1)).float().mean().item()
    passed = cos > FP32_COS_MIN and top1 == 1.0
    ok &= passed
    print(
        f"  logits: cos={cos:.4f} rel={rel:.2e} top1-agree={top1:.2f}  {'PASS' if passed else 'FAIL'}"
    )

    _free(model, cache)
    return ok


TESTS = {"bridge": test_bridge_gpt_oss, "ht-mxfp4": test_ht_mxfp4, "olmo3": test_olmo3}


def main() -> int:
    selected = sys.argv[1:] or list(TESTS)
    unknown = [name for name in selected if name not in TESTS]
    if unknown:
        print(f"Unknown test(s): {unknown}. Choose from {list(TESTS)}.")
        return 2

    results = {}
    for name in selected:
        try:
            results[name] = TESTS[name]()
        except Exception as err:  # a crash is a hard FAIL for that test
            print(f"  EXCEPTION in {name}: {type(err).__name__}: {err}")
            results[name] = False
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Summary ===")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
