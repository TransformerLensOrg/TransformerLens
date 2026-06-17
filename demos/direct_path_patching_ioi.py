"""
Direct Path Patching — Real Experiment on GPT-2 Small (IOI task)
=================================================================

Indirect Object Identification (IOI) task from Wang et al. 2022:
  Clean:     "When Mary and John went to the store, John gave a drink to"  → Mary
  Corrupted: "When Mary and John went to the store, Mary gave a drink to"  → John

We measure logit(Mary) - logit(John) as our metric.

For each source head known to be important in the IOI circuit
(S-inhibition heads: 8.6, 8.10, 7.3, 7.9), we patch its output
directly into the queries of all downstream heads and see which
(src → dst) paths carry the most information.
"""

import importlib.util
import os
import sys

import einops
import torch

from transformer_lens import HookedTransformer

# Load our local module
_path = os.path.join(os.path.dirname(__file__), "..", "transformer_lens", "direct_path_patching.py")
_spec = importlib.util.spec_from_file_location("direct_path_patching", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_act_patch_direct_path = _mod.get_act_patch_direct_path

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print("Loading GPT-2 small…")
model = HookedTransformer.from_pretrained(
    "gpt2", center_unembed=True, center_writing_weights=True, fold_ln=True
)
model.eval()

# ---------------------------------------------------------------------------
# 2. Define IOI prompts
# ---------------------------------------------------------------------------
# Classic IOI pair from Wang et al. 2022
CLEAN_PROMPT = "When Mary and John went to the store, John gave a drink to"
CORRUPTED_PROMPT = "When Mary and John went to the store, Mary gave a drink to"

clean_tokens = model.to_tokens(CLEAN_PROMPT)  # [1, seq]
corrupted_tokens = model.to_tokens(CORRUPTED_PROMPT)  # [1, seq]

# Token IDs for Mary and John
mary_token = model.to_single_token(" Mary")
john_token = model.to_single_token(" John")
print(f"Mary token id: {mary_token}, John token id: {john_token}")


# ---------------------------------------------------------------------------
# 3. Metric: logit(Mary) - logit(John) at the last token position
# ---------------------------------------------------------------------------
def logit_diff(logits):
    """Higher = more correct (predicts Mary over John)."""
    last = logits[0, -1, :]
    return last[mary_token] - last[john_token]


# Baselines
with torch.no_grad():
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_baseline = logit_diff(clean_logits).item()
corrupted_baseline = logit_diff(corrupted_logits).item()
print(f"\nClean logit diff:     {clean_baseline:+.3f}  (correctly prefers Mary)")
print(f"Corrupted logit diff: {corrupted_baseline:+.3f}  (incorrectly prefers John)")


# Normalised metric: 0 = corrupted baseline, 1 = clean baseline
def normalised_metric(logits):
    raw = logit_diff(logits)
    return (raw - corrupted_baseline) / (clean_baseline - corrupted_baseline)


# ---------------------------------------------------------------------------
# 4. Run direct path patching for known IOI circuit heads
# ---------------------------------------------------------------------------
# S-inhibition heads that write to the Name Mover heads' queries
ioi_src_heads = [
    (7, 3),  # S-inhibition head
    (7, 9),  # S-inhibition head
    (8, 6),  # S-inhibition head
    (8, 10),  # S-inhibition head
]

print("\n" + "=" * 60)
print("DIRECT PATH PATCHING RESULTS")
print("Metric: normalised logit diff (0=corrupted, 1=clean)")
print("=" * 60)

all_results = {}
for sl, sh in ioi_src_heads:
    print(f"\nSource head ({sl},{sh}) → all downstream heads [Q input]")
    with torch.no_grad():
        results = get_act_patch_direct_path(
            model=model,
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            corrupted_cache=corrupted_cache,
            patching_metric=normalised_metric,
            src_layer=sl,
            src_head=sh,
            component="q",
            verbose=True,
        )
    all_results[(sl, sh)] = results

    # Top 5 (dst_layer, dst_head) destinations
    flat = results.view(-1)
    top5_idx = flat.topk(5).indices
    print(f"  Top 5 destinations (by normalised metric):")
    for idx in top5_idx:
        dl = idx.item() // model.cfg.n_heads
        dh = idx.item() % model.cfg.n_heads
        val = results[dl, dh].item()
        print(f"    ({sl},{sh}) → ({dl},{dh}):  {val:+.4f}")

# ---------------------------------------------------------------------------
# 5. Summary: Known name-mover head query inputs
# ---------------------------------------------------------------------------
name_movers = [(9, 9), (9, 6), (10, 0)]  # confirmed in IOI paper

print("\n" + "=" * 60)
print("DIRECT PATH: S-inhibition → Name-Mover (Q) scores")
print("Expected: strong signal for known circuit edges")
print("=" * 60)
print(f"{'Src head':>10}  {'Dst head':>10}  {'Score':>8}")
print("-" * 34)
for (sl, sh), results in all_results.items():
    for dl, dh in name_movers:
        if dl > sl:
            val = results[dl, dh].item()
            print(f"   ({sl:2d},{sh:2d})   →   ({dl:2d},{dh:2d})    {val:+.4f}")

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
torch.save(
    {k: v for k, v in all_results.items()},
    os.path.join(os.path.dirname(__file__), "results_direct_path_ioi.pt"),
)
print("\nResults saved to demos/results_direct_path_ioi.pt")
print("\nDone.")
