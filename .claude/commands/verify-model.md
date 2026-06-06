---
description: Run verify_models.py against a single model (non-parallel). Always dry-run first.
argument-hint: <model_name_or_hf_repo>
---

Verify model `$ARGUMENTS`. If empty, ask for an HF repo path (e.g. `gpt2`, `meta-llama/Llama-2-7b-hf`) or registry alias.

## Always dry-run first

Verification loads the full model and runs Phases 1–4 — 30 s to 30 min, needs memory to hold the model. **Never invoke the real run blindly.**

```
set -a; source .env; set +a
uv run python -m transformer_lens.tools.model_registry.verify_models --model "$ARGUMENTS" --dry-run
```

Capture: estimated parameter count, projected memory (GB), HF_TOKEN requirement, architecture class.

| Model | Action |
|---|---|
| Cached small (`gpt2`, `attn-only-*`, `tiny-stories-1M`, `distilgpt2`, …) | Proceed; report dry-run in your response so user can intervene |
| ≥1B params, gated, or anything else | Present dry-run, ask before running |

## Run the verification

```
set -a; source .env; set +a
uv run python -m transformer_lens.tools.model_registry.verify_models --model "$ARGUMENTS"
```

## Optional flags

Full reference: [tools/model_registry/AGENTS.md §Flag reference](../../transformer_lens/tools/model_registry/AGENTS.md#flag-reference).

- `--device cpu|cuda|mps` — override device selection
- `--dtype float32|bfloat16` — override dtype
- `--max-memory <gb>` — skip if param estimate exceeds; e.g. `16` on a 24 GB GPU leaves headroom for activations
- `--phases 1 2 3` — restrict (P4 is slowest; restrict when debugging P1 forward parity)
- `--dry-run` — see above; always first
- `--no-hf-reference` / `--no-ht-reference` — skip HF / HT comparison (faster, lower confidence)
- `--reverify` — re-test `status==1`
- `--retry-failed` — re-test `status==3` (read existing `note` first)

Batch flags (`--architectures`, `--per-arch`, `--limit`, `--resume`) don't apply to `--model <repo>` — use [§Canonical invocations](../../transformer_lens/tools/model_registry/AGENTS.md#canonical-invocations).

## Interpreting the output

Hard thresholds (`_MIN_PHASE_SCORES` in `verify_models.py`):

| Phase | Min score | Required tests | Below = |
|---|---|---|---|
| 1 | 100% | — | `STATUS_FAILED` |
| 2 | 75% | `logits_equivalence`, `loss_equivalence` | `STATUS_FAILED` |
| 3 | 75% | `logits_equivalence`, `loss_equivalence` | `STATUS_FAILED` |
| 4 | 50% | — | **Non-gating** — adds `"low text quality"` to `note`; never fails. |
| 7 | 75% | `multimodal_forward` | `STATUS_FAILED`. NULL = fail. |
| 8 | 75% | `audio_forward` | `STATUS_FAILED`. NULL = fail. |

`STATUS_VERIFIED` means hard gates passed. `note` carries quality flags or failure details.

**Adapter-author caveat:** P4's 50% bar is intentionally lenient (coherence, not correctness). P4 well below 100% on a small parity-test model can indicate a real bug the system doesn't gate on — most often a missing [`preprocess_weights` fold](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#when-to-override-preprocess_weights) or wrong [`default_prepend_bos`](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#tokenizer-policy). Investigate even on VERIFIED.

Full reference: [§Phase-score thresholds](../../transformer_lens/tools/model_registry/AGENTS.md#phase-score-thresholds).

## Hard rules

**Use `verify_models`, never `main_benchmark`** — only `verify_models` writes `data/supported_models.json` ([tools/model_registry/AGENTS.md](../../transformer_lens/tools/model_registry/AGENTS.md)).

One model at a time — concurrent loads OOM. Report actual per-phase scores; investigate failures per [AGENTS.md §10](../../AGENTS.md#10-hard-rules).
