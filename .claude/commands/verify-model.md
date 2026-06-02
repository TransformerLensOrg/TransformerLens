---
description: Run verify_models.py against a single model (non-parallel). Always dry-run first.
argument-hint: <model_name_or_hf_repo>
---

Verify model `$ARGUMENTS`. If `$ARGUMENTS` is empty, ask the user which model to verify (an HF repo path like `gpt2` or `meta-llama/Llama-2-7b-hf`, or a registry alias) before running.

## Always dry-run first

Verification loads the full model and runs Phases 1–4 — 30 s to 30 min and needs enough free memory to hold the model. **Never invoke the real run blindly.**

```
set -a; source .env; set +a
uv run python -m transformer_lens.tools.model_registry.verify_models --model "$ARGUMENTS" --dry-run
```

Capture: estimated parameter count, projected memory (GB), HF_TOKEN requirement, architecture class.

Decide whether to confirm with the user:

| Model | Action |
|---|---|
| Cached small (`gpt2`, `attn-only-*`, `tiny-stories-1M`, `distilgpt2`, etc.) | Proceed directly; report the dry-run in your response so the user can intervene |
| ≥1B params, or gated, or anything else | Present the dry-run to the user and ask before running |

## Run the verification

```
set -a; source .env; set +a
uv run python -m transformer_lens.tools.model_registry.verify_models --model "$ARGUMENTS"
```

## Optional flags

Full reference: [tools/model_registry/AGENTS.md §Flag reference](../../transformer_lens/tools/model_registry/AGENTS.md#flag-reference).

- `--device cpu|cuda|mps` — override automatic device selection
- `--dtype float32|bfloat16` — override automatic dtype selection
- `--max-memory <gb>` — skip if parameter estimate exceeds; on a 24 GB GPU, `--max-memory 16` leaves headroom for activations
- `--phases 1 2 3` — restrict to a subset (Phase 4 generation is the slowest; restrict when debugging Phase 1 forward parity)
- `--dry-run` — project memory + parameter count without loading the model (see above; always run this first)
- `--no-hf-reference` / `--no-ht-reference` — skip the HF / HT comparison passes (faster, lower confidence)
- `--reverify` — re-test a model already at `status == 1`
- `--retry-failed` — re-test a model at `status == 3` (read its existing `note` first to see the prior failure mode)

Batch flags (`--architectures`, `--per-arch`, `--limit`, `--resume`) are not relevant for `--model <repo>` — use the batch invocation in [tools/model_registry/AGENTS.md §Canonical invocations](../../transformer_lens/tools/model_registry/AGENTS.md#canonical-invocations) if you need them.

## Interpreting the output

`verify_models` enforces hard thresholds (from `_MIN_PHASE_SCORES` in `verify_models.py`):

| Phase | Min score | Required tests | Below = |
|---|---|---|---|
| 1 | 100% | — | `STATUS_FAILED` |
| 2 | 75% | `logits_equivalence`, `loss_equivalence` | `STATUS_FAILED` |
| 3 | 75% | `logits_equivalence`, `loss_equivalence` | `STATUS_FAILED` |
| 4 | 50% | — | **Non-gating** — only adds `"low text quality"` to the `note`; never causes `STATUS_FAILED`. |
| 7 | 75% | `multimodal_forward` | `STATUS_FAILED`. NULL = fail. |
| 8 | 75% | `audio_forward` | `STATUS_FAILED`. NULL = fail. |

`STATUS_VERIFIED` means the system's hard contract is met. The `note` field carries quality flags (`"low text quality"`) or failure details.

**One caveat for adapter authors:** Phase 4's 50% bar is intentionally lenient (it's a coherence metric, not correctness). A small parity-test model that gets Phase 4 well below 100% can still indicate a real adapter bug the system doesn't gate on — most often a missing `preprocess_weights` fold ([supported_architectures/AGENTS.md §When to override preprocess_weights](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#when-to-override-preprocess_weights)) or wrong `default_prepend_bos` ([§Tokenizer policy](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#tokenizer-policy)). Worth investigating even on VERIFIED runs.

Full reference: [tools/model_registry/AGENTS.md §Phase-score thresholds](../../transformer_lens/tools/model_registry/AGENTS.md#phase-score-thresholds).

## Hard rules

**Use `verify_models`, never `main_benchmark`** — only `verify_models` writes `data/supported_models.json`. See [transformer_lens/tools/model_registry/AGENTS.md](../../transformer_lens/tools/model_registry/AGENTS.md) for the canonical-invocations table, file roles, and the resume/checkpoint mechanism.

Run one model at a time — concurrent loads OOM a single device. Report the actual benchmark output (including per-phase scores) and investigate any failure or drift per [AGENTS.md §10](../../AGENTS.md#10-hard-rules).
