---
description: Run verify_models.py against a single model (non-parallel).
argument-hint: <model_name_or_hf_repo>
---

Run model verification for `$ARGUMENTS`:

```
set -a; source .env; set +a
uv run python -m transformer_lens.tools.model_registry.verify_models --model "$ARGUMENTS"
```

If `$ARGUMENTS` is empty, ask the user which model to verify (an HF repo path like `gpt2` or `meta-llama/Llama-2-7b-hf`, or a registry alias) before running.

**Optional flags for a single-model run** (full reference: [tools/model_registry/AGENTS.md §Flag reference](../../transformer_lens/tools/model_registry/AGENTS.md#flag-reference)):

- `--device cpu|cuda|mps` — override automatic device selection
- `--dtype float32|bfloat16` — override automatic dtype selection
- `--max-memory <gb>` — skip if parameter estimate exceeds; on a 24 GB GPU, `--max-memory 16` leaves headroom for activations
- `--phases 1 2 3` — restrict to a subset (Phase 4 generation is the slowest; restrict when debugging Phase 1 forward parity)
- `--dry-run` — project memory + parameter count without loading the model. Use this before any expensive run, and as the data source for the confirmation prompt in [`/add-model-support`](add-model-support.md).
- `--no-hf-reference` / `--no-ht-reference` — skip the HF / HT comparison passes (faster, lower confidence)
- `--reverify` — re-test a model already at `status == 1`
- `--retry-failed` — re-test a model at `status == 3` (read its existing `note` first to see the prior failure mode)

Batch flags (`--architectures`, `--per-arch`, `--limit`, `--resume`) are not relevant for `--model <repo>` — use the batch invocation in [tools/model_registry/AGENTS.md §Canonical invocations](../../transformer_lens/tools/model_registry/AGENTS.md#canonical-invocations) if you need them.

**Use `verify_models`, never `main_benchmark`** — only `verify_models` writes `data/supported_models.json`. See [transformer_lens/tools/model_registry/AGENTS.md](../../transformer_lens/tools/model_registry/AGENTS.md) for the canonical-invocations table, file roles, and the resume/checkpoint mechanism.

Run one model at a time — concurrent loads OOM a single device. Report the actual benchmark output and investigate any failure or drift per [AGENTS.md §10](../../AGENTS.md#10-hard-rules). For phase failures or logit-parity drift, the bisection workflow lives in [docs/source/content/debugging_numerical_divergence.md](../../docs/source/content/debugging_numerical_divergence.md).
