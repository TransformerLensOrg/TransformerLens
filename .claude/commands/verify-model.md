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

**Use `verify_models`, never `main_benchmark`** — only `verify_models` writes `data/supported_models.json`. See [transformer_lens/tools/model_registry/AGENTS.md](../../transformer_lens/tools/model_registry/AGENTS.md) for the canonical-invocations table, file roles, and the resume/checkpoint mechanism.

Run one model at a time — concurrent loads OOM a single device. Report the actual benchmark output and investigate any failure or drift per [AGENTS.md §10](../../AGENTS.md#10-hard-rules).
