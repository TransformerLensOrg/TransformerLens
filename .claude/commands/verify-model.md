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

**Critical**: run verification for **one model at a time**. Never parallelize across models — a single CUDA/MPS device does not have memory for concurrent loads. See [AGENTS.md §10](../../AGENTS.md#10-hard-rules).

Report the actual benchmark output. If any phase fails or shows unexpected drift, investigate before declaring done. Do not claim drift is "fp noise" without empirical evidence.
