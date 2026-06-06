---
name: Verify model support
about: Track a request to run verify_models on a specific model that someone without appropriate hardware can't run themselves
title: "[Verify Model] org/model-id"
labels: verification-request

---

<!--
File this issue when you've followed the /add-model-support workflow but couldn't run verification yourself (insufficient memory / no GPU / gated model without HF_TOKEN access). A maintainer with appropriate hardware will pick it up.

The architecture adapter must already exist for this template to fit. If it doesn't, file a feature request / proposal instead.
-->

## Model

- HF repo: `https://huggingface.co/REPLACE_WITH_MODEL_ID`
- Architecture class (from `config.architectures[0]`): `REPLACE_WITH_HF_ARCH_CLASS`
- Estimated parameters: `REPLACE_WITH_PARAM_COUNT`
- Projected memory: `REPLACE_WITH_GB` GB (from `verify_models --dry-run`)
- Gated repo: `yes / no` (HF_TOKEN required: `yes / no`)

## Registry state

- Architecture adapter exists: `yes` (file: `transformer_lens/model_bridge/supported_architectures/REPLACE_WITH_ADAPTER.py`)
- Currently in `data/supported_models.json`:
  - [ ] No — entry added in the PR linked from this issue
  - [ ] Yes, status: `REPLACE_WITH_STATUS` (0=unverified, 2=skipped, 3=failed)

## Motivation

<!-- What are you trying to do? Any symptom you've observed on this model that motivated the request? -->


## How to run

```bash
set -a; source .env; set +a
uv run python -m transformer_lens.tools.model_registry.verify_models --model REPLACE_WITH_MODEL_ID
```

For the full workflow, see [Creating Architecture Adapters in contributing.md](../../docs/source/content/contributing.md#creating-architecture-adapters).

## Result

<!-- A maintainer running the verification fills this in:
- Phases passed / failed
- Final status written to supported_models.json
- Any notes added to the registry entry
- Link to the PR that closes this issue
-->
