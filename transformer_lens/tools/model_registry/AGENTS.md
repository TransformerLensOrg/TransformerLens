# Model Registry — AGENTS.md

Read [the root AGENTS.md](../../../AGENTS.md) for project-wide rules. This file covers only conventions specific to `transformer_lens/tools/model_registry/`.

---

## TL;DR

> **Use `verify_models`, not `main_benchmark`.** Only `python -m transformer_lens.tools.model_registry.verify_models …` updates `data/supported_models.json`. `python -m transformer_lens.benchmarks.main_benchmark …` runs the same benchmark math but defaults to **not** writing the registry (needs `--update-registry`, and even with that flag misses Phase 7 / 8 scores and the resume checkpoint). **If you ran `main_benchmark`, the registry is stale — re-run via `verify_models`.**

- **The only function that writes `supported_models.json` is `update_model_status()` in `registry_io.py`.** Anything that doesn't route through it leaves the registry stale.
- **Never edit `supported_models.json` by hand.** Always go through `update_model_status()`.
- **Never run `verify_models` in parallel.** A single CUDA/MPS device cannot hold concurrent loads (see [AGENTS.md §10](../../../AGENTS.md#10-hard-rules)).
- **HF token required.** `verify_models` calls `AutoConfig.from_pretrained(..., token=get_hf_token())` for parameter estimation; gated models fail without `HF_TOKEN`. Source `.env` first: `set -a; source .env; set +a`.

---

## Canonical invocations

| Goal | Command |
|---|---|
| Verify one specific model + update registry | `python -m transformer_lens.tools.model_registry.verify_models --model <hf_repo>` |
| Verify N models per architecture family | `python -m transformer_lens.tools.model_registry.verify_models --architectures <HFClassName> --per-arch <n>` |
| Verify N models across all architectures | `python -m transformer_lens.tools.model_registry.verify_models --per-arch <n>` |
| Resume after Ctrl-C / crash | Re-run the same command with `--resume` (reads `data/verification_checkpoint.json`) |
| Re-verify already-verified models for an arch | `--reverify --architectures <HFClassName>` |
| See what would run without doing it | Add `--dry-run` |
| Restrict to specific phases | `--phases 1 2 3` |
| Override device / dtype / memory cap | `--device cuda --dtype float32 --max-memory 16` |

`HFClassName` matches the strings in `HF_SUPPORTED_ARCHITECTURES` (see `__init__.py`) — e.g. `LlamaForCausalLM`, `GPT2LMHeadModel`, `Olmo2ForCausalLM`.

---

## File roles

| File | Role |
|---|---|
| `verify_models.py` | **Canonical CLI** for batch verification + registry updates |
| `registry_io.py` | I/O for `supported_models.json`; `update_model_status()` is the only writer |
| `verification.py` | `VerificationRecord` / `VerificationHistory` dataclasses (audit-trail schema) |
| `validate.py` | JSON-schema validation for registry files |
| `api.py` | Read-only programmatic access (`is_model_supported`, `get_architecture_models`, …) |
| `schemas.py` | Dataclasses for model entries, scan info, architecture stats |
| `exceptions.py` | Custom exception types |
| `alias_drift.py` | Detects when legacy `MODEL_ALIASES` and the registry have diverged |
| `discover_architectures.py` | Lightweight HF scan to enumerate architecture classes |
| `hf_scraper.py` | Full HF Hub scan; builds initial supported/unsupported model lists |
| `relevancy.py` | Filters models by download count, foundation-org provenance |
| `generate_report.py` | Renders human-readable status summaries |

`__init__.py` exports the canonical `HF_SUPPORTED_ARCHITECTURES` set and `CANONICAL_AUTHORS_BY_ARCH` map; agents adding a new HF architecture must update both.

---

## `data/verification_checkpoint.json` (gitignored)

Resume state for long-running `verify_models` runs. Lists tested / verified / failed / skipped model IDs plus start timestamp. Behaviour:

- On Ctrl-C, the SIGINT handler in `verify_models.py` finishes the current model, persists the checkpoint, and exits cleanly.
- Re-running with `--resume` reads it and skips already-tested models.
- Deleted automatically on a successful full run.
- Missing or corrupt → fresh run from scratch (safe, just slow).

Agents should never edit it manually; let `verify_models` manage it.

---

## Phase reference

`verify_models` runs the model through phases and writes per-phase scores back into the registry entry. Phases (some don't apply to every architecture — see `applicable_phases` on the adapter):

| Phase | Checks |
|---|---|
| 1 | Core forward correctness vs HuggingFace logits |
| 2 | Hook firing + gradient flow |
| 3 | Weight processing (compatibility mode, fold/centre) |
| 4 | Text-generation quality |
| 7 | Multimodal (vision/text alignment) — only Llava / Gemma3-multimodal |
| 8 | Audio — only Hubert |

A model entry's `status` is `1` (verified) only after all applicable phases pass.

---

## Hard "don'ts"

- **No `main_benchmark` for registry updates.** It misses Phase 7 / 8, doesn't manage the checkpoint, and silently skips the registry write unless `--update-registry` is passed.
- **No parallel `verify_models` runs.** Device OOM. See [AGENTS.md §10](../../../AGENTS.md#10-hard-rules).
- **No manual edits to `data/supported_models.json`.** Always `update_model_status()`.
- **No deleting `data/verification_checkpoint.json` mid-run.** Let the SIGINT handler clean up.
- **No skipping `.env` sourcing.** Gated-model arch verification (Llama / Mistral / Gemma / gated Qwen) needs `HF_TOKEN`.
