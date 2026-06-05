# Benchmarks — AGENTS.md

This is the benchmark library that `verify_models` calls into. Read [the root AGENTS.md](../../AGENTS.md) for project-wide rules.

## ⚠ Not the registry-update path

**Use [`verify_models`](../tools/model_registry/AGENTS.md), not [`main_benchmark.py`](main_benchmark.py), for registry updates.** `main_benchmark` runs the same benchmark math but defaults to NOT writing `data/supported_models.json` (needs `--update-registry`, and even with that flag misses Phase 7 / 8 scores and the resume checkpoint).

If an agent is here because the user asked to "update the registry" or "verify a model," they took the wrong entry point — redirect to [tools/model_registry/AGENTS.md](../tools/model_registry/AGENTS.md).

## What this directory IS for

- The phase-by-phase benchmark implementations (`forward_pass.py`, `generation.py`, `hook_registration.py`, `weight_processing.py`, `multimodal.py`, `audio.py`, `text_quality.py`, `granular_weight_processing.py`, `component_outputs.py`, `backward_gradients.py`, `activation_cache.py`, `component_benchmark.py`, `hook_structure.py`).
- `main_benchmark.py` — exploratory benchmark runner for ad-hoc comparison. Useful for debugging a single model's phase scores without touching the registry.
- `utils.py` — shared helpers including `BenchmarkSeverity`.

## When you ARE in the right place

You're modifying a benchmark implementation (e.g., adding a new check to Phase 2 hook firing), debugging why a phase score is wrong, or adding a benchmark for a new architectural feature. In that case, follow project conventions in [AGENTS.md §10](../../AGENTS.md#10-hard-rules) and ensure the change is exercised by an existing or new `verify_models` phase.
