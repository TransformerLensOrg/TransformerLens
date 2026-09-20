# Scripts Reference

Every standalone tool under [`scripts/`](../scripts/). The agent prompts invoke several of these directly via `$TL_ADAPTER_BUILDER_ROOT/scripts/…`; the rest are for human use. Shell scripts are bash; Python scripts expect to run via `uv run python` from inside the target worktree when they touch the TransformerLens registry.

## Adapter analysis and scaffolding

### [`analyze-hf-model.py`](../scripts/analyze-hf-model.py)

Analyzes a HuggingFace model to extract adapter-relevant information. This is the Programmer's main tool during Step 1a.2 (generate a scaffold adapter).

```bash
# Config-only analysis (no model download)
python scripts/analyze-hf-model.py meta-llama/Llama-3.1-8B

# Generate a pre-filled adapter file (meta device, no weights downloaded)
python scripts/analyze-hf-model.py Salesforce/codegen-350M-mono --scaffold
python scripts/analyze-hf-model.py Salesforce/codegen-350M-mono --scaffold --scaffold-out adapter.py

# Full model inspection (downloads model)
python scripts/analyze-hf-model.py meta-llama/Llama-3.1-8B --modules --state-dict

# JSON output for programmatic use
python scripts/analyze-hf-model.py meta-llama/Llama-3.1-8B --json
```

The `--scaffold` flag is the key feature: it instantiates the model on a meta device (no weight download), inspects the module tree and state-dict keys, and emits a ready-to-refine adapter `.py` file with config attributes pre-set, component-mapping paths detected from real module names, and weight-conversion stubs. The Programmer uses this as the starting point — not the generic [`docs/adapter-template.py`](adapter-template.py).

### [`scan-hf-architecture.py`](../scripts/scan-hf-architecture.py)

Exhaustively scans HuggingFace for every model of a given architecture class and writes the results to `transformer_lens/tools/model_registry/data/supported_models_<arch_short>.json`. Invoked by the Programmer during Step 1a.2.

```bash
# Run from the worktree root
uv run python "$TL_ADAPTER_BUILDER_ROOT/scripts/scan-hf-architecture.py" \
    Qwen3MoeForCausalLM --arch-short qwen3_moe
```

Produces an authoritative per-architecture artifact that stays in place for human review and later gets merged into the main registry via `port-arch-models.py`. Replaces the ~70-line inline Python that used to live in `programmer.md`.

Tuning knobs: `--limit N` caps the HF listing pagination (default 10000), `--output-dir DIR` overrides the registry path (default matches the worktree layout).

### [`validate-architecture.py`](../scripts/validate-architecture.py)

Pre-flight check used by [`launch.sh`](../agents/launch.sh) to validate that an architecture class name actually exists before creating a worktree. Catches typos like `CohoreForCausalLM` before they burn agent time.

```bash
# Run from any cwd that has uv + the TransformerLens project env available
uv run --project "$DEFAULT_TARGET_REPO" \
  python "$TL_ADAPTER_BUILDER_ROOT/scripts/validate-architecture.py" CohereForCausalLM
```

Two checks in order:

1. **`transformers` package** — `getattr(transformers, <arch>)` to exercise the lazy-loading machinery. Fast, local, ~1–2s including cold import.
2. **HuggingFace Hub** — bounded scan of the top 500 models by download count (configurable via `--hf-limit`), breaking on first match. Only runs when the transformers check misses. ~1–3s on top of the transformers import.

**Flags:**

- `--hf-limit N` — cap on the HF Hub scan (default 500)
- `--skip-hf` — only check the transformers package; don't fall back to HF Hub
- `--quiet` — suppress the "OK: …" message on success (errors still print)

**Exit codes:**

- `0` — found in transformers or on HF Hub
- `1` — definitively not found (both checks ran, neither found a match)
- `2` — could not verify (missing deps, network error); caller decides

The launcher treats `0` as proceed, `1` as abort, and `2` as warn+proceed. See [cli-reference.md § Pre-flight: Architecture Existence Check](cli-reference.md#pre-flight-architecture-existence-check) for details and the `--skip-arch-check` bypass.

### [`port-arch-models.py`](../scripts/port-arch-models.py)

Merges the per-architecture model list produced by `scan-hf-architecture.py` into `supported_models.json` so that `verify_models` can pick the models up. Invoked by the Programmer during Step 3.0 (before verification).

```bash
uv run python "$TL_ADAPTER_BUILDER_ROOT/scripts/port-arch-models.py" \
    --arch-short qwen3_moe
```

Idempotent — models already present in the registry are skipped. The per-arch file is not modified. If this step is skipped, `verify_models` silently no-ops (the `guard-verify-models.sh` hook catches this case and blocks the call with an error message pointing here).

## Adapter validation

### [`validate-adapter.sh`](../scripts/validate-adapter.sh)

Validates an adapter's structure and optionally its semantics against a real model.

```bash
# Structural validation only (no model download)
./scripts/validate-adapter.sh llama

# Deep validation — cross-references against a real model (config-only, meta device)
./scripts/validate-adapter.sh llama --model meta-llama/Llama-3.2-1B
```

**Structural checks:** adapter file exists, class extends `ArchitectureAdapter`, `component_mapping` is set, `weight_processing_conversions` is set, registered in `supported_architectures/__init__.py` and the factory, config attributes set, import test passes.

**Deep checks (with `--model`):** loads the model on a meta device, verifies component-mapping paths resolve to real HF modules, weight conversion keys match state-dict parameters, no unmapped projection weights. Catches mismatches before `verify_models` downloads the full model.

### [`validate-adapter-deep.py`](../scripts/validate-adapter-deep.py)

The Python backend for `--model` deep validation. Usually invoked via `validate-adapter.sh` but can be run standalone for scripting.

### [`compare-adapters.sh`](../scripts/compare-adapters.sh)

Structured diff between two existing adapters. The Programmer uses this in Step 1a.3 to study differences between candidate reference adapters.

```bash
./scripts/compare-adapters.sh llama qwen2
./scripts/compare-adapters.sh gemma1 gemma2
```

Outputs: config attribute differences, bridge-component differences, component-mapping path differences, weight-conversion differences, optional override differences, and a truncated code diff.

## Operational tooling

### [`format-timeline.py`](../scripts/format-timeline.py)

Formats `.adapter-workspace/timeline.jsonl` entries for human-readable display. Reads JSONL from stdin, writes one formatted line per entry to stdout.

Used internally by `ops.sh`:

```bash
# From status (single latest event)
tail -1 timeline.jsonl | python3 scripts/format-timeline.py

# From logs (live stream)
tail -f timeline.jsonl | python3 scripts/format-timeline.py
```

Each line renders as `HH:MM:SS  EventName      label`, where `label` is the agent-authored `description`, a truncated Bash command, a file path, or the raw event name in that order. Malformed lines are annotated rather than dropped.

### [`notify.sh`](../scripts/notify.sh)

Sends completion notifications via a fallback chain:

1. **Slack/Discord webhook** — if `NOTIFICATION_WEBHOOK_URL` is set (auto-detects payload format)
2. **iMessage** — if `NOTIFICATION_NUMBER` is set and on macOS
3. **macOS Notification Center** — last resort on macOS
4. **stdout** — always works as a fallback

Invoked automatically by `agents/hooks/notify-on-completion.sh` when a session ends with `verification_passed: true`. The orchestrator also calls it directly for the "stuck after 3 review rounds" and "verification skipped — all models too large" escalations.

### [`dry-run-test.sh`](../scripts/dry-run-test.sh)

Self-test suite that validates this project's integrity. Safe to run anytime — it doesn't touch the TransformerLens repo.

```bash
./scripts/dry-run-test.sh
```

Auto-discovers and syntax-checks every `.sh` and `.py` file under `agents/`, `scripts/`, and `docs/`. Validates that domain docs exist, agent definitions have required sections, signals are consistent across `signals.sh` and the agent prompts, `validate-adapter.sh` runs against an existing adapter, and the TransformerLens repo is accessible.
