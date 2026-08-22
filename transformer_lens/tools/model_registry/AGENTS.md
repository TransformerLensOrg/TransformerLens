# Model Registry — AGENTS.md

Read [the root AGENTS.md](../../../AGENTS.md) for project-wide rules. This file covers only conventions specific to `transformer_lens/tools/model_registry/`.

---

## TL;DR

> **Use `verify_models`, not `main_benchmark`.** Only `verify_models` writes `data/supported_models.json`. `main_benchmark` runs the same math but defaults to NOT writing the registry (needs `--update-registry`, and even with it misses Phase 7/8/9 scores and the resume checkpoint). If you ran `main_benchmark`, the registry is stale.

- **`update_model_status()` in `registry_io.py` is the only mutator of status/phase/note on existing entries.** Never set by hand.
- **Adding a new model-ID entry is allowed** (required before `verify_models --model <repo>` can find it). See [Adding a new model entry](#adding-a-new-model-entry).
- **Never run in parallel** — single CUDA/MPS device OOMs ([AGENTS.md §10](../../../AGENTS.md#10-hard-rules)).
- **HF token required** for parameter estimation on gated models. Source `.env`: `set -a; source .env; set +a`.

---

## Canonical invocations

| Goal | Command |
|---|---|
| Verify one specific model + update registry | `uv run python -m transformer_lens.tools.model_registry.verify_models --model <hf_repo>` |
| Verify N models per architecture family | `uv run python -m transformer_lens.tools.model_registry.verify_models --architectures <HFClassName> --per-arch <n>` |
| Verify N models across all architectures | `uv run python -m transformer_lens.tools.model_registry.verify_models --per-arch <n>` |
| Resume after Ctrl-C / crash | Re-run the same command with `--resume` (reads `data/verification_checkpoint.json`) |
| Re-verify already-verified models for an arch | `--reverify --architectures <HFClassName>` |
| See what would run without doing it | Add `--dry-run` |
| Restrict to specific phases | `--phases 1 2 3` |
| Override device / dtype / memory cap | `--device cuda --dtype float32 --max-memory 16` |

`HFClassName` matches the strings in `HF_SUPPORTED_ARCHITECTURES` (see `__init__.py`) — e.g. `LlamaForCausalLM`, `GPT2LMHeadModel`, `Olmo2ForCausalLM`.

### Flag reference

| Flag | Meaning |
|---|---|
| `--model <repo>` | Verify a single HF repo (must already exist as an entry in `supported_models.json`) |
| `--architectures <ClassName...>` | Restrict to one or more HF architecture classes |
| `--per-arch <n>` | Verify the top-N unverified models per architecture (default 10) |
| `--limit <n>` | Cap total models verified across all architectures |
| `--device <cpu\|cuda\|mps>` | Override automatic device selection |
| `--dtype <float32\|bfloat16>` | Override automatic dtype selection |
| `--max-memory <gb>` | Skip models whose parameter-count estimate exceeds this GB cap (default: tries every model that fits available device memory). Use this to avoid OOM on a small device — e.g. `--max-memory 16` on a 24 GB GPU leaves head-room for activations. |
| `--phases <n...>` | Restrict to specific phases (default `1 2 3 4`; Phase 7/8/9 are auto-skipped for non-applicable architectures) |
| `--resume` | Read `data/verification_checkpoint.json` and skip models already tested in the in-flight run |
| `--reverify` | Re-test already-verified models (default skips status=1 entries) |
| `--retry-failed` | Re-test status=3 (failed) entries |
| `--dry-run` | Print what would be tested without running |
| `--no-hf-reference` | Skip the HF reference — Phase 1 is structural-only, never numerically compared to HF. A passing run records `status=4` (PROVISIONAL), which does **not** count as verified. Re-run without the flag to upgrade to VERIFIED. |
| `--no-ht-reference` | Skip the HookedTransformer comparison passes (faster, lower confidence) |
| `--quiet` | Suppress per-model logging |

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
| `generate_report.py` | Renders human-readable status summaries; holds `ARCHITECTURE_DESCRIPTIONS` |

`__init__.py` exports the canonical `HF_SUPPORTED_ARCHITECTURES` set and `CANONICAL_AUTHORS_BY_ARCH` map; agents adding a new HF architecture must update both.

---

## Adding a new model entry

To verify a model not yet in `data/supported_models.json`, hand-add the entry first. This is the **only** allowed hand-edit:

```json
{
  "architecture_id": "MyArchForCausalLM",
  "model_id": "org/repo-name",
  "status": 0,
  "verified_date": null, "metadata": null, "note": null,
  "phase1_score": null, "phase2_score": null, "phase3_score": null,
  "phase4_score": null, "phase7_score": null, "phase8_score": null,
  "phase9_score": null
}
```

`verify_models --model org/repo-name` then populates status/score/note via `update_model_status()`. Never set those fields manually.

---

## `data/verification_checkpoint.json` (gitignored)

Resume state for long-running runs (tested/verified/failed/skipped IDs + timestamp):

- Ctrl-C → SIGINT handler finishes current model, persists checkpoint, exits cleanly.
- `--resume` reads it, skips already-tested models.
- Deleted on successful full run; missing/corrupt → fresh run (safe).

Never edit manually.

---

## Phase reference

`verify_models` runs the model through phases and writes per-phase scores back into the registry entry. Phases (some don't apply to every architecture — see `applicable_phases` on the adapter):

| Phase | Checks |
|---|---|
| 1 | Core forward correctness vs HuggingFace logits |
| 2 | Hook firing + gradient flow |
| 3 | Weight processing (compatibility mode, fold/centre) |
| 4 | Text-generation quality (per-model prompt profile, scored by a pinned multilingual judge) |
| 7 | Multimodal (vision/text alignment) — only Llava / Gemma3-multimodal |
| 8 | Audio — Hubert (waveform) and AST (spectrogram) |
| 9 | Vision — ViT/DeiT pixel forward, hook/cache firing, representation stability, classification decode |

SSM / recurrent families and the hybrids (Mamba-1/2, gated-delta-net, NemotronH, GraniteMoeHybrid, Jamba, Qwen3.5/Qwen3-Next) declare `applicable_phases = [1, 2, 3, 4]` — all four apply. P2/P3 run but skip their HookedTransformer-comparison sub-tests (SSMs have no HT), which is scored as a pass.

**Non-text modalities.** `classify_architecture` routes audio architectures to `{1, 8}` and vision architectures (ViT/DeiT) to `{1, 9}` — vision has no tokenizer for P4, and neither modality has a HookedTransformer counterpart for P2/P3. Phases 1/8/9 build their input with `build_modality_input()` ([`benchmarks/utils.py`](../../benchmarks/utils.py)), which shapes it from the HF config: `[batch, max_length, num_mel_bins]` for spectrogram encoders, `[batch, samples]` for waveform encoders, `[batch, channels, image_size, image_size]` for vision. Add a new non-text architecture there rather than hardcoding a shape at the call site.

### Phase-score thresholds

`verify_models` enforces hard pass/fail at the thresholds in `_MIN_PHASE_SCORES` ([`verify_models.py:508`](verify_models.py)). Below threshold OR a required-test failure → `STATUS_FAILED`. The contract:

| Phase | Min score | Required tests | Effect when below threshold or required tests fail |
|---|---|---|---|
| 1 | **100%** | — | `STATUS_FAILED` |
| 2 | 75% | `logits_equivalence`, `loss_equivalence` | `STATUS_FAILED` |
| 3 | 75% | `logits_equivalence`, `loss_equivalence` | `STATUS_FAILED` |
| 4 | 54.5% — the measured pass line `p4_pass_threshold()` (score of the bake-off noise floor `JUDGE_R_GOOD`) | — | **Non-gating.** Below the line adds `"text quality poor (P4=…)"` to the registry `note`; never causes `STATUS_FAILED`. |
| 7 | 75% | `multimodal_forward` | `STATUS_FAILED`. NULL score (processor unavailable) also fails. |
| 8 | 75% | `audio_forward` | `STATUS_FAILED`. NULL score also fails. |
| 9 | 75% | `vision_forward`, `vision_cache` | `STATUS_FAILED`. NULL score also fails. |

P4 prompts each model with its resolved **prompt profile** — chat template, translation, code, own-language continuation, or another task kind — via `resolve_profile()` in [`benchmarks/text_quality_profiles.py`](../../benchmarks/text_quality_profiles.py) (precedence: per-model override > architecture rule > live HF Hub signals > stored registry value > default). The resolved profile is cached sparsely on the entry as `prompt_profile` (key omitted when it's just the default). Each generation is scored against a known-good reference by one pinned multilingual judge via the perplexity ratio `PPL(generated)/PPL(reference)`, which cancels the judge's per-language handicap; the pass/fail constants are measured, not hand-picked, in [`benchmarks/text_quality.py`](../../benchmarks/text_quality.py).

Phase 4 is a quality metric, not a correctness check. Its floor is not hand-picked: it equals the benchmark pass line, derived from the judge bake-off's fluent-vs-fluent noise floor (`p4_pass_threshold()` in [`benchmarks/text_quality_profiles.py`](../../benchmarks/text_quality_profiles.py)), so the registry note and the benchmark verdict can never disagree.

**For adapter authors:** a `STATUS_VERIFIED` entry with P4 well below 100% on a small parity-test model can still indicate a real bug the system doesn't gate on (e.g. missing `preprocess_weights` fold). Investigate manually even when VERIFIED.

**Reading the result:**

- `status==1` + `note="Full verification completed"` → all gates passed, no quality flag. Good.
- `status==1` + `note` mentions `"text quality poor"` → P4 below the pass line; investigate (`scripts/phase4_review.py` orders the candidates and separates old-scale scores).
- `status==1` + P4 < 100% on a small model, no quality flag → potential weight-fold/tokenizer bug; investigate.
- `status==3` (FAILED) → `note` carries the failure reason; debug from there.
- `status==4` (PROVISIONAL) → structural-only pass via `--no-hf-reference`; Phase 1 was never numerically compared to HF, so it does **not** count as verified (`note` is prefixed `Structural only (no HF reference)`). Re-run without the flag for a real verification.

P1/P3 failures: [supported_architectures/AGENTS.md §When to override preprocess_weights](../../model_bridge/supported_architectures/AGENTS.md#when-to-override-preprocess_weights), [debugging_numerical_divergence.md](../../../docs/source/content/debugging_numerical_divergence.md). P4 drift: [§Tokenizer policy](../../model_bridge/supported_architectures/AGENTS.md#tokenizer-policy) (logit-scale / embedding-scale folds typically degrade P4 without crossing the pass line).

---

## Hard "don'ts"

- **No `main_benchmark` for registry updates** — misses P7/P8, no checkpoint, no registry write without `--update-registry`.
- **No parallel `verify_models`** — device OOM ([AGENTS.md §10](../../../AGENTS.md#10-hard-rules)).
- **No manual edits to existing entries' `status`/`verified_date`/`note`/`phaseN_score`** — only `update_model_status()` writes those. (New entries OK — see [Adding a new model entry](#adding-a-new-model-entry).)
- **No deleting `data/verification_checkpoint.json` mid-run** — let SIGINT clean up.
- **No skipping `.env`** — gated-model verification needs `HF_TOKEN`.
