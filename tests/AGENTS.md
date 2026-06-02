# Tests — AGENTS.md

Read [the root AGENTS.md](../AGENTS.md) for project-wide rules. This file covers conventions specific to `tests/`.

---

## TL;DR

- **Tier placement matters.** Use [the tier table](#test-tiers) below — a unit test loading a model belongs in `integration/`, not `unit/`.
- **No mocking of model loads or the HF Hub.** Tests load real models. The session-scoped fixtures amortize the cost.
- **Use cached models for fast tests** (`gpt2`, `attn-only-{1,2,3,4}l`, `tiny-stories-1M`). Anything else gets `@pytest.mark.slow`.
- **MPS is a carve-out, not a green light.** `TRANSFORMERLENS_ALLOW_MPS=1` is required; only [`tests/mps/`](mps/) runs there.
- **Hard rules in [AGENTS.md §10](../AGENTS.md#10-hard-rules) apply**: no `xfail`/`skipif` to dodge CI, no dismissing failing tests as "pre-existing," no platform skips outside MPS.
- **Before debugging a failing test, check [QUARANTINES.md](QUARANTINES.md)** — if the failure matches a documented quarantine, it's known. If not, treat it as a real bug.

---

## Test tiers

| Tier | Path | Run | Loads models? | Hits HF Hub? | Scope | Example |
|---|---|---|---|---|---|---|
| `unit` | [`tests/unit/`](unit/) | `make unit-test` | None / synthetic (rare exceptions) | No | Function or single module | [`tests/unit/test_key_value_cache_entry.py`](unit/test_key_value_cache_entry.py) |
| `integration` | [`tests/integration/`](integration/) | `make integration-test` | 1–2 cached models, module-scoped | Yes | Cross-component | [`tests/integration/test_generation_compatibility.py`](integration/test_generation_compatibility.py) |
| `acceptance` | [`tests/acceptance/`](acceptance/) | `make acceptance-test` | Full models (`gpt2`, `bloom-560m`), session-scoped | Yes | End-to-end behaviour | [`tests/acceptance/conftest.py`](acceptance/conftest.py) |
| `benchmarks` | [`tests/benchmarks/`](benchmarks/) | `make benchmark-test` | Varies; performance focus | Yes | Throughput / memory | [`tests/benchmarks/test_boot_memory.py`](benchmarks/test_boot_memory.py) |
| `mps` | [`tests/mps/`](mps/) | `pytest tests/mps -v` (needs `TRANSFORMERLENS_ALLOW_MPS=1`) | TinyStories-1M, fp32 only | Yes | macOS-MPS smoke only | [`tests/mps/test_mps_basic.py`](mps/test_mps_basic.py) |

Common combinations: `make test-pr` (unit + docstring + acceptance + integration — the PR-review surface), `make test` (everything including benchmarks + notebooks).

**Rule of thumb:** new tests that load a model should land in `integration/` by default. The `unit/` tier has a few legitimate model-loading exceptions (e.g. `test_bridge_vs_hooked_transformer_*.py` compares numerics across architectures, which is conceptually unit-scoped) — match that pattern only when the test really is testing isolated behaviour that happens to need a model.

---

## Conftest hierarchy

| Path | Provides |
|---|---|
| [`tests/conftest.py`](conftest.py) | `cleanup_memory` (function autouse) and `cleanup_class_memory` — CUDA/MPS cache + GC; `_enable_hf_retry_for_tests` (session autouse) — wraps HF `from_pretrained` with 429 retry; seeded RNG (numpy/torch/Python @ 42); `gpt2_tokenizer` + `gpt2_hooked_processed` (session); `temp_dir` |
| [`tests/acceptance/conftest.py`](acceptance/conftest.py) | `gpt2_model`, `bloom_560m_hooked`, `bloom_560m_hf_model`, `bloom_560m_hf_tokenizer` (all session) |
| [`tests/acceptance/model_bridge/conftest.py`](acceptance/model_bridge/conftest.py) | Bridge variants of gpt2 with/without compat mode |
| [`tests/integration/model_bridge/conftest.py`](integration/model_bridge/conftest.py) | distilgpt2 + gpt2 Bridge variants × {compat, no-compat, no-processing} |

**Notes:**

- All `transformer_lens` imports inside conftest fixtures live in fixture bodies, not at module top — jaxtyping's `pytest_configure` hook must install before the package is first imported.
- The session-scoped model fixtures (`gpt2_hooked_processed`, `gpt2_bridge`, …) are read-only — mutating them leaks across the entire test session.

---

## Cached-model allowlist

The CI cache in [`.github/workflows/checks.yml`](../.github/workflows/checks.yml) covers: `gpt2`, `gpt2-xl`, `distilgpt2`, `pythia-70m`, `gpt-neo-125M`, `gemma-2-2b-it`, `bloom-560m`, `Qwen2-0.5B`, `bert-base-cased`, `NeelNanda/Attn_Only*`, `roneneldan/TinyStories-1M*`, `NeelNanda/SoLU*`, `redwood_attn_2l`, `tiny-random-llama-2`, `DialoGPT-medium`.

Of those, prefer **`attn-only-{1,2,3,4}l`** and **`tiny-stories-1M`** for fast tests — `gpt2` is "quite slow" on CI's CPU runners ([contributing.md](../docs/source/content/contributing.md)). Use `gpt2` when you specifically need GPT-2 numerics.

Anything outside this set → `@pytest.mark.slow`.

---

## The `slow` marker

Defined in [`pyproject.toml`](../pyproject.toml) as `slow: marks tests as slow (deselect with '-m "not slow"')`. Add it when the test:

- loads a non-cached model
- iterates exhaustively over many parameter combinations
- takes more than ~5 seconds per invocation

Deselect with `pytest -m "not slow"`. Note: the default `make` targets do NOT filter out `slow`; the marker is for ad-hoc local runs.

---

## MPS rules

- The macOS CI job [`mps-checks`](../.github/workflows/checks.yml) sets `TRANSFORMERLENS_ALLOW_MPS=1` and runs `tests/unit`, `tests/integration`, and `tests/mps` on `macos-latest`.
- `get_device()` returns `"cpu"` unless `TRANSFORMERLENS_ALLOW_MPS=1` is set — the default protects against silent MPS divergence.
- The long `--ignore=` list in the workflow is for tests known to diverge on MPS (MoE, optimizer compatibility, KV-cache layout). It is **not** a license to add new skips — it documents existing limitations.
- [`tests/mps/test_mps_basic.py`](mps/test_mps_basic.py) is the template: float32 only (MPS lacks bfloat16), TinyStories-1M only (50 MB fits the runner), `torch.mps.empty_cache()` + `gc.collect()` between tests.
- Top of the module: `pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")` — required on any MPS-only test.

---

## Hard "don'ts"

Repo-wide rules in [AGENTS.md §10](../AGENTS.md#10-hard-rules) apply. Specifics for this directory:

- **No mocking model loads.** Use the session-scoped fixtures; they're cheap enough.
- **No mocking the HF Hub.** Tests hit the real hub with `enable_hf_retry()` handling 429s.
- **No platform `skipif` other than MPS.** No `skipif(sys.platform == 'win32')` or `skipif(not torch.cuda.is_available())` to dodge CI.
- **No `xfail` to dodge a failing test.** Fix the underlying bug — even if it predates your PR.
- **No copying acceptance-tier tests as templates for unit tests.** Their model fixtures will time out or OOM in the unit tier.
