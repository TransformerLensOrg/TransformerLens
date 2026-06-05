# Tests — AGENTS.md

Read [the root AGENTS.md](../AGENTS.md) for project-wide rules. This file covers conventions specific to `tests/`.

> **Just running tests?** Run `make test-pr` for the standard PR-review surface (unit + docstring + acceptance + integration). `make unit-test` for fast feedback. Full tier breakdown in [Test tiers](#test-tiers) below.

---

## TL;DR

- **Tier placement matters** — a unit test loading a model belongs in `integration/`. See [tier table](#test-tiers).
- **No mocking model loads or HF Hub.** Session-scoped fixtures amortize the cost.
- **Cached models for fast tests** (`gpt2`, `attn-only-{1,2,3,4}l`, `tiny-stories-1M`). Anything else → `@pytest.mark.slow`.
- **MPS is a carve-out** — `TRANSFORMERLENS_ALLOW_MPS=1` required; only [`tests/mps/`](mps/) runs there.
- **[AGENTS.md §10](../AGENTS.md#10-hard-rules) applies**: no `xfail`/`skipif` to dodge CI; no platform skips outside MPS.
- **Check [QUARANTINES.md](QUARANTINES.md) before debugging a failing test** — known quarantines have documented reasons.

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

[`tests/conftest.py`](conftest.py) — root, provides:

- `cleanup_memory` (function autouse), `cleanup_class_memory` — CUDA/MPS cache + GC
- `_enable_hf_retry_for_tests` (session autouse) — wraps HF `from_pretrained` with 429 retry
- Seeded RNG (numpy/torch/Python @ 42)
- `gpt2_tokenizer` (session)
- `gpt2_hooked_processed` (session)
- `temp_dir`

Sub-folder conftests:

| Path | Provides |
|---|---|
| [`tests/acceptance/conftest.py`](acceptance/conftest.py) | `gpt2_model`, `bloom_560m_hooked`, `bloom_560m_hf_model`, `bloom_560m_hf_tokenizer` (all session) |
| [`tests/acceptance/model_bridge/conftest.py`](acceptance/model_bridge/conftest.py) | Bridge variants of gpt2 with/without compat mode |
| [`tests/integration/model_bridge/conftest.py`](integration/model_bridge/conftest.py) | distilgpt2 + gpt2 Bridge variants × {compat, no-compat, no-processing} |

Two cross-cutting rules:

- All `transformer_lens` imports inside conftest fixtures live in fixture bodies, not at module top — jaxtyping's `pytest_configure` hook must install before the package is first imported.
- Session-scoped model fixtures (`gpt2_hooked_processed`, `gpt2_bridge`, …) are read-only — mutating them leaks across the entire test session.

---

## Cached-model allowlist

CI cache ([`checks.yml`](../.github/workflows/checks.yml)) covers: `gpt2`, `gpt2-xl`, `distilgpt2`, `pythia-70m`, `gpt-neo-125M`, `gemma-2-2b-it`, `bloom-560m`, `Qwen2-0.5B`, `bert-base-cased`, `NeelNanda/Attn_Only*`, `roneneldan/TinyStories-1M*`, `NeelNanda/SoLU*`, `redwood_attn_2l`, `tiny-random-llama-2`, `DialoGPT-medium`.

Prefer `attn-only-{1,2,3,4}l` and `tiny-stories-1M` for fast tests — `gpt2` is slow on CI's CPU runners. Use `gpt2` only when you need GPT-2 numerics. Anything outside the cached set → `@pytest.mark.slow`.

---

## The `slow` marker

`pyproject.toml`: `"slow: marks tests as slow (deselect with '-m \"not slow\"')"`. Add when the test:

- loads a non-cached model
- iterates exhaustively over many param combos
- takes >5 s per invocation

Deselect with `pytest -m "not slow"`. Default `make` targets do NOT filter; the marker is for ad-hoc runs.

---

## MPS rules

- [`mps-checks`](../.github/workflows/checks.yml) sets `TRANSFORMERLENS_ALLOW_MPS=1` and runs `tests/unit`, `tests/integration`, `tests/mps` on `macos-latest`.
- `get_device()` returns `"cpu"` unless `TRANSFORMERLENS_ALLOW_MPS=1` — protects against silent MPS divergence.
- The workflow's long `--ignore=` list documents existing MPS divergence (MoE, optimizer compat, KV-cache layout); it's **not** a license to add new skips.
- [`tests/mps/test_mps_basic.py`](mps/test_mps_basic.py) is the template: float32 only (no bfloat16 on MPS), TinyStories-1M only (50 MB fits the runner), `torch.mps.empty_cache()` + `gc.collect()` between tests.
- MPS-only modules need: `pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")`.

---

## Hard "don'ts"

Plus [AGENTS.md §10](../AGENTS.md#10-hard-rules):

- **No mocking model loads** — session-scoped fixtures are cheap enough.
- **No mocking the HF Hub** — tests hit the real hub with `enable_hf_retry()` handling 429s.
- **No platform `skipif` outside MPS** — no `skipif(sys.platform == 'win32')` or `skipif(not torch.cuda.is_available())` to dodge CI.
- **No `xfail` to dodge a failing test** — fix the bug, even if pre-existing.
- **No copying acceptance-tier tests as unit-test templates** — their model fixtures time out / OOM at the unit tier.
