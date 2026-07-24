# AGENTS.md

Guidance for AI coding agents contributing to **TransformerLens**.

This file is the single source of truth. Vendor-specific files ([CLAUDE.md](CLAUDE.md), [.github/copilot-instructions.md](.github/copilot-instructions.md), [.cursor/rules/transformerlens.mdc](.cursor/rules/transformerlens.mdc)) and sub-folder `AGENTS.md` files all defer here.

> **Just want to use the library?** Start with the [README](README.md). **Just want to run the tests?** Skip to [§3 Quickstart](#3-quickstart). **Contributing?** Read on.

## TL;DR

1. **Use `uv`**, not `pip` or `poetry` (`uv sync`).
2. **Source `.env`** (`set -a; source .env; set +a`) before any HF-Hub command.
3. **Base PRs on `dev`**, not `main`. Never name a branch `main` or `dev`.
4. **Mirror HookedTransformer → TransformerBridge** when behaviour exists in both ([§2](#2-two-systems-live-in-this-repo)).
5. **`make format` + `uv run mypy .` before push** — no pre-commit hook.
6. **Never add `# type: ignore`** ([§10](#10-hard-rules)).
7. **Never dismiss a failing test as "pre-existing"** ([§10](#10-hard-rules)).

Sub-folder rules: [tests/AGENTS.md](tests/AGENTS.md) · [supported_architectures/AGENTS.md](transformer_lens/model_bridge/supported_architectures/AGENTS.md) · [tools/model_registry/AGENTS.md](transformer_lens/tools/model_registry/AGENTS.md).

---

## 1. What this repo is

**TransformerLens** — mechanistic-interpretability library. Loads 9,000+ models across 50+ architecture families (see [supported_models.json](transformer_lens/tools/model_registry/data/supported_models.json)) and exposes internal activations through a hook system for caching, editing, and ablating intermediate state. Built on HuggingFace `transformers`.

## 2. Two systems live in this repo

| System | Status | Lives in | Numerics | Registry |
|---|---|---|---|---|
| **`TransformerBridge`** | v3 — default for new work | [transformer_lens/model_bridge/](transformer_lens/model_bridge/) | Raw HF weights by default; `bridge.enable_compatibility_mode()` for HT-equivalent | [transformer_lens/tools/model_registry/data/supported_models.json](transformer_lens/tools/model_registry/data/supported_models.json) |
| **`HookedTransformer`** | Legacy, maintenance mode, deprecated in 3.0 | [transformer_lens/HookedTransformer.py](transformer_lens/HookedTransformer.py) + [transformer_lens/components/](transformer_lens/components/) | Folds LayerNorm + centres weights → does NOT match HF | [transformer_lens/supported_models.py](transformer_lens/supported_models.py) (**HT-only**) |

> ⚠ The **HookedTransformer acceptance suite is quarantined** ([test_hooked_transformer.py](tests/acceptance/test_hooked_transformer.py), [test_hooked_encoder.py](tests/acceptance/test_hooked_encoder.py), [test_hooked_encoder_decoder.py](tests/acceptance/test_hooked_encoder_decoder.py); see [QUARANTINES.md](tests/QUARANTINES.md)). HT changes land untested at the acceptance level — extra manual care required.

Bridge architecture-adapter pattern: each HF architecture has one file in [supported_architectures/](transformer_lens/model_bridge/supported_architectures/) mapping HF module paths to canonical names. Bridge hooks are architecture-native (e.g. `blocks.{i}.hook_out`); HT-style aliases live in [bridge.py](transformer_lens/model_bridge/bridge.py).

**Mirroring rule:** if you change `HookedTransformer` behaviour that has a `TransformerBridge` counterpart, update both in the same PR. [supported_models.py](transformer_lens/supported_models.py) is HT-only — Bridge-only models go in the Bridge registry data file.

## 3. Quickstart

```bash
# Bootstrap uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install — uv only (not pip, not poetry)
uv sync
source .venv/bin/activate

# First-time only: create .env from the template, then fill in HF_TOKEN
cp .env.example .env

# Source HF token before any HF-Hub-hitting command
set -a; source .env; set +a

# Tests
make unit-test          # fast, no model loads
make integration-test   # cross-component
make acceptance-test    # end-to-end
make docstring-test     # doctest + doctest-plus
make notebook-test      # slow; subset run in CI
make test-pr            # unit + docstring + acceptance + integration (PR-review surface)
make test               # everything (long; includes benchmarks + notebooks)

# Format + typecheck — no pre-commit hook; run manually before push
make format             # pycln + isort + black
make check-format       # CI-equivalent check
uv run mypy .

# Docs
uv run docs-hot-reload  # live preview
uv run build-docs       # build to docs/build/
```

Python: **>=3.10, <4.0**. CI tests 3.10, 3.11, 3.12. Format/type/docstring checks run on 3.12.

**Windows:** use WSL2 — the bash invocations above (`source`, `set -a; source .env; set +a`, heredocs in slash commands) don't translate to PowerShell.

## 4. Repo map

| Path | What's there |
|---|---|
| [transformer_lens/](transformer_lens/) | Core package |
| [transformer_lens/HookedTransformer.py](transformer_lens/HookedTransformer.py) | Legacy `HookedTransformer` API |
| [transformer_lens/HookedEncoder.py](transformer_lens/HookedEncoder.py), [HookedEncoderDecoder.py](transformer_lens/HookedEncoderDecoder.py), [HookedAudioEncoder.py](transformer_lens/HookedAudioEncoder.py) | Encoder-only / seq2seq / audio variants |
| [transformer_lens/model_bridge/](transformer_lens/model_bridge/) | `TransformerBridge` system |
| [transformer_lens/model_bridge/supported_architectures/](transformer_lens/model_bridge/supported_architectures/) | One adapter file per HF architecture |
| [transformer_lens/model_bridge/generalized_components/](transformer_lens/model_bridge/generalized_components/) | Bridge-side reusable components |
| [transformer_lens/components/](transformer_lens/components/) | HT-side components (attention, MLP, LN, embed) |
| [transformer_lens/factories/](transformer_lens/factories/) | `architecture_adapter_factory.py`, `mlp_factory.py`, `activation_function_factory.py` |
| [transformer_lens/config/](transformer_lens/config/) | `HookedTransformerConfig` and `TransformerBridgeConfig` |
| [transformer_lens/utilities/](transformer_lens/utilities/) | Device management, weight processing, HF utilities |
| [transformer_lens/hook_points.py](transformer_lens/hook_points.py) | `HookPoint` class and `LensHandle` |
| [transformer_lens/supported_models.py](transformer_lens/supported_models.py) | **HT-only** registry (`OFFICIAL_MODEL_NAMES`, `MODEL_ALIASES`) |
| [transformer_lens/tools/model_registry/](transformer_lens/tools/model_registry/) | Bridge-side registry + `verify_models.py` benchmark suite |
| [transformer_lens/tools/analysis/](transformer_lens/tools/analysis/) | High-level single-call analyses over the cache (e.g. `direct_logit_attribution`); works with both HT and Bridge |
| [transformer_lens/patching.py](transformer_lens/patching.py), [evals.py](transformer_lens/evals.py) | Activation patching, IOI, ROME, etc. |
| [tests/unit/](tests/unit/), [tests/integration/](tests/integration/), [tests/acceptance/](tests/acceptance/), [tests/benchmarks/](tests/benchmarks/), [tests/mps/](tests/mps/) | Test tiers |
| [demos/](demos/) | Jupyter notebooks; a subset runs in CI under `nbval` with sanitization from [demos/doc_sanitize.cfg](demos/doc_sanitize.cfg) |
| [docs/source/content/](docs/source/content/) | Sphinx markdown sources |
| [docs/source/content/adapter_development/](docs/source/content/adapter_development/) | Adapter-authoring guides — read these before adding a new architecture |
| [makefile](makefile) | Canonical test/format/docs targets |
| [pyproject.toml](pyproject.toml) | Deps, pytest / mypy / format / build config |
| [.github/workflows/checks.yml](.github/workflows/checks.yml) | CI gates |
| [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) | PR template + base-branch rules |

## 5. Hook naming — HT vs Bridge

- **HT canonical**: uniform across architectures — `hook_embed`, `blocks.{i}.hook_resid_pre`, `blocks.{i}.attn.hook_q`, `blocks.{i}.hook_resid_post`.
- **Bridge-native**: architecture-shaped — `blocks.{i}.hook_out`, `blocks.{i}.attn.q.hook_out`. HT aliases registered via `build_alias_to_canonical_map()` in [bridge.py](transformer_lens/model_bridge/bridge.py).

Prefer Bridge-native names in new code. Raw-HF-forward drivers comparing against `boot_transformers` must match its load configuration (fp32, eager attention) and probe for optional features like `resid_mid` rather than assume.

## 6. Adding a model

Adapters are written **per architecture family**, not per individual model — adding `gpt2` registers all GPT-2 variants. Full workflow (starter-adapter table, 4-place registration, common gotchas, anti-patterns): **[supported_architectures/AGENTS.md](transformer_lens/model_bridge/supported_architectures/AGENTS.md)**. Verification flow: **[tools/model_registry/AGENTS.md](transformer_lens/tools/model_registry/AGENTS.md)**. Claude Code users: invoke `/add-model-support <hf_repo>`.

## 7. Prioritization

When picking solutions to any problem, prioritize by **research impact**, not implementation ease. A correct, broadly-applicable feature is worth more than a one-off shortcut.

## 8. PR conventions

**Base branch**:
- `dev` — default for all PRs (new features, refactors, docs, most bug fixes).
- `main` — **only** for bug fixes against the currently-released version. PRs to `main` are made by maintainers; request permission before basing off `main`.

**Branch naming**:
- Do NOT name your branch `main` or `dev` — these conflict with the canonical branches when maintainers periodically refresh PRs against upstream.
- Include the word `docs` in the branch name if your PR is primarily a docs change (this triggers the docs build job).
- New branches must track their own remote — `git push -u origin <branch>` from your branch, not from `main`/`dev`.

**Pre-push checks**: no pre-commit hook — `make format` + `uv run mypy .` manually.

**PR template**: [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md). No conventional-commits enforcement.

**Changelog**: no per-PR file. Note user-facing changes in your PR description and commit messages — release essays in [news/](docs/source/content/news/) and GitHub Releases are drafted by a maintainer from those at release time.

## 9. CI gates a PR must pass

From [.github/workflows/checks.yml](.github/workflows/checks.yml):

| Job | Runs | When it fails, run locally |
|---|---|---|
| `compatibility-checks` | `make unit-test` + `make acceptance-test` + `uv build` × py 3.10 / 3.11 / 3.12 | `make unit-test` / `make acceptance-test` / `uv build`. Repro Python-specific failures with `uv python install <ver>` then `uv run --python <ver> pytest …` |
| `mps-checks` | macOS MPS unit + integration + smoke (PRs targeting `main` or pushes to `main` only) | `TRANSFORMERLENS_ALLOW_MPS=1 uv run pytest tests/mps` on a Mac with MPS |
| `format-check` | `make check-format` | `make format` (writes the fix) then commit |
| `type-check` | `uv run mypy .` | `uv run mypy .` — fix the error; never add `# type: ignore` |
| `docstring-test` | `make docstring-test` | `make docstring-test` — failures are doctest mismatches in `transformer_lens/` |
| `coverage-test` | Full suite + coverage artifact | `make test-pr` covers most of this; full reproduction with `make coverage-report-test` |
| `notebook-checks` | `nbval` over subset of `demos/*.ipynb`; `HF_TOKEN`-gated notebooks skip when secret absent | `uv run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/<notebook>.ipynb` |
| `build-docs` | Sphinx build (push to `main`/`dev` or branch containing `docs`) | `uv run build-docs` (sources `.env` if needed for HF-gated doctest examples) |
| `deploy-docs` | GitHub Pages (push to `main` only) | Maintainer-only; rarely the contributor's fault — usually a `build-docs` artifact issue |

In-progress PR runs cancel on new commit; tag/release runs do not.

## 10. Hard rules

**Test failures:**

- Never dismiss a failing test as "pre-existing" — investigate every failure.
- Never fabricate test counts or claim passes without running. Reviewers re-run.
- Never add `@pytest.mark.skip` / `xfail` / platform-skip to make CI green. Debug the bug.
- If new tests surface pre-existing bugs, fix them — don't punt.

**Code quality:**

- Never add `# type: ignore` — use `isinstance` / `typing.cast`.
- Comments: terse one-liners; inline comments explain WHY, not WHAT.
- Never reference plan-file details (audit IDs, "see plan section X") in source.

**Numerical work:**

- Never claim drift is "fp noise" without empirical evidence — bugs and accumulated rounding look identical at noise scale.
- Never run model verification or benchmarks in parallel — a single CUDA/MPS device OOMs.

**Environment:**

- `uv` only — never `pip` or `poetry`. Run via `uv run <cmd>` or `make`.
- Source `.env` before any HF-Hub-hitting command. `HF_TOKEN` required for Llama, Mistral, Gemma, gated Qwen.
- No pre-commit hook — `make format` + `uv run mypy .` manually before push.

## 11. "Done" checklist

Before declaring a task complete:

| Task type | Must do |
|---|---|
| Bug fix | Reproduce in a test, fix, confirm test passes, `make format`, `uv run mypy .` |
| New adapter | Adapter file + 4-place registration + integration test asserting HF logit parity + `verify_models` run (see [supported_architectures/AGENTS.md](transformer_lens/model_bridge/supported_architectures/AGENTS.md)) |
| Docs change | `uv run build-docs` succeeds; branch name contains `docs` so the docs job triggers in CI |
| Notebook change | `pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/<notebook>.ipynb` passes locally |
| Anything else | `make format` + `uv run mypy .` + `make test-pr` clean before push |

Claude Code: `/task-complete` automates the last row. See [§15 Workflow shortcuts](#15-workflow-shortcuts).

## 12. Pointers for further reading

- [docs/source/content/migrating_to_v3.md](docs/source/content/migrating_to_v3.md) — HT → Bridge migration recipes
- [docs/source/content/adapter_development/](docs/source/content/adapter_development/) — adapter authoring deep dive
- [docs/source/content/compatibility_mode.md](docs/source/content/compatibility_mode.md) — when to call `bridge.enable_compatibility_mode()`, what each flag does, four-quadrant test matrix
- [docs/source/content/debugging_numerical_divergence.md](docs/source/content/debugging_numerical_divergence.md) — bisection workflow for HT-vs-Bridge / Bridge-vs-HF logit drift
- [tests/QUARANTINES.md](tests/QUARANTINES.md) — inventory of every `skip` / `xfail` and when each can be un-skipped

## 13. Local-only conventions

Gitignored, first-class (entries checked in):

- **`transformer_lens/scratch.py`** — one-off bisection scripts, ad-hoc imports.
- **`.adapter-workspace/`** — adapter WIP (notes, dumps, repros).

## 14. Upstream dependency pins

Load-bearing pins live in [pyproject.toml](pyproject.toml):

| Pin | Where | Why it matters |
|---|---|---|
| `transformers>=5.4.0` | `[project] dependencies` | The Bridge adapter contract is written against HF module layouts; every minor HF release can break adapter component-mappings. Bumping is a real test pass. |
| `torch>=2.6` | `[project] dependencies` | Hook system relies on PyTorch's forward / backward hook semantics; major torch bumps occasionally change ordering. |
| `accelerate>=0.23.0` | `[project] dependencies` | Required for Llama-family loading. |
| `numpy>=1.24` / `>=1.26` | `[project] dependencies` (python-version-conditional) | Doctest float formatting can drift across NumPy versions. |
| `isort==5.8.0` | `[dependency-groups] dev` (exact) | Format check pins to exactly this version; a bump flips the formatting of every file. |

**Bumping upstream pins:**

1. Bump in `pyproject.toml`, `uv lock`.
2. `make test-pr` — expect real breakages, don't pin around them.
3. `uv run python -m transformer_lens.tools.model_registry.verify_models --architectures <canonical>` to catch numerical regressions.
4. Land in a focused PR — pin bumps are reviewed separately (wide blast radius).

HF specifically: a `transformers` minor bump that adds or renames an architecture impacts `_HF_PASSTHROUGH_ATTRS` in [`_bridge_builder.py`](transformer_lens/model_bridge/sources/_bridge_builder.py) and `SUPPORTED_ARCHITECTURES` in [`architecture_adapter_factory.py`](transformer_lens/factories/architecture_adapter_factory.py).

## 15. Workflow shortcuts

Claude Code users have slash commands in [.claude/commands/](.claude/commands/) that wrap common workflows. Non-Claude agents (Cursor, Codex, Copilot, Aider, etc.) can run the manual equivalents:

| Claude shortcut | Manual equivalent | Reference |
|---|---|---|
| `/test-unit` | `make unit-test` | [§3 Quickstart](#3-quickstart) |
| `/test-all` | `make test` (long) | [§3 Quickstart](#3-quickstart) |
| `/format` | `make format && uv run mypy .` | [§3 Quickstart](#3-quickstart) |
| `/typecheck` | `uv run mypy .` | [§3 Quickstart](#3-quickstart) |
| `/build-docs` | `set -a; source .env; set +a && uv run build-docs` | [§3 Quickstart](#3-quickstart) |
| `/verify-model <repo>` | `set -a; source .env; set +a` then `uv run python -m transformer_lens.tools.model_registry.verify_models --model <repo> --dry-run` first, confirm, then drop `--dry-run` | [tools/model_registry/AGENTS.md](transformer_lens/tools/model_registry/AGENTS.md) |
| `/add-model-support <hf_repo>` | Follow the 4-way branch + adapter-authoring workflow | [supported_architectures/AGENTS.md](transformer_lens/model_bridge/supported_architectures/AGENTS.md) |
| `/task-complete` | `make format && uv run mypy . && make test-pr` (loop until clean) | [§11 Done checklist](#11-done-checklist) |
