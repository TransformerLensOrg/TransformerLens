# AGENTS.md

Guidance for AI coding agents contributing to **TransformerLens**.

This file is the single source of truth. Vendor-specific files ([CLAUDE.md](CLAUDE.md), [.github/copilot-instructions.md](.github/copilot-instructions.md), [.cursor/rules/transformerlens.mdc](.cursor/rules/transformerlens.mdc)) and sub-folder `AGENTS.md` files all defer here.

## TL;DR — read before doing anything

1. **Use `uv`**, not `pip` or `poetry`. Install with `uv sync`.
2. **Source `.env`** (`set -a; source .env; set +a`) before any HuggingFace-Hub-hitting command. Gated models need `HF_TOKEN`.
3. **Base PRs on `dev`**, not `main`. Never name a branch `main` or `dev`.
4. **Mirror HookedTransformer → TransformerBridge** when changing behaviour that exists in both ([§2](#2-two-systems-live-in-this-repo)).
5. **Run `make format` + `uv run mypy .` before push** — no pre-commit hook is installed.
6. **Never add `# type: ignore`** — use `isinstance` / `typing.cast` ([§10](#10-hard-rules)).
7. **Never dismiss a failing test as "pre-existing"** — investigate every failure ([§10](#10-hard-rules)).

Sub-folder rules: [tests/AGENTS.md](tests/AGENTS.md) · [supported_architectures/AGENTS.md](transformer_lens/model_bridge/supported_architectures/AGENTS.md) · [tools/model_registry/AGENTS.md](transformer_lens/tools/model_registry/AGENTS.md).

---

## 1. What this repo is

**TransformerLens** — mechanistic-interpretability library. Loads 9,000+ models across 50+ architecture families (see [supported_models.json](transformer_lens/tools/model_registry/data/supported_models.json)) and exposes internal activations through a hook system for caching, editing, and ablating intermediate state. Built on HuggingFace `transformers`.

## 2. Two systems live in this repo

| System | Status | Lives in | Numerics | Registry |
|---|---|---|---|---|
| **`TransformerBridge`** | v3 — default for new work | [transformer_lens/model_bridge/](transformer_lens/model_bridge/) | Raw HF weights by default; `bridge.enable_compatibility_mode()` for HT-equivalent | [transformer_lens/tools/model_registry/data/supported_models.json](transformer_lens/tools/model_registry/data/supported_models.json) |
| **`HookedTransformer`** | Legacy, maintenance mode, deprecated in 3.0 | [transformer_lens/HookedTransformer.py](transformer_lens/HookedTransformer.py) + [transformer_lens/components/](transformer_lens/components/) | Folds LayerNorm + centres weights → does NOT match HF | [transformer_lens/supported_models.py](transformer_lens/supported_models.py) (**HT-only**) |

Bridge architecture-adapter pattern: each HF architecture has one file in [transformer_lens/model_bridge/supported_architectures/](transformer_lens/model_bridge/supported_architectures/) mapping HF module paths to canonical TransformerLens names. Bridge hook names are architecture-native (e.g. `blocks.{i}.hook_out`); HT-style aliases are registered separately in [transformer_lens/model_bridge/bridge.py](transformer_lens/model_bridge/bridge.py).

### Mirroring rule

If you change behaviour in `HookedTransformer` that has a counterpart in `TransformerBridge`, update both in the same PR. Parallel implementations; drift between them is a recurring source of bugs. The HT-only [supported_models.py](transformer_lens/supported_models.py) is NOT the place to add Bridge-only models — those go in the Bridge registry data file.

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
| [transformer_lens/patching.py](transformer_lens/patching.py), [evals.py](transformer_lens/evals.py) | Activation patching, IOI, ROME, etc. |
| [tests/unit/](tests/unit/), [tests/integration/](tests/integration/), [tests/acceptance/](tests/acceptance/), [tests/benchmarks/](tests/benchmarks/), [tests/mps/](tests/mps/) | Test tiers |
| [demos/](demos/) | Jupyter notebooks; a subset runs in CI under `nbval` with sanitization from [demos/doc_sanitize.cfg](demos/doc_sanitize.cfg) |
| [docs/source/content/](docs/source/content/) | Sphinx markdown sources |
| [docs/source/content/adapter_development/](docs/source/content/adapter_development/) | Adapter-authoring guides — read these before adding a new architecture |
| [makefile](makefile) | Canonical test/format/docs targets |
| [pyproject.toml](pyproject.toml) | Deps, pytest / mypy / format / build config. |
| [.github/workflows/checks.yml](.github/workflows/checks.yml) | CI gates |
| [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) | PR template + base-branch rules |

## 5. Hook naming — HT vs Bridge

- **HT canonical names**: uniform across architectures. `hook_embed`, `blocks.{i}.hook_resid_pre`, `blocks.{i}.attn.hook_q`, `blocks.{i}.hook_resid_post`, etc.
- **Bridge-native names**: architecture-shaped. E.g. `blocks.{i}.hook_out`, `blocks.{i}.attn.q.hook_out`. Aliases are registered via `build_alias_to_canonical_map()` in [transformer_lens/model_bridge/bridge.py](transformer_lens/model_bridge/bridge.py) so HT names continue to resolve.

When writing new code, prefer Bridge-native names.

When raw-HF-forward drivers compare against `boot_transformers`, match its load configuration (fp32, eager attention). Probe for structural features like `resid_mid` rather than assuming all architectures expose them.

## 6. Adding a model

Adapters are written **per architecture family**, not per individual model — adding `gpt2` registers all GPT-2 variants. Full workflow (starter-adapter table, 4-place registration, common gotchas, anti-patterns): **[supported_architectures/AGENTS.md](transformer_lens/model_bridge/supported_architectures/AGENTS.md)**. Verification flow: **[tools/model_registry/AGENTS.md](transformer_lens/tools/model_registry/AGENTS.md)**. Claude Code users: invoke `/add-model-support <hf_repo>`.

## 7. Prioritization

When picking solutions to any problem, prioritize by **research impact**, not implementation ease. A correct, broadly-applicable feature is worth more than a one-off shortcut.

## 8. PR conventions

- **Base branch**:
  - `dev` — default for all PRs (new features, refactors, docs, most bug fixes).
  - `main` — **only** for bug fixes against the currently-released version. PRs to main made at Maintainer's discretion.
- **Branch names**: do NOT name your branch `main` or `dev` — these conflict with the canonical branches when maintainers periodically refresh PRs against upstream. Include the word `docs` in the branch name if your PR is primarily a docs change (this triggers the docs build job).
- **New branches must track their own remote**, not the source branch's remote. When you `git push -u`, push to the new branch's namespace, not the branch you forked from.
- **PR template**: [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md). No conventional-commits enforcement.
- **No pre-commit hook is installed.** Run `make format` and `uv run mypy .` manually before pushing; CI will fail otherwise.
- **Changelog**: no per-PR changelog file. User-facing changes go in the PR description and are rolled up into the next release essay in [docs/source/content/news/](docs/source/content/news/) (`release-2.0.md`, `release-3.0.md`, … — one file per major version). If your PR introduces a breaking change or a notable user-facing feature, note it explicitly in the PR description so the next release essay picks it up.

## 9. CI gates a PR must pass

From [.github/workflows/checks.yml](.github/workflows/checks.yml):

| Job | Runs |
|---|---|
| `compatibility-checks` | `make unit-test` + `make acceptance-test` + `uv build` × py 3.10 / 3.11 / 3.12 |
| `mps-checks` | macOS MPS unit + integration + smoke (PRs to `main` only) |
| `format-check` | `make check-format` |
| `type-check` | `uv run mypy .` |
| `docstring-test` | `make docstring-test` |
| `coverage-test` | Full suite + coverage artifact |
| `notebook-checks` | `nbval` over subset of `demos/*.ipynb`; `HF_TOKEN`-gated notebooks skip when secret absent |
| `build-docs` | Sphinx build (push to `main`/`dev` or branch containing `docs`) |
| `deploy-docs` | GitHub Pages (push to `main` only) |

In-progress PR runs cancel on new commit; tag/release runs do not.

## 10. Hard rules

These are the explicit "do / do not" rules. They are informed by maintainer experience.

**On test failures:**

- Never dismiss a failing test as "pre-existing" or "unrelated" — investigate every failure and fix the underlying issue.
- Never fabricate test counts or claim tests passed without running them. Reviewers re-run tests; agent self-reports are not trusted as evidence.
- Never add platform skips, `@pytest.mark.skip`, or `xfail` to make a failing CI green. Debug the actual bug.
- If new tests surface pre-existing bugs, fix them. Don't punt to a future PR or `xfail` marker.

**On code quality:**

- Never add `# type: ignore`. Prefer `isinstance` assertions or `typing.cast` for narrowing.
- Comments: terse, one-line docstrings; inline comments should explain WHY, not WHAT. No multi-paragraph explanations.
- Never reference plan-file details (audit IDs, finding IDs, "see plan section X") in source comments or docstrings. Plan artifacts belong in PR descriptions, not the codebase.

**On numerical work:**

- Never claim observed drift is "fp noise" without empirical evidence. Real bugs and accumulated rounding error look identical at noise scale.
- Never run model verification or benchmarks in parallel, unless explicitly requested to — a single CUDA/MPS device does not have memory for concurrent loads. Serialize them.

**On environment and tooling:**

- Use `uv`, never `pip` or `poetry`. Run commands via `uv run <cmd>` or the appropriate `make` target.
- Source `.env` before any HF-Hub-hitting command (docs build, notebook runs, `boot_transformers` against gated models). The `HF_TOKEN` is required for Llama, Mistral, Gemma, and gated Qwen variants.
- No pre-commit hook is installed; run `make format` + `uv run mypy .` manually before push.

## 11. "Done" checklist

Before declaring a task complete:

| Task type | Must do |
|---|---|
| Bug fix | Reproduce in a test, fix, confirm test passes, `make format`, `uv run mypy .` |
| New adapter | Adapter file + 4-place registration + integration test asserting HF logit parity + `verify_models` run (see [supported_architectures/AGENTS.md](transformer_lens/model_bridge/supported_architectures/AGENTS.md)) |
| Docs change | `uv run build-docs` succeeds; branch name contains `docs` so the docs job triggers in CI |
| Notebook change | `pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/<notebook>.ipynb` passes locally |
| Anything else | `make format` + `uv run mypy .` + `make test-pr` clean before push |

If you're using Claude Code, the `/task-complete` slash command in [.claude/commands/](.claude/commands/) automates the last row (`make test-pr` = unit + docstring + acceptance + integration).

## 12. Pointers for further reading

- [docs/source/content/migrating_to_v3.md](docs/source/content/migrating_to_v3.md) — HT → Bridge migration recipes
- [docs/source/content/adapter_development/](docs/source/content/adapter_development/) — adapter authoring deep dive
- [docs/source/content/compatibility_mode.md](docs/source/content/compatibility_mode.md) — when to call `bridge.enable_compatibility_mode()`, what each flag does, four-quadrant test matrix
- [docs/source/content/debugging_numerical_divergence.md](docs/source/content/debugging_numerical_divergence.md) — bisection workflow for HT-vs-Bridge / Bridge-vs-HF logit drift
- [tests/QUARANTINES.md](tests/QUARANTINES.md) — inventory of every `skip` / `xfail` and when each can be un-skipped

## 13. Local-only conventions

Two gitignored paths exist for ephemeral work; use them so your work-in-progress doesn't show up in `git status`:

- **`transformer_lens/scratch.py`** — sibling-of-package scratch file for one-off bisection scripts and ad-hoc imports. Already in `.gitignore`; never commit.
- **`.adapter-workspace/`** — directory for adapter WIP (notes, config dumps, repro scripts) while you're iterating. Already in `.gitignore`.

Both are first-class conventions, not personal preferences — the gitignore entries are checked in. New contributors should know about them so they don't reinvent the convention or accidentally commit experimental code.

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

1. Bump the pin in `pyproject.toml`, refresh `uv.lock` (`uv lock`).
2. Run `make test-pr` locally; expect adapter / hook tests to surface real breakages.
3. For each break, fix the adapter or component — do not pin around the regression.
4. Run `uv run python -m transformer_lens.tools.model_registry.verify_models --architectures <a-few-canonical-ones>` to catch numerical regressions the unit tests miss.
5. Land in a focused PR — pin bumps are reviewed separately from feature work because the blast radius is wide.

Specific to HF: when a `transformers` minor bump introduces a new architecture or renames an existing one, check the impact on `_HF_PASSTHROUGH_ATTRS` in [`transformer_lens/model_bridge/sources/_bridge_builder.py`](transformer_lens/model_bridge/sources/_bridge_builder.py) and `SUPPORTED_ARCHITECTURES` in [`transformer_lens/factories/architecture_adapter_factory.py`](transformer_lens/factories/architecture_adapter_factory.py).
