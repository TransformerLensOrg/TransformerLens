# AGENTS.md

Guidance for AI coding agents contributing to **TransformerLens**.

This file is the single source of truth. Vendor-specific files ([CLAUDE.md](CLAUDE.md), [.github/copilot-instructions.md](.github/copilot-instructions.md), [.cursor/rules/transformerlens.mdc](.cursor/rules/transformerlens.mdc)) all defer here.

---

## 1. What this repo is

**TransformerLens** is a library for mechanistic interpretability of generative language models. It loads 9,000+ open-source models across 50+ architecture families (GPT-2, Llama, Gemma, Mistral, Qwen, T5, BERT, Mamba, etc. — see [transformer_lens/tools/model_registry/data/supported_models.json](transformer_lens/tools/model_registry/data/supported_models.json) for the full inventory) and exposes their internal activations through a hook system so researchers can cache, edit, and ablate intermediate state.

Upstream dependency: HuggingFace `transformers`. Most models are wrappers over HF implementations.

## 2. Two systems live in this repo

The library is in the middle of a v2 → v3 transition. **Both systems are still present** and changes to HookedTransformer often need to be mirrored to TransformerBridge.

### `TransformerBridge` (v3, recommended for new work)

- Lives in [transformer_lens/model_bridge/](transformer_lens/model_bridge/).
- Architecture-adapter pattern: each HF architecture has an adapter file in [transformer_lens/model_bridge/supported_architectures/](transformer_lens/model_bridge/supported_architectures/) that maps HF module paths to canonical TransformerLens names.
- Preserves raw HF weights by default (logits match HF). Call `bridge.enable_compatibility_mode()` after booting for HookedTransformer-equivalent numerics.
- Hook names are architecture-native (e.g. `blocks.{i}.hook_out`), with aliases provided for HT-style names where applicable.

### `HookedTransformer` (legacy, maintenance mode)

- Lives in [transformer_lens/HookedTransformer.py](transformer_lens/HookedTransformer.py) and [transformer_lens/components/](transformer_lens/components/).
- Folds LayerNorm and centers weights by default — useful for circuit analysis but means weights and logits **do not** match HF.
- Model registry is [transformer_lens/supported_models.py](transformer_lens/supported_models.py) — **HT-only**. Bridge-only models do not belong here; they live in the Bridge registry data file under [transformer_lens/tools/model_registry/](transformer_lens/tools/model_registry/).
- Deprecated as of TransformerLens 3.0; will be removed in the next major version. Still fully functional via the compatibility layer.

### Mirroring rule

If you change behavior in `HookedTransformer` that has a counterpart in `TransformerBridge`, update both in the same PR. They are parallel implementations of the same surface; drift between them is a recurring source of bugs. Conversely, `supported_models.py` is HT-only — do not add Bridge-only models there.

## 3. Quickstart

```bash
# Install (this repo uses uv, NOT pip and NOT poetry)
uv sync
source .venv/bin/activate

# Source HF token (required for gated models: Llama, Mistral, Gemma, gated Qwen, etc.)
set -a; source .env; set +a

# Run tests
make unit-test          # fast, no model loads — run this on every change
make integration-test   # cross-component, hits HF Hub
make acceptance-test    # end-to-end
make notebook-test      # demos/*.ipynb (slow; CI parallelizes)
make docstring-test     # doctest + doctest-plus
make test               # all of the above (long)

# Format + typecheck (no pre-commit hook is installed — run manually before push)
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

This trips up agents and humans alike.

- **HT canonical names**: uniform across architectures. `hook_embed`, `blocks.{i}.hook_resid_pre`, `blocks.{i}.attn.hook_q`, `blocks.{i}.hook_resid_post`, etc.
- **Bridge-native names**: architecture-shaped. E.g. `blocks.{i}.hook_out`, `blocks.{i}.attn.q.hook_out`. Aliases are registered via `build_alias_to_canonical_map()` in [transformer_lens/model_bridge/bridge.py](transformer_lens/model_bridge/bridge.py) so HT names continue to resolve.

When writing new code, prefer Bridge-native names.

When raw-HF-forward drivers compare against `boot_transformers`, match its load configuration (fp32, eager attention). Probe for structural features like `resid_mid` rather than assuming all architectures expose them.

## 6. Adding a model

TransformerBridge adapters are written per architecture family, not per individual model. Adding `gpt2` registers all GPT-2 variants; you generally won't add a new file for a sibling checkpoint of an already-supported architecture.

Start with [docs/source/content/adapter_development/](docs/source/content/adapter_development/) — there are step-by-step guides covering the full workflow. Summary:

1. Read the HF model's `config.json` and source to identify components (embed, attention, MLP, normalization, output head).
2. Copy [docs/source/_static/adapter-template.py](docs/source/_static/adapter-template.py) into [transformer_lens/model_bridge/supported_architectures/](transformer_lens/model_bridge/supported_architectures/) as `<model_name>.py`.
3. Fill in component mappings.
4. Register in both [transformer_lens/model_bridge/supported_architectures/__init__.py](transformer_lens/model_bridge/supported_architectures/__init__.py) and [transformer_lens/factories/architecture_adapter_factory.py](transformer_lens/factories/architecture_adapter_factory.py).
5. Add the HF repo path to the Bridge registry (model registry data file) and run `verify_models.py` against the new architecture.
6. Add an integration test under [tests/integration/](tests/integration/) that asserts logit parity with HF.

The existing adapters under [supported_architectures/](transformer_lens/model_bridge/supported_architectures/) are the best references for tricky cases.

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

## 9. CI gates a PR must pass

From [.github/workflows/checks.yml](.github/workflows/checks.yml):

| Job | What it runs |
|---|---|
| `compatibility-checks` | `make unit-test` + `make acceptance-test` + `uv build` across Python 3.10 / 3.11 / 3.12 |
| `mps-checks` | macOS unit + integration + smoke tests with MPS (PRs to `main` only) |
| `format-check` | `make check-format` |
| `type-check` | `uv run mypy .` |
| `docstring-test` | `make docstring-test` |
| `coverage-test` | Full test suite with coverage report (uploaded as `test-coverage` artifact) |
| `notebook-checks` | Subset of `demos/*.ipynb` validated via `nbval` against [demos/doc_sanitize.cfg](demos/doc_sanitize.cfg). Notebooks that require `HF_TOKEN` are skipped when the secret is absent. |
| `build-docs` | Sphinx build (only on push to `main`/`dev` or branches containing `docs`) |
| `deploy-docs` | Deploys to GitHub Pages (only on push to `main`) |

In-progress PR runs are cancelled when a new commit lands; tag/release runs are not.

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
| New adapter | Adapter file + factory registration + Bridge registry entry + integration test asserting HF logit parity + `verify_models.py` run for the new architecture |
| Docs change | `uv run build-docs` succeeds; branch name contains `docs` so the docs job triggers in CI |
| Notebook change | `pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/<notebook>.ipynb` passes locally |
| Anything else | `make unit-test` + `make format` + `uv run mypy .` clean before push |

If you're using Claude Code, the `/task-complete` slash command in [.claude/commands/](.claude/commands/) automates the last row.

## 12. Pointers for further reading

- [docs/source/content/contributing.md](docs/source/content/contributing.md) — the human-facing contributor guide
- [docs/source/content/migrating_to_v3.md](docs/source/content/migrating_to_v3.md) — HT → Bridge migration recipes
- [docs/source/content/getting_started.md](docs/source/content/getting_started.md) — full env-vars list and first-time setup
- [docs/source/content/adapter_development/](docs/source/content/adapter_development/) — adapter authoring deep dive
- [demos/Main_Demo.ipynb](demos/Main_Demo.ipynb), [demos/Bridge_Evals_Demo.ipynb](demos/Bridge_Evals_Demo.ipynb) — start here to see the library in use
