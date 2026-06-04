# Contributing

```{warning}
`HookedTransformer` is deprecated as of TransformerLens 3.0 and will be removed in the next major version. New code should use [`TransformerBridge`](migrating_to_v3.md) instead. Existing `HookedTransformer` code continues to work through the 3.x branch via a compatibility layer. See the [migration guide](migrating_to_v3.md) for conversion recipes.

The HookedTransformer **acceptance test suite is currently quarantined** due to a CI test-pollution issue (see `tests/QUARANTINES.md` in the repo). Changes that touch HookedTransformer internals therefore land essentially untested at the acceptance level — extra manual care is required until the suite is re-enabled.
```

## Setup

### DevContainer

For a one-click setup of your development environment, this project includes a
[DevContainer](https://containers.dev/). It can be used locally with [VS
Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) or
with [GitHub Codespaces](https://github.com/features/codespaces).

### Manual Setup

As of TransformerLens 3.0, this project uses [UV](https://docs.astral.sh/uv/getting-started/installation/) for package and environment management (it previously used Poetry). Install UV first, then run:

```bash
# resolves and installs dependencies into .venv
uv sync
# activate the virtual environment
source .venv/bin/activate
# first-time only: create .env from the template, then fill in HF_TOKEN for gated models
cp .env.example .env
```

Dependency groups are defined in `pyproject.toml` under `[dependency-groups]`. The project sets `default-groups = ["dev", "docs", "jupyter"]`, so `uv sync` installs all three out of the box — you do not need to pass `--group` flags for the standard contributor setup.

- Standard contributor setup (recommended default): `uv sync`
- Include the optional `quantization` group (bitsandbytes, optimum-quanto): `uv sync --all-groups`

You can also add individual groups with `uv sync --group <name>`, or install without optional groups using `uv sync --no-default-groups`.

Requires Python 3.10 or higher.

**Windows users:** the bash invocations above don't translate to PowerShell. Use WSL2.

### Environment variables

Source `.env` (`set -a; source .env; set +a`) before any HuggingFace-Hub-hitting command. `HF_TOKEN` is required for gated models (Llama, Mistral, Gemma, gated Qwen variants). See `.env.example` in the repo root for the full list.

## Testing

If adding a feature, please add unit tests for it. If you need a model, please use one of the ones
that are cached by GitHub Actions (so that it runs quickly on the CD). These are `gpt2`,
`attn-only-1l`, `attn-only-2l`, `attn-only-3l`, `attn-only-4l`, `tiny-stories-1M`. Note `gpt2` is
quite slow (as we only have CPU actions) so the smaller models like `attn-only-1l` and
`tiny-stories-1M` are preferred if possible.

### Running the tests

- Standard PR-review surface (unit + docstring + acceptance + integration): `make test-pr`
- Just unit tests (fast feedback): `make unit-test`
- Integration tests: `make integration-test`
- Acceptance tests: `make acceptance-test`
- Docstring tests: `make docstring-test`
- Notebook tests: `make notebook-test`
- All test suites including benchmarks + notebooks (long): `make test`

### Test tiers

| Tier | Path | Loads models? | Hits HF Hub? | Scope |
|---|---|---|---|---|
| `unit` | `tests/unit/` | None / synthetic (rare exceptions) | No | Function or single module |
| `integration` | `tests/integration/` | 1–2 cached models, module-scoped | Yes | Cross-component |
| `acceptance` | `tests/acceptance/` | Full models, session-scoped | Yes | End-to-end behaviour |
| `benchmarks` | `tests/benchmarks/` | Varies; performance focus | Yes | Throughput / memory |
| `mps` | `tests/mps/` | TinyStories-1M, fp32 only | Yes | macOS-MPS smoke only |

Rule of thumb: new tests that load a model should land in `integration/` by default. Tests that need real (large) weights to verify go through the model registry's `verify_models` workflow rather than running in pytest.

The flaky-retry policy (`--reruns 2 --reruns-delay 5`) wraps every `make` target and exists to absorb HF Hub 429s. The root `tests/conftest.py` also enables `enable_hf_retry()` session-wide.

### Quarantined tests

Some tests carry persistent `skip` / `skipif` / `xfail` markers — for optional dependencies (LIT, bitsandbytes), hardware requirements (CUDA, MPS, multi-GPU), CI cost / network budget, or upstream platform bugs. The `tests/QUARANTINES.md` file in the repo inventories every one with an "un-skip when…" line. **Before debugging a failing test, check whether it's a known quarantine.**

The HookedTransformer acceptance suite (`tests/acceptance/test_hooked_transformer.py`, `test_hooked_encoder.py`, `test_hooked_encoder_decoder.py`) is currently a whole-file quarantine — see the warning at the top of this page.

## Formatting

This project uses `pycln`, `isort` and `black` for formatting, pull requests are checked in github
actions.

- Format all files via `make format`
- Only check the formatting via `make check-format`
- Type-check via `uv run mypy .`

Note that `black` line length is set to 100 in `pyproject.toml` (instead of the default 88).

**No pre-commit hook is installed.** Run `make format` and `uv run mypy .` manually before pushing — CI will fail otherwise.

## Project conventions

A few conventions have grown out of recurring failure modes. Following them tends to make reviews go faster.

### Test failures

Treat every failing test as a signal worth investigating, even if the failure looks unrelated to your change. "Pre-existing failure" is rarely true once you dig in, and the cost of finding out is small compared to shipping a regression. If a new test surfaces a pre-existing bug, it's worth fixing in the same PR rather than deferring.

Avoid reaching for `@pytest.mark.skip`, `xfail`, or platform-gated `skipif` to make a red CI go green. If a skip is genuinely required — an optional dependency, a hardware requirement, an upstream platform bug — document it in `tests/QUARANTINES.md` so the next person debugging knows why it's quarantined.

### Code quality

We avoid `# type: ignore` comments throughout the codebase; if the type system disagrees with you, the usual escape hatches are `isinstance` narrowing or `typing.cast`. Keep comments terse — one-line docstrings, and inline comments that explain *why* code does what it does rather than restating *what* it does.

### Numerical work

When a forward pass disagrees with HuggingFace, it's tempting to call the difference "floating-point noise" and move on. The trouble is that genuine bugs and accumulated rounding error are indistinguishable at small magnitudes until you measure. A cheap check: rerun the comparison in `dtype=torch.float64` on both sides. If the diff stays the same magnitude, it's a bug; if it drops by roughly eight orders of magnitude, it was noise. The [Debugging Numerical Divergence](debugging_numerical_divergence.md) guide walks the rest of the bisection.

Model verification and benchmark runs each load a full model; a single CUDA or MPS device generally lacks the memory to host two at once. Run them serially.

### Environment and tooling

Use `uv` rather than `pip` or `poetry` — commands run via `uv run <cmd>` or the appropriate `make` target. Source `.env` (`set -a; source .env; set +a`) before any command that hits the HuggingFace Hub.

## Two systems live in this repo

The library is mid-transition between two parallel paths:

| System | Status | Lives in | Numerics | Registry |
|---|---|---|---|---|
| `TransformerBridge` | v3 — default for new work | `transformer_lens/model_bridge/` | Raw HF weights by default; `bridge.enable_compatibility_mode()` for HT-equivalent — see [Compatibility Mode](compatibility_mode.md) | `transformer_lens/tools/model_registry/data/supported_models.json` |
| `HookedTransformer` | Legacy, maintenance mode, deprecated in 3.0 | `transformer_lens/HookedTransformer.py` + `transformer_lens/components/` | Folds LayerNorm + centres weights → does NOT match HF | `transformer_lens/supported_models.py` (**HT-only**) |

Because the two systems are parallel implementations of the same surface, behavioural changes on one side usually need a matching change on the other. If you change a feature in `HookedTransformer` that has a counterpart in `TransformerBridge` (or vice versa), update both in the same PR — drift between them has historically been a steady source of bugs. The registries are *not* parallel, though: `supported_models.py` is HookedTransformer-only, while Bridge-only models live in the Bridge registry data file under `transformer_lens/tools/model_registry/`.

## PR conventions

### Base branch

- `dev` — default for all PRs (new features, refactors, docs, most bug fixes).
- `main` — **only** for bug fixes against the currently-released version. PRs to `main` are made by maintainers; request permission before basing off `main`.

### Branch naming

- Do NOT name your branch `main` or `dev` — these conflict with the canonical branches when maintainers periodically refresh PRs against upstream.
- Include the word `docs` in the branch name if your PR is primarily a docs change (this triggers the docs build job).
- New branches must track their own remote (`git push -u origin <branch>` from your branch, not from `main`/`dev`).

### Changelog

There is no per-PR changelog file. **Note user-facing changes in your PR description and commit messages** — release essays in `docs/source/content/news/` and GitHub Releases are drafted by a maintainer by reviewing those at release time. Breaking changes and notable user-facing features should be called out explicitly so the rollup picks them up.

## Documentation

Please make sure to add thorough documentation for any features you add. You should do this directly
in the docstring, and this will then automatically generate the API docs when merged into `main`.
They will also be automatically checked with [pytest](https://docs.pytest.org/) (via
[doctest](https://docs.python.org/3/library/doctest.html)).

If you want to view your documentation changes, run `uv run docs-hot-reload`. This will give you
hot-reloading docs (they change in real time as you edit docstrings).

For documentation generation to work, install with `uv sync --group docs`.

### Docstring Style Guide

We follow the Google Python Docstring Style for writing docstrings, with some added features from
reStructuredText (reST).

#### Sections and Order

You should follow this order:

```python
"""Title In Title Case.

A description of what the function/class does, including as much detail as is necessary to fully understand it.

Warning:

Any warnings to the user (e.g. common pitfalls).

Examples:

Include any examples here. They will be checked with doctest.

  >>> print(1 + 2)
  3

Args:
    param_without_type_signature:
        Each description should be indented once more.
    param_2:
        Another example parameter.

Returns:
    Returns description without type signature.

Raises:
    Information about the error it may raise (if any).
"""
```

#### Supported Sphinx Properties

##### References to Other Functions/Classes

You can reference other parts of the codebase using
[cross-referencing](https://www.sphinx-doc.org/en/master/usage/domains/python.html#cross-referencing-python-objects)
(noting that you can omit the full path if it is in the same file).

```reStructuredText
:mod:transformer_lens # Function or module

:const:`transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES`

:class:`transformer_lens.HookedTransformer`

:meth:`transformer_lens.HookedTransformer.from_pretrained`

:attr:`transformer_lens.HookedTransformer.cfg`
```

##### Maths

You can use LaTeX, but note that as you're placing this in python strings the backwards slash (`\`)
must be repeated (i.e. `\\`). You can write LaTeX inline, or in "display mode".

```reStructuredText
:math:`(a + b)^2 = a^2 + 2ab + b^2`
```

```reStructuredText
.. math::
   :nowrap:

   \\begin{eqnarray}
      y    & = & ax^2 + bx + c \\
      f(x) & = & x^2 + 2xy + y^2
   \\end{eqnarray}
```

#### Markup

- Italics - `*text*`
- Bold - `**text**`
- Code - ` ``code`` `
- List items - `*item`
- Numbered items - `1. Item`
- Quotes - indent one level
- External links = ``` `Link text <https://domain.invalid/>` ```

## Creating Architecture Adapters

If a HuggingFace model is not yet supported by `TransformerBridge`, you can add support by writing an Architecture Adapter. An adapter is a Python class that tells the bridge how a particular HF model maps to TransformerLens's canonical component names (`embed`, `blocks`, `attn.q`, etc.). Once registered, `TransformerBridge.boot_transformers("<your-model>")` will load the model end-to-end with full hook support.

The work is mostly bookkeeping: identify each component on the HF side (embeddings, attention, MLP, normalization), point a Bridge instance at the corresponding HF module path, and supply tensor-reshape rules where the weight layout differs from TransformerLens conventions. Most of the per-architecture decisions are already encoded in the existing adapters under `transformer_lens/model_bridge/supported_architectures/`, which are good starting points to copy from.

Two guides walk through the process:

- [Architecture Adapter Creation Guide](adapter_development/adapter-creation-guide.md) — start here. A step-by-step workflow for taking an HF model from unsupported to tested, registered adapter.
- [HuggingFace Model Analysis Guide](adapter_development/hf-model-analysis-guide.md) — a reference for reading an HF model's `config.json` and source files to extract the attributes you'll set on `self.cfg`.
- [HuggingFace Model Scraper](adapter_development/hf-scraper.md) — how to run the scraper that discovers HF models for the registry, including the per-architecture targeted-scrape mode used after merging a new adapter.

Adapters live in `transformer_lens/model_bridge/supported_architectures/<model_name>.py` and need to be registered in **four** places. Each registration site has a different consequence if you skip it, which is why the next section's invariant test is worth running before you open the PR.

1. **`transformer_lens/model_bridge/supported_architectures/__init__.py`** — import the adapter class and add it to `__all__`. The package fails to import at boot if this is missed.
2. **`transformer_lens/factories/architecture_adapter_factory.py`** — import the class and add it to the `SUPPORTED_ARCHITECTURES` dict. The dict key must match `config.architectures[0]` exactly; otherwise `boot_transformers` raises "unsupported architecture."
3. **`transformer_lens/tools/model_registry/__init__.py`** — add the architecture name to `HF_SUPPORTED_ARCHITECTURES` (the set) and `CANONICAL_AUTHORS_BY_ARCH` (the foundation-orgs map). Without this, the HF scraper won't discover canonical models for the new architecture.
4. **`transformer_lens/tools/model_registry/generate_report.py`** — add a one-line entry to `ARCHITECTURE_DESCRIPTIONS` so the generated coverage table covers the new architecture.

If you want a starter file, copy [adapter-template.py](../_static/adapter-template.py) into `supported_architectures/` and rename it.

After registering, verify the four-place wiring by running:

```bash
uv run pytest tests/unit/tools/test_model_registry.py -k TestRegistrySyncedWithFactory
```

The `TestRegistrySyncedWithFactory` class bidirectionally asserts that `SUPPORTED_ARCHITECTURES`, `HF_SUPPORTED_ARCHITECTURES`, and `CANONICAL_AUTHORS_BY_ARCH` stay in sync — the failure message names exactly which set is missing your new key.

### Required tests for a new adapter

Two test layers, both required:

1. **Unit adapter test** at `tests/unit/model_bridge/supported_architectures/test_<arch>_adapter.py`. ~26 of these exist; copy the closest sibling. The pattern: a `_make_cfg()` factory, an `adapter` fixture, and one test per architecture-specific quirk. Unit adapter tests instantiate the adapter from a synthetic config and assert structural properties — they don't load weights and don't hit HF Hub.
2. **Integration parity test** at `tests/integration/model_bridge/test_<arch>_adapter.py`. Loads a real cached HF model and asserts logit parity vs HuggingFace at fp32 + eager attention.

### Common adapter gotchas

- **HF raw config attributes are invisible to TL-side consumers unless explicitly propagated to `self.cfg`.** Walk the HF `config.json` and mirror any non-standard knobs (`final_logit_softcapping`, `attn_logit_softcapping`, `query_pre_attn_scalar`, `sliding_window`, `layer_types`, custom `eps_attr` names) onto `self.cfg` so weight processing and forward passes can see them.
- **Some config attrs need both surface-on-cfg AND fold-into-weight** via a `preprocess_weights()` override. The trigger: a numerical operation HF's forward applies natively must also be baked into the raw weights, or `bridge.enable_compatibility_mode()` (which calls `process_weights` on raw weights) produces wrong results. Concrete examples in-tree: Cohere `logit_scale` → `unembed.weight`; Gemma embedding scale (`√d_model`) → `embed.weight`. Skip the fold and Phase 3 / Phase 4 of `verify_models` will silently degrade.
- **Tokenizer policy is per-model, not per-architecture.** Sibling models in the same family routinely differ — the chat-instruct variant may prepend BOS where the base does not, padding side can flip, EOS handling can differ. It's worth re-checking `default_prepend_bos`, padding side, and EOS handling against the specific target rather than copying them from a starter adapter. `tokenizer_config.json` is not always reliable on its own — some architectures (Cohere is a notable example) declare `add_bos_token=False` but HF's `__call__` prepends BOS anyway. The most reliable check is to invoke the tokenizer directly:

  ```python
  from transformers import AutoTokenizer
  t = AutoTokenizer.from_pretrained("<hf_repo>")
  print(t("hello").input_ids)        # what generation actually uses
  print(t.bos_token_id)
  ```

  If `t("hello").input_ids[0] == t.bos_token_id`, set `cfg.default_prepend_bos = True`; otherwise leave the flag unset.
- **Hook names inside adapters are Bridge-native** (e.g., `blocks.{i}.hook_out`). HookedTransformer-style aliases (e.g., `blocks.{i}.hook_resid_post`) are registered elsewhere — in `transformer_lens/model_bridge/bridge.py` via `build_alias_to_canonical_map()`. Adapters declare canonical names only.
- **`ComponentMapping` types do not need `# type: ignore`.** If the type system disagrees, prefer `isinstance` narrowing or `typing.cast`; the project as a whole avoids `# type: ignore`.

### Verifying a new model

After your adapter is registered, the model registry's `verify_models` runner exercises it end-to-end:

```bash
set -a; source .env; set +a
uv run python -m transformer_lens.tools.model_registry.verify_models --model <hf_repo>
```

`verify_models` runs phases 1–4 (forward correctness vs HF, hook firing + gradients, weight processing, generation quality) and updates `data/supported_models.json` with the resulting status and per-phase scores. We recommend running `--dry-run` first to project memory and parameter count without loading the model, and verifying one model at a time — concurrent loads tend to OOM a single device.

A note on entry points: `verify_models` is the script that writes the registry. `main_benchmark` runs the same underlying benchmarks but defaults to *not* writing the registry (it requires `--update-registry`, and even then it doesn't record Phase 7 / 8 scores or the resume checkpoint). If you want the registry updated after your run, use `verify_models`.

It's worth reading the per-phase scores in addition to the final status — the verifier enforces hard pass/fail thresholds, and a model that just clears the bar tells you something different than one that breezes through. The current thresholds:

| Phase | Min score | Required tests | Effect when below threshold |
|---|---|---|---|
| 1 | 100% | — | Verification fails |
| 2 | 75% | `logits_equivalence`, `loss_equivalence` | Verification fails |
| 3 | 75% | `logits_equivalence`, `loss_equivalence` | Verification fails |
| 4 | 50% | — | **Non-gating** — below 50% adds `"low text quality"` to the registry `note`; never fails verification. |
| 7 | 75% | `multimodal_forward` | Verification fails. A NULL score also fails. |
| 8 | 75% | `audio_forward` | Verification fails. A NULL score also fails. |

Phase 4 is intentionally lenient — it's a coherence metric, not a correctness check. A sub-100% Phase-4 score on a small parity-test model can still indicate a real adapter bug that the gates don't catch (missing `preprocess_weights` fold, wrong `default_prepend_bos`, and so on); the model can pass verification overall and still be worth a manual look.

If verification fails by `~1e-3` or more against the HF reference, the bisection workflow lives at [Debugging Numerical Divergence](debugging_numerical_divergence.md).

```{toctree}
:hidden:
:maxdepth: 1

adapter_development/adapter-creation-guide
adapter_development/hf-model-analysis-guide
adapter_development/hf-scraper
adapter_development/external-adapter-registration
```
