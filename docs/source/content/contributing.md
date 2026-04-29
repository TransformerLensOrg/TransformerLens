# Contributing

```{warning}
`HookedTransformer` is deprecated as of TransformerLens 3.0 and will be removed in the next major version. New code should use [`TransformerBridge`](migrating_to_v3.md) instead. Existing `HookedTransformer` code continues to work through the 3.x branch via a compatibility layer. See the [migration guide](migrating_to_v3.md) for conversion recipes.
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
```

Dependency groups are defined in `pyproject.toml` under `[dependency-groups]`. The project sets `default-groups = ["dev", "docs", "jupyter"]`, so `uv sync` installs all three out of the box — you do not need to pass `--group` flags for the standard contributor setup.

- Standard contributor setup (recommended default): `uv sync`
- Include the optional `quantization` group (bitsandbytes, optimum-quanto): `uv sync --all-groups`

You can also add individual groups with `uv sync --group <name>`, or install without optional groups using `uv sync --no-default-groups`.

Requires Python 3.10 or higher.

## Testing

If adding a feature, please add unit tests for it. If you need a model, please use one of the ones
that are cached by GitHub Actions (so that it runs quickly on the CD). These are `gpt2`,
`attn-only-1l`, `attn-only-2l`, `attn-only-3l`, `attn-only-4l`, `tiny-stories-1M`. Note `gpt2` is
quite slow (as we only have CPU actions) so the smaller models like `attn-only-1l` and
`tiny-stories-1M` are preferred if possible.

### Running the tests

- Unit tests only via `make unit-test`
- Acceptance tests only via `make acceptance-test`
- Docstring tests only via `make docstring-test`
- Notebook tests only via `make notebook-test`
- Run all test suites mentioned `make test`

## Formatting

This project uses `pycln`, `isort` and `black` for formatting, pull requests are checked in github
actions.

- Format all files via `make format`
- Only check the formatting via `make check-format`

Note that `black` line length is set to 100 in `pyproject.toml` (instead of the default 88).

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

Adapters live in `transformer_lens/model_bridge/supported_architectures/<model_name>.py` and are registered in two places: `supported_architectures/__init__.py` and `factories/architecture_adapter_factory.py`. Both steps are covered in the creation guide. If you want a starter file, copy [adapter-template.py](../_static/adapter-template.py) into `supported_architectures/` and rename it.

```{toctree}
:hidden:
:maxdepth: 1

adapter_development/adapter-creation-guide
adapter_development/hf-model-analysis-guide
```
