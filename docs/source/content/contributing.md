# Contributing

## Setup

### DevContainer

For a one-click setup of your development environment, this project includes a
[DevContainer](https://containers.dev/). It can be used locally with [VS
Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) or
with [GitHub Codespaces](https://github.com/features/codespaces).

### Manual Setup

This project uses [Poetry](https://python-poetry.org/docs/#installation) for package management.
Install as follows (this will also setup your virtual environment):

```bash
poetry config virtualenvs.in-project true
poetry install --with dev,docs,jupyter
```

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

## Formatting

This project uses `pycln`, `isort` and `black` for formatting, pull requests are checked in github
actions.

- Format all files via `make format`
- Only check the formatting via `make check-format`

## Documentation

Please make sure to add thorough documentation for any features you add. You should do this directly
in the docstring, and this will then automatically generate the API docs when merged into `main`.
They will also be automatically checked with [pytest](https://docs.pytest.org/) (via
[doctest](https://docs.python.org/3/library/doctest.html)).

If you want to view your documentation changes, run `poetry run docs-hot-reload`. This will give you
hot-reloading docs (they change in real time as you edit docstrings).

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
