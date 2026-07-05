
# Transformer-Lens Docs


This repo contains the [website](https://TransformerLensOrg.github.io/TransformerLens/) for [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). This site is currently in Beta and we are in the process of adding/editing information.

The documentation uses Sphinx. However, the documentation is written in regular md, NOT rst.

## Build the Documentation

For the standard contributor setup, install the default dependency groups:

```bash
uv sync
```

For a docs-focused environment without the other default groups, install only the docs group:

```bash
uv sync --no-default-groups --group docs
```

Then for hot-reloading, run this (note the model properties table won't hot reload, but everything
else will):

```bash
uv run docs-hot-reload
```

Alternatively to build once, run:

```bash
uv run build-docs
```
