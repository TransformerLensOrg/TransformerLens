
# Transformer-Lens Docs


This repo contains the [website](https://TransformerLensOrg.github.io/TransformerLens/) for [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). This site is currently in Beta and we are in the process of adding/editing information.

The documentation uses Sphinx. However, the documentation is written in regular md, NOT rst.

## Build the Documentation

First install the docs packages:

```bash
poetry install --with docs
```

Then for hot-reloading, run this (note the model properties table won't hot reload, but everything
else will):

```bash
poetry run docs-hot-reload
```

Alternatively to build once, run:

```bash
poetry run build-docs
```
