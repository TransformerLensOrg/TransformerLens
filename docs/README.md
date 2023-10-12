
# Transformer-Lens Docs


This repo contains the [website](https://neelnanda-io.github.io/TransformerLens/) for [TransformerLens](https://github.com/neelnanda-io/TransformerLens). This site is currently in Beta and we are in the process of adding/editing information.

The documentation uses Sphinx. However, the documentation is written in regular md, NOT rst.

## Build the Documentation

First install the docs packages:

```bash
poetry install --with docs
```

Then for hot-reloading, run:

```bash
poetry run autobuild
```

Alternatively to build once, run:

```bash
poetry run docs
```