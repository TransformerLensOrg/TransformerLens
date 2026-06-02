---
description: Source .env then build the Sphinx docs.
---

Build the documentation locally:

```
set -a; source .env; set +a
uv run build-docs
```

Sourcing `.env` is required so `HF_TOKEN` is available — some doctests and notebook embeddings load gated models. Output goes to [docs/build/](../../docs/build/).

For an interactive live-reloading preview instead, run `uv run docs-hot-reload`.

Docs follow Google docstring style with reST extensions; see [docs/source/content/contributing.md](../../docs/source/content/contributing.md) for the style guide.
