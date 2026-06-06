# GitHub Copilot Instructions

**Read [AGENTS.md](../AGENTS.md) at the repo root for the full set of project conventions, quickstart commands, repo layout, and hard rules — start with its TL;DR section.** This file inlines the highest-friction defaults Copilot most often gets wrong.

## Top rules to remember

1. **Use `uv`, not `pip` or `poetry`.** `uv sync` to install; `uv run <cmd>` or a `make` target to run anything.
2. **Mirror `HookedTransformer` → `TransformerBridge`** in the same PR when behaviour exists in both. The HT registry [`transformer_lens/supported_models.py`](../transformer_lens/supported_models.py) is HT-only — Bridge-only models go in the Bridge registry under [`transformer_lens/tools/model_registry/`](../transformer_lens/tools/model_registry/).
3. **Base PRs against `dev`**, not `main`. PRs to `main` are maintainer-only.

## Common commands

```bash
uv sync                  # install
make unit-test           # fast tests
make format              # pycln + isort + black
uv run mypy .            # type check
uv run docs-hot-reload   # live docs preview
```

## Copilot-specific anti-patterns

- Don't add `# type: ignore`. Prefer `isinstance` / `typing.cast`.
- Don't dismiss failing tests as "pre-existing" — investigate every failure.

The full set of hard rules (numerics, parallel benchmarks, plan-file references, etc.) lives in [AGENTS.md §10](../AGENTS.md#10-hard-rules).
