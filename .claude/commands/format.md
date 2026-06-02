---
description: Format the codebase (pycln + isort + black) and run mypy.
---

Format and type-check the working tree:

```
make format
uv run mypy .
```

`make format` runs `pycln --all` (unused imports), `isort` (import sorting), and `black` (line length 100). `uv run mypy .` runs the type checker with the config in [pyproject.toml](../../pyproject.toml).

If mypy reports new errors, fix them — do not add `# type: ignore`. Prefer `isinstance` assertions or `typing.cast` for narrowing. See [AGENTS.md §10](../../AGENTS.md#10-hard-rules).
