---
description: Type-check then format the working tree.
---

Run mypy first, then format. Mypy fixes (`isinstance`, `cast`, signatures) can introduce format drift — running format after means a single pass.

```
uv run mypy .
make format
```

`uv run mypy .` uses the config in [pyproject.toml](../../pyproject.toml). `make format` runs `pycln --all` (unused imports), `isort`, and `black` (line length 100).

If mypy reports errors, fix the underlying typing issue — never add `# type: ignore`. Prefer `isinstance` / `typing.cast` ([AGENTS.md §10](../../AGENTS.md#10-hard-rules)).
