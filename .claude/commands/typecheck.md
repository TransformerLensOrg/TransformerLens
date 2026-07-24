---
description: Run mypy across the project.
---

Run the type checker:

```
uv run mypy .
```

Config lives in `[tool.mypy]` of [pyproject.toml](../../pyproject.toml). If mypy reports errors, fix the underlying typing issue — do not add `# type: ignore`. Prefer `isinstance` assertions or `typing.cast` for narrowing.
