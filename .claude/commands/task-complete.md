---
description: End-of-task gate. Clean up new comments, format, type-check, and run the standard test tiers (unit + docstring + acceptance + integration) — fixing issues along the way.
---

Run the end-of-task gate. Do not declare the task complete until every step below passes cleanly.

### 1. Clean up new comments

Review every comment and docstring **added or modified during this task** against the rules in [AGENTS.md §10](../../AGENTS.md#10-hard-rules):

- Comments should be terse one-liners; docstrings are one-line where possible.
- Inline comments explain WHY, not WHAT — delete any that just restate the code.
- Multi-paragraph explanations belong in PR descriptions or design docs, not source.
- Remove any references to plan files, audit IDs, finding IDs, or "see plan section X" — those rot as the codebase evolves and belong only in the PR description.

Use `git diff` against the merge-base to scope the review to genuinely new comments — do NOT rewrite unrelated comments elsewhere in the file.

### 2. Format

```
make format
uv run mypy .
```

If mypy reports new errors, fix the underlying typing issue. Do not add `# type: ignore` — prefer `isinstance` assertions or `typing.cast`.

### 3. Run the standard test tiers

```
set -a; source .env; set +a
make unit-test
make docstring-test
make acceptance-test
make integration-test
```

These are the tiers that gate PR review for almost every change. Notebook and benchmark suites are intentionally skipped — they are slow, hit gated models, and are not required to land typical PRs (CI runs them separately). If your change specifically touches a notebook or a benchmark, run that file directly (`pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/<notebook>.ipynb` or `make benchmark-test`) in addition.

Investigate every failure. Do not dismiss any failure as "pre-existing" or "unrelated" — fix the underlying issue, even if it predates this task (see [AGENTS.md §10](../../AGENTS.md#10-hard-rules)). Do not add platform skips or `xfail` markers to dodge a failing test.

### 4. Re-loop on failure

If any of steps 1–3 surface issues, fix them, then **restart from step 1** — fixes can introduce new comments, new format/type drift, or new test failures. Repeat until all three steps pass without changes.

### 5. Report

Report the actual final command output, not a summary. Reviewers re-run tests; agent self-reports are not evidence ([AGENTS.md §10](../../AGENTS.md#10-hard-rules)).
