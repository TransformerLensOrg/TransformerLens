# CLAUDE.md

**Read [AGENTS.md](AGENTS.md) first** — single source of truth. Everything below is Claude-Code-specific.

## Slash commands ([.claude/commands/](.claude/commands/))

- `/test-unit` — `make unit-test`
- `/test-all` — `make test` (long; benchmarks + notebooks too)
- `/format` — `make format` + `uv run mypy .`
- `/typecheck` — `uv run mypy .`
- `/build-docs` — sources `.env`, runs `uv run build-docs`
- `/verify-model <model_name>` — guided single-model `verify_models.py` run
- `/add-model-support <hf_repo>` — checklist-driven new-adapter workflow
- `/task-complete` — end-of-task gate: clean new comments, `/format`, `make test-pr` (unit + docstring + acceptance + integration), loop until clean. Skips notebook + benchmark tiers.

## Settings

- [.claude/settings.json](.claude/settings.json) is the checked-in scaffold (hooks, env); ships with no permission allowlist.
- **First-time permission setup**: every command prompts on first use. Either (1) run `/fewer-permission-prompts` to populate `.claude/settings.local.json` from your transcripts, or (2) approve with "Don't ask again." Both write to the gitignored local file.
- `.claude/settings.local.json`, `.claude/agents/`, `.claude/worktrees/` are gitignored — per-user.

## Pointers

- [AGENTS.md §10](AGENTS.md#10-hard-rules) — hard rules; load-bearing.
- [AGENTS.md §2](AGENTS.md#2-two-systems-live-in-this-repo) — HT → Bridge mirroring; most common PR-review pushback.
- [tests/QUARANTINES.md](tests/QUARANTINES.md) — check before debugging any failing test. The macOS-arm64 KV-cache skip is the most common time-sink.
- [debugging_numerical_divergence.md](docs/source/content/debugging_numerical_divergence.md) — Bridge-vs-HF logit drift bisection.
- [compatibility_mode.md](docs/source/content/compatibility_mode.md) — `bridge.enable_compatibility_mode()` contract; read before adding tests that use it.

## Starter tasks

- [`good first issue`](https://github.com/TransformerLensOrg/TransformerLens/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) · [`help wanted`](https://github.com/TransformerLensOrg/TransformerLens/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
- [`verification-request`](https://github.com/TransformerLensOrg/TransformerLens/issues?q=is%3Aissue+is%3Aopen+label%3Averification-request) — models awaiting verification; pick one that fits your machine, run `/verify-model <model_id>`.
- Backfilling per-adapter unit tests in [`tests/unit/model_bridge/supported_architectures/`](tests/unit/model_bridge/supported_architectures/) is high-leverage; copy a sibling adapter test.
