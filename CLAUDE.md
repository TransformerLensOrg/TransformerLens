# CLAUDE.md

**Read [AGENTS.md](AGENTS.md) first.** It is the single source of truth for project conventions, quickstart commands, repo layout, hard rules, and PR conventions. Everything below is Claude-Code-specific tooling on top of it.

## Claude Code conveniences in this repo

- **Slash commands** — see [.claude/commands/](.claude/commands/) for canonical workflows:
  - `/test-unit` — `make unit-test`
  - `/test-all` — `make test` (long; runs benchmarks + notebooks too)
  - `/format` — `make format` then `uv run mypy .`
  - `/typecheck` — `uv run mypy .`
  - `/build-docs` — sources `.env`, runs `uv run build-docs`
  - `/verify-model <model_name>` — guided single-model run of `verify_models.py`
  - `/add-model-support <hf_repo>` — checklist-driven new-adapter workflow
  - `/task-complete` — end-of-task gate: clean up new comments per the comment rules in AGENTS.md, then `/format`, then `make test-pr` (unit + docstring + acceptance + integration), looping until clean. Skips notebook + benchmark tiers (run those directly if your change touched them).
- **Project settings** — [.claude/settings.json](.claude/settings.json) is the checked-in scaffold for project-wide Claude Code settings (hooks, env). It intentionally ships with no permission allowlist; configure your own permissions in `.claude/settings.local.json` (gitignored).
- **First-time permission setup** — the empty checked-in allowlist means every command (`uv sync`, `make unit-test`, `git status`, `gh pr view`, etc.) prompts on first use. Two ways to reduce the noise: (1) run the built-in `/fewer-permission-prompts` skill, which scans your recent transcripts and adds the read-only commands you actually use to `.claude/settings.local.json`; or (2) approve commands with "Don't ask again" the first time each is used. Both routes write to the gitignored local settings file — nothing committed.
- **User-specific settings** — `.claude/settings.local.json`, `.claude/agents/`, `.claude/worktrees/` are all gitignored. Configure them per-user.

## Pointers when working with Claude Code in this repo

- [AGENTS.md §10](AGENTS.md#10-hard-rules) — hard rules; load-bearing.
- [AGENTS.md §2](AGENTS.md#2-two-systems-live-in-this-repo) — HT → Bridge mirroring rule; single most common source of PR-review pushback.
- [tests/QUARANTINES.md](tests/QUARANTINES.md) — check this before debugging any failing test. Known quarantines have "un-skip when…" lines; the macOS-arm64 KV-cache skip is the most common time-sink.
- [docs/source/content/debugging_numerical_divergence.md](docs/source/content/debugging_numerical_divergence.md) — bisection workflow for Bridge-vs-HF logit drift.
- [docs/source/content/compatibility_mode.md](docs/source/content/compatibility_mode.md) — what `bridge.enable_compatibility_mode()` actually does. Read before deciding whether a new test needs it.

## Looking for a starter task?

- [Issues labeled `good first issue`](https://github.com/TransformerLensOrg/TransformerLens/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- [Issues labeled `help wanted`](https://github.com/TransformerLensOrg/TransformerLens/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
- [Issues labeled `verification-request`](https://github.com/TransformerLensOrg/TransformerLens/issues?q=is%3Aissue+is%3Aopen+label%3Averification-request) — models awaiting verification on hardware the requester didn't have. Pick one that fits your machine and run `/verify-model <model_id>`.
- Backfilling per-adapter unit tests under [`tests/unit/model_bridge/supported_architectures/`](tests/unit/model_bridge/supported_architectures/) is high-leverage — recent commits show this is an active push and the pattern is well-established (copy a sibling adapter test).
