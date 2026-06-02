# CLAUDE.md

**Read [AGENTS.md](AGENTS.md) first.** It is the single source of truth for project conventions, quickstart commands, repo layout, hard rules, and PR conventions. Everything below is Claude-Code-specific tooling on top of it.

## Claude Code conveniences in this repo

- **Slash commands** — see [.claude/commands/](.claude/commands/) for canonical workflows:
  - `/test-unit` — `make unit-test`
  - `/test-all` — `make test` (slow; warns before running)
  - `/format` — `make format` then `uv run mypy .`
  - `/typecheck` — `uv run mypy .`
  - `/build-docs` — sources `.env`, runs `uv run build-docs`
  - `/verify-model <model_name>` — guided single-model run of `verify_models.py`
  - `/add-model-support <hf_repo>` — checklist-driven new-adapter workflow
  - `/task-complete` — end-of-task gate: clean up new comments per the comment rules in AGENTS.md, then `/format`, then run `unit-test` + `docstring-test` + `acceptance-test` + `integration-test`, fixing any issues until all three steps pass cleanly. Skips notebook + benchmark tiers (run those directly if your change touched them).
- **Project settings** — [.claude/settings.json](.claude/settings.json) is the checked-in scaffold for project-wide Claude Code settings (hooks, env). It intentionally ships with no permission allowlist; configure your own permissions in `.claude/settings.local.json` (gitignored).
- **User-specific settings** — `.claude/settings.local.json`, `.claude/agents/`, `.claude/worktrees/` are all gitignored. Configure them per-user.

## Pointers when working with Claude Code in this repo

- The hard rules in [AGENTS.md §10](AGENTS.md#10-hard-rules) reflect rules the maintainer has repeatedly enforced. Treat them as load-bearing.
- The mirroring rule in [AGENTS.md §2](AGENTS.md#2-two-systems-live-in-this-repo) — `HookedTransformer` to `TransformerBridge` — is the single most common source of "you forgot to mirror this to TransformerBridge" PR feedback.
