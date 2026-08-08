# TL Adapter Builder

## Purpose

This directory automates creating **Architecture Adapters** for the `TransformerBridge` system. It is a **control plane** — it holds agent definitions, domain knowledge, launch scripts, and tooling. It lives at `devtools/adapter_builder/` inside the TransformerLens repo but is contributor tooling, not part of the shipped package. Agents work in **git worktrees of this repo** created outside the checkout; nothing runs against the main working tree.

For the manual (non-agent-team) path to the same outcome, see the repo's `/add-model-support` slash command — the adapter spec and registration checklist in the repo docs are canonical; this tool's docs cover orchestration only.

## Project Layout

```text
agents/                        Agent definitions and orchestration
  launch-agent-pair.sh         User-facing dispatcher — routes to launch.sh, launch-solo-pair.sh, or ops.sh
  launch.sh                    Launch flow — agent-teams mode (Max tier, orchestrator + subagents)
  launch-solo-pair.sh          Launch flow — solo mode (any tier, two independent sessions, file-based coordination)
  ops.sh                       Ops subcommands (status/logs/attach/send/stop/clean)
  watch-completion.sh          Background watcher — auto-sends /exit on genuine completion
  solo-coordinator.sh          Solo-mode signal router daemon (consumes signals, wakes panes)
  check-signals.sh             Non-blocking signal state snapshot (solo re-orientation)
  solo-programmer.md           Programmer prompt for solo mode (signal + end turn)
  solo-reviewer.md             Reviewer prompt for solo mode (message-driven)
  lib/common.sh                Shared bash helpers (log/ok/warn/err/require_cmd)
  orchestrator.md              Orchestration prompt template ({{PLACEHOLDER}} tokens)
  programmer.md                Programmer agent (3-step lifecycle)
  reviewer.md                  Reviewer agent (5-phase review, Opus model)
  signals.sh                   Signal protocol — single source of truth
  progress.sh                  Progress tracking for crash recovery
  overlord-request.sh          flock-based memory lock for heavy operations
  hooks/                       Claude Code hooks for auto-enforcement
    timeline-capture.sh          All events — structured JSONL logging
    guard-hooked-transformer.sh  PreToolUse — blocks edits to deprecated files
    guard-git.sh                 PreToolUse — blocks `git commit`, `git push`, `gh pr create`
    guard-review-rounds.sh       PreToolUse — blocks review files past round 3
    guard-verify-models.sh       PreToolUse — blocks verify_models on >7B or unregistered models
    gate-reviewer-writes-file.sh SubagentStop — blocks reviewer results that have no file artifact
    gate-lint-checks.sh          Stop — blocks exit until mypy+format pass
    notify-on-completion.sh      SessionEnd — fires Slack notification on success
docs/                          Domain knowledge for agents
  adapter-specification.md     What an adapter is and how to build one
  adapter-template.py          Skeleton adapter (Llama-style pattern)
  artifact-templates.md        File templates for all build artifacts (brief, plan, reviews, etc.)
  memory-lock.md               Memory lock protocol (flock-based, run subcommand only)
  hf-model-analysis-guide.md   How to analyze an HF model for adapter creation
  review-specification.md      Five-phase review methodology (source of truth)
scripts/                       Tooling
  analyze-hf-model.py          Analyze HF model config, generate scaffold adapters
  validate-architecture.py     Pre-flight check: is this arch real? (transformers + HF Hub)
  scan-hf-architecture.py      Exhaustive HF scan for all models of an arch class
  port-arch-models.py          Merge per-arch model list into supported_models.json
  validate-adapter.sh          Structural + deep validation of adapters
  validate-adapter-deep.py     Semantic validation against real models (meta device)
  compare-adapters.sh          Structured diff between two existing adapters
  format-timeline.py           Render timeline.jsonl entries for status/logs
  notify.sh                    Slack/iMessage/macOS notification on completion
  strip-adapter.py             Remove an adapter + registrations + registry entries (golden-master rebuild tests)
  dry-run-test.sh              Self-test suite for this project
```

## TransformerLens Repo

- **Path:** the repo containing this checkout, derived automatically (`git rev-parse --show-toplevel`). Override with `--target-repo` or `DEFAULT_TARGET_REPO` in `.env` to drive a different checkout.
- **Key paths:**
  - `transformer_lens/model_bridge/architecture_adapter.py` — base class
  - `transformer_lens/model_bridge/supported_architectures/` — all existing adapters
  - `transformer_lens/model_bridge/generalized_components/` — bridge components
  - `transformer_lens/factories/architecture_adapter_factory.py` — adapter registry
  - `transformer_lens/config/transformer_bridge_config.py` — TransformerBridgeConfig

## Agent System

Uses Claude Code experimental agent teams (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`).

**Lifecycle:**

1. `launch-agent-pair.sh` creates a git worktree in the TransformerLens repo
2. **Step 1a:** Programmer analyzes architecture, writes brief (`.adapter-workspace/adapter-brief.md`), Reviewer fact-checks
3. **Step 1b:** Programmer writes phased plan (`.adapter-workspace/adapter-plan.md`), Reviewer approves
4. **Step 2:** Programmer implements phase-by-phase, Reviewer reviews each phase
5. **Step 3:** Programmer runs verify_models one model at a time, then mypy + format checks
6. Reviewer writes completion report, notification sent to Slack

**Key constraints:**

- Max 3 review rounds per checkpoint before escalating to user
- No git commits or pushes by agents — user commits manually
- `# type: ignore` not acceptable for mypy errors
- New bridge components only when forward pass is fundamentally different

**Auto-enforced by Claude Code hooks** (in `.claude/settings.json` per worktree):

- Timeline capture: every tool call + session event → `.adapter-workspace/timeline.jsonl`
- HookedTransformer guardrail: edits to deprecated files are blocked at the framework level
- Lint gate: session can't end until `mypy` + `make check-format` pass
- Completion notifier: Slack fires automatically on `verification_passed: true`

**Memory lock:** flock-based (`overlord-request.sh`) at `/tmp/tl-adapter-builder.lock`. Only for verify_models and full model loading. 30-minute TTL for stale lock cleanup.

**Crash recovery:** `.adapter-progress.json` in worktree tracks lifecycle step, completed phases, and verification attempts. Relaunching resumes from saved state. `--retry` flag is safe for planning/programming crashes; verification crashes require manual intervention.

**Background mode (tmux):** `--background` runs the session inside a detached tmux session named `tl-adapter-<branch>`. This gives a real pty (so Claude's interactive mode works), a persistent process that survives terminal disconnects, and lets you attach/inject at any time via:

- `./agents/launch-agent-pair.sh attach <arch>` — interactive tmux attach
- `./agents/launch-agent-pair.sh send <arch> "message"` — inject a prompt as if you typed it

Nothing is copied into the TransformerLens worktree except `.claude/agents/` (required by Claude Code agent teams) and `.claude/settings.json` (registers hooks). All docs, scripts, and tooling are accessed via absolute paths using the `TL_ADAPTER_BUILDER_ROOT` env var.

## Environment

- `.env` — see `.env.example` for all variables; gitignored, never commit it
- `HF_TOKEN` — HuggingFace API token (falls back to the repo root `.env` if unset here)
- `DEFAULT_TARGET_REPO` — optional override; defaults to the containing repo
- `DEFAULT_BASE_BRANCH` — default branch (dev-4.x)
- `DEFAULT_MAX_MEMORY_GB` — memory limit for verification (96)
- `WORKTREE_BASE` — where agent pair worktrees live; optional, defaults to `<parent-of-TransformerLens>/worktrees`
- `NOTIFICATION_WEBHOOK_URL` — Slack webhook for notifications
- `NOTIFICATION_NUMBER` — iMessage fallback
- Python managed via `uv`

Runtime output (PID files, debug logs, raw tmux dumps, wrapper scripts, orchestration prompts) is written to `.logs/` in the project root and is gitignored.
