# TL Adapter Builder

An AI agent team system that automates creating **Architecture Adapters** for the TransformerBridge system. This is **contributor tooling** — it lives in `devtools/adapter_builder/`, is not part of the shipped `transformer_lens` package, and is not supported API. For the manual path to the same outcome, use the repo's `/add-model-support` slash command.

Give it a HuggingFace architecture class name, and a team of Claude Code agents will plan the adapter, implement it phase-by-phase with code review at every step, and verify it against real models — all autonomously.

## Quick Start

Run from anywhere inside the repo (paths shown from the repo root):

```bash
# Interactive (foreground)
devtools/adapter_builder/agents/launch-agent-pair.sh --architecture CohereForCausalLM

# Background (detached tmux) — run multiple in parallel
devtools/adapter_builder/agents/launch-agent-pair.sh --architecture CohereForCausalLM --background
devtools/adapter_builder/agents/launch-agent-pair.sh --architecture CodeGenForCausalLM --background

# Check status of all running pairs
devtools/adapter_builder/agents/launch-agent-pair.sh status

# Tail the structured timeline for a specific architecture
devtools/adapter_builder/agents/launch-agent-pair.sh logs codegen

# Clean up one worktree, or all verified-complete ones
devtools/adapter_builder/agents/launch-agent-pair.sh clean CodeGenForCausalLM
devtools/adapter_builder/agents/launch-agent-pair.sh clean
```

If the architecture already has an adapter in TransformerLens, the launcher exits immediately. If a worktree already exists for the branch, it resumes from `.adapter-progress.json`. Full CLI details in [`docs/cli-reference.md`](docs/cli-reference.md).

## How It Works

This directory is a **control plane**. It does not contain adapter code itself — it launches Claude Code agent teams that work in isolated git worktrees of this repository, created outside the checkout (default: `<parent-of-repo>/worktrees`). The main working tree is never touched. **Nothing is copied into the worktree** except `.claude/agents/` (Claude Code tool config) and `.claude/settings.json` (hook registration). All docs, scripts, and tooling are accessed via absolute paths using `$TL_ADAPTER_BUILDER_ROOT`, which the launcher derives from its own location.

### The Lifecycle

Every adapter goes through a strict sequence of checkpoints:

1. **Step 1a — Architecture Analysis.** Programmer reads the HF Transformers source, extracts module paths and architectural properties per [`docs/hf-model-analysis-guide.md`](docs/hf-model-analysis-guide.md), generates a scaffold adapter via `analyze-hf-model.py --scaffold`, scans HuggingFace for every model of the target architecture via `scan-hf-architecture.py`, and writes an architecture brief. Reviewer independently re-reads the HF source and fact-checks the brief before any plan exists.
2. **Step 1b — Planning.** Using the validated brief, Programmer writes a phased implementation plan. New bridge components get their own phase. Reviewer checks design, completeness, and consistency with the brief.
3. **Step 2 — Programming.** Programmer implements one phase at a time, starting from the scaffold. Each phase goes through the Reviewer's five-phase review process ([`docs/review-specification.md`](docs/review-specification.md)) before the next begins.
4. **Step 3 — Verification.** Programmer ports the architecture's models into `supported_models.json` via `port-arch-models.py`, then runs `verify_models` one model at a time on targets ≤7B parameters. Any failure halts and returns to planning. On success, `mypy` and `make check-format` must pass before the session can end.

The agent prompts are authoritative and point at the `docs/` files rather than inlining their content — see [`agents/programmer.md`](agents/programmer.md) and [`agents/reviewer.md`](agents/reviewer.md) for the full workflow.

### The Agent Team

| Agent | Role | Model |
| ----- | ---- | ----- |
| **Orchestrator** | Routes messages, drives the lifecycle, enforces iteration limits | claude-sonnet-4-6 |
| **Programmer** | Analyzes architectures, plans, writes code, runs verification | claude-sonnet-4-6 |
| **Reviewer** | Reviews briefs, plans, and code using a 5-phase review process | claude-opus-4-6 |

Agents do **not** make git commits, push to remotes, or create pull requests. All changes stay uncommitted for manual review. This is enforced at the framework level by `guard-git.sh` (see below), not just by the prompts.

### Runtime Enforcement via Claude Code Hooks

Critical rules are enforced by Claude Code hooks, not by asking agents to remember them. The launcher installs `.claude/settings.json` in each worktree registering the hooks listed below. Full details, matchers, and block conditions are in [`docs/hooks-reference.md`](docs/hooks-reference.md).

| Hook | Enforces |
| ---- | -------- |
| **guard-hooked-transformer** | Blocks writes to deprecated `HookedTransformer.py`, `loading_from_pretrained.py`, `components/`, `pretrained/weight_conversions/` |
| **guard-git** | Blocks `git commit`, `git push`, `gh pr create`, `gh release create` — agents cannot publish changes |
| **guard-review-rounds** | Blocks review files past round 3 per checkpoint, forcing escalation to the user when loops stall |
| **guard-verify-models** | Blocks `verify_models --model …` invocations targeting unregistered models or anything above `MAX_VERIFY_PARAMS` (default 7.5B) |
| **gate-reviewer-writes-file** | Blocks reviewer subagent returns that didn't persist a review file to `.adapter-workspace/reviews/` (or `completion-report.md`) |
| **gate-lint-checks** | Refuses to let the session end until `mypy` and `make check-format` both pass |
| **timeline-capture** | Logs every tool call and session event to `.adapter-workspace/timeline.jsonl` (grepable, structured) |
| **notify-on-completion** | Fires the Slack notification automatically when `verification_passed: true` |

The effect is that the prompt-level rules (no commits, 3-round limit, ≤7B verification) are also runtime-level stops. An agent that misreads the prompt gets blocked by the framework with a message pointing at the exact next action.

### Crash Recovery

`.adapter-progress.json` in the worktree tracks lifecycle step, plan approval status, current/completed phases, verification attempts, and the final result. Relaunching with the same `--architecture` reads this file and constructs a resume prompt telling agents exactly where they left off.

Use `--retry` to explicitly resume a crashed session. It's safe for planning and programming crashes but refuses to auto-resume verification crashes (the registry may be in a partial state, and reruns can waste hours). Verification crashes trigger a "needs human review" notification and require manual inspection.

### Background Mode

`--background` launches Claude Code inside a detached `tl-adapter-<branch>` tmux session — real pty (so interactive mode works), persistent across terminal disconnects, attachable and injectable. Full details in [`docs/cli-reference.md § Background Mode`](docs/cli-reference.md#background-mode-tmux).

## Project Structure

```text
.
├── agents/
│   ├── launch-agent-pair.sh      # User-facing dispatcher — routes to launch.sh or ops.sh
│   ├── launch.sh                 # Launch flow — creates worktree, boots agent pair
│   ├── ops.sh                    # Ops subcommands: status / logs / attach / send / stop / clean
│   ├── watch-completion.sh       # Background watcher — auto-sends /exit on genuine completion
│   ├── lib/common.sh             # Shared bash helpers (log/ok/warn/err, require_cmd, iso_now)
│   ├── orchestrator.md           # Orchestration prompt template ({{PLACEHOLDER}} tokens)
│   ├── programmer.md             # Programmer agent (3-step lifecycle)
│   ├── reviewer.md               # Reviewer agent (references docs/review-specification.md)
│   ├── signals.sh                # Signal protocol — single source of truth
│   ├── progress.sh               # Progress tracking for crash recovery
│   ├── overlord-request.sh       # flock-based memory lock
│   └── hooks/                    # Runtime-enforcement hooks (see docs/hooks-reference.md)
│       ├── timeline-capture.sh
│       ├── guard-hooked-transformer.sh
│       ├── guard-git.sh
│       ├── guard-review-rounds.sh
│       ├── guard-verify-models.sh
│       ├── gate-reviewer-writes-file.sh
│       ├── gate-lint-checks.sh
│       └── notify-on-completion.sh
├── docs/
│   ├── adapter-specification.md  # Full adapter spec, all bridge components, common patterns
│   ├── adapter-template.py       # Skeleton adapter with TODOs (Llama-style)
│   ├── hf-model-analysis-guide.md # How to analyze an HF model for adapter creation
│   ├── review-specification.md   # Five-phase review methodology (source of truth)
│   ├── cli-reference.md          # Complete CLI reference (subcommands, flags, env vars, tmux)
│   ├── scripts-reference.md      # Detailed script docs
│   └── hooks-reference.md        # Hook catalog with triggers and rationale
├── scripts/
│   ├── analyze-hf-model.py       # Analyze HF model config, generate scaffold adapters
│   ├── scan-hf-architecture.py   # Exhaustive HF scan for all models of an arch class
│   ├── port-arch-models.py       # Merge per-arch list into supported_models.json
│   ├── validate-adapter.sh       # Structural + deep validation of adapters
│   ├── validate-adapter-deep.py  # Semantic validation against real models (meta device)
│   ├── compare-adapters.sh       # Structured diff between two existing adapters
│   ├── format-timeline.py        # Render timeline.jsonl entries for status/logs
│   ├── notify.sh                 # Slack/iMessage/macOS notification on completion
│   └── dry-run-test.sh           # Self-test suite for this project
├── .logs/                        # Runtime output (gitignored) — PIDs, raw tmux dumps, debug logs
├── .env                          # Local defaults (see .env.example)
├── .env.example                  # Template with placeholder values
├── CLAUDE.md                     # Project context for Claude Code
└── README.md                     # This file
```

## Configuration

Local defaults live in `.env`; see [`.env.example`](.env.example) for a template.

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `DEFAULT_TARGET_REPO` | containing repo | Optional override; defaults to the repo holding this checkout |
| `DEFAULT_BASE_BRANCH` | `dev-4.x` | Base branch for worktrees |
| `DEFAULT_MAX_MEMORY_GB` | `96` | Memory limit in GB for verify_models |
| `WORKTREE_BASE` | `<parent-of-TransformerLens>/worktrees` | Where agent pair worktrees are created |
| `NOTIFICATION_WEBHOOK_URL` | — | Slack/Discord webhook URL for notifications |
| `NOTIFICATION_NUMBER` | — | Phone number for iMessage fallback (macOS only) |
| `HF_TOKEN` | repo root `.env` | HuggingFace API token; falls back to the repo root `.env` |

Runtime hook tunables (optional, for advanced use): `MAX_REVIEW_ROUNDS` (default 3) and `MAX_VERIFY_PARAMS` (default 7.5B). See [`docs/cli-reference.md § Hook Tunables`](docs/cli-reference.md#hook-tunables-environment-variables).

## Prerequisites

- **Claude Code Max subscription** — Uses experimental agent teams. A single run involves sustained multi-agent sessions with multiple review loops and verification. Pro/Team tier will not have sufficient limits.
- **TransformerLens** repo cloned locally
- **git**, **claude**, **jq**, and **tmux** (required for `--background` mode) on `PATH`
- **Python 3** with `transformers` and `huggingface_hub`
- **uv** for running TransformerLens tests and verify_models
- **Sufficient system memory** — defaults to 96GB (`DEFAULT_MAX_MEMORY_GB`); adjustable via `--max-memory`

## How Adapters Work

An Architecture Adapter maps between a HuggingFace model's internal structure and TransformerLens's canonical component names. Every adapter defines:

1. **Config attributes** — normalization type, positional embedding type, GQA support, etc.
2. **Component mapping** — maps TL names (`embed`, `blocks`, `attn`, `mlp`, etc.) to HF module paths via Bridge components
3. **Weight processing conversions** — tensor reshaping rules for loading HF weights into TL format

For models with Grouped Query Attention (GQA), the adapter sets `n_key_value_heads` and weight conversions use `n_kv_heads` (not `n_heads`) for K/V rearrangement.

The full spec is in [`docs/adapter-specification.md`](docs/adapter-specification.md). The skeleton template is in [`docs/adapter-template.py`](docs/adapter-template.py). TransformerLens currently supports 40+ architectures — see the [factory](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/factories/architecture_adapter_factory.py) for the full list.

## Further Reading

- **[`docs/cli-reference.md`](docs/cli-reference.md)** — all subcommands, flags, env vars, tmux details, signal protocol
- **[`docs/scripts-reference.md`](docs/scripts-reference.md)** — every tool in `scripts/` with examples
- **[`docs/hooks-reference.md`](docs/hooks-reference.md)** — every hook in `agents/hooks/`, what it blocks, and why
- **[`docs/adapter-specification.md`](docs/adapter-specification.md)** — authoritative adapter spec (bridge components, patterns, registration checklist)
- **[`docs/hf-model-analysis-guide.md`](docs/hf-model-analysis-guide.md)** — how to extract architectural facts from an HF model
- **[`docs/review-specification.md`](docs/review-specification.md)** — five-phase review methodology used by the Reviewer agent
- **[`CLAUDE.md`](CLAUDE.md)** — project context file read by Claude Code when you run it in this repo
