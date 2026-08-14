# Automated Adapter Builder

The repo ships a contributor tool that automates the entire adapter-creation
workflow described in the [Adapter Creation Guide](adapter-creation-guide.md):
give it a HuggingFace architecture class name, and a pair of Claude Code
agents analyzes the HF source, plans the adapter, implements it
phase-by-phase with independent code review at every checkpoint, and verifies
it against real models.

It lives at `devtools/adapter_builder/` — it is developer tooling, not part
of the shipped `transformer_lens` package. This page is an orientation; the
authoritative reference is the tool's own
[README](https://github.com/TransformerLensOrg/TransformerLens/blob/main/devtools/adapter_builder/README.md)
and the docs beside it.

## When to use it

- **Manual path**: for a single adapter you want to write yourself, follow
  the [Adapter Creation Guide](adapter-creation-guide.md) (Claude Code users:
  the `/add-model-support` slash command walks the same checklist).
- **Automated path**: for batch adapter work, or when you want an
  autonomously built and reviewed first draft, use the builder. The output is
  left **uncommitted in an isolated git worktree** for human review — the
  agents cannot commit, push, or open PRs.

## How it works, briefly

The builder launches a **Programmer** and a **Reviewer** agent that work in a
dedicated git worktree of the repo (your main working tree is never touched).
Every lifecycle step is gated: the architecture brief is fact-checked against
the HF source before planning, each implementation phase is code-reviewed
before the next begins, and verification requires full HuggingFace output
parity — model targets are chosen for config diversity (always including the
architecture's smallest model, verified first) so numerical edge cases fail
in seconds.

Two coordination modes exist: **agent-teams** (an orchestrator session;
requires Claude Code Max) and **solo** (two independent sessions coordinated
by a signal-routing daemon; works on any tier).

Critical rules — no git writes, no edits to deprecated `HookedTransformer`
paths, no oversized or structural-only verification runs, mandatory
mypy/format gates — are enforced by Claude Code hooks at the framework
level, not just by prompt instructions.

## Launching

From the repo root:

```bash
# Agent-teams mode (Max tier)
devtools/adapter_builder/agents/launch-agent-pair.sh --architecture CohereForCausalLM

# Solo mode (any tier)
devtools/adapter_builder/agents/launch-agent-pair.sh --mode solo --architecture CohereForCausalLM

# Check everything without creating anything
devtools/adapter_builder/agents/launch-agent-pair.sh --architecture CohereForCausalLM --dry-run
```

When a run completes, the worktree (default: `../worktrees/<branch>/`)
contains the adapter, its unit tests, registry entries, and a full audit
trail (`.adapter-workspace/`: brief, plan, every review round, verification
results, and a structured timeline of every tool call). Review it, then
commit and open the PR yourself — registration and testing expectations are
the same as the manual path.

## Requirements

Claude Code on `PATH` (Max tier for agent-teams mode; any tier for solo),
plus `git`, `jq`, `tmux`, and `uv`. Verification loads real model weights:
budget memory accordingly (`--max-memory`, default 96GB).
