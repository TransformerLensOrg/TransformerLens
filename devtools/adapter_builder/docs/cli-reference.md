# CLI Reference

Complete reference for `./agents/launch-agent-pair.sh` — the user-facing entry point. The dispatcher routes operational subcommands to [`agents/ops.sh`](../agents/ops.sh) and the default launch flow to [`agents/launch.sh`](../agents/launch.sh). Shared bash helpers live in [`agents/lib/common.sh`](../agents/lib/common.sh).

## Subcommands

```text
./agents/launch-agent-pair.sh --architecture <class> [flags]   # Launch an adapter build
./agents/launch-agent-pair.sh status                            # Show all running/stopped pairs
./agents/launch-agent-pair.sh logs <name> [--raw]               # Tail structured timeline (or raw tty log)
./agents/launch-agent-pair.sh attach <architecture>             # Attach to a running tmux session
./agents/launch-agent-pair.sh send <architecture> <message>     # Inject a message into a running session
./agents/launch-agent-pair.sh stop [architecture|all]           # Cleanly kill running session(s)
./agents/launch-agent-pair.sh clean [architecture]              # Remove worktree + branch + logs
```

### Default (launch)

```bash
./agents/launch-agent-pair.sh --architecture CohereForCausalLM              # interactive
./agents/launch-agent-pair.sh --architecture CohereForCausalLM --background # detached tmux
```

If the architecture already has an adapter in TransformerLens, the script exits immediately — no wasted work. If a worktree already exists for the branch, the launcher resumes the existing session using the `.adapter-progress.json` crash-recovery state.

### `status`

Shows every running or recently-stopped pair with:

- PID and `[running]` / `[stopped]` state
- Lifecycle step (`planning` / `programming` / `verification`)
- Current phase and completed phases
- Timeline event count and the formatted last event
- **Time since the last timeline event** — human-readable age like `12s ago`, `3m 40s ago`, `2h 15m ago`, `1d 4h ago`. Useful for spotting stuck sessions at a glance: a pair with no new events in hours is likely blocked or idling.
- tmux attach command for the session
- Memory-lock status (held or free)

The last-event column is rendered by piping the final line of the worktree's `.adapter-workspace/timeline.jsonl` through [`scripts/format-timeline.py`](../scripts/format-timeline.py). The age is computed from the same entry's `ts` field.

### `logs <arch> [--raw]`

Matches partial architecture names — `logs codegen` finds the running codegen session.

- **Default (no flag):** tails the structured JSONL timeline through `format-timeline.py`, showing `HH:MM:SS  EventName      label` lines. This is the preferred view — it skips terminal spinner noise.
- **`--raw`:** tails the raw pty dump from `.logs/<branch>.raw` with ANSI escapes stripped. Use when you need to see exactly what Claude Code rendered in the session.

Both modes are `tail -f` — Ctrl-C to stop.

### `attach <arch>`

Opens an interactive tmux connection to a running background session. You can watch the agents work in real time and type directly into Claude's interactive prompt. Detach with `Ctrl-b d` — the session keeps running.

Requires the session to have been launched with `--background`.

### `send <arch> "<message>"`

Injects a text message into a running session as if you typed it at the Claude prompt. Useful for mid-run corrections without stopping the agents:

```bash
./agents/launch-agent-pair.sh send qwen3moe "Use float32 instead of bfloat16 for verification"
./agents/launch-agent-pair.sh send codegen "Skip the 13B variant — start with the 2B model"
```

### `stop [arch|all]`

- `stop` (no arg) or `stop all` — cleanly kills every running background pair
- `stop <arch>` — stops the pair matching the given architecture name

Worktrees and `.adapter-progress.json` are preserved so you can resume later. Prefer `stop` over `kill -9` — `stop` cleanly tears down the tmux session, which lets the SessionEnd hooks fire.

### `clean [arch]`

- `clean` (no arg) — removes every worktree with `verification_passed: true` in its progress file. Skips in-progress builds.
- `clean <arch>` — removes that specific worktree, branch, and log files regardless of state.

Use this after you've committed the adapter from a successful run and want to tidy up.

## Launch Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--architecture <class>` | (required) | HF architecture class name, e.g. `CohereForCausalLM`, `CodeGenForCausalLM` |
| `--target-repo <path>` | `DEFAULT_TARGET_REPO` from `.env` | Path to TransformerLens repository |
| `--base-branch <branch>` | `DEFAULT_BASE_BRANCH` from `.env` (`dev-4.x`) | Base branch to create the worktree from |
| `--new-branch <branch>` | `feature/<arch>-adapter` | Feature branch name (auto-derived from architecture) |
| `--max-memory <gb>` | `DEFAULT_MAX_MEMORY_GB` from `.env` (`96`) | Memory limit in GB for verify_models |
| `--worktree-dir <path>` | `${WORKTREE_BASE}/<branch>` | Custom worktree location |
| `--programmer-prompt <text>` | auto-generated from architecture | Override the task prompt sent to the Programmer |
| `--reviewer-prompt <text>` | generic review focus | Extra review criteria for the Reviewer |
| `--background` | off | Run detached inside a tmux session — enables parallel pairs, persistent sessions, and `attach`/`send` |
| `--retry` | off | Resume a crashed session. Planning/programming crashes resume normally; verification crashes require manual inspection and trigger a notification instead |
| `--auto-approve` | off | Skip Claude Code permission prompts |
| `--skip-arch-check` | off | Skip the pre-flight architecture existence check (see below). Use when launching an architecture that's so new it isn't yet in the installed `transformers` package *and* isn't on HuggingFace Hub |
| `--dry-run` | off | Run all pre-flight checks (adapter-exists on the base branch, architecture validation, base-branch exists), print what would be created, and exit without creating a worktree or launching sessions |
| `-h`, `--help` | — | Print the full usage header from `launch.sh` |

## Pre-flight: Architecture Existence Check

Before creating a worktree, the launcher validates that the architecture class name actually exists. This catches typos (e.g. `CohoreForCausalLM`) that would otherwise burn a worktree, an agent startup, ~3 brief-review cycles, and a stuck-report notification before failing.

The check runs [`scripts/validate-architecture.py`](../scripts/validate-architecture.py) in the target repo's uv env and looks in two places in order:

1. **`transformers` package** — `getattr(transformers, <arch>)`. Fast, local, ~1–2s including cold import.
2. **HuggingFace Hub** — bounded scan of the top 500 models by download count, checking each `config.architectures` field for a match. Only runs if the transformers check comes up empty. ~1–3s on top of the transformers import.

**Exit codes from the validator:**

- `0` — found in transformers or on HF Hub → launcher proceeds
- `1` — definitively not found in either → launcher aborts with a "does not exist" error pointing at common causes (case sensitivity, typos)
- `2` — could not verify (missing deps, network error) → launcher emits a warning and proceeds

**Bypass:** pass `--skip-arch-check` to skip the validation entirely. Useful for architectures not yet released on HuggingFace Hub or not yet in the pinned `transformers` version.

## Configuration (`.env`)

Defaults are set in `.env`; see [`.env.example`](../.env.example) for a template.

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_TARGET_REPO` | — | Path to the TransformerLens repo |
| `DEFAULT_BASE_BRANCH` | `dev-4.x` | Base branch for worktrees |
| `DEFAULT_MAX_MEMORY_GB` | `96` | Memory limit in GB for verify_models |
| `WORKTREE_BASE` | `<parent-of-TransformerLens>/worktrees` | Where agent pair worktrees are created |
| `NOTIFICATION_WEBHOOK_URL` | — | Slack/Discord/custom webhook URL for notifications |
| `NOTIFICATION_NUMBER` | — | Phone number for iMessage fallback (macOS only) |
| `HF_TOKEN` | — | HuggingFace API token for model access |

## Hook Tunables (environment variables)

These control the runtime-enforcement hooks. See [hooks-reference.md](hooks-reference.md) for what each hook does.

| Variable | Default | Consumed by | Effect |
|----------|---------|-------------|--------|
| `MAX_REVIEW_ROUNDS` | `3` | `guard-review-rounds.sh` | Max review iterations per checkpoint before the hook blocks writes and forces escalation |
| `MAX_VERIFY_PARAMS` | `7500000000` (7.5B) | `guard-verify-models.sh` | Ceiling for models passed to `verify_models --model …` |

## Runtime Environment Variables

Set automatically by `launch.sh` and exported into every agent session:

| Variable | Description |
|----------|-------------|
| `TL_ADAPTER_BUILDER_ROOT` | Absolute path to this project (used by agents and hooks to find docs/, scripts/, and `overlord-request.sh`) |
| `MAX_MEMORY_GB` | Memory limit for verify_models |
| `NOTIFICATION_WEBHOOK_URL` | Webhook URL for notifications |
| `NOTIFICATION_NUMBER` | Phone number for iMessage fallback |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | HuggingFace auth (both names exported for compatibility with `huggingface_hub` and `transformers`) |
| `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` | Always `1` |

## Background Mode (tmux)

`--background` launches the Claude Code session inside a detached tmux session named `tl-adapter-<branch>`. This provides:

- **A real pty** — Claude Code's interactive mode works properly; piped stdio breaks subagent spawning.
- **Process persistence** — the session survives terminal disconnects, SSH drops, and laptop sleeps.
- **Interactive attach** — jump into a running session at any time and watch the agents work.
- **Message injection** — send text to the orchestrator mid-run without stopping anything.

Under the hood each session's pty output is piped to `.logs/<branch>.raw` via `tmux pipe-pane`, so the raw archive is always available for grepping. The structured timeline at `.adapter-workspace/timeline.jsonl` (inside the worktree) is the preferred source — it skips spinner noise and is what `logs` tails by default.

**Requirements:** `tmux` must be on `PATH` (`brew install tmux` on macOS, `apt install tmux` on Linux).

### Artifacts written under `.logs/`

For each background session `<branch>`:

| File | Purpose |
|------|---------|
| `<branch>.pid` | Top-level tmux pane PID; read by `status` and `stop` |
| `<branch>.log` | Legacy log target (still emitted for compatibility) |
| `<branch>.raw` | Full pty capture from `tmux pipe-pane` (ANSI-preserved) |
| `<branch>.debug.log` | Claude Code's `--debug hooks` debug output |
| `<branch>.wrapper.sh` | The small bash wrapper tmux invokes (exports env, reads prompt file, calls `claude`) |
| `<branch>.prompt.txt` | The rendered orchestration prompt (too large for shell quoting) |
| `<branch>.watch.log` | Poll log from the completion watcher (see below) |

`.logs/` is gitignored. Files accumulate across sessions — clean them up manually when needed.

### Completion watcher

The wrapper launches [`agents/watch-completion.sh`](../agents/watch-completion.sh) as a background process alongside `claude`. It closes a gap in interactive Claude Code: an orchestrator has no tool to self-terminate — it can print "Session terminating — SessionEnd hook will fire the Slack notification" but that only ends its turn, leaving the pane at the `❯` prompt. Without the watcher, the session sits forever, the SessionEnd hook never fires, and no Slack notification ever gets sent.

The watcher polls `.adapter-progress.json` every `WATCH_POLL_SECS` seconds (default `30`) and fires `tmux send-keys "/exit" Enter` when:

1. `verification_passed == true`
2. `final_review_passed == true`
3. The worktree's `.adapter-workspace/timeline.jsonl` has been idle for at least `WATCH_IDLE_SECS` seconds (default `60`)

The idle-grace guard (#3) is the critical protection against false positives. Cases where a progress flag gets flipped prematurely while the reviewer is still actively working will keep the timeline hot, so the watcher defers until genuine quiescence. Once `/exit` is sent, Claude Code processes it as a clean session terminate, which fires the `SessionEnd` hook chain (`timeline-capture.sh` → `notify-on-completion.sh` → `notify.sh` → Slack).

**Tunables** (both optional, exported via the wrapper):

| Variable | Default | Effect |
|---|---|---|
| `WATCH_POLL_SECS` | `30` | Poll interval |
| `WATCH_IDLE_SECS` | `60` | Required timeline idleness before sending `/exit` |

The watcher logs each poll outcome to `.logs/<branch>.watch.log` for debugging. It exits automatically when the tmux session disappears, when it successfully sends `/exit`, or when the wrapper kills it on claude exit (via `trap EXIT` + explicit `kill`). Foreground mode (`--background` not set) does not launch the watcher — in foreground you're expected to type `/exit` manually when you see the orchestrator print its "terminating" farewell.

## Signal Protocol

All agent signals are defined in [`agents/signals.sh`](../agents/signals.sh) — the single source of truth. The orchestrator prompt is generated from `signals.sh` at launch, so the protocol can't drift.

**Programmer signals**: `BRIEF READY FOR REVIEW`, `PLAN READY FOR REVIEW`, `PHASE <X> READY FOR REVIEW`, `VERIFICATION COMPLETE`, `VERIFICATION FAILED`, `VERIFICATION SKIPPED — ALL MODELS TOO LARGE`, `READY FOR REVIEW`.

**Reviewer signals**: `BRIEF APPROVED` / `BRIEF CHANGES REQUESTED`, `PLAN APPROVED` / `PLAN CHANGES REQUESTED`, `APPROVED` / `CHANGES REQUESTED`.

The full state machine diagram and routing rules are in [`agents/signals.sh`](../agents/signals.sh) and get injected into the orchestrator prompt as the `{{PROTOCOL_BLOCK}}` placeholder during `launch.sh` rendering.

## Solo Mode (`--mode solo`)

```bash
./agents/launch-agent-pair.sh --mode solo --architecture CohereForCausalLM
```

For accounts without agent-teams access (Pro/Team tier). Launches two
independent Claude Code sessions in one tmux session — programmer (left
pane `:0.0`) and reviewer (right pane `:0.1`) — plus a
**coordinator daemon** ([`agents/solo-coordinator.sh`](../agents/solo-coordinator.sh)).

Coordination is file-based but never blocking: an agent `touch`es a signal
file in `.adapter-workspace/signals/` and ends its turn. The coordinator
consumes the signal (archived to `signals/.archive/<epoch>.<name>`) and
wakes the counterpart pane with an injected `[coordinator]` message.
Agents must never poll or wait — in-agent blocking calls die at the Bash
tool timeout, which is why the routing lives in a daemon.

File-signal names (distinct from the agent-teams message signals above):

| Signal | Emitted by | Routed to |
| ------ | ---------- | --------- |
| `brief-ready`, `plan-ready`, `phase-<X>-ready`, `verification-complete` | programmer | reviewer |
| `<checkpoint>-approved`, `<checkpoint>-changes-<N>`, `final-approved` | reviewer | programmer |
| `verification-failed` | programmer | reviewer (informational) |
| `verification-skipped`, `stuck` | programmer | user notification; sessions stay alive |
| `done` | programmer | completion: `/exit` to both panes after progress flags + timeline idle |

The coordinator also completes passively (flags set + idle timeline, no
`done` signal), subsuming the teams-mode completion watcher. Its log is
`.logs/<branch>.watch.log`; its PID file `.logs/<branch>.coordinator.pid`
(cleaned by `stop`). Agents can snapshot signal state non-blockingly with
[`agents/check-signals.sh`](../agents/check-signals.sh). Per-role models
come from the solo prompt files' frontmatter and are passed via `--model`.

Testing hooks: `solo-coordinator.sh --route <signal>` prints the routing
decision; `solo-coordinator.sh --once <worktree>` does a single
scan-archive-print pass with no tmux. Both are exercised by
`scripts/dry-run-test.sh`.
