# Hooks Reference

Critical rules are enforced at the framework level via Claude Code hooks, not by asking agents to remember them. `launch.sh` writes a `.claude/settings.json` in each worktree that registers absolute paths to the hook scripts under [`agents/hooks/`](../agents/hooks/). When Claude Code fires the relevant event, the hook reads the payload from stdin, decides whether to allow or block, and emits a JSON response on stdout.

Every hook includes a `RUNTIME CONTEXT:` header reminding the reader that the script *lives* under `devtools/adapter_builder/agents/hooks/` but *runs* inside the target TransformerLens worktree (so `pwd` is the worktree, not the hook's own directory).

## Hook summary

| Hook | Event | Matcher | Can block? |
|------|-------|---------|------------|
| [timeline-capture.sh](../agents/hooks/timeline-capture.sh) | `SessionStart`, `SessionEnd`, `SubagentStart`, `SubagentStop`, `PreToolUse`, `PostToolUse` | (all) | No — logging only |
| [guard-hooked-transformer.sh](../agents/hooks/guard-hooked-transformer.sh) | `PreToolUse` | `Edit\|Write\|MultiEdit\|NotebookEdit` | Yes |
| [guard-review-rounds.sh](../agents/hooks/guard-review-rounds.sh) | `PreToolUse` | `Edit\|Write\|MultiEdit\|NotebookEdit` | Yes |
| [guard-git.sh](../agents/hooks/guard-git.sh) | `PreToolUse` | `Bash` | Yes |
| [guard-verify-models.sh](../agents/hooks/guard-verify-models.sh) | `PreToolUse` | `Bash` | Yes |
| [gate-reviewer-writes-file.sh](../agents/hooks/gate-reviewer-writes-file.sh) | `SubagentStop` | (all) | Yes |
| [gate-lint-checks.sh](../agents/hooks/gate-lint-checks.sh) | `Stop` | (all) | Yes |
| [notify-on-completion.sh](../agents/hooks/notify-on-completion.sh) | `SessionEnd` | (all) | No — side effect only |

Multiple hooks may register on the same matcher; they run in order and any one block denies the tool call.

## timeline-capture.sh

Appends a structured JSON line to `.adapter-workspace/timeline.jsonl` on every tool call, subagent spawn, and session event. Captures timestamp, event name, agent id/type, tool name, and `tool_input` (including bash commands). The `logs` and `status` subcommands format this file via [`scripts/format-timeline.py`](../scripts/format-timeline.py).

Never blocks — always emits `{"continue": true}`. If `jq` or the timeline file isn't available, the append silently fails and the tool call still proceeds.

## guard-hooked-transformer.sh

Blocks edits to deprecated HookedTransformer files. Agents may read them for reference but must not modify them — their work must only touch the TransformerBridge system.

**Blocks writes** whose `tool_input.file_path` contains any of:

- `transformer_lens/HookedTransformer.py`
- `transformer_lens/loading_from_pretrained.py`
- `transformer_lens/components/`
- `transformer_lens/pretrained/weight_conversions/`

On block, emits `{"continue": false, "decision": "block", "reason": …}` with a message explaining the adapter should only touch `transformer_lens/model_bridge/` and the factory.

## guard-review-rounds.sh

Enforces the 3-round review-iteration limit from [`agents/orchestrator.md`](../agents/orchestrator.md) § Iteration Limits. Turns a prompt-level rule into a runtime stop so runaway loops can't happen.

**Blocks writes** to review files at round 4 or higher:

- `.adapter-workspace/reviews/brief-review-<N>.md` where N > 3
- `.adapter-workspace/reviews/plan-review-<N>.md` where N > 3
- `.adapter-workspace/reviews/phase-<letter>-review-<N>.md` where N > 3

On block, the error message tells the orchestrator to: (1) collect unresolved CRITICAL issues from previous reviews, (2) write `.adapter-workspace/stuck-report.md`, (3) fire the "stuck" notification via `scripts/notify.sh`, and (4) terminate the session.

Unrecognized files under `.adapter-workspace/reviews/` pass through — the reviewer owns the naming convention and unknown files are presumed unrelated.

**Tunable:** `MAX_REVIEW_ROUNDS` (default `3`). Exported from the session environment via the wrapper script.

## guard-git.sh

Blocks any Bash command that commits, pushes, or publishes changes. Agents must leave everything uncommitted for manual review.

**Blocks** commands matching (with a shell-separator lead anchor so chained invocations like `make test && git commit -m …` are caught):

- `git commit` (any flags — `-m`, `--amend`, `-a`)
- `git push` (any form)
- `gh pr create | merge | close | review | ready | edit`
- `gh release create | delete | edit`

**Passes through** read-only git operations (`status`, `diff`, `log`, `show`, `branch -l`, `worktree list`) and unrelated substrings like `gitignore` — the word-boundary regex requires `git` followed by whitespace followed by the forbidden subcommand, so `cat .gitignore` is not affected.

On block, the error message references `agents/programmer.md § What NOT to Do`.

## guard-verify-models.sh

Enforces the ≤7B verification rule from [`agents/programmer.md`](../agents/programmer.md) Step 3.1. Turns the "skip large models" guidance into a runtime stop so a misread prompt can't waste hours on a 70B model load.

**Intercepts** Bash commands invoking `verify_models` (matches both `verify_models` directly and `python -m transformer_lens.tools.model_registry.verify_models`).

**Always allowed:**

- `--dry-run` invocations — they don't load weights
- Bulk `--architectures` runs without a specific `--model` arg — `verify_models` has its own memory-based skip for bulk mode

**Blocks** when the command specifies `--model <id>` (or `--model=<id>`) and either:

1. The model is not registered in `transformer_lens/tools/model_registry/data/supported_models.json`. The error message tells the agent to run [`scan-hf-architecture.py`](../scripts/scan-hf-architecture.py) + [`port-arch-models.py`](../scripts/port-arch-models.py) first.
2. The registered `metadata.total_params` exceeds `MAX_VERIFY_PARAMS` (default 7.5B — a small buffer above "7B" models like Mistral-7B at 7.24B; blocks Llama-3-8B and anything larger).

**Tunables:** `MAX_VERIFY_PARAMS` (default `7500000000`). Exported from the session environment.

## gate-reviewer-writes-file.sh

Enforces the "every review must be persisted to a file" rule from [`agents/reviewer.md`](../agents/reviewer.md) § Feedback Format. Turns a prompt-level convention into a runtime stop so reviewers can't silently return verbal approvals while leaving `.adapter-workspace/reviews/` empty.

Fires on `SubagentStop`. Takes action only when the stopped subagent is `agent_type == "reviewer"`:

1. Reads the reviewer's matching `SubagentStart` timestamp from `.adapter-workspace/timeline.jsonl` (by `agent_id`).
2. Checks whether any file in `.adapter-workspace/reviews/` or the single `.adapter-workspace/completion-report.md` has an `mtime` at or after that `SubagentStart` timestamp.
3. **If yes** → allow. The reviewer wrote a file; normal processing continues.
4. **If no** → blocks with a `{continue: false, decision: "block", reason: …}` response. The reason string instructs the orchestrator to treat the reviewer's result as void, resume the reviewer by `agent_id` with a message prefixed `Resuming you —` insisting on a file, and re-run the Review File Integrity Check after the next `SubagentStop`.
5. In the drift case, also appends a `ReviewerFileDrift` event to `timeline.jsonl` for visibility in `status` / `logs` / post-hoc inspection.

Non-reviewer `SubagentStop` events pass through unchanged (the hook exits at the `agent_type` check).

Complemented by the orchestrator-level Review File Integrity Check section in the protocol block (see `agents/signals.sh generate_protocol_block`), which tells the orchestrator to verify file existence manually without waiting for the hook.

## gate-lint-checks.sh

Refuses to let the session end until `mypy` and `make check-format` both pass — but only *after* verification has passed, so it doesn't block legitimate mid-build stops.

Flow when the `Stop` event fires:

1. If `.adapter-progress.json` doesn't exist → allow stop (not an adapter build)
2. If `verification_passed != true` → allow stop (agent is stopping legitimately mid-build)
3. If `final_review_passed == true` → allow stop (already gated this session, don't re-run the checks)
4. Run `uv run mypy .` — if it fails, block with the last 30 lines of output and a reminder that `# type: ignore` is not an acceptable fix
5. Run `make check-format` — if it fails, block with the last 30 lines of output
6. Both pass → mark `final_review_passed: true` in the progress file and allow stop

The mark ensures subsequent `Stop` events on the same session skip straight to allow, so the expensive checks run exactly once per completed build.

## notify-on-completion.sh

Fires the Slack notification on `SessionEnd` when the progress file shows `verification_passed: true`. Uses a `.adapter-workspace/notification-sent` sentinel file to avoid double-sending on resumed sessions.

Never blocks — this is a side-effect hook. Finds `scripts/notify.sh` via `TL_ADAPTER_BUILDER_ROOT` (exported by the launch wrapper) or by searching upward from the worktree as a fallback.

## Hook settings registration

[`agents/launch.sh`](../agents/launch.sh) writes `.claude/settings.json` in each worktree with this structure:

```json
{
  "hooks": {
    "SessionStart":  [{ "hooks": [{ "command": "<timeline>" }] }],
    "SessionEnd":    [{ "hooks": [{ "command": "<timeline>" }, { "command": "<notify>" }] }],
    "SubagentStart": [{ "hooks": [{ "command": "<timeline>" }] }],
    "SubagentStop":  [{ "hooks": [{ "command": "<timeline>" }, { "command": "<gate-reviewer-writes-file>" }] }],
    "PreToolUse": [
      { "matcher": "Edit|Write|MultiEdit|NotebookEdit",
        "hooks": [{ "command": "<guard-ht>" }, { "command": "<guard-review-rounds>" }] },
      { "matcher": "Bash",
        "hooks": [{ "command": "<guard-git>" }, { "command": "<guard-verify-models>" }] },
      { "matcher": "",
        "hooks": [{ "command": "<timeline>" }] }
    ],
    "PostToolUse": [
      { "matcher": "",
        "hooks": [{ "command": "<timeline>" }] }
    ],
    "Stop": [{ "hooks": [{ "command": "<gate-lint>" }] }]
  }
}
```

Hook commands are stored as absolute paths, single-quoted to preserve the space in "TL Adapter Builder". Nothing is copied from this repo into the worktree except this `.claude/settings.json` and `.claude/agents/{programmer,reviewer}.md`.
