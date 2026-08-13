#!/usr/bin/env bash
# =============================================================================
# signals.sh — Single source of truth for the agent team protocol
#
# Defines all signals, their meanings, and the state machine transitions.
# Sourced by launch-agent-pair.sh and injected into agent prompts.
# =============================================================================

# --------------------------------------------------------------------------- #
# Programmer signals
# --------------------------------------------------------------------------- #
SIG_BRIEF_READY="PROGRAMMER: BRIEF READY FOR REVIEW"
SIG_PLAN_READY="PROGRAMMER: PLAN READY FOR REVIEW"
SIG_PHASE_READY="PROGRAMMER: PHASE <X> READY FOR REVIEW"  # <X> replaced at runtime
SIG_VERIFICATION_COMPLETE="PROGRAMMER: VERIFICATION COMPLETE"
SIG_VERIFICATION_FAILED="PROGRAMMER: VERIFICATION FAILED"
SIG_VERIFICATION_SKIPPED="PROGRAMMER: VERIFICATION SKIPPED — ALL MODELS TOO LARGE"
SIG_FINAL_READY="PROGRAMMER: READY FOR REVIEW"

# --------------------------------------------------------------------------- #
# Reviewer signals
# --------------------------------------------------------------------------- #
SIG_BRIEF_APPROVED="REVIEWER: BRIEF APPROVED"
SIG_BRIEF_CHANGES="REVIEWER: BRIEF CHANGES REQUESTED"
SIG_PLAN_APPROVED="REVIEWER: PLAN APPROVED"
SIG_PLAN_CHANGES="REVIEWER: PLAN CHANGES REQUESTED"
SIG_CODE_APPROVED="REVIEWER: APPROVED"
SIG_CODE_CHANGES="REVIEWER: CHANGES REQUESTED"

# --------------------------------------------------------------------------- #
# State machine (for injection into orchestrator prompt)
# --------------------------------------------------------------------------- #
generate_protocol_block() {
  cat <<'PROTOCOL'
## Protocol

### Signals
Programmer: `BRIEF READY FOR REVIEW`, `PLAN READY FOR REVIEW`, `PHASE <X> READY FOR REVIEW`, `VERIFICATION COMPLETE`, `VERIFICATION FAILED`, `VERIFICATION SKIPPED — ALL MODELS TOO LARGE`, `READY FOR REVIEW`
Reviewer: `BRIEF APPROVED`/`CHANGES REQUESTED`, `PLAN APPROVED`/`CHANGES REQUESTED`, `APPROVED`/`CHANGES REQUESTED`

### Subagent resume (CRITICAL)
`SendMessage to=<type_name>` is a **silent no-op** against a stopped subagent. Always:
1. Record each subagent's `agent_id` from its `SubagentStart` event
2. Before `SendMessage`, check if target is running (last event = `SubagentStart`, not `SubagentStop`)
3. If stopped → `SendMessage to=<agent_id>` with body prefixed `Resuming you — `
4. If running → `SendMessage to=<type_name>` is fine
5. To find an ID: `tail -r .adapter-workspace/timeline.jsonl | jq -r 'select(.event=="SubagentStart" and .agent_type=="programmer") | .agent_id' | head -1`
Never send pings. Never retry by type name against a stopped subagent. Fresh spawn via `Task` is also valid but loses prior context.

### Review file check
After every reviewer turn, verify a new file exists in `.adapter-workspace/reviews/` (or `completion-report.md`). A hook also enforces this. If no file → review is VOID. Resume reviewer by ID: `Resuming you — write your review to a file before returning.`

### Hook block monitoring
All blocking hooks append a `HookBlocked` event to `.adapter-workspace/timeline.jsonl` with `hook`, `reason`, and `tool` fields. If a subagent seems stuck or repeating the same action, check for recent `HookBlocked` events:
```bash
grep HookBlocked .adapter-workspace/timeline.jsonl | tail -3
```
If you see repeated blocks, the subagent is retrying a denied action. Resume it with explicit guidance on what to do instead (the `reason` field explains the constraint).

### Subagent activity monitoring
If a subagent has been running for >15 minutes with no new timeline events, it may be hung. Check:
```bash
tail -1 .adapter-workspace/timeline.jsonl | jq -r .ts
```
If the last event is old AND the subagent is still `SubagentStart`'d (no `SubagentStop`), consider sending a status ping via `SendMessage` to the agent_id. If doing legitimate long work (verify_models downloading a model), the timeline will show a long-running Bash PreToolUse without a PostToolUse — that's normal. True hangs show no events at all.

### Routing rules
All routing applies the resume protocol (use agent_id for stopped subagents) and the file check (verify file exists before accepting reviewer decisions).

1. `BRIEF READY FOR REVIEW` → spawn Reviewer
2. `BRIEF CHANGES REQUESTED` → route to Programmer
3. `BRIEF APPROVED` → tell Programmer: start planning (Step 1b)
4. `PLAN READY FOR REVIEW` → spawn Reviewer
5. `PLAN CHANGES REQUESTED` → route to Programmer
6. `PLAN APPROVED` → tell Programmer: begin implementation (Step 2)
7. `PHASE <X> READY FOR REVIEW` → spawn Reviewer
8. `CHANGES REQUESTED` → route to Programmer
9. `APPROVED` → tell Programmer: next phase (or verification if all done)
10. `VERIFICATION FAILED` → Programmer writes a **new** fix plan (`.adapter-workspace/adapter-plan.md`), signals `PLAN READY FOR REVIEW`. Reviewer reviews the fix plan (rule 4-6). Then re-implement + re-verify. Do NOT skip the plan review — the failure may reveal architectural issues that need reviewer input.
11. `VERIFICATION COMPLETE` + `READY FOR REVIEW` → spawn Reviewer (final)
12. `VERIFICATION SKIPPED — ALL MODELS TOO LARGE` → notify, write stuck-report, terminate
13. Final `APPROVED` → notify, summarize, exit
PROTOCOL
}
