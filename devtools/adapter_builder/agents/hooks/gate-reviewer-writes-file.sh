#!/usr/bin/env bash
# =============================================================================
# gate-reviewer-writes-file.sh — SubagentStop hook for reviewer drift detection
#
# RUNTIME CONTEXT:
#   This script lives under devtools/adapter_builder/agents/hooks/ but is executed
#   by Claude Code *inside the target TransformerLens worktree* via the
#   .claude/settings.json written by launch.sh. The hook payload is read
#   from stdin and `pwd` at runtime is the worktree.
#
# reviewer.md § Feedback Format mandates that EVERY review must be written
# to a file in .adapter-workspace/reviews/ (or .adapter-workspace/completion-report.md
# for the final review) before the reviewer subagent returns. Empirically,
# reviewers have been returning review text in their subagent result WITHOUT
# persisting to disk, which leaves reviews/ empty, defeats the audit trail,
# and bypasses guard-review-rounds.sh (which counts review files).
#
# This hook fires on SubagentStop. When the stopped subagent is a reviewer
# AND no review file has been written in .adapter-workspace/reviews/ OR
# .adapter-workspace/completion-report.md since the matching SubagentStart
# timestamp, it:
#
#   1. Appends a `ReviewerFileDrift` event to timeline.jsonl so the drift
#      is visible in subsequent `status`, `logs`, and post-hoc inspection.
#   2. Emits a `{continue: false, decision: "block", reason: …}` response
#      pointing the orchestrator at the exact remediation (resume the
#      reviewer with a `Resuming you — ` message telling it to write the
#      review file before returning).
#
# The `continue: false` signal on SubagentStop may or may not be strictly
# blocking in Claude Code's agent-teams model, but the appended timeline
# event is durable and the `reason` field surfaces to the orchestrator via
# the hook output — either way the drift becomes visible and actionable.
# =============================================================================

set -euo pipefail

payload=$(cat)

event=$(echo "$payload" | jq -r '.hook_event_name // empty')
agent_type=$(echo "$payload" | jq -r '.agent_type // empty')
agent_id=$(echo "$payload" | jq -r '.agent_id // empty')

allow() {
  echo '{"continue": true}'
  exit 0
}

# Only fire for reviewer SubagentStop events.
[[ "$event" != "SubagentStop" ]] && allow
[[ "$agent_type" != "reviewer" ]] && allow
[[ -z "$agent_id" ]] && allow

WORKTREE_DIR="$(pwd)"
TIMELINE="$WORKTREE_DIR/.adapter-workspace/timeline.jsonl"
REVIEWS_DIR="$WORKTREE_DIR/.adapter-workspace/reviews"
COMPLETION_REPORT="$WORKTREE_DIR/.adapter-workspace/completion-report.md"

# If there's no timeline we can't correlate — let it through.
[[ -f "$TIMELINE" ]] || allow

# Find the matching SubagentStart timestamp for this agent_id. This is the
# earliest moment during which the reviewer could have written its file.
start_ts=$(python3 -c "
import json
last = None
for line in open('$TIMELINE'):
    try:
        e = json.loads(line)
    except Exception:
        continue
    if e.get('event') == 'SubagentStart' and e.get('agent_id') == '$agent_id':
        last = e.get('ts')  # keep the latest matching SubagentStart
print(last or '')
" 2>/dev/null || echo "")

# No matching start — can't tell if drift happened. Let it through.
[[ -z "$start_ts" ]] && allow

# Check whether any review file or completion-report.md has been written
# or modified since start_ts. Uses file mtime compared against the parsed
# ISO 8601 timestamp.
new_files_written=$(python3 -c "
import pathlib
from datetime import datetime, timezone
try:
    start = datetime.fromisoformat('$start_ts'.replace('Z', '+00:00')).timestamp()
except Exception:
    print('unknown')
    raise SystemExit(0)

def any_modified_since(p, since):
    if not p.exists():
        return False
    if p.is_file():
        return p.stat().st_mtime >= since
    if p.is_dir():
        for child in p.iterdir():
            if child.is_file() and child.stat().st_mtime >= since:
                return True
    return False

reviews = pathlib.Path('$REVIEWS_DIR')
report = pathlib.Path('$COMPLETION_REPORT')
print('yes' if any_modified_since(reviews, start) or any_modified_since(report, start) else 'no')
" 2>/dev/null || echo "unknown")

# If we found a new file, or can't determine, let it through.
[[ "$new_files_written" == "yes" ]] && allow
[[ "$new_files_written" == "unknown" ]] && allow

# --- Drift detected -------------------------------------------------------

# Append a visible drift marker to the timeline so `status` and `logs` see it.
now_ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
python3 -c "
import json
with open('$TIMELINE', 'a') as f:
    f.write(json.dumps({
        'ts': '$now_ts',
        'event': 'ReviewerFileDrift',
        'agent_id': '$agent_id',
        'agent_type': 'reviewer',
        'tool': None,
        'tool_input': {
            'description': 'Reviewer subagent stopped without writing any review file since its SubagentStart at $start_ts',
        },
    }) + '\n')
" 2>/dev/null || true

reason="REVIEWER DRIFT: reviewer subagent $agent_id stopped without writing any file to .adapter-workspace/reviews/ or .adapter-workspace/completion-report.md between its SubagentStart at $start_ts and this SubagentStop. Per reviewer.md § Feedback Format, every review must be persisted to a file before returning — reviews that only exist in the subagent's result text are not valid and must not be accepted as approvals. REMEDIATION: (1) do NOT treat this reviewer's returned text as a decision; (2) re-engage the reviewer via SendMessage to=$agent_id with the body prefixed 'Resuming you — You did not write your review to a file. Write it to .adapter-workspace/reviews/<appropriate-name>.md (or completion-report.md for final reviews) BEFORE returning a decision. Your previous result is void.'; (3) then route the file-backed review through the normal approval flow."

jq -n --arg reason "$reason" '{continue: false, decision: "block", reason: $reason}'
exit 2
