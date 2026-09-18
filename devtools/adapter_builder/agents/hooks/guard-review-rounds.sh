#!/usr/bin/env bash
# =============================================================================
# guard-review-rounds.sh — PreToolUse hook for Edit/Write/MultiEdit/NotebookEdit
#
# RUNTIME CONTEXT:
#   This script lives under devtools/adapter_builder/agents/hooks/ but is executed
#   by Claude Code *inside the target TransformerLens worktree* via the
#   .claude/settings.json written by launch.sh. The hook payload is read
#   from stdin.
#
# Enforces the 3-round review iteration limit from agents/orchestrator.md.
# Blocks any Write/Edit targeting a review file at round 4 or higher:
#   .adapter-workspace/reviews/brief-review-4.md
#   .adapter-workspace/reviews/plan-review-4.md
#   .adapter-workspace/reviews/phase-A-review-4.md
#   ...
#
# When triggered, the orchestrator must escalate unresolved issues to the
# user and terminate the session instead of looping further. This turns a
# prompt-level rule ("do not exceed 3 rounds") into a runtime stop.
# =============================================================================

set -euo pipefail

# Hard ceiling of 5 prevents env override from disabling the guard entirely.
_env_rounds="${MAX_REVIEW_ROUNDS:-3}"
MAX_ROUNDS=$(( _env_rounds > 5 ? 5 : _env_rounds ))

payload=$(cat)

tool_name=$(echo "$payload" | jq -r '.tool_name // empty')
file_path=$(echo "$payload" | jq -r '.tool_input.file_path // .tool_input.path // empty')

allow() {
  echo '{"continue": true}'
  exit 0
}

block() {
  local tl="$(pwd)/.adapter-workspace/timeline.jsonl"
  if [[ -f "$tl" ]]; then
    jq -n --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg hook "guard-review-rounds" --arg reason "$1" \
      '{ts:$ts, event:"HookBlocked", tool:"Write", hook:$hook, reason:$reason}' >> "$tl" 2>/dev/null || true
  fi
  jq -n --arg reason "$1" '{continue: false, decision: "block", reason: $reason}'
  exit 2
}

case "$tool_name" in
  Write|Edit|MultiEdit|NotebookEdit) ;;
  *) allow ;;
esac

[[ -z "$file_path" ]] && allow

# Only care about review files under .adapter-workspace/reviews/.
[[ "$file_path" != *".adapter-workspace/reviews/"* ]] && allow

# Expected filename patterns (set by reviewer.md § Feedback Format):
#   brief-review-<N>.md
#   plan-review-<N>.md
#   phase-<letter>-review-<N>.md
fname=$(basename "$file_path")
round=$(echo "$fname" | sed -nE 's/.*-review-([0-9]+)\.md$/\1/p')

# Unrecognized filename → let it through (probably an unrelated write into
# the reviews/ directory; the reviewer owns the naming convention).
[[ -z "$round" ]] && allow

if (( round > MAX_ROUNDS )); then
  checkpoint=$(echo "$fname" | sed -nE 's/^(.*)-review-[0-9]+\.md$/\1/p')
  reason=$(printf '%s' "BLOCKED: Review round ${round} exceeds the ${MAX_ROUNDS}-round iteration limit for '${checkpoint:-this checkpoint}'. Per agents/orchestrator.md § Iteration Limits, the orchestrator must now: (1) collect unresolved CRITICAL issues from the previous reviews, (2) write a summary to .adapter-workspace/stuck-report.md, (3) run the 'stuck' notification, and (4) terminate the session. Do NOT continue the review loop.")
  block "$reason"
fi

allow
