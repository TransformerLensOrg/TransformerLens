#!/usr/bin/env bash
# =============================================================================
# timeline-capture.sh — Generic hook for capturing events to the timeline
#
# RUNTIME CONTEXT:
#   This script lives under devtools/adapter_builder/agents/hooks/ but is executed
#   by Claude Code *inside the target TransformerLens worktree* via the
#   .claude/settings.json written by launch.sh. When this script runs, `pwd`
#   is the worktree, NOT the directory containing this file. All paths below
#   are resolved relative to the worktree intentionally.
#
# Reads the hook payload from stdin and appends a JSON line to the worktree's
# .adapter-workspace/timeline.jsonl file. Captures timestamp, event name,
# agent info, tool name, and tool input (including bash commands).
#
# Registered by launch.sh on: SessionStart, SessionEnd, SubagentStart,
# SubagentStop, PreToolUse, PostToolUse.
#
# IMPORTANT: downstream hooks (gate-reviewer-writes-file.sh) depend on the
# timeline existing and being current. If this hook fails silently, those
# guards lose their data source and drift detection stops working. We
# therefore validate prerequisites and emit a warning to stderr (visible in
# the debug log) on failure rather than swallowing errors.
#
# Claude Code expects hook output on stdout to be valid JSON (typically
# {"continue": true}). We always emit that — a broken timeline must not
# block the agent's work, but we do log the failure.
# =============================================================================

set -euo pipefail

WORKSPACE="$(pwd)/.adapter-workspace"
TIMELINE_FILE="$WORKSPACE/timeline.jsonl"

ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Read stdin once (hooks can only read payload once)
payload=$(cat)

# --- Validate prerequisites ------------------------------------------------

# The workspace directory must exist (launch.sh creates it at boot).
# If it's missing, something went wrong during setup.
if [[ ! -d "$WORKSPACE" ]]; then
  echo "[timeline-capture] ERROR: $WORKSPACE does not exist — timeline cannot be written. Downstream guards (gate-reviewer-writes-file.sh) will not function." >&2
  echo '{"continue": true}'
  exit 0
fi

# jq must be available for structured JSON extraction.
if ! command -v jq &>/dev/null; then
  # Fall back to appending the raw payload with a timestamp prefix so we
  # don't lose the event entirely. Downstream hooks that parse timeline
  # fields may get malformed lines, but at least the file exists and has
  # entries (which prevents false "empty timeline" conditions).
  echo "[timeline-capture] WARNING: jq is not installed — appending raw payload to timeline. Install jq for proper structured capture." >&2
  echo "{\"ts\":\"$ts\",\"raw\":$(echo "$payload" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null || echo '""')}" >> "$TIMELINE_FILE" 2>/dev/null || true
  echo '{"continue": true}'
  exit 0
fi

# --- Append the structured timeline entry ----------------------------------

if ! echo "$payload" | jq -c --arg ts "$ts" '{
  ts: $ts,
  event: .hook_event_name,
  agent_id: .agent_id,
  agent_type: .agent_type,
  tool: .tool_name,
  tool_input: .tool_input
}' >> "$TIMELINE_FILE" 2>/tmp/tl-timeline-err.$$; then
  echo "[timeline-capture] ERROR: jq failed to process payload — $(cat /tmp/tl-timeline-err.$$ 2>/dev/null)" >&2
  rm -f /tmp/tl-timeline-err.$$
fi
rm -f /tmp/tl-timeline-err.$$ 2>/dev/null

# Always allow the tool call to proceed regardless of timeline errors.
echo '{"continue": true}'
exit 0
