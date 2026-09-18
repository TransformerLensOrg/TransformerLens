#!/usr/bin/env bash
# =============================================================================
# notify-on-completion.sh — SessionEnd hook
#
# RUNTIME CONTEXT:
#   This script lives under devtools/adapter_builder/agents/hooks/ but is executed
#   by Claude Code *inside the target TransformerLens worktree* — the
#   .claude/settings.json written by launch.sh references this file via an
#   absolute path. `pwd` at runtime is the worktree, which is why
#   WORKTREE_DIR="$(pwd)" points at .adapter-progress.json inside it. We use
#   the TL_ADAPTER_BUILDER_ROOT env var (exported by launch.sh) to find the
#   notify.sh helper back in the control-plane repo.
#
# Fires a Slack notification when the session ends IF the progress file
# shows verification has passed. Uses a sentinel file to avoid double-sending
# notifications on resumed sessions.
# =============================================================================

set -euo pipefail

WORKTREE_DIR="$(pwd)"
PROGRESS_FILE="$WORKTREE_DIR/.adapter-progress.json"
SENT_MARKER="$WORKTREE_DIR/.adapter-workspace/notification-sent"

# No progress file = not an adapter build
[[ -f "$PROGRESS_FILE" ]] || exit 0

# Already notified = skip
[[ -f "$SENT_MARKER" ]] && exit 0

verification_passed=$(jq -r '.verification_passed // false' "$PROGRESS_FILE" 2>/dev/null || echo "false")
architecture=$(jq -r '.architecture // "unknown"' "$PROGRESS_FILE" 2>/dev/null || echo "unknown")

if [[ "$verification_passed" == "true" ]]; then
  # Find the notify script via TL_ADAPTER_BUILDER_ROOT or relative
  if [[ -n "${TL_ADAPTER_BUILDER_ROOT:-}" ]]; then
    notify_script="$TL_ADAPTER_BUILDER_ROOT/scripts/notify.sh"
  else
    # Fall back: search upward for the scripts directory
    notify_script=""
    candidate="$WORKTREE_DIR"
    for _ in 1 2 3 4; do
      candidate=$(dirname "$candidate")
      if [[ -f "$candidate/scripts/notify.sh" ]]; then
        notify_script="$candidate/scripts/notify.sh"
        break
      fi
    done
  fi

  if [[ -x "$notify_script" ]]; then
    "$notify_script" "Adapter completed for $architecture" >/dev/null 2>&1 || true
    touch "$SENT_MARKER"
  fi
fi

# Emit a valid JSON response for Claude Code's hook protocol
echo '{"continue": true}'
exit 0
