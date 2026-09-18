#!/usr/bin/env bash
# =============================================================================
# watch-completion.sh — Background watcher for --background tmux sessions
#
# In interactive Claude Code, an orchestrator has no tool to self-terminate.
# It can print "Session terminating — SessionEnd hook will fire the Slack
# notification" but that just ends its turn; the session then sits at the
# `❯` prompt waiting for user input forever. The SessionEnd hook never fires
# (because the session hasn't actually ended) and the Slack notification
# never gets sent.
#
# This watcher closes the gap. It's launched in the background by launch.sh's
# --background wrapper, runs for the lifetime of the session, and polls the
# worktree's .adapter-progress.json file. When:
#
#   1. verification_passed == true, AND
#   2. final_review_passed == true, AND
#   3. the timeline.jsonl has been idle for at least $WATCH_IDLE_SECS seconds
#      (default 60), confirming the orchestrator has finished printing its
#      farewell and isn't still spawning subagents,
#
# …the watcher sends `/exit` to the tmux session via `tmux send-keys`.
# Claude Code processes /exit as a clean session terminate, which fires the
# SessionEnd hook chain (timeline-capture → notify-on-completion → notify.sh
# → Slack), marks the session exit code 0, and lets the wrapper script
# proceed to its post-session tail.
#
# The idle-grace guard is the critical protection against false positives.
# xglm-style cases where the flags get flipped prematurely while the reviewer
# is still actively working will keep the timeline hot, so the watcher waits
# for genuine quiescence before acting.
#
# Usage (invoked by launch.sh's background wrapper):
#   watch-completion.sh <worktree_dir> <tmux_session> [architecture]
#
# Environment:
#   WATCH_POLL_SECS    — poll interval (default 30)
#   WATCH_IDLE_SECS    — required timeline idle before /exit (default 60)
# =============================================================================

set -euo pipefail

WORKTREE_DIR="${1:?Usage: watch-completion.sh <worktree_dir> <tmux_session> [architecture]}"
TMUX_SESSION="${2:?Usage: watch-completion.sh <worktree_dir> <tmux_session> [architecture]}"
ARCHITECTURE="${3:-unknown}"

POLL_INTERVAL="${WATCH_POLL_SECS:-30}"
IDLE_GRACE="${WATCH_IDLE_SECS:-60}"

# All TL Adapter Builder tmux sessions live on a dedicated socket so they
# don't share a server with any tmux session the user has running.
TMUX_SOCKET="tl-adapter"
TM() { tmux -L "$TMUX_SOCKET" "$@"; }

PROGRESS="$WORKTREE_DIR/.adapter-progress.json"
TIMELINE="$WORKTREE_DIR/.adapter-workspace/timeline.jsonl"
TAG="[watch-completion:${ARCHITECTURE}]"

log() {
  echo "$TAG $(date -u +%H:%M:%S) $*"
}

log "Started. Poll interval ${POLL_INTERVAL}s, idle grace ${IDLE_GRACE}s. Watching $TMUX_SESSION."

# The watcher exits cleanly under any of three conditions:
#   - it successfully sends /exit (normal completion path)
#   - the tmux session disappears (claude was killed externally, or we already sent /exit)
#   - its parent wrapper signals it via SIGTERM (cleanup trap)
trap 'log "Received termination signal — watcher exiting."; exit 0' TERM INT

while true; do
  sleep "$POLL_INTERVAL"

  # If tmux is gone or the session vanished, we're done.
  if ! command -v tmux &>/dev/null || ! TM has-session -t "$TMUX_SESSION" 2>/dev/null; then
    log "tmux session '$TMUX_SESSION' is gone — watcher exiting."
    exit 0
  fi

  [[ -f "$PROGRESS" ]] || continue

  # Are both completion flags set?
  flags_ok=$(python3 -c "
import json
try:
    p = json.load(open('$PROGRESS'))
except Exception:
    print('no')
    raise SystemExit(0)
verif = p.get('verification_passed') is True
review = p.get('final_review_passed') is True
print('yes' if verif and review else 'no')
" 2>/dev/null || echo "no")

  if [[ "$flags_ok" != "yes" ]]; then
    continue
  fi

  # Flags set. Has the timeline been quiet long enough to assume the
  # orchestrator is done printing its farewell and not mid-subagent?
  #
  # IMPORTANT: if the timeline file doesn't exist at all, that means hooks
  # haven't fired yet (session just started, or workspace wasn't set up).
  # Default to NOT idle — we must never send /exit before work begins.
  if [[ ! -f "$TIMELINE" ]]; then
    log "Completion flags set but timeline file missing — deferring (session may not have started)."
    continue
  fi

  idle_ok=$(python3 -c "
import os, time
try:
    age = time.time() - os.path.getmtime('$TIMELINE')
    print('yes' if age >= $IDLE_GRACE else 'no')
except Exception:
    print('no')
" 2>/dev/null || echo "no")

  if [[ "$idle_ok" != "yes" ]]; then
    log "Completion flags set but timeline still active — deferring /exit."
    continue
  fi

  # Double-check the mtime again after a short sleep to close the race
  # window where an agent could write to the timeline between our check
  # and the /exit send.
  sleep 2
  idle_recheck=$(python3 -c "
import os, time
try:
    age = time.time() - os.path.getmtime('$TIMELINE')
    print('yes' if age >= $IDLE_GRACE else 'no')
except Exception:
    print('no')
" 2>/dev/null || echo "no")

  if [[ "$idle_recheck" != "yes" ]]; then
    log "Timeline became active during recheck — deferring /exit."
    continue
  fi

  log "Adapter complete + timeline idle ≥${IDLE_GRACE}s (confirmed twice). Sending /exit to $TMUX_SESSION."
  TM send-keys -t "$TMUX_SESSION" "/exit" Enter

  # Give Claude Code a moment to process the /exit command and run hooks.
  sleep 10

  # If the session is still alive, try one more /exit (in case the first
  # one landed in a paste buffer or Claude was mid-tool-call).
  if TM has-session -t "$TMUX_SESSION" 2>/dev/null; then
    log "Session still alive 10s after /exit — retrying once."
    TM send-keys -t "$TMUX_SESSION" Enter
    sleep 1
    TM send-keys -t "$TMUX_SESSION" "/exit" Enter
    sleep 10
  fi

  log "Watcher done. tmux session alive? $(TM has-session -t "$TMUX_SESSION" 2>/dev/null && echo yes || echo no)"
  exit 0
done
