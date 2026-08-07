#!/usr/bin/env bash
# =============================================================================
# solo-coordinator.sh — Signal router for solo-mode agent pairs
#
# Solo mode runs two independent Claude Code sessions (programmer pane :0.0,
# reviewer pane :0.1) that cannot block on each other: long waits inside an
# agent's Bash tool die at the tool timeout, and polling burns turns. This
# daemon is the solo analogue of the agent-teams orchestrator — agents only
# ever `touch` a signal file and end their turn; the coordinator consumes
# each signal and wakes the counterpart with an injected tmux message.
#
# Lifecycle per signal:
#   1. Signal file appears in .adapter-workspace/signals/
#   2. Coordinator archives it to signals/.archive/<epoch>.<name> (consume —
#      this is what makes multi-round loops re-runnable; stale signals never
#      linger to satisfy a later wait)
#   3. Routes it: a message is pasted into the target pane as if typed
#
# Routing table (see route_signal):
#   *-ready, verification-complete        → reviewer   (do the matching review)
#   *-approved, *-changes-N               → programmer (proceed / address review)
#   verification-failed                   → reviewer   (informational standby)
#   verification-skipped, stuck           → notify-human (Slack/iMessage, panes
#                                           informed, sessions left alive)
#   done                                  → complete   (flags + idle check,
#                                           then /exit to BOTH panes)
#
# Completion also has a passive fallback: if verification_passed and
# final_review_passed are both true and the timeline has been idle for
# $WATCH_IDLE_SECS, the coordinator completes even without a `done` signal
# (same safeguard logic as watch-completion.sh, which this replaces in solo
# mode).
#
# Usage:
#   solo-coordinator.sh <worktree_dir> <tmux_session> [architecture]  # daemon
#   solo-coordinator.sh --route <signal-name>   # print "<target>\t<message>"
#   solo-coordinator.sh --once <worktree_dir>   # one scan: archive + print
#                                               # actions, no tmux (for tests)
#
# Environment:
#   COORD_POLL_SECS  — poll interval (default 10)
#   WATCH_IDLE_SECS  — required timeline idle before /exit (default 60)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

POLL="${COORD_POLL_SECS:-10}"
IDLE_GRACE="${WATCH_IDLE_SECS:-60}"

TMUX_SOCKET="tl-adapter"
TM() { tmux -L "$TMUX_SOCKET" "$@"; }

# --------------------------------------------------------------------------- #
# route_signal <name> — pure routing decision, no side effects.
# Prints "<target>\t<message>" where target is one of:
#   reviewer | programmer | notify-human | complete | unknown
# Kept side-effect-free so tests can assert the table via --route.
# --------------------------------------------------------------------------- #
route_signal() {
  local sig="$1"
  local target="" msg=""

  case "$sig" in
    brief-ready)
      target="reviewer"
      msg="Signal 'brief-ready' received. Review .adapter-workspace/adapter-brief.md per your 'Brief review' step: verify against HF source, write .adapter-workspace/reviews/brief-review-<N>.md, then touch signals/brief-approved or signals/brief-changes-<N> and end your turn." ;;
    plan-ready)
      target="reviewer"
      msg="Signal 'plan-ready' received. Review .adapter-workspace/adapter-plan.md per your 'Plan review' step, write .adapter-workspace/reviews/plan-review-<N>.md, then touch signals/plan-approved or signals/plan-changes-<N> and end your turn." ;;
    phase-*-ready)
      local phase="${sig#phase-}"; phase="${phase%-ready}"
      target="reviewer"
      msg="Signal 'phase-${phase}-ready' received. Review phase ${phase} per your 'Phase code review' step (scope: this phase's changes only; report at .adapter-workspace/phase-reports/phase-${phase}-report.md), write .adapter-workspace/reviews/phase-${phase}-review-<N>.md, then touch signals/phase-${phase}-approved or signals/phase-${phase}-changes-<N> and end your turn." ;;
    verification-complete)
      target="reviewer"
      msg="Signal 'verification-complete' received. Run your 'Final review' step (holistic, all phases; results at .adapter-workspace/verification-results.md), write .adapter-workspace/completion-report.md, then touch signals/final-approved and end your turn." ;;
    verification-failed)
      target="reviewer"
      msg="Signal 'verification-failed' received. The Programmer is analyzing the failure and re-planning — no action needed from you now. You will be messaged again at the next checkpoint (plan-ready)." ;;
    brief-approved)
      target="programmer"
      msg="Signal 'brief-approved' received. The brief is approved — proceed to Step 1b (plan), then touch signals/plan-ready and end your turn." ;;
    plan-approved)
      target="programmer"
      msg="Signal 'plan-approved' received. The plan is approved — proceed to Step 2 (implement the first phase), then touch signals/phase-<X>-ready and end your turn." ;;
    phase-*-approved)
      local phase="${sig#phase-}"; phase="${phase%-approved}"
      target="programmer"
      msg="Signal 'phase-${phase}-approved' received. Append '${phase}' to completed_phases in .adapter-progress.json, then implement the next phase (touch signals/phase-<X>-ready) or, if all phases are done, run Step 3 (verification) and touch signals/verification-complete. End your turn after signaling." ;;
    final-approved)
      target="programmer"
      msg="Signal 'final-approved' received. The final review passed. Set verification_passed to true in .adapter-progress.json, touch signals/done, and end your turn — the coordinator will close both sessions." ;;
    brief-changes-*|plan-changes-*|phase-*-changes-*)
      local checkpoint="${sig%-changes-*}"
      local round="${sig##*-}"
      target="programmer"
      msg="Signal '${sig}' received. Read .adapter-workspace/reviews/${checkpoint}-review-${round}.md, address every finding, then re-touch signals/${checkpoint}-ready and end your turn. If this was round 3, write .adapter-workspace/stuck-report.md and touch signals/stuck instead." ;;
    verification-skipped)
      target="notify-human"
      msg="Adapter built but verification skipped — all models exceed the memory limit. Human review needed: inspect the worktree, verify on bigger hardware or approve manually." ;;
    stuck)
      target="notify-human"
      msg="Review loop hit the 3-round limit. See .adapter-workspace/stuck-report.md. Human intervention needed — attach to the tmux session to unblock." ;;
    done)
      target="complete"
      msg="Programmer signaled done — verifying completion flags, then closing both sessions." ;;
    *)
      target="unknown"
      msg="Unrecognized signal '${sig}' — archived without routing." ;;
  esac

  printf '%s\t%s\n' "$target" "$msg"
}

# --------------------------------------------------------------------------- #
# --route mode: print the routing decision and exit (test hook)
# --------------------------------------------------------------------------- #
if [[ "${1:-}" == "--route" ]]; then
  route_signal "${2:?--route requires a signal name}"
  exit 0
fi

# --------------------------------------------------------------------------- #
# Shared scan: archive each pending signal, emit "<name>\t<target>\t<msg>"
# --------------------------------------------------------------------------- #
scan_signals() {
  local signals_dir="$1"
  local archive_dir="$signals_dir/.archive"
  mkdir -p "$archive_dir"

  # Oldest first (ls -tr) so multi-signal bursts route in causal order.
  # Signal names are coordinator-controlled (no spaces); .archive is hidden
  # so plain ls skips it.
  local name
  while IFS= read -r name; do
    [[ -n "$name" && -f "$signals_dir/$name" ]] || continue
    mv "$signals_dir/$name" "$archive_dir/$(date +%s).$name"
    printf '%s\t' "$name"
    route_signal "$name"
  done < <(ls -tr "$signals_dir" 2>/dev/null)
}

# --------------------------------------------------------------------------- #
# --once mode: single scan against a worktree, print actions, no tmux (tests)
# --------------------------------------------------------------------------- #
if [[ "${1:-}" == "--once" ]]; then
  WORKTREE_DIR="${2:?--once requires a worktree dir}"
  scan_signals "$WORKTREE_DIR/.adapter-workspace/signals"
  exit 0
fi

# --------------------------------------------------------------------------- #
# Daemon mode
# --------------------------------------------------------------------------- #
WORKTREE_DIR="${1:?Usage: solo-coordinator.sh <worktree_dir> <tmux_session> [architecture]}"
TMUX_SESSION="${2:?Usage: solo-coordinator.sh <worktree_dir> <tmux_session> [architecture]}"
ARCHITECTURE="${3:-unknown}"

SIGNALS_DIR="$WORKTREE_DIR/.adapter-workspace/signals"
PROGRESS="$WORKTREE_DIR/.adapter-progress.json"
TIMELINE="$WORKTREE_DIR/.adapter-workspace/timeline.jsonl"
TAG="[solo-coordinator:${ARCHITECTURE}]"

log() { echo "$TAG $(date -u +%H:%M:%S) $*"; }

# Panes as created by launch-solo-pair.sh: left=programmer, right=reviewer.
PANE_PROGRAMMER="$TMUX_SESSION:0.0"
PANE_REVIEWER="$TMUX_SESSION:0.1"

# Paste-buffer injection (same two-step idiom as ops.sh send: long payloads
# staged via send-keys get stuck as an unsubmitted bracketed paste, so load
# a buffer, paste it, then send Enter as a discrete key press).
inject() {
  local pane="$1" message="$2"
  local buf="tl-coord-$$-$(date +%s)"
  printf '%s' "[coordinator] $message" | TM load-buffer -b "$buf" -
  TM paste-buffer -d -b "$buf" -t "$pane"
  sleep 0.3
  TM send-keys -t "$pane" Enter
}

notify_human() {
  local message="$1"
  "$PROJECT_ROOT/scripts/notify.sh" "[$ARCHITECTURE] $message" >/dev/null 2>&1 || true
  # Inform both panes; sessions stay alive so the user can attach and act.
  inject "$PANE_PROGRAMMER" "$message The coordinator has notified the user; hold for instructions." || true
  inject "$PANE_REVIEWER" "$message The coordinator has notified the user; hold for instructions." || true
}

flags_complete() {
  python3 -c "
import json
try:
    p = json.load(open('$PROGRESS'))
    print('yes' if p.get('verification_passed') is True and p.get('final_review_passed') is True else 'no')
except Exception:
    print('no')
" 2>/dev/null || echo "no"
}

timeline_idle() {
  python3 -c "
import os, time
try:
    print('yes' if time.time() - os.path.getmtime('$TIMELINE') >= $IDLE_GRACE else 'no')
except Exception:
    print('no')
" 2>/dev/null || echo "no"
}

# Wait for flags + quiet timeline, then /exit both panes. The double idle
# check closes the race where an agent writes between check and /exit.
complete_run() {
  log "Completion requested — waiting for progress flags + ${IDLE_GRACE}s timeline idle."
  while TM has-session -t "$TMUX_SESSION" 2>/dev/null; do
    if [[ "$(flags_complete)" == "yes" && "$(timeline_idle)" == "yes" ]]; then
      sleep 2
      [[ "$(timeline_idle)" == "yes" ]] || continue
      log "Flags set and timeline idle (confirmed twice) — sending /exit to both panes."
      for pane in "$PANE_REVIEWER" "$PANE_PROGRAMMER"; do
        TM send-keys -t "$pane" "/exit" Enter 2>/dev/null || true
        sleep 5
        # Retry once if the pane's session half is still alive.
        if TM list-panes -t "$pane" &>/dev/null; then
          TM send-keys -t "$pane" Enter 2>/dev/null || true
          sleep 1
          TM send-keys -t "$pane" "/exit" Enter 2>/dev/null || true
          sleep 5
        fi
      done
      log "Completion sequence done. Session alive? $(TM has-session -t "$TMUX_SESSION" 2>/dev/null && echo yes || echo no)"
      return 0
    fi
    sleep "$POLL"
  done
  log "tmux session vanished during completion wait."
}

log "Started. Poll ${POLL}s, idle grace ${IDLE_GRACE}s. Watching $SIGNALS_DIR → $TMUX_SESSION."
trap 'log "Received termination signal — exiting."; exit 0' TERM INT

while true; do
  sleep "$POLL"

  if ! TM has-session -t "$TMUX_SESSION" 2>/dev/null; then
    log "tmux session '$TMUX_SESSION' is gone — coordinator exiting."
    exit 0
  fi

  # Passive completion fallback: agents finished but forgot/failed to signal
  # done (same protection watch-completion.sh gave the teams mode).
  if [[ "$(flags_complete)" == "yes" && "$(timeline_idle)" == "yes" ]]; then
    log "Progress flags complete without 'done' signal — entering completion."
    complete_run
    exit 0
  fi

  [[ -d "$SIGNALS_DIR" ]] || continue

  while IFS=$'\t' read -r name target msg; do
    [[ -n "${name:-}" ]] || continue
    log "signal='$name' → $target"
    case "$target" in
      reviewer)     inject "$PANE_REVIEWER" "$msg" || log "WARN: inject to reviewer pane failed" ;;
      programmer)   inject "$PANE_PROGRAMMER" "$msg" || log "WARN: inject to programmer pane failed" ;;
      notify-human) notify_human "$msg" ;;
      complete)     complete_run; exit 0 ;;
      unknown)      log "WARN: $msg" ;;
    esac
  done < <(scan_signals "$SIGNALS_DIR")
done
