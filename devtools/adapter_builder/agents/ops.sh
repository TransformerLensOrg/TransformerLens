#!/usr/bin/env bash
# =============================================================================
# ops.sh — Operational subcommands for the TL Adapter Builder
#
# Handles status / logs / attach / send / stop / clean. The launch path lives
# in launch.sh, and the user-facing entry point is launch-agent-pair.sh which
# routes subcommands here.
#
# Usage:
#   ./ops.sh status
#   ./ops.sh logs <architecture> [--raw]
#   ./ops.sh attach <architecture>
#   ./ops.sh send <architecture> <message>
#   ./ops.sh stop [architecture|all]
#   ./ops.sh clean [architecture]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/.logs"
FORMAT_TIMELINE="$PROJECT_ROOT/scripts/format-timeline.py"

# Load .env once so subcommands can share WORKTREE_BASE / DEFAULT_TARGET_REPO.
ENV_FILE="$PROJECT_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

# Where worktrees live. Override via WORKTREE_BASE in .env; otherwise fall
# back to "<parent-of-repo>/worktrees" — outside the repo, matching launch.sh.
REPO_ROOT="$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel 2>/dev/null || dirname "$PROJECT_ROOT")"
WORKTREE_BASE="${WORKTREE_BASE:-$(dirname "$REPO_ROOT")/worktrees}"

# Friendly command name shown in user-facing messages (e.g. "$TL_CMD attach X").
# launch-agent-pair.sh exports this so users see the dispatcher name rather
# than ops.sh when subcommands are invoked through it.
TL_CMD="${TL_CMD:-$0}"

# Shared helpers (log/ok/warn/err/require_cmd/iso_now).
TL_LOG_TAG="ops"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

# Dedicated tmux socket for all TL Adapter Builder sessions. Isolates our
# tmux server from any tmux session the user might have running, so a crashed
# pane or escape sequence can't kill the user's tmux.
TMUX_SOCKET="tl-adapter"
TM() { tmux -L "$TMUX_SOCKET" "$@"; }

# --------------------------------------------------------------------------- #
# last_event_age — compute a human-readable "time since" string from the
# last JSONL entry in a timeline file. Emits forms like "12s ago", "3m 40s
# ago", "2h 15m ago", "1d 4h ago", or "?" on error. Uses python3 for
# cross-platform ISO 8601 parsing (macOS `date -j` vs GNU `date -d`).
# --------------------------------------------------------------------------- #
last_event_age() {
  local filepath="$1"
  tail -1 "$filepath" 2>/dev/null | python3 -c "
import json, sys
from datetime import datetime, timezone
try:
    ts = json.loads(sys.stdin.read()).get('ts', '')
    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    s = int(max((datetime.now(timezone.utc) - dt).total_seconds(), 0))
    if s < 60:
        out = f'{s}s ago'
    elif s < 3600:
        out = f'{s//60}m {s%60}s ago'
    elif s < 86400:
        out = f'{s//3600}h {(s%3600)//60}m ago'
    else:
        out = f'{s//86400}d {(s%86400)//3600}h ago'
    print(out)
except Exception:
    print('?')
" 2>/dev/null || echo "?"
}

# --------------------------------------------------------------------------- #
# status — show all running agent pairs and their progress + memory-lock state
# --------------------------------------------------------------------------- #
cmd_status() {
  echo ""
  echo "=========================================="
  echo "  TL Adapter Builder — Running Pairs"
  echo "=========================================="
  echo ""

  local found=false
  if [[ -d "$LOGS_DIR" ]]; then
    for pidfile in "$LOGS_DIR"/*.pid; do
      [[ -f "$pidfile" ]] || continue
      # Daemon PID files (solo coordinator, teams watcher) are not pairs.
      case "$pidfile" in *.coordinator.pid|*.watcher.pid) continue ;; esac
      found=true
      local name pid worktree_dir progressfile="" timelinefile=""
      name=$(basename "$pidfile" .pid)
      pid=$(cat "$pidfile")
      worktree_dir="$WORKTREE_BASE/$name"

      [[ -f "$worktree_dir/.adapter-progress.json" ]] \
        && progressfile="$worktree_dir/.adapter-progress.json"
      [[ -f "$worktree_dir/.adapter-workspace/timeline.jsonl" ]] \
        && timelinefile="$worktree_dir/.adapter-workspace/timeline.jsonl"

      if kill -0 "$pid" 2>/dev/null; then
        echo -e "  \033[1;32m[running]\033[0m $name (PID: $pid)"
      else
        echo -e "  \033[1;31m[stopped]\033[0m $name (PID: $pid — process dead)"
      fi

      if [[ -n "$progressfile" ]]; then
        local step current_phase completed
        step=$(python3 -c "import json; print(json.load(open('$progressfile')).get('step','?'))" 2>/dev/null || echo "?")
        current_phase=$(python3 -c "import json; print(json.load(open('$progressfile')).get('current_phase','?'))" 2>/dev/null || echo "?")
        completed=$(python3 -c "import json; print(','.join(json.load(open('$progressfile')).get('completed_phases',[])))" 2>/dev/null || echo "?")
        echo "             step: $step | phase: $current_phase | completed: ${completed:-none}"
      fi

      if [[ -n "$timelinefile" ]]; then
        local event_count last_event last_age
        event_count=$(wc -l < "$timelinefile" 2>/dev/null | tr -d ' ')
        if [[ "${event_count:-0}" -gt 0 ]]; then
          last_event=$(tail -1 "$timelinefile" 2>/dev/null | python3 "$FORMAT_TIMELINE" 2>/dev/null || echo "?")
          last_age=$(last_event_age "$timelinefile")
          echo "             timeline: $event_count events — last: ${last_event:-?}"
          echo "             time since last event: ${last_age:-?}"
        else
          echo "             timeline: empty"
        fi
      fi

      local tmux_session="tl-adapter-$name"
      if command -v tmux &>/dev/null && TM has-session -t "$tmux_session" 2>/dev/null; then
        echo "             tmux: attach with '$TL_CMD attach $name'"
      fi
      echo ""
    done
  fi

  if [[ "$found" == false ]]; then
    echo "  No running pairs found."
    echo ""
  fi

  # Show memory lock status
  "$SCRIPT_DIR/overlord-request.sh" status 2>/dev/null || true
  echo ""
}

# --------------------------------------------------------------------------- #
# logs — tail the structured timeline (default) or the raw ANSI log (--raw)
# --------------------------------------------------------------------------- #
cmd_logs() {
  local arch="${1:-}"
  [[ -z "$arch" ]] && { echo "Usage: $TL_CMD logs <architecture-name> [--raw]"; exit 1; }

  local mode="timeline"
  [[ "${2:-}" == "--raw" ]] && mode="raw"

  arch=$(echo "$arch" | tr '[:upper:]' '[:lower:]')

  if [[ "$mode" == "timeline" ]]; then
    local worktree_dir=""
    for d in "$WORKTREE_BASE"/*"${arch}"*; do
      [[ -d "$d" ]] && worktree_dir="$d" && break
    done

    if [[ -z "$worktree_dir" ]]; then
      echo "No worktree found matching '$arch'"
      exit 1
    fi

    local timeline_file="$worktree_dir/.adapter-workspace/timeline.jsonl"
    if [[ ! -f "$timeline_file" ]]; then
      echo "No timeline file at: $timeline_file"
      echo "Falling back to raw log. Use '--raw' to skip this fallback."
      mode="raw"
    else
      echo "Tailing structured timeline: $timeline_file (Ctrl+C to stop)"
      echo "(use '$TL_CMD logs $arch --raw' for raw terminal log)"
      echo ""
      tail -f "$timeline_file" | python3 "$FORMAT_TIMELINE"
      return
    fi
  fi

  # Raw mode (or timeline fallback)
  local logfile=""
  for f in "$LOGS_DIR"/*"${arch}"*.raw; do
    [[ -f "$f" ]] && logfile="$f" && break
  done
  if [[ -z "$logfile" ]]; then
    for f in "$LOGS_DIR"/*"${arch}"*.log; do
      [[ -f "$f" ]] && logfile="$f" && break
    done
  fi

  if [[ -z "$logfile" ]]; then
    echo "No log file found matching '$arch' in $LOGS_DIR/"
    ls "$LOGS_DIR"/*.raw "$LOGS_DIR"/*.log 2>/dev/null || echo "(no log files exist)"
    exit 1
  fi

  echo "Tailing raw log: $logfile (Ctrl+C to stop, ANSI codes stripped)"
  tail -f "$logfile" | LC_ALL=C sed -u $'s/\x1b\[[0-9;?]*[a-zA-Z]//g; s/\x1b[][><=()][^\x1b]*//g; s/\r//g'
}

# --------------------------------------------------------------------------- #
# clean — remove a worktree (by architecture) or all verified-completed ones
# --------------------------------------------------------------------------- #
cmd_clean() {
  local arch="${1:-}"
  local force=false
  local incomplete_only=false

  # Accept flags in either position: `clean --force` or `clean <arch> --force`
  for a in "$@"; do
    case "$a" in
      --force|--all)    force=true; arch="" ;;
      --incomplete)     incomplete_only=true; arch="" ;;
    esac
  done

  # Needed for `git -C <repo>` operations on the branch being cleaned.
  # DEFAULT_TARGET_REPO (from .env) overrides the containing-repo default.
  local repo="${DEFAULT_TARGET_REPO:-$REPO_ROOT}"
  [[ -z "$repo" ]] && { echo "Error: could not derive repo root (set DEFAULT_TARGET_REPO in .env)"; exit 1; }

  if [[ -n "$arch" ]]; then
    # Clean a specific architecture
    local branch_name
    branch_name=$(echo "$arch" | sed -E 's/(For(Causal|Conditional|Masked).*|LMHead.*)$//' | tr '[:upper:]' '[:lower:]')
    local safe_branch="feature-${branch_name}-adapter"
    local worktree_dir="${WORKTREE_BASE}/${safe_branch}"

    echo ""
    echo "Cleaning up: $arch"
    echo "  worktree : $worktree_dir"
    echo "  branch   : feature/${branch_name}-adapter"
    echo ""

    if [[ -d "$worktree_dir" ]]; then
      git -C "$repo" worktree remove --force "$worktree_dir" 2>/dev/null && \
        echo "  Worktree removed." || echo "  Worktree removal failed (may not exist)."
    else
      echo "  No worktree found at $worktree_dir"
    fi

    git -C "$repo" branch -D "feature/${branch_name}-adapter" 2>/dev/null && \
      echo "  Branch deleted." || echo "  Branch not found (may already be deleted)."

    rm -f "$LOGS_DIR/${safe_branch}".{log,pid,raw,debug.log,wrapper.sh,prompt.txt,watch.log} 2>/dev/null
    echo "  Logs cleaned."
    echo ""
  else
    # Bulk clean. Default: verification_passed: true only. With --force/--all:
    # every adapter worktree regardless of state. With --incomplete: only the
    # ones where verification_passed is NOT true (useful after finishing
    # adapters manually outside the agent pipeline).
    echo ""
    if [[ "$force" == true ]]; then
      echo "Cleaning ALL adapter worktrees (--force)..."
    elif [[ "$incomplete_only" == true ]]; then
      echo "Cleaning incomplete adapter worktrees (--incomplete)..."
    else
      echo "Cleaning completed adapter worktrees (verification_passed: true)..."
    fi
    echo ""

    local cleaned=0 skipped=0
    if [[ -d "$WORKTREE_BASE" ]]; then
      for wt in "$WORKTREE_BASE"/feature-*-adapter; do
        [[ -d "$wt" ]] || continue
        local wt_name branch_ref passed="unknown"
        wt_name=$(basename "$wt")
        branch_ref=$(echo "$wt_name" | sed 's/^feature-//;s/-adapter$//')
        local progress="$wt/.adapter-progress.json"
        if [[ -f "$progress" ]]; then
          passed=$(python3 -c "import json; print(json.load(open('$progress')).get('verification_passed', False))" 2>/dev/null || echo "False")
        fi

        local should_clean=false
        if [[ "$force" == true ]]; then
          should_clean=true
        elif [[ "$incomplete_only" == true ]]; then
          [[ "$passed" != "True" ]] && should_clean=true
        else
          [[ "$passed" == "True" ]] && should_clean=true
        fi

        if [[ "$should_clean" == true ]]; then
          # Kill any live tmux session for this pair first so the worktree
          # remove doesn't fail on locked files.
          local tmux_session="tl-adapter-$wt_name"
          if command -v tmux &>/dev/null && TM has-session -t "$tmux_session" 2>/dev/null; then
            TM kill-session -t "$tmux_session" 2>/dev/null || true
            echo "  Killed tmux session: $tmux_session"
          fi
          echo "  Removing: $wt_name (verification_passed=$passed)"
          git -C "$repo" worktree remove --force "$wt" 2>/dev/null || true
          git -C "$repo" branch -D "feature/${branch_ref}-adapter" 2>/dev/null || true
          rm -f "$LOGS_DIR/${wt_name}".{log,pid,raw,debug.log,wrapper.sh,prompt.txt,watch.log,programmer.raw,reviewer.raw,programmer-wrapper.sh,reviewer-wrapper.sh,programmer-prompt.txt,reviewer-prompt.txt,watcher.pid} 2>/dev/null
          cleaned=$((cleaned + 1))
        else
          echo "  Skipping: $wt_name (verification_passed=$passed)"
          skipped=$((skipped + 1))
        fi
      done
      # Prune any dangling worktree refs whose directories are gone.
      git -C "$repo" worktree prune 2>/dev/null || true
    fi

    echo ""
    echo "Cleaned $cleaned worktree(s), skipped $skipped."
    echo ""
  fi
}

# --------------------------------------------------------------------------- #
# stop — kill a running pair (tmux session or bare PID) and keep its worktree
# --------------------------------------------------------------------------- #
cmd_stop() {
  local target="${1:-all}"

  # Kill companion daemons (solo coordinator, teams watcher) for a pair.
  # Both self-exit when the tmux session vanishes, but killing them directly
  # avoids a dangling poll cycle and cleans up their PID files.
  _stop_daemons() {
    local name="$1" daemon dpid
    for daemon in "$LOGS_DIR/$name.coordinator.pid" "$LOGS_DIR/$name.watcher.pid"; do
      [[ -f "$daemon" ]] || continue
      dpid=$(cat "$daemon" 2>/dev/null || echo 0)
      if kill -0 "$dpid" 2>/dev/null; then
        kill "$dpid" 2>/dev/null || true
        echo "  Stopped: $(basename "$daemon" .pid) (PID $dpid)"
      fi
      rm -f "$daemon"
    done
  }

  _stop_pair() {
    local pidfile="$1"
    local name pid tmux_session
    name=$(basename "$pidfile" .pid)
    pid=$(cat "$pidfile" 2>/dev/null || echo 0)
    tmux_session="tl-adapter-$name"

    # Prefer tmux session termination if we can find one.
    if command -v tmux &>/dev/null && TM has-session -t "$tmux_session" 2>/dev/null; then
      TM kill-session -t "$tmux_session" 2>/dev/null
      echo "  Stopped: $name (tmux session killed)"
      rm -f "$pidfile"
      _stop_daemons "$name"
      return 0
    fi

    # Fall back to killing by PID (legacy non-tmux sessions).
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null
      echo "  Stopped: $name (PID $pid)"
    else
      echo "  Already dead: $name (PID $pid)"
    fi
    rm -f "$pidfile"
    _stop_daemons "$name"
  }

  if [[ "$target" == "all" ]]; then
    echo ""
    echo "Stopping all background agent pairs..."
    echo ""
    local stopped=0
    for pidfile in "$LOGS_DIR"/*.pid; do
      [[ -f "$pidfile" ]] || continue
      # Daemon PID files are cleaned up by _stop_daemons via their pair.
      case "$pidfile" in *.coordinator.pid|*.watcher.pid) continue ;; esac
      _stop_pair "$pidfile"
      stopped=$((stopped + 1))
    done
    echo ""
    echo "Stopped $stopped process(es). Worktrees preserved for resume."
    echo ""
  else
    local arch
    arch=$(echo "$target" | sed -E 's/(For(Causal|Conditional|Masked).*|LMHead.*)$//' | tr '[:upper:]' '[:lower:]')

    local matched=false
    for pidfile in "$LOGS_DIR"/*"${arch}"*.pid; do
      [[ -f "$pidfile" ]] || continue
      case "$pidfile" in *.coordinator.pid|*.watcher.pid) continue ;; esac
      matched=true
      _stop_pair "$pidfile"
    done

    if [[ "$matched" == false ]]; then
      echo "No running process found matching '$target'"
      ls "$LOGS_DIR"/*.pid 2>/dev/null || echo "(no PID files exist)"
    else
      echo "Worktree preserved for resume."
    fi
  fi
}

# --------------------------------------------------------------------------- #
# _find_tmux_session — resolve an architecture name to an exact tmux session.
#
# Matching priority (to avoid "cohere" matching "cohere2"):
#   1. Exact: tl-adapter-feature-<arch>-adapter
#   2. Unique prefix: only one session starts with tl-adapter-*<arch>*
#   3. Ambiguous → error listing all matches
# --------------------------------------------------------------------------- #
_find_tmux_session() {
  local arch="$1"
  local exact="tl-adapter-feature-${arch}-adapter"

  # Try exact match first
  if TM has-session -t "$exact" 2>/dev/null; then
    echo "$exact"
    return 0
  fi

  # Fall back to prefix match, but require uniqueness
  local matches=()
  for s in $(tmux list-sessions -F '#{session_name}' 2>/dev/null); do
    if [[ "$s" == tl-adapter-*"${arch}"* ]]; then
      matches+=("$s")
    fi
  done

  if [[ ${#matches[@]} -eq 1 ]]; then
    echo "${matches[0]}"
    return 0
  elif [[ ${#matches[@]} -gt 1 ]]; then
    echo "Ambiguous: '$arch' matches ${#matches[@]} sessions:" >&2
    printf "  %s\n" "${matches[@]}" >&2
    echo "Use a more specific name." >&2
    return 1
  fi

  return 1
}

# --------------------------------------------------------------------------- #
# attach — interactively attach to the tmux session for an architecture
# --------------------------------------------------------------------------- #
cmd_attach() {
  local arch="${1:-}"
  [[ -z "$arch" ]] && { echo "Usage: $TL_CMD attach <architecture>"; exit 1; }

  command -v tmux &>/dev/null || { echo "tmux not installed"; exit 1; }

  arch=$(echo "$arch" | sed -E 's/(For(Causal|Conditional|Masked).*|LMHead.*)$//' | tr '[:upper:]' '[:lower:]')

  local session
  session=$(_find_tmux_session "$arch") || {
    echo "No tmux session matching '$arch'. Active sessions:"
    TM list-sessions 2>/dev/null || echo "(none)"
    exit 1
  }

  echo "Attaching to $session (detach with Ctrl-b d)"
  sleep 0.5
  exec TM attach-session -t "$session"
}

# --------------------------------------------------------------------------- #
# send — inject a message into a running tmux session as if typed at the prompt
# --------------------------------------------------------------------------- #
cmd_send() {
  local arch="${1:-}"
  shift || true
  local message="$*"

  [[ -z "$arch" || -z "$message" ]] && {
    echo "Usage: $TL_CMD send <architecture> <message>"
    echo ""
    echo "Sends a message to a running Claude Code session as if you typed it"
    echo "in the interactive prompt. Useful for injecting mid-run corrections"
    echo "without stopping the session."
    echo ""
    echo "Example:"
    echo "  $TL_CMD send qwen3moe \"Use float32 instead of bfloat16 for verification\""
    exit 1
  }

  command -v tmux &>/dev/null || { echo "tmux not installed"; exit 1; }

  arch=$(echo "$arch" | sed -E 's/(For(Causal|Conditional|Masked).*|LMHead.*)$//' | tr '[:upper:]' '[:lower:]')

  local session
  session=$(_find_tmux_session "$arch") || {
    echo "No tmux session matching '$arch'"
    exit 1
  }

  # Claude Code's interactive prompt treats large TM send-keys payloads as
  # a bracketed paste, which stages them as `[Pasted text #1]` and swallows
  # any `Enter` sent in the same tmux call. The fix is a two-step submission:
  #
  #   1. Deliver the message body via a dedicated paste buffer
  #      (tmux load-buffer + paste-buffer) so bracketed-paste markers are
  #      emitted cleanly. This is more reliable than `send-keys -l "$msg"`
  #      for multi-line / very long payloads.
  #   2. Sleep briefly so tmux finalizes the paste before the submit key.
  #   3. Send `Enter` as a standalone send-keys call so it reaches the prompt
  #      as a discrete key press (Claude Code submits the staged paste).
  #
  # Empirically, the old single-shot `send-keys "$msg"; send-keys Enter`
  # pattern left long messages (≳200 chars) unsubmitted with `[Pasted text #1]`
  # visible in the pane, forcing the user to press Enter manually.
  local tmp_buffer
  tmp_buffer="tl-send-$$-$(date +%s%N 2>/dev/null || date +%s)"
  printf '%s' "$message" | TM load-buffer -b "$tmp_buffer" -
  TM paste-buffer -d -b "$tmp_buffer" -t "$session"
  sleep 0.3
  TM send-keys -t "$session" Enter

  echo "Message sent to $session:"
  echo "  $message"
}

# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
case "${1:-}" in
  status) shift; cmd_status "$@" ;;
  logs)   shift; cmd_logs   "$@" ;;
  stop)   shift; cmd_stop   "$@" ;;
  attach) shift; cmd_attach "$@" ;;
  send)   shift; cmd_send   "$@" ;;
  clean)  shift; cmd_clean  "$@" ;;
  "")     echo "Usage: $TL_CMD {status|logs|stop|attach|send|clean} [args...]" >&2; exit 1 ;;
  *)      echo "Unknown subcommand: $1" >&2; exit 1 ;;
esac
