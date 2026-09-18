#!/usr/bin/env bash
# =============================================================================
# overlord-request.sh — Simple flock-based memory slot lock
#
# Ensures only one memory-intensive operation (verify_models, benchmarks,
# full model loading) runs at a time across all agent pairs.
#
# Usage (preferred — single command, atomic acquire/run/release):
#   ./overlord-request.sh run "verify_models" -- uv run python -m ...
#
# Usage (status check):
#   ./overlord-request.sh status
#
# NOTE: when `source`d by another script, err() must not call `exit` or it
# kills the parent. We override err() locally for the sourced case.
# =============================================================================

set -euo pipefail

LOCK_FILE="/tmp/tl-adapter-builder.lock"
STATUS_FILE="/tmp/tl-adapter-builder.status"
LOCK_TTL_SECONDS=1800  # 30 minutes — auto-expire stale locks

# Shared helpers. Override err() to use return instead of exit when sourced,
# so a lock timeout doesn't kill the calling script.
OVERLORD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TL_LOG_TAG="memory-lock"
# shellcheck disable=SC1091
source "$OVERLORD_SCRIPT_DIR/lib/common.sh"

# If sourced, redefine err() to return 1 instead of exit 1.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  err() { echo -e "\033[1;31m[memory-lock ERROR]\033[0m $*" >&2; return 1; }
fi

iso_now() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# --------------------------------------------------------------------------- #
# Stale lock cleanup — runs AFTER flock, not before. This avoids the TOCTOU
# where cleanup removes the lock file between another process's flock and
# its status write. Instead, we acquire the lock first (blocking), then
# check if the status file is stale (meaning the previous holder crashed
# without releasing). If stale, we already hold the lock so no race.
# --------------------------------------------------------------------------- #
_cleanup_stale_status() {
  # Only meaningful when we already hold the lock.
  if [[ -f "$STATUS_FILE" ]]; then
    local file_age
    if [[ "$(uname)" == "Darwin" ]]; then
      file_age=$(( $(date +%s) - $(stat -f %m "$STATUS_FILE") ))
    else
      file_age=$(( $(date +%s) - $(stat -c %Y "$STATUS_FILE") ))
    fi
    if (( file_age > LOCK_TTL_SECONDS )); then
      local stale_holder
      stale_holder=$(python3 -c "import json; print(json.load(open('$STATUS_FILE')).get('operation','unknown'))" 2>/dev/null || echo "unknown")
      log "Stale status detected (${file_age}s old, holder: ${stale_holder}). Clearing."
      rm -f "$STATUS_FILE"
    fi
  fi
}

# --------------------------------------------------------------------------- #
# acquire: take the lock, clean stale status, write new status
# --------------------------------------------------------------------------- #
overlord_acquire() {
  local operation="${1:-unspecified}"
  local timeout="${2:-300}"

  log "Requesting memory slot for: $operation"

  # Open the lock file on fd 9
  exec 9>"$LOCK_FILE"

  # Block until lock is available (or timeout)
  if ! flock -w "$timeout" 9; then
    err "Timeout (${timeout}s) waiting for memory slot. Another heavy operation is running."
    return 1
  fi

  # Now that we hold the lock, clean any stale status from a crashed holder.
  _cleanup_stale_status

  # Write status so others can see who holds the lock
  cat > "$STATUS_FILE" <<EOF
{
  "holder": "${OVERLORD_AGENT_ROLE:-unknown}",
  "operation": "${operation}",
  "acquired_at": "$(iso_now)",
  "pid": $$
}
EOF

  log "Acquired memory slot for: $operation"
}

# --------------------------------------------------------------------------- #
# release: drop the lock, clear status
# --------------------------------------------------------------------------- #
overlord_release() {
  rm -f "$STATUS_FILE"
  exec 9>&- 2>/dev/null || true
  log "Released memory slot."
}

# --------------------------------------------------------------------------- #
# status: show who holds the lock (if anyone)
# --------------------------------------------------------------------------- #
cmd_status() {
  # Acquire the lock briefly to check stale status atomically, then release.
  if exec 9>"$LOCK_FILE" && flock -n 9 2>/dev/null; then
    _cleanup_stale_status
    exec 9>&- 2>/dev/null || true
  fi

  if [[ -f "$STATUS_FILE" ]]; then
    log "Memory slot is HELD:"
    cat "$STATUS_FILE"
  else
    log "Memory slot is FREE."
  fi
}

# --------------------------------------------------------------------------- #
# run: acquire lock, run a command, release lock (all in one process)
# --------------------------------------------------------------------------- #
cmd_run() {
  local operation="${1:-unspecified}"
  shift
  [[ "${1:-}" == "--" ]] && shift

  overlord_acquire "$operation"
  local exit_code=0
  "$@" || exit_code=$?
  overlord_release
  return $exit_code
}

# --------------------------------------------------------------------------- #
# Direct CLI invocation
# --------------------------------------------------------------------------- #
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  case "${1:-}" in
    acquire) shift; overlord_acquire "$@" ;;
    release) overlord_release ;;
    status)  cmd_status ;;
    run)     shift; cmd_run "$@" ;;
    *) echo "Usage: $0 {run <op> -- <cmd>|status|acquire <op>|release}" >&2; exit 1 ;;
  esac
fi
