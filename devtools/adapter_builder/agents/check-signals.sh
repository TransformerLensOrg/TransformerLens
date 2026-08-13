#!/usr/bin/env bash
# =============================================================================
# check-signals.sh — Non-blocking signal state snapshot for solo mode
#
# Agents never wait or poll for signals (the coordinator wakes them with an
# injected message). This helper is for re-orientation on wake-up or after a
# crash-resume: one call, instant output, no blocking.
#
# Prints:
#   - pending signals (not yet consumed by the coordinator)
#   - the most recent processed signals from the coordinator's archive
#   - review round counts per checkpoint (from .adapter-workspace/reviews/)
#
# Usage (from the worktree root):
#   "$TL_ADAPTER_BUILDER_ROOT/agents/check-signals.sh"
# =============================================================================

set -euo pipefail

WORKSPACE="$(pwd)/.adapter-workspace"
SIGNALS_DIR="$WORKSPACE/signals"
ARCHIVE_DIR="$SIGNALS_DIR/.archive"
REVIEWS_DIR="$WORKSPACE/reviews"

echo "== Pending signals (awaiting coordinator) =="
if [[ -d "$SIGNALS_DIR" ]] && ls "$SIGNALS_DIR" 2>/dev/null | grep -q .; then
  ls -tr "$SIGNALS_DIR"
else
  echo "(none)"
fi

echo ""
echo "== Recently processed (newest last) =="
if [[ -d "$ARCHIVE_DIR" ]] && ls "$ARCHIVE_DIR" 2>/dev/null | grep -q .; then
  # Archive names are <epoch>.<signal>; show the last 10 in order.
  ls "$ARCHIVE_DIR" | sort -n | tail -10 | sed -E 's/^[0-9]+\.//'
else
  echo "(none)"
fi

echo ""
echo "== Review rounds per checkpoint =="
if [[ -d "$REVIEWS_DIR" ]] && ls "$REVIEWS_DIR"/*.md >/dev/null 2>&1; then
  # brief-review-2.md -> "brief 2"; keep the highest round per checkpoint.
  ls "$REVIEWS_DIR" | sed -nE 's/^(.+)-review-([0-9]+)\.md$/\1 \2/p' \
    | sort -k1,1 -k2,2n | awk '{last[$1]=$2} END {for (c in last) print c ": round " last[c]}' | sort
else
  echo "(none)"
fi
