#!/usr/bin/env bash
# =============================================================================
# launch-agent-pair.sh — User-facing entry point (dispatcher)
#
# Routes subcommands and launch modes:
#   - Operational subcommands → ops.sh
#   - --mode solo             → launch-solo-pair.sh (file-based, any tier)
#   - Default                 → launch.sh (agent-teams, Max tier)
#
# Usage:
#   ./launch-agent-pair.sh --architecture <class> [options]          # agent-teams (Max tier)
#   ./launch-agent-pair.sh --mode solo --architecture <class> [opts] # file-based (any tier)
#   ./launch-agent-pair.sh status                                    # running pairs
#   ./launch-agent-pair.sh logs <architecture> [--raw]               # tail timeline
#   ./launch-agent-pair.sh attach <architecture>                     # tmux attach
#   ./launch-agent-pair.sh send <architecture> <message>             # inject text
#   ./launch-agent-pair.sh stop [architecture|all]                   # kill pair(s)
#   ./launch-agent-pair.sh clean [architecture]                      # remove worktree
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TL_CMD="$0"

# Check for --mode flag before dispatching
MODE=""
REMAINING_ARGS=()
for arg in "$@"; do
  if [[ "$MODE" == "NEXT" ]]; then
    MODE="$arg"
  elif [[ "$arg" == "--mode" ]]; then
    MODE="NEXT"
  else
    REMAINING_ARGS+=("$arg")
  fi
done
# If --mode was the last arg with no value
[[ "$MODE" == "NEXT" ]] && MODE=""

case "${1:-}" in
  status|logs|stop|attach|send|clean)
    exec "$SCRIPT_DIR/ops.sh" "$@"
    ;;
esac

case "$MODE" in
  solo)
    exec "$SCRIPT_DIR/launch-solo-pair.sh" "${REMAINING_ARGS[@]}"
    ;;
  team|"")
    exec "$SCRIPT_DIR/launch.sh" "$@"
    ;;
  *)
    echo "Unknown mode: $MODE (valid: solo, team)" >&2
    exit 1
    ;;
esac
