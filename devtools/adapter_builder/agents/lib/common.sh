#!/usr/bin/env bash
# =============================================================================
# lib/common.sh — Shared bash helpers for TL Adapter Builder scripts
#
# Provides colored logging, command presence checks, and a UTC timestamp.
# Sourced by launch.sh, ops.sh, overlord-request.sh, and any helper that
# needs consistent terminal output.
#
# The [TAG] prefix for `log` is controlled by TL_LOG_TAG. Callers should set
# it before sourcing:
#
#   TL_LOG_TAG="launch"
#   source "$SCRIPT_DIR/lib/common.sh"
#
# This file intentionally does NOT set `set -euo pipefail` — that is the
# caller's responsibility, so sourcing into an existing shell is side-effect
# free beyond the helper definitions.
# =============================================================================

: "${TL_LOG_TAG:=tl}"

log()  { echo -e "\033[1;34m[${TL_LOG_TAG}]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[  ok  ]\033[0m $*"; }
warn() { echo -e "\033[1;33m[ warn ]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ err  ]\033[0m $*" >&2; exit 1; }

require_cmd() { command -v "$1" &>/dev/null || err "'$1' is not installed or not in PATH."; }

iso_now() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
