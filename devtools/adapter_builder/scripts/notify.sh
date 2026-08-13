#!/usr/bin/env bash
# =============================================================================
# notify.sh — Send a completion notification
#
# Tries in order:
#   1. Webhook (Slack/Discord/custom) — if NOTIFICATION_WEBHOOK_URL is set
#   2. iMessage (macOS) — if NOTIFICATION_NUMBER is set and on macOS
#   3. macOS Notification Center — last resort on macOS
#   4. Stdout — always works
#
# Usage:
#   ./scripts/notify.sh "Adapter completed for CohereForCausalLM"
# =============================================================================

set -euo pipefail

MESSAGE="${1:-Adapter build complete}"
WEBHOOK_URL="${NOTIFICATION_WEBHOOK_URL:-}"
PHONE="${NOTIFICATION_NUMBER:-}"

log() { echo -e "\033[1;35m[notify]\033[0m $*"; }

# --------------------------------------------------------------------------- #
# 1. Webhook (Slack, Discord, or custom)
# --------------------------------------------------------------------------- #
if [[ -n "$WEBHOOK_URL" ]]; then
  # Detect service from URL and format payload accordingly
  if [[ "$WEBHOOK_URL" == *"slack"* ]]; then
    PAYLOAD=$(printf '{"text": "%s"}' "$MESSAGE")
  elif [[ "$WEBHOOK_URL" == *"discord"* ]]; then
    PAYLOAD=$(printf '{"content": "%s"}' "$MESSAGE")
  else
    # Generic — send both common field names
    PAYLOAD=$(printf '{"text": "%s", "content": "%s", "message": "%s"}' "$MESSAGE" "$MESSAGE" "$MESSAGE")
  fi

  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "$WEBHOOK_URL" 2>/dev/null) || HTTP_CODE="000"

  if [[ "$HTTP_CODE" =~ ^2 ]]; then
    log "Webhook notification sent (HTTP $HTTP_CODE)"
    exit 0
  else
    log "Webhook failed (HTTP $HTTP_CODE) — trying fallback"
  fi
fi

# --------------------------------------------------------------------------- #
# 2. iMessage (macOS only)
# --------------------------------------------------------------------------- #
if [[ -n "$PHONE" && "$(uname)" == "Darwin" ]] && command -v osascript &>/dev/null; then
  osascript -e "
    tell application \"Messages\"
      set targetService to 1st account whose service type = iMessage
      set targetBuddy to participant \"${PHONE}\" of targetService
      send \"${MESSAGE}\" to targetBuddy
    end tell
  " 2>/dev/null && {
    log "iMessage sent to $PHONE"
    exit 0
  } || {
    log "iMessage failed — trying fallback"
  }
fi

# --------------------------------------------------------------------------- #
# 3. macOS Notification Center
# --------------------------------------------------------------------------- #
if [[ "$(uname)" == "Darwin" ]]; then
  osascript -e "display notification \"${MESSAGE}\" with title \"TL Adapter Builder\"" 2>/dev/null && {
    log "macOS notification sent"
    exit 0
  } || true
fi

# --------------------------------------------------------------------------- #
# 4. Stdout (always works)
# --------------------------------------------------------------------------- #
log "$MESSAGE"
log "(no notification service configured — set NOTIFICATION_WEBHOOK_URL in .env)"
