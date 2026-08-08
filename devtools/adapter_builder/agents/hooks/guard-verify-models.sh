#!/usr/bin/env bash
# =============================================================================
# guard-verify-models.sh — PreToolUse hook for Bash
#
# RUNTIME CONTEXT:
#   This script lives under devtools/adapter_builder/agents/hooks/ but is executed
#   by Claude Code *inside the target TransformerLens worktree* via the
#   .claude/settings.json written by launch.sh. The hook payload is read
#   from stdin and the cwd at runtime is the worktree root, so relative
#   paths to supported_models.json resolve against the worktree's
#   transformer_lens/tools/model_registry/data/.
#
# Enforces two rules on verify_models invocations, turning prompt-level
# guidance into runtime stops:
#
#   1. The target model must be registered in supported_models.json.
#      Otherwise verify_models silently skips it and the agent gets no
#      signal (documented failure mode in agents/programmer.md Step 3.0).
#
#   2. The target model must be ≤ MAX_VERIFY_PARAMS (default 7.5B). Anything
#      larger is a waste of verification time per the ≤7B rule in
#      agents/programmer.md Step 3.1.
#
# Dry-run invocations (--dry-run) and bulk --architectures runs without a
# specific --model arg pass through: dry runs don't load weights, and bulk
# runs have verify_models' own memory-based skip.
# =============================================================================

set -euo pipefail

MAX_PARAMS="${MAX_VERIFY_PARAMS:-7500000000}"  # 7.5B gives a small buffer above "7B" models
REGISTRY="transformer_lens/tools/model_registry/data/supported_models.json"

payload=$(cat)

tool_name=$(echo "$payload" | jq -r '.tool_name // empty')
command=$(echo "$payload" | jq -r '.tool_input.command // empty')

allow() {
  echo '{"continue": true}'
  exit 0
}

block() {
  local tl="$(pwd)/.adapter-workspace/timeline.jsonl"
  if [[ -f "$tl" ]]; then
    jq -n --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg hook "guard-verify-models" --arg reason "$1" \
      '{ts:$ts, event:"HookBlocked", tool:"Bash", hook:$hook, reason:$reason}' >> "$tl" 2>/dev/null || true
  fi
  jq -n --arg reason "$1" '{continue: false, decision: "block", reason: $reason}'
  exit 2
}

[[ "$tool_name" != "Bash" ]] && allow
[[ -z "$command" ]] && allow

# Only intercept verify_models invocations. This matches both
# `verify_models` directly and `python -m transformer_lens.tools.model_registry.verify_models`.
echo "$command" | grep -qE 'verify_models(\b|[[:space:]]|$)' || allow

# Dry runs are harmless — they don't load weights.
if echo "$command" | grep -qE '(^|[[:space:]])--dry-run(\b|[[:space:]]|=|$)'; then
  allow
fi

# Structural-only mode is never acceptable for adapter verification — it
# skips the HuggingFace parity comparison and writes status 4 (provisional),
# which the review gate does not count as verified.
if echo "$command" | grep -qE '(^|[[:space:]])--no-hf-reference(\b|[[:space:]]|=|$)'; then
  block "BLOCKED: --no-hf-reference downgrades verification to structural-only (status 4 provisional). Adapter verification requires the HuggingFace parity comparison — rerun without this flag (the HF reference is on by default)."
fi

# Extract --model <id> or --model=<id>. If absent, this is likely a bulk
# --architectures run; defer to verify_models' own filtering.
model=$(echo "$command" | sed -nE 's/.*--model[[:space:]]+([^[:space:]]+).*/\1/p' | head -1)
if [[ -z "$model" ]]; then
  model=$(echo "$command" | sed -nE 's/.*--model=([^[:space:]]+).*/\1/p' | head -1)
fi
[[ -z "$model" ]] && allow

if [[ ! -f "$REGISTRY" ]]; then
  block "BLOCKED: $REGISTRY is missing. Run scripts/port-arch-models.py (Step 3.0 in agents/programmer.md) to populate the registry before calling verify_models."
fi

# Look up the model's total_params. Returns one of:
#   NOT_FOUND       — not in registry
#   <integer>       — params count
#   0               — registered but no param metadata
lookup=$(python3 -c "
import json
try:
    data = json.load(open('$REGISTRY'))
except Exception:
    print('NOT_FOUND')
    raise SystemExit(0)
for m in data.get('models', []):
    if m.get('model_id') == '$model':
        md = m.get('metadata') or {}
        print(md.get('total_params') or 0)
        raise SystemExit(0)
print('NOT_FOUND')
" 2>/dev/null)

if [[ "$lookup" == "NOT_FOUND" ]]; then
  block "BLOCKED: Model '$model' is not registered in $REGISTRY. Run scripts/scan-hf-architecture.py then scripts/port-arch-models.py to port this architecture's models into the registry (Step 1a.2 + Step 3.0 in agents/programmer.md), then retry."
fi

if [[ "$lookup" =~ ^[0-9]+$ ]] && (( lookup > MAX_PARAMS )); then
  params_b=$(python3 -c "print(f'{$lookup/1e9:.1f}B')" 2>/dev/null || echo "${lookup}")
  cap_b=$(python3 -c "print(f'{$MAX_PARAMS/1e9:.1f}B')" 2>/dev/null || echo "${MAX_PARAMS}")
  block "BLOCKED: Model '$model' has ${params_b} parameters, exceeding the ${cap_b} verification ceiling. Large models waste verification time without additional signal. Pick a smaller model from $REGISTRY, or if no models of this architecture fit, document the situation in .adapter-workspace/verification-note.md and signal 'VERIFICATION SKIPPED — ALL MODELS TOO LARGE' per agents/programmer.md Step 3.1."
fi

allow
