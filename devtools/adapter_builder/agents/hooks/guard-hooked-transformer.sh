#!/usr/bin/env bash
# =============================================================================
# guard-hooked-transformer.sh — PreToolUse hook
#
# RUNTIME CONTEXT:
#   This script lives under devtools/adapter_builder/agents/hooks/ but is executed
#   by Claude Code *inside the target TransformerLens worktree* — the
#   .claude/settings.json written by launch.sh references this file via an
#   absolute path. `pwd` at runtime is the worktree, NOT the directory
#   containing this file. The deny_patterns below match against paths
#   inside that worktree.
#
# Denies any Edit/Write tool call targeting deprecated HookedTransformer files.
# Agents may read these files (for reference) but must not modify them.
#
# The hook receives the tool call payload on stdin. If the tool is Edit/Write
# and the target file is a deprecated path, we exit with code 2 and print a
# message to stderr — Claude Code interprets this as a deny decision.
# =============================================================================

set -euo pipefail

# Read the hook payload from stdin
payload=$(cat)

# Extract the tool name and file path using jq (fall back gracefully if missing)
tool_name=$(echo "$payload" | jq -r '.tool_name // empty')
file_path=$(echo "$payload" | jq -r '.tool_input.file_path // .tool_input.path // empty')

allow_response() {
  echo '{"continue": true}'
  exit 0
}

# Only intercept Edit, Write, MultiEdit, NotebookEdit operations
case "$tool_name" in
  Edit|Write|MultiEdit|NotebookEdit) ;;
  *) allow_response ;;
esac

# No file path — let it through (some tools may not have one)
[[ -z "$file_path" ]] && allow_response

# Check against the deprecated path patterns. Paths are matched as suffixes
# (the file_path from tool_input is typically absolute, so we check whether
# it ends with the pattern). This prevents false positives on unrelated paths
# like "my_transformer_lens_backup/foo.py".
deny_patterns=(
  "transformer_lens/HookedTransformer.py"
  "transformer_lens/loading_from_pretrained.py"
  "transformer_lens/components/"
  "transformer_lens/pretrained/weight_conversions/"
)

for pattern in "${deny_patterns[@]}"; do
  # Match: path ends with the pattern, or contains /pattern (subpath match).
  # The leading / ensures we match at a directory boundary, not mid-word.
  if [[ "$file_path" == */"$pattern"* ]] || [[ "$file_path" == "$pattern"* ]]; then
    local reason="BLOCKED: This file is part of the deprecated HookedTransformer system. Adapter work must only modify files under transformer_lens/model_bridge/ and transformer_lens/factories/architecture_adapter_factory.py. HookedTransformer files may be READ for reference but not modified."
    local tl="$(pwd)/.adapter-workspace/timeline.jsonl"
    if [[ -f "$tl" ]]; then
      jq -n --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg hook "guard-hooked-transformer" --arg reason "$reason" --arg path "$file_path" \
        '{ts:$ts, event:"HookBlocked", tool:"Edit", hook:$hook, file_path:$path, reason:$reason}' >> "$tl" 2>/dev/null || true
    fi
    jq -n --arg reason "$reason" '{continue: false, decision: "block", reason: $reason}'
    exit 2
  fi
done

allow_response
