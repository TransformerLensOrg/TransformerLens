#!/usr/bin/env bash
# =============================================================================
# guard-git.sh — PreToolUse hook for Bash
#
# RUNTIME CONTEXT:
#   This script lives under devtools/adapter_builder/agents/hooks/ but is executed
#   by Claude Code *inside the target TransformerLens worktree* via the
#   .claude/settings.json written by launch.sh. The hook payload is read
#   from stdin and the cwd at runtime is the worktree, not this file's
#   directory.
#
# Blocks any Bash command that commits, pushes, or publishes changes. The
# user commits manually after reviewing the agent's uncommitted work —
# agents must never commit on their own. This is a belt-and-braces
# enforcement of the prompt-level "no git commits by agents" rule.
#
# Denied operations (exit 2 with {continue: false}):
#   - git commit (any form, including -m, --amend, -a, etc.)
#   - git push  (any form)
#   - gh pr create / merge / close / review
#   - gh release create
#
# Everything else passes, including read-only git operations like status,
# diff, log, show, branch -l, worktree list.
# =============================================================================

set -euo pipefail

payload=$(cat)

tool_name=$(echo "$payload" | jq -r '.tool_name // empty')
command=$(echo "$payload" | jq -r '.tool_input.command // empty')

allow() {
  echo '{"continue": true}'
  exit 0
}

block() {
  # Append a HookBlocked marker to the timeline so the orchestrator can
  # detect that a subagent's tool call was denied by a hook.
  local tl="$(pwd)/.adapter-workspace/timeline.jsonl"
  if [[ -f "$tl" ]]; then
    jq -n --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg hook "guard-git" --arg reason "$1" \
      '{ts:$ts, event:"HookBlocked", tool:"Bash", hook:$hook, reason:$reason}' >> "$tl" 2>/dev/null || true
  fi
  jq -n --arg reason "$1" '{continue: false, decision: "block", reason: $reason}'
  exit 2
}

# Only intercept Bash tool calls with a non-empty command.
[[ "$tool_name" != "Bash" ]] && allow
[[ -z "$command" ]] && allow

# Word-boundary regexes. The leading anchor allows the command to appear at
# the start of the string, after whitespace, after a shell separator
# (;, &, |, (, `), or after &&/||. This catches chained invocations like
# `make test && git commit -m ...` without false-positives on things like
# `gitignore` or "commit" appearing in a commit-message string (those don't
# form a `git commit` token).
lead='(^|[[:space:];&|(`])'

if echo "$command" | grep -qE "${lead}git[[:space:]]+commit([[:space:]]|$)"; then
  block "BLOCKED: 'git commit' is not allowed. Agents must leave all changes uncommitted; the user commits manually after reviewing the diff. See agents/programmer.md § What NOT to Do."
fi

if echo "$command" | grep -qE "${lead}git[[:space:]]+push([[:space:]]|$)"; then
  block "BLOCKED: 'git push' is not allowed. Agents must not publish changes to any remote."
fi

if echo "$command" | grep -qE "${lead}gh[[:space:]]+pr[[:space:]]+(create|merge|close|review|ready|edit)([[:space:]]|$)"; then
  block "BLOCKED: Creating/merging/closing/reviewing PRs via 'gh pr' is not allowed. Agents must not publish changes."
fi

if echo "$command" | grep -qE "${lead}gh[[:space:]]+release[[:space:]]+(create|delete|edit)([[:space:]]|$)"; then
  block "BLOCKED: 'gh release' operations are not allowed. Agents must not publish releases."
fi

allow
