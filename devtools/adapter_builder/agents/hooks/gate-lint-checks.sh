#!/usr/bin/env bash
# =============================================================================
# gate-lint-checks.sh — Stop hook
#
# RUNTIME CONTEXT:
#   This script lives under devtools/adapter_builder/agents/hooks/ but is executed
#   by Claude Code *inside the target TransformerLens worktree* — the
#   .claude/settings.json written by launch.sh references this file via an
#   absolute path. `pwd` at runtime is the worktree, which is why
#   WORKTREE_DIR="$(pwd)" and `uv run mypy .` resolve against the worktree,
#   not this file's directory.
#
# Runs mypy and make check-format before allowing the session to end.
# Only runs when the progress file indicates verification has passed
# (we don't want to block mid-build stops for legitimate reasons).
#
# Exits with code 2 (deny stop) if checks fail, causing the agent to keep
# working until it fixes the issues.
# =============================================================================

set -euo pipefail

# We run from the worktree (cwd when the session exits)
WORKTREE_DIR="$(pwd)"
PROGRESS_FILE="$WORKTREE_DIR/.adapter-progress.json"

allow_stop() {
  echo '{"continue": true}'
  exit 0
}

# If no progress file, this isn't an adapter build — let it exit
[[ -f "$PROGRESS_FILE" ]] || allow_stop

# Only gate when verification has passed; earlier stops are legitimate
verification_passed=$(jq -r '.verification_passed // false' "$PROGRESS_FILE" 2>/dev/null || echo "false")
if [[ "$verification_passed" != "true" ]]; then
  allow_stop
fi

# If we've already gated this session (final_review_passed), let it exit
final_review=$(jq -r '.final_review_passed // false' "$PROGRESS_FILE" 2>/dev/null || echo "false")
if [[ "$final_review" == "true" ]]; then
  allow_stop
fi

# Run mypy
mypy_output=$(uv run mypy . 2>&1) || mypy_exit=$?
mypy_exit="${mypy_exit:-0}"

_lint_block() {
  local reason="$1"
  local tl="$WORKTREE_DIR/.adapter-workspace/timeline.jsonl"
  if [[ -f "$tl" ]]; then
    jq -n --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg hook "gate-lint-checks" --arg reason "$reason" \
      '{ts:$ts, event:"HookBlocked", tool:"Stop", hook:$hook, reason:$reason}' >> "$tl" 2>/dev/null || true
  fi
  python3 -c "
import json
print(json.dumps({'continue': False, 'decision': 'block', 'reason': '''$reason'''}))
"
  exit 2
}

if [[ "$mypy_exit" != "0" ]]; then
  mypy_tail=$(echo "$mypy_output" | tail -30)
  _lint_block "BLOCKED: mypy checks failing. Fix type errors before completing.

$mypy_tail

Remember: # type: ignore is NOT an acceptable fix."
fi

# Run format check
format_output=$(make check-format 2>&1) || format_exit=$?
format_exit="${format_exit:-0}"

if [[ "$format_exit" != "0" ]]; then
  format_tail=$(echo "$format_output" | tail -30)
  _lint_block "BLOCKED: Format check failing. Run make format to fix.

$format_tail"
fi

# Both checks pass — mark final_review_passed so we don't re-run on subsequent stops.
# Uses fcntl.flock on the same co-located lock file as progress.sh.
python3 -c "
import json, fcntl
lockfile = '${PROGRESS_FILE%.json}.lock'
lockfd = open(lockfile, 'w')
try:
    fcntl.flock(lockfd, fcntl.LOCK_EX)
    p = json.load(open('$PROGRESS_FILE'))
    p['final_review_passed'] = True
    p['last_updated'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
    json.dump(p, open('$PROGRESS_FILE','w'), indent=2)
finally:
    fcntl.flock(lockfd, fcntl.LOCK_UN)
    lockfd.close()
" 2>/dev/null

echo '{"continue": true}'
exit 0
