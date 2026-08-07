#!/usr/bin/env bash
# =============================================================================
# dry-run-test.sh
#
# Tests the adapter builder system to verify all components are correctly
# structured without actually launching agents.
#
# Checks:
#   1. All .sh scripts have valid bash syntax
#   2. All .py scripts have valid Python syntax
#   3. Domain knowledge docs are present and non-empty
#   4. Agent definitions have required sections
#   5. Signal consistency across files
#   6. validate-adapter.sh passes for an existing adapter
#   7. Project files exist
#   8. TransformerLens repo is accessible
#
# Usage:
#   ./scripts/dry-run-test.sh [--repo /path/to/TransformerLens]
# =============================================================================

set -euo pipefail

log()  { echo -e "\033[1;34m[test]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[ ok ]\033[0m $*"; }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $*"; FAILURES=$((FAILURES + 1)); }

FAILURES=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TL_REPO="$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel 2>/dev/null || { cd "$PROJECT_ROOT/../.." && pwd; })"

# Parse --repo flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) TL_REPO="$2"; shift 2 ;;
    *) shift ;;
  esac
done

echo ""
echo "=========================================="
echo "  Dry-Run Test Suite"
echo "  TL Repo: $TL_REPO"
echo "=========================================="
echo ""

# --------------------------------------------------------------------------- #
# 1. Bash syntax check — all .sh files in agents/ and scripts/
# --------------------------------------------------------------------------- #
log "Checking bash syntax (all .sh files)..."
while IFS= read -r -d '' shfile; do
  relpath="${shfile#"$PROJECT_ROOT/"}"
  if bash -n "$shfile" 2>/dev/null; then
    ok "$relpath"
  else
    fail "$relpath has syntax errors"
  fi
done < <(find "$PROJECT_ROOT/agents" "$PROJECT_ROOT/scripts" -name '*.sh' -print0 2>/dev/null)

# --------------------------------------------------------------------------- #
# 2. Python syntax check — all .py files in scripts/
# --------------------------------------------------------------------------- #
log "Checking Python syntax (all .py files)..."
while IFS= read -r -d '' pyfile; do
  relpath="${pyfile#"$PROJECT_ROOT/"}"
  if python3 -c "import ast; ast.parse(open('$pyfile').read())" 2>/dev/null; then
    ok "$relpath"
  else
    fail "$relpath has syntax errors"
  fi
done < <(find "$PROJECT_ROOT/scripts" "$PROJECT_ROOT/docs" -name '*.py' -print0 2>/dev/null)

# --------------------------------------------------------------------------- #
# 3. Domain knowledge docs
# --------------------------------------------------------------------------- #
log "Checking domain knowledge docs..."
for doc in docs/adapter-specification.md docs/adapter-template.py docs/hf-model-analysis-guide.md docs/review-specification.md; do
  if [[ -s "$PROJECT_ROOT/$doc" ]]; then
    ok "$doc exists and non-empty"
  else
    fail "$doc missing or empty"
  fi
done

# --------------------------------------------------------------------------- #
# 4. Agent definitions have required sections
# --------------------------------------------------------------------------- #
log "Checking agent definitions..."
for agent_file in "$PROJECT_ROOT"/agents/*.md; do
  [[ -f "$agent_file" ]] || continue
  name=$(basename "$agent_file")
  if head -1 "$agent_file" | grep -q '^---'; then
    ok "$name has YAML frontmatter"
  else
    # orchestrator.md doesn't need frontmatter (it's a template, not an agent def)
    if [[ "$name" == "orchestrator.md" ]]; then
      ok "$name is a template (no frontmatter expected)"
    else
      fail "$name missing YAML frontmatter"
    fi
  fi
done

# Check programmer.md has adapter-specific content
if grep -q "Architecture Adapter" "$PROJECT_ROOT/agents/programmer.md" 2>/dev/null; then
  ok "programmer.md has adapter domain knowledge"
else
  fail "programmer.md missing adapter domain knowledge"
fi

# Check the five-phase review process: reviewer.md references the spec, and
# the spec itself contains the phases (content lives in the doc, not the prompt)
if grep -q "review-specification.md" "$PROJECT_ROOT/agents/reviewer.md" 2>/dev/null \
   && grep -q "Phase 1: Factual Verification" "$PROJECT_ROOT/docs/review-specification.md" 2>/dev/null; then
  ok "reviewer.md references five-phase review spec (docs/review-specification.md)"
else
  fail "reviewer.md missing reference to five-phase review spec"
fi

# Check reviewer.md has brief review mode
if grep -q "BRIEF APPROVED" "$PROJECT_ROOT/agents/reviewer.md" 2>/dev/null; then
  ok "reviewer.md has brief review mode"
else
  fail "reviewer.md missing brief review mode"
fi

# --------------------------------------------------------------------------- #
# 5. Signal consistency
# --------------------------------------------------------------------------- #
log "Checking signal consistency..."
source "$PROJECT_ROOT/agents/signals.sh"

# Programmer signals in programmer.md
for sig in "$SIG_BRIEF_READY" "$SIG_PLAN_READY" "$SIG_PHASE_READY" "$SIG_VERIFICATION_COMPLETE" "$SIG_VERIFICATION_FAILED" "$SIG_FINAL_READY"; do
  if grep -qF "$sig" "$PROJECT_ROOT/agents/programmer.md" 2>/dev/null; then
    ok "programmer.md: $sig"
  else
    fail "programmer.md missing signal: $sig"
  fi
done

# Reviewer signals in reviewer.md
for sig in "$SIG_BRIEF_APPROVED" "$SIG_BRIEF_CHANGES" "$SIG_PLAN_APPROVED" "$SIG_PLAN_CHANGES" "$SIG_CODE_APPROVED" "$SIG_CODE_CHANGES"; do
  if grep -qF "$sig" "$PROJECT_ROOT/agents/reviewer.md" 2>/dev/null; then
    ok "reviewer.md: $sig"
  else
    fail "reviewer.md missing signal: $sig"
  fi
done

# --------------------------------------------------------------------------- #
# 6. validate-adapter.sh against existing Llama adapter
# --------------------------------------------------------------------------- #
log "Running validate-adapter.sh against llama..."
if "$PROJECT_ROOT/scripts/validate-adapter.sh" llama --repo "$TL_REPO" >/dev/null 2>&1; then
  ok "validate-adapter.sh passes for llama"
else
  fail "validate-adapter.sh failed for llama"
fi

# --------------------------------------------------------------------------- #
# 7. Project files exist
# --------------------------------------------------------------------------- #
log "Checking project files..."
for f in CLAUDE.md .gitignore .env.example; do
  if [[ -s "$PROJECT_ROOT/$f" ]]; then
    ok "$f exists"
  else
    fail "$f missing or empty"
  fi
done

# --------------------------------------------------------------------------- #
# 8. TransformerLens repo is accessible
# --------------------------------------------------------------------------- #
log "Checking TransformerLens repo..."
if [[ -d "$TL_REPO/transformer_lens/model_bridge/supported_architectures" ]]; then
  ok "TransformerLens repo accessible at $TL_REPO"
  ADAPTER_COUNT=$(find "$TL_REPO/transformer_lens/model_bridge/supported_architectures" -name '*.py' -not -name '__init__*' | wc -l | tr -d ' ')
  ok "Found $ADAPTER_COUNT existing adapters"
else
  fail "TransformerLens repo not found or missing expected directories"
fi

# --------------------------------------------------------------------------- #
# State machine validation
# --------------------------------------------------------------------------- #
log "Validating routing rules in signals.sh..."

# Source signals.sh to get the protocol block, then validate:
# - Every programmer signal has at least one routing rule that handles it
# - Every reviewer signal has at least one routing rule that handles it
# - No signal variable in signals.sh is absent from the routing rules
source "$PROJECT_ROOT/agents/signals.sh" 2>/dev/null

PROTOCOL=$(generate_protocol_block 2>/dev/null || echo "")
if [[ -z "$PROTOCOL" ]]; then
  fail "generate_protocol_block() returned empty output"
else
  # Check that every signal defined at the top of signals.sh appears in routing
  for sig_var in SIG_BRIEF_READY SIG_PLAN_READY SIG_PHASE_READY SIG_VERIFICATION_COMPLETE \
                 SIG_VERIFICATION_FAILED SIG_VERIFICATION_SKIPPED SIG_FINAL_READY \
                 SIG_BRIEF_APPROVED SIG_BRIEF_CHANGES SIG_PLAN_APPROVED SIG_PLAN_CHANGES \
                 SIG_CODE_APPROVED SIG_CODE_CHANGES; do
    # Extract the key phrase from the signal value (e.g. "BRIEF READY FOR REVIEW")
    sig_value="${!sig_var}"
    # Strip the "PROGRAMMER: " or "REVIEWER: " prefix to get the routing-rule keyword
    keyword=$(echo "$sig_value" | sed 's/^PROGRAMMER: //; s/^REVIEWER: //')
    if echo "$PROTOCOL" | grep -qF "$keyword"; then
      ok "Signal $sig_var ('$keyword') has a routing rule"
    else
      fail "Signal $sig_var ('$keyword') has NO routing rule in the protocol block"
    fi
  done

  # Check that routing rules reference only defined signals (no typos)
  rule_signals=$(echo "$PROTOCOL" | grep -oE '`[A-Z][A-Z ]+`' | tr -d '`' | sort -u)
  while IFS= read -r rule_sig; do
    [[ -z "$rule_sig" ]] && continue
    # Check if any SIG_* variable contains this phrase
    found=false
    for sig_var in SIG_BRIEF_READY SIG_PLAN_READY SIG_PHASE_READY SIG_VERIFICATION_COMPLETE \
                   SIG_VERIFICATION_FAILED SIG_VERIFICATION_SKIPPED SIG_FINAL_READY \
                   SIG_BRIEF_APPROVED SIG_BRIEF_CHANGES SIG_PLAN_APPROVED SIG_PLAN_CHANGES \
                   SIG_CODE_APPROVED SIG_CODE_CHANGES; do
      if [[ "${!sig_var}" == *"$rule_sig"* ]]; then
        found=true
        break
      fi
    done
    if [[ "$found" == false ]]; then
      # Might be a partial match or a combined signal — only fail if it looks
      # like a full signal (starts with a capital letter, >10 chars)
      if [[ ${#rule_sig} -gt 10 ]]; then
        fail "Routing rule references '$rule_sig' which matches no defined signal"
      fi
    fi
  done <<< "$rule_signals"
fi

# --------------------------------------------------------------------------- #
# Solo-mode coordinator: routing table + signal consumption
# --------------------------------------------------------------------------- #
log "Validating solo-coordinator routing table..."

COORDINATOR="$PROJECT_ROOT/agents/solo-coordinator.sh"
_route_expect() {
  local sig="$1" expected="$2" got
  got=$("$COORDINATOR" --route "$sig" 2>/dev/null | cut -f1)
  if [[ "$got" == "$expected" ]]; then
    ok "route: $sig → $expected"
  else
    fail "route: $sig → '$got' (expected '$expected')"
  fi
}

_route_expect brief-ready              reviewer
_route_expect plan-ready               reviewer
_route_expect phase-A-ready            reviewer
_route_expect phase-A+B-ready          reviewer
_route_expect verification-complete    reviewer
_route_expect verification-failed      reviewer
_route_expect brief-approved           programmer
_route_expect brief-changes-2          programmer
_route_expect plan-approved            programmer
_route_expect plan-changes-1           programmer
_route_expect phase-A-approved         programmer
_route_expect phase-A+B-changes-3      programmer
_route_expect final-approved           programmer
_route_expect verification-skipped     notify-human
_route_expect stuck                    notify-human
_route_expect done                     complete
_route_expect no-such-signal           unknown

log "Validating solo-coordinator signal consumption (--once)..."

TEST_WT=$(mktemp -d)
mkdir -p "$TEST_WT/.adapter-workspace/signals"
touch "$TEST_WT/.adapter-workspace/signals/brief-ready"
sleep 1  # distinct mtimes so causal (oldest-first) ordering is observable
touch "$TEST_WT/.adapter-workspace/signals/plan-ready"

ONCE_OUT=$("$COORDINATOR" --once "$TEST_WT" 2>/dev/null || true)

# Both signals routed, oldest first
if [[ $(echo "$ONCE_OUT" | wc -l | tr -d ' ') == "2" ]] \
   && [[ $(echo "$ONCE_OUT" | sed -n 1p | cut -f1,2) == $'brief-ready\treviewer' ]] \
   && [[ $(echo "$ONCE_OUT" | sed -n 2p | cut -f1,2) == $'plan-ready\treviewer' ]]; then
  ok "--once routes pending signals oldest-first"
else
  fail "--once output unexpected: $ONCE_OUT"
fi

# Signals consumed (dir empty) and archived
if ls "$TEST_WT/.adapter-workspace/signals" | grep -q .; then
  fail "--once left unconsumed signals behind"
else
  ok "--once consumed all pending signals"
fi
archived=$(ls "$TEST_WT/.adapter-workspace/signals/.archive" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$archived" == "2" ]]; then
  ok "--once archived both signals"
else
  fail "--once archive has $archived entries (expected 2)"
fi

# Re-running is a no-op: consumption means no stale re-delivery
ONCE_AGAIN=$("$COORDINATOR" --once "$TEST_WT" 2>/dev/null || true)
if [[ -z "$ONCE_AGAIN" ]]; then
  ok "re-scan after consumption routes nothing (no stale signals)"
else
  fail "re-scan re-delivered consumed signals: $ONCE_AGAIN"
fi

# check-signals.sh reflects the archive
CHECK_OUT=$(cd "$TEST_WT" && "$PROJECT_ROOT/agents/check-signals.sh" 2>/dev/null || true)
if echo "$CHECK_OUT" | grep -q "brief-ready" && echo "$CHECK_OUT" | grep -q "plan-ready"; then
  ok "check-signals.sh reports processed signals"
else
  fail "check-signals.sh missing processed signals: $CHECK_OUT"
fi

rm -rf "$TEST_WT"

# Solo prompts must not contain blocking waits
log "Checking solo prompts are non-blocking..."
for f in solo-programmer.md solo-reviewer.md; do
  if grep -q "wait-for-signal" "$PROJECT_ROOT/agents/$f"; then
    fail "$f still references wait-for-signal (blocking waits are forbidden)"
  else
    ok "$f has no blocking wait references"
  fi
  if grep -q "end your turn" "$PROJECT_ROOT/agents/$f"; then
    ok "$f instructs signal-then-end-turn"
  else
    fail "$f missing signal-then-end-turn instruction"
  fi
done

# Launcher must start the coordinator, not the teams watcher
if grep -q "solo-coordinator.sh" "$PROJECT_ROOT/agents/launch-solo-pair.sh"; then
  ok "launch-solo-pair.sh starts solo-coordinator.sh"
else
  fail "launch-solo-pair.sh does not start solo-coordinator.sh"
fi

# Summary
echo ""
echo "=========================================="
if [[ $FAILURES -eq 0 ]]; then
  echo -e "  \033[1;32mAll dry-run tests passed!\033[0m"
else
  echo -e "  \033[1;31m${FAILURES} test(s) failed\033[0m"
fi
echo "=========================================="
echo ""

exit $FAILURES
