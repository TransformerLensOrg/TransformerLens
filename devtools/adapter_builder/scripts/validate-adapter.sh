#!/usr/bin/env bash
# =============================================================================
# validate-adapter.sh
#
# Validates that a generated Architecture Adapter is correctly structured and
# registered in the TransformerLens codebase.
#
# Usage:
#   ./scripts/validate-adapter.sh <adapter_module_name> [--repo <path>] [--model <hf_model_id>]
#
# Example:
#   ./scripts/validate-adapter.sh llama                                    # structural only
#   ./scripts/validate-adapter.sh llama --model meta-llama/Llama-3.2-1B    # + deep validation
#   ./scripts/validate-adapter.sh cohere --repo /path/to/TransformerLens
# =============================================================================

set -euo pipefail

log()  { echo -e "\033[1;34m[check]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[  ok ]\033[0m $*"; }
fail() { echo -e "\033[1;31m[ FAIL]\033[0m $*"; FAILURES=$((FAILURES + 1)); }

FAILURES=0
ADAPTER_NAME=""
HF_MODEL_ID=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${TL_REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || { cd "$SCRIPT_DIR/../../.." && pwd; })}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_ROOT="$2"; shift 2 ;;
    --model) HF_MODEL_ID="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 <adapter_module_name> [--repo <path>] [--model <hf_model_id>]"
      echo "Example: $0 llama --model meta-llama/Llama-3.2-1B"
      exit 0
      ;;
    *) ADAPTER_NAME="$1"; shift ;;
  esac
done

[[ -z "$ADAPTER_NAME" ]] && { echo "Error: adapter module name required"; exit 1; }

ADAPTERS_DIR="$REPO_ROOT/transformer_lens/model_bridge/supported_architectures"
ADAPTER_FILE="$ADAPTERS_DIR/${ADAPTER_NAME}.py"
INIT_FILE="$ADAPTERS_DIR/__init__.py"
FACTORY_FILE="$REPO_ROOT/transformer_lens/factories/architecture_adapter_factory.py"

echo ""
echo "=========================================="
echo "  Validating adapter: ${ADAPTER_NAME}"
echo "=========================================="
echo ""

# 1. Check adapter file exists
log "Checking adapter file exists..."
if [[ -f "$ADAPTER_FILE" ]]; then
  ok "Found: $ADAPTER_FILE"
else
  fail "Adapter file not found: $ADAPTER_FILE"
fi

# 2. Check class definition
log "Checking class definition..."
CLASS_PATTERN="class.*ArchitectureAdapter"
if grep -qE "$CLASS_PATTERN" "$ADAPTER_FILE" 2>/dev/null; then
  CLASS_NAME=$(grep -oE "class \w+ArchitectureAdapter" "$ADAPTER_FILE" | head -1 | sed 's/class //')
  ok "Found class: $CLASS_NAME"
else
  fail "No ArchitectureAdapter class found in $ADAPTER_FILE"
  CLASS_NAME=""
fi

# 3. Check extends ArchitectureAdapter
log "Checking inheritance..."
if grep -qE "class.*\(ArchitectureAdapter\)" "$ADAPTER_FILE" 2>/dev/null; then
  ok "Extends ArchitectureAdapter"
else
  fail "Class does not extend ArchitectureAdapter"
fi

# 4. Check component_mapping is set
log "Checking component_mapping..."
if grep -q "self.component_mapping" "$ADAPTER_FILE" 2>/dev/null; then
  ok "component_mapping is set"
else
  fail "component_mapping not found in adapter"
fi

# 5. Check weight_processing_conversions is set
log "Checking weight_processing_conversions..."
if grep -q "self.weight_processing_conversions" "$ADAPTER_FILE" 2>/dev/null; then
  ok "weight_processing_conversions is set"
else
  fail "weight_processing_conversions not found in adapter"
fi

# 6. Check registered in __init__.py
log "Checking __init__.py registration..."
if [[ -n "$CLASS_NAME" ]]; then
  if grep -q "$CLASS_NAME" "$INIT_FILE" 2>/dev/null; then
    ok "Found in __init__.py"
  else
    fail "$CLASS_NAME not found in $INIT_FILE"
  fi

  if grep -q "$CLASS_NAME" "$INIT_FILE" 2>/dev/null && grep -q "__all__" "$INIT_FILE" 2>/dev/null; then
    if grep -A 200 "__all__" "$INIT_FILE" | grep -q "\"$CLASS_NAME\""; then
      ok "Found in __all__ list"
    else
      fail "$CLASS_NAME not found in __all__ list"
    fi
  fi
fi

# 7. Check registered in factory
log "Checking factory registration..."
if [[ -n "$CLASS_NAME" ]]; then
  if grep -q "$CLASS_NAME" "$FACTORY_FILE" 2>/dev/null; then
    ok "Found in architecture_adapter_factory.py"
  else
    fail "$CLASS_NAME not found in $FACTORY_FILE"
  fi
fi

# 8. Check config attributes are set — either explicitly (exotic archs) or via
# a base-class defaults helper like _set_rms_rotary_defaults() (the common case)
log "Checking config attributes..."
if grep -qE "self\.cfg\.(normalization_type|positional_embedding_type)|self\._set_[a-z_]*defaults\(" "$ADAPTER_FILE" 2>/dev/null; then
  ok "Sets normalization/positional config (defaults helper or explicit)"
else
  fail "Does not set normalization/positional config (no _set_*_defaults() call or explicit self.cfg assignment)"
fi

# 9. Run adapter-related tests (if pytest is available)
log "Running adapter tests..."
if command -v uv &>/dev/null && [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
  # Run a quick import test to verify the adapter can be loaded
  TEST_OUTPUT=$(cd "$REPO_ROOT" && uv run python -c "
from transformer_lens.model_bridge.supported_architectures import ${CLASS_NAME}
print('Import OK: ${CLASS_NAME}')
" 2>&1) && {
    ok "Import test passed: $CLASS_NAME"
  } || {
    fail "Import test failed: $TEST_OUTPUT"
  }

  # Run any existing tests matching the adapter name
  ADAPTER_TEST_FILE="$REPO_ROOT/tests/unit/test_${ADAPTER_NAME}_adapter.py"
  ADAPTER_TEST_FILE2="$REPO_ROOT/tests/unit/architecture_adapters/test_${ADAPTER_NAME}.py"
  if [[ -f "$ADAPTER_TEST_FILE" ]]; then
    log "Found test file: $ADAPTER_TEST_FILE"
    TEST_RESULT=$(cd "$REPO_ROOT" && uv run pytest "$ADAPTER_TEST_FILE" -x -q 2>&1) && {
      ok "Adapter tests passed"
    } || {
      fail "Adapter tests failed:\n$TEST_RESULT"
    }
  elif [[ -f "$ADAPTER_TEST_FILE2" ]]; then
    log "Found test file: $ADAPTER_TEST_FILE2"
    TEST_RESULT=$(cd "$REPO_ROOT" && uv run pytest "$ADAPTER_TEST_FILE2" -x -q 2>&1) && {
      ok "Adapter tests passed"
    } || {
      fail "Adapter tests failed:\n$TEST_RESULT"
    }
  else
    log "No adapter-specific test file found (checked test_${ADAPTER_NAME}_adapter.py and test_${ADAPTER_NAME}.py) — skipping."
  fi
else
  log "uv not available or no pyproject.toml — skipping test execution."
fi

# 10. Deep validation against a real model (if --model provided)
if [[ -n "$HF_MODEL_ID" && -n "$CLASS_NAME" ]]; then
  log "Running deep validation against model: $HF_MODEL_ID"

  # Find the architecture class name from the factory
  ARCH_CLASS=$(grep -B1 "$CLASS_NAME" "$FACTORY_FILE" 2>/dev/null | grep -oE '"[^"]+ForCausalLM"|"[^"]+ForConditionalGeneration"|"[^"]+ForMaskedLM"|"[^"]+LMHeadModel"' | head -1 | tr -d '"')

  if [[ -z "$ARCH_CLASS" ]]; then
    log "Could not determine architecture class from factory — using class name heuristic."
    # Try common patterns from the factory
    ARCH_CLASS=$(grep "$CLASS_NAME" "$FACTORY_FILE" 2>/dev/null | grep -oE '"[^"]+"' | head -1 | tr -d '"')
  fi

  if [[ -n "$ARCH_CLASS" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    DEEP_RESULT=$(cd "$REPO_ROOT" && uv run python "$SCRIPT_DIR/validate-adapter-deep.py" "$ARCH_CLASS" "$HF_MODEL_ID" --repo "$REPO_ROOT" 2>&1) && {
      ok "Deep validation passed"
      echo "$DEEP_RESULT" | tail -20
    } || {
      fail "Deep validation found issues"
      echo "$DEEP_RESULT" | tail -30
    }
  else
    log "Could not determine architecture class — skipping deep validation."
  fi
elif [[ -n "$HF_MODEL_ID" ]]; then
  log "Skipping deep validation — no class name available."
fi

# Summary
echo ""
echo "=========================================="
if [[ $FAILURES -eq 0 ]]; then
  echo -e "  \033[1;32mAll checks passed!\033[0m"
else
  echo -e "  \033[1;31m${FAILURES} check(s) failed\033[0m"
fi
echo "=========================================="
echo ""

exit $FAILURES
