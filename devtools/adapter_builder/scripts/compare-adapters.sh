#!/usr/bin/env bash
# =============================================================================
# compare-adapters.sh — Structured diff between two existing adapters
#
# Shows config attribute differences, component mapping differences, and
# weight conversion differences between two adapters. Designed to support
# the "find nearest reference adapter and adapt" workflow.
#
# Usage:
#   ./scripts/compare-adapters.sh <adapter_a> <adapter_b> [--repo <path>]
#
# Example:
#   ./scripts/compare-adapters.sh gemma1 gemma2
#   ./scripts/compare-adapters.sh llama qwen2
#   ./scripts/compare-adapters.sh gpt2 gptj --repo /path/to/TransformerLens
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${TL_REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || { cd "$SCRIPT_DIR/../../.." && pwd; })}"
ADAPTER_A=""
ADAPTER_B=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_ROOT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 <adapter_a> <adapter_b> [--repo <path>]"
      echo "Example: $0 gemma1 gemma2"
      exit 0
      ;;
    *)
      if [[ -z "$ADAPTER_A" ]]; then
        ADAPTER_A="$1"
      elif [[ -z "$ADAPTER_B" ]]; then
        ADAPTER_B="$1"
      fi
      shift
      ;;
  esac
done

[[ -z "$ADAPTER_A" || -z "$ADAPTER_B" ]] && { echo "Error: two adapter names required"; exit 1; }

ADAPTERS_DIR="$REPO_ROOT/transformer_lens/model_bridge/supported_architectures"
FILE_A="$ADAPTERS_DIR/${ADAPTER_A}.py"
FILE_B="$ADAPTERS_DIR/${ADAPTER_B}.py"

[[ -f "$FILE_A" ]] || { echo "Error: $FILE_A not found"; exit 1; }
[[ -f "$FILE_B" ]] || { echo "Error: $FILE_B not found"; exit 1; }

echo ""
echo "================================================================"
echo "  Adapter Comparison: ${ADAPTER_A} vs ${ADAPTER_B}"
echo "================================================================"

# --- Config Attributes ---
echo ""
echo -e "\033[1;36m--- Config Attributes ---\033[0m"
echo ""

CONFIG_ATTRS=(
  normalization_type
  positional_embedding_type
  final_rms
  gated_mlp
  attn_only
  uses_rms_norm
  eps_attr
  default_prepend_bos
  uses_split_attention
  split_attention_weights
)

printf "  %-30s  %-25s  %-25s\n" "Attribute" "$ADAPTER_A" "$ADAPTER_B"
printf "  %-30s  %-25s  %-25s\n" "------------------------------" "-------------------------" "-------------------------"

for attr in "${CONFIG_ATTRS[@]}"; do
  val_a=$(grep "self\.cfg\.$attr" "$FILE_A" 2>/dev/null | head -1 | sed 's/.*=\s*//' | tr -d ' "'"'"'
' || echo "—")
  val_b=$(grep "self\.cfg\.$attr" "$FILE_B" 2>/dev/null | head -1 | sed 's/.*=\s*//' | tr -d ' "'"'"'
' || echo "—")
  [[ -z "$val_a" ]] && val_a="—"
  [[ -z "$val_b" ]] && val_b="—"
  if [[ "$val_a" == "$val_b" ]]; then
    printf "  %-30s  %-25s  %-25s\n" "$attr" "$val_a" "(same)"
  else
    printf "  %-30s  \033[33m%-25s\033[0m  \033[33m%-25s\033[0m\n" "$attr" "$val_a" "$val_b"
  fi
done

# --- Imports (Bridge Components Used) ---
echo ""
echo -e "\033[1;36m--- Bridge Components Used ---\033[0m"
echo ""

imports_a=$(grep -oE '[A-Z][A-Za-z]*Bridge' "$FILE_A" 2>/dev/null | sort -u)
imports_b=$(grep -oE '[A-Z][A-Za-z]*Bridge' "$FILE_B" 2>/dev/null | sort -u)

# Find differences
only_a=$(comm -23 <(echo "$imports_a") <(echo "$imports_b"))
only_b=$(comm -13 <(echo "$imports_a") <(echo "$imports_b"))
both=$(comm -12 <(echo "$imports_a") <(echo "$imports_b"))

for bridge in $both; do
  echo "  [both]     $bridge"
done
for bridge in $only_a; do
  echo -e "  \033[33m[${ADAPTER_A} only]\033[0m  $bridge"
done
for bridge in $only_b; do
  echo -e "  \033[33m[${ADAPTER_B} only]\033[0m  $bridge"
done

# --- Component Mapping Paths ---
echo ""
echo -e "\033[1;36m--- Component Mapping (HF paths) ---\033[0m"
echo ""

# Extract name="..." from component mapping
extract_paths() {
  local file="$1"
  grep -oE 'name="[^"]*"' "$file" 2>/dev/null | sed 's/name="//;s/"//' | sort
}

paths_a=$(extract_paths "$FILE_A")
paths_b=$(extract_paths "$FILE_B")

only_pa=$(comm -23 <(echo "$paths_a") <(echo "$paths_b"))
only_pb=$(comm -13 <(echo "$paths_a") <(echo "$paths_b"))
both_p=$(comm -12 <(echo "$paths_a") <(echo "$paths_b"))

for p in $both_p; do
  echo "  [both]     $p"
done
for p in $only_pa; do
  echo -e "  \033[33m[${ADAPTER_A} only]\033[0m  $p"
done
for p in $only_pb; do
  echo -e "  \033[33m[${ADAPTER_B} only]\033[0m  $p"
done

# --- Weight Conversions ---
echo ""
echo -e "\033[1;36m--- Weight Processing Conversions ---\033[0m"
echo ""

extract_conversions() {
  local file="$1"
  grep -oE '"blocks\.\{i\}\.[^"]*"|"[a-z_]+\.[a-z_]+"' "$file" 2>/dev/null | tr -d '"' | sort -u
}

conv_a=$(extract_conversions "$FILE_A")
conv_b=$(extract_conversions "$FILE_B")

only_ca=$(comm -23 <(echo "$conv_a") <(echo "$conv_b"))
only_cb=$(comm -13 <(echo "$conv_a") <(echo "$conv_b"))
both_c=$(comm -12 <(echo "$conv_a") <(echo "$conv_b"))

for c in $both_c; do
  echo "  [both]     $c"
done
for c in $only_ca; do
  echo -e "  \033[33m[${ADAPTER_A} only]\033[0m  $c"
done
for c in $only_cb; do
  echo -e "  \033[33m[${ADAPTER_B} only]\033[0m  $c"
done

# --- Optional Overrides ---
echo ""
echo -e "\033[1;36m--- Optional Overrides ---\033[0m"
echo ""

OVERRIDES=(
  "setup_component_testing"
  "preprocess_weights"
  "prepare_loading"
  "prepare_model"
)

for override in "${OVERRIDES[@]}"; do
  label_a=$(grep -q "def $override" "$FILE_A" 2>/dev/null && echo "yes" || echo "—")
  label_b=$(grep -q "def $override" "$FILE_B" 2>/dev/null && echo "yes" || echo "—")
  if [[ "$label_a" == "$label_b" ]]; then
    printf "  %-30s  %-10s  %-10s\n" "$override" "$label_a" "(same)"
  else
    printf "  %-30s  \033[33m%-10s\033[0m  \033[33m%-10s\033[0m\n" "$override" "$label_a" "$label_b"
  fi
done

# --- Line Count ---
echo ""
echo -e "\033[1;36m--- Size ---\033[0m"
echo ""
lines_a=$(wc -l < "$FILE_A" | tr -d ' ')
lines_b=$(wc -l < "$FILE_B" | tr -d ' ')
echo "  ${ADAPTER_A}: ${lines_a} lines"
echo "  ${ADAPTER_B}: ${lines_b} lines"

# --- Full Diff (collapsed) ---
echo ""
echo -e "\033[1;36m--- Full Diff ---\033[0m"
echo ""
diff --color=always -u "$FILE_A" "$FILE_B" 2>/dev/null | head -80 || true
TOTAL_DIFF=$(diff -u "$FILE_A" "$FILE_B" 2>/dev/null | wc -l | tr -d ' \n')
if (( TOTAL_DIFF > 80 )); then
  echo ""
  echo "  ... ($((TOTAL_DIFF - 80)) more lines — run 'diff -u $FILE_A $FILE_B' for full diff)"
fi

echo ""
echo "================================================================"
echo ""
