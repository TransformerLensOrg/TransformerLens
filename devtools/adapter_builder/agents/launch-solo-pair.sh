#!/usr/bin/env bash
# =============================================================================
# launch-solo-pair.sh
#
# Launches two independent Claude Code sessions sharing a worktree:
#   - Programmer: builds the adapter (analysis, planning, coding, verification)
#   - Reviewer: reviews at each checkpoint (brief, plan, phases, final)
#
# The two sessions communicate through signal files in
# .adapter-workspace/signals/, routed by solo-coordinator.sh — a daemon that
# consumes each signal and wakes the counterpart pane with an injected
# message. Agents never block or poll. No agent-teams, no
# CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS. Works on Pro/Team tier.
#
# The same hooks, scripts, progress tracking, and memory lock infrastructure
# from the agent-teams mode apply — only the coordination mechanism changes.
#
# Usage:
#   ./agents/launch-solo-pair.sh --architecture <class> [options]
#
# Options:
#     [--seed-model <hf-id>]       Known model for scaffolding
#     [--target-repo <path>]       Path to TransformerLens repo (default from .env)
#     [--base-branch <branch>]     Base branch (default from .env)
#     [--new-branch  <branch>]     Feature branch (default: feature/<arch>-adapter)
#     [--max-memory <gb>]          Memory limit in GB (default from .env)
#     [--worktree-dir <path>]      Where to create the worktree
#     [--skip-arch-check]          Skip architecture existence check
#     [--auto-approve]             Skip Claude Code permission prompts
#     [--dry-run]                  Run all pre-flight checks, then exit without
#                                  creating a worktree or launching sessions
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/.logs"

TL_LOG_TAG="solo"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

TL_CMD="${TL_CMD:-$0}"

usage() {
  grep '^#' "$0" | sed 's/^# \{0,2\}//' | tail -n +2
  exit 1
}

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
TARGET_REPO=""
ARCHITECTURE=""
SEED_MODEL=""
BASE_BRANCH=""
NEW_BRANCH=""
MAX_MEMORY_GB=""
WORKTREE_DIR=""
AUTO_APPROVE=false
SKIP_ARCH_CHECK=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-repo)        TARGET_REPO="$2";        shift 2 ;;
    --architecture)       ARCHITECTURE="$2";       shift 2 ;;
    --seed-model)         SEED_MODEL="$2";         shift 2 ;;
    --base-branch)        BASE_BRANCH="$2";        shift 2 ;;
    --new-branch)         NEW_BRANCH="$2";         shift 2 ;;
    --max-memory)         MAX_MEMORY_GB="$2";      shift 2 ;;
    --worktree-dir)       WORKTREE_DIR="$2";       shift 2 ;;
    --skip-arch-check)    SKIP_ARCH_CHECK=true;     shift   ;;
    --auto-approve)       AUTO_APPROVE=true;        shift   ;;
    --dry-run)            DRY_RUN=true;             shift   ;;
    -h|--help)            usage ;;
    *) err "Unknown argument: $1" ;;
  esac
done

# Load defaults from .env
ENV_FILE="$PROJECT_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

# The builder lives in devtools/adapter_builder inside TransformerLens, so the
# containing repo is the default target. --target-repo / DEFAULT_TARGET_REPO
# remain as overrides for driving a different checkout.
TARGET_REPO="${TARGET_REPO:-${DEFAULT_TARGET_REPO:-}}"
if [[ -z "$TARGET_REPO" ]]; then
  TARGET_REPO="$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel 2>/dev/null)" \
    || err "Could not derive the TransformerLens repo root (pass --target-repo or set DEFAULT_TARGET_REPO in .env)."
fi
[[ -z "$ARCHITECTURE" ]] && err "--architecture is required"

BASE_BRANCH="${BASE_BRANCH:-${DEFAULT_BASE_BRANCH:-dev-4.x}}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-${DEFAULT_MAX_MEMORY_GB:-96}}"

if [[ -z "$NEW_BRANCH" ]]; then
  ARCH_SHORT=$(echo "$ARCHITECTURE" | sed -E 's/(For(Causal|Conditional|Masked).*|LMHead.*)$//' | tr '[:upper:]' '[:lower:]')
  NEW_BRANCH="feature/${ARCH_SHORT}-adapter"
fi

# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
require_cmd git
require_cmd claude
require_cmd tmux
require_cmd jq

REPO_ROOT="$(cd "$TARGET_REPO" && git rev-parse --show-toplevel 2>/dev/null)" \
  || err "--target-repo '$TARGET_REPO' is not a git repository."

WORKTREE_BASE="${WORKTREE_BASE:-$(dirname "$REPO_ROOT")/worktrees}"

# --------------------------------------------------------------------------- #
# Architecture validation
# --------------------------------------------------------------------------- #
if [[ -n "$SEED_MODEL" && "$SKIP_ARCH_CHECK" == false ]]; then
  log "Seed model '${SEED_MODEL}' provided — skipping architecture existence check."
  SKIP_ARCH_CHECK=true
fi

# The worktree is created from BASE_BRANCH, so the "already supported" check
# must read the factory file from that branch — not the main checkout's
# working tree, which may differ (e.g. golden-master test branches with the
# adapter stripped). Fall back to the on-disk file if the branch ref can't
# be read; the branch is hard-validated later.
FACTORY_REL="transformer_lens/factories/architecture_adapter_factory.py"
FACTORY_CONTENT="$(git -C "$REPO_ROOT" show "${BASE_BRANCH}:${FACTORY_REL}" 2>/dev/null \
  || cat "$REPO_ROOT/$FACTORY_REL" 2>/dev/null || true)"
if [[ -n "$FACTORY_CONTENT" ]] && grep -q "\"${ARCHITECTURE}\"" <<< "$FACTORY_CONTENT"; then
  ok "Architecture '${ARCHITECTURE}' already has an adapter on branch '${BASE_BRANCH}'."
  log "No work needed. Exiting."
  exit 0
fi

if [[ "$SKIP_ARCH_CHECK" == true ]]; then
  warn "Skipping architecture existence check."
else
  log "Validating '${ARCHITECTURE}'..."
  set +e
  env -u VIRTUAL_ENV uv run --project "$REPO_ROOT" \
    python "$PROJECT_ROOT/scripts/validate-architecture.py" "$ARCHITECTURE"
  validate_exit=$?
  set -e
  case "$validate_exit" in
    0) ok "Architecture validated." ;;
    1) err "Architecture '${ARCHITECTURE}' does not exist. Check spelling or use --skip-arch-check." ;;
    2) warn "Could not fully verify '${ARCHITECTURE}'. Proceeding." ;;
  esac
fi

# --------------------------------------------------------------------------- #
# Worktree setup
# --------------------------------------------------------------------------- #
if [[ -z "$WORKTREE_DIR" ]]; then
  SAFE_BRANCH=$(echo "$NEW_BRANCH" | tr '/' '-')
  WORKTREE_DIR="${WORKTREE_BASE}/${SAFE_BRANCH}"
fi

git -C "$REPO_ROOT" rev-parse --verify "$BASE_BRANCH" &>/dev/null \
  || err "Base branch '$BASE_BRANCH' does not exist."

if [[ "$DRY_RUN" == true ]]; then
  ok "Pre-flight passed (dry run) — no worktree created, no sessions launched."
  log "  would create : $WORKTREE_DIR (branch $NEW_BRANCH from $BASE_BRANCH)"
  log "  architecture : $ARCHITECTURE"
  log "  seed model   : ${SEED_MODEL:-none}"
  exit 0
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/progress.sh"

REUSE_WORKTREE=false
if [[ -d "$WORKTREE_DIR" ]] && git -C "$REPO_ROOT" rev-parse --verify "$NEW_BRANCH" &>/dev/null; then
  ok "Existing worktree found — resuming."
  REUSE_WORKTREE=true
else
  log "Creating worktree at: $WORKTREE_DIR"

  _cleanup_partial_worktree() {
    warn "Cleaning up partial worktree..."
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
    git -C "$REPO_ROOT" branch -D "$NEW_BRANCH" 2>/dev/null || true
  }

  if ! git -C "$REPO_ROOT" worktree add -b "$NEW_BRANCH" "$WORKTREE_DIR" "$BASE_BRANCH" 2>/tmp/tl-wt-err.$$; then
    wt_err=$(cat /tmp/tl-wt-err.$$ 2>/dev/null); rm -f /tmp/tl-wt-err.$$
    err "Failed to create worktree: $wt_err"
  fi
  rm -f /tmp/tl-wt-err.$$
  trap '_cleanup_partial_worktree' ERR

  ok "Worktree created."
  progress_init "$WORKTREE_DIR" "$ARCHITECTURE"
  ok "Progress tracking initialized."
  trap - ERR
fi

# --------------------------------------------------------------------------- #
# Workspace and signals directory
# --------------------------------------------------------------------------- #
mkdir -p "$WORKTREE_DIR/.adapter-workspace/reviews"
mkdir -p "$WORKTREE_DIR/.adapter-workspace/phase-reports"
mkdir -p "$WORKTREE_DIR/.adapter-workspace/signals"
touch "$WORKTREE_DIR/.adapter-workspace/adapter-brief.md" 2>/dev/null || true
touch "$WORKTREE_DIR/.adapter-workspace/adapter-plan.md" 2>/dev/null || true
touch "$WORKTREE_DIR/.adapter-workspace/adapter-scaffold.py" 2>/dev/null || true
touch "$WORKTREE_DIR/.adapter-workspace/timeline.jsonl" 2>/dev/null || true

# --------------------------------------------------------------------------- #
# Install hooks (same as agent-teams mode)
# --------------------------------------------------------------------------- #
SETTINGS_FILE="$WORKTREE_DIR/.claude/settings.json"
mkdir -p "$WORKTREE_DIR/.claude"

_hook_cmd() { echo "'$SCRIPT_DIR/hooks/$1'"; }

TIMELINE_HOOK=$(_hook_cmd timeline-capture.sh)
GUARD_HT_HOOK=$(_hook_cmd guard-hooked-transformer.sh)
GUARD_GIT_HOOK=$(_hook_cmd guard-git.sh)
GUARD_REVIEW_ROUNDS_HOOK=$(_hook_cmd guard-review-rounds.sh)
GUARD_VERIFY_MODELS_HOOK=$(_hook_cmd guard-verify-models.sh)
GATE_LINT_HOOK=$(_hook_cmd gate-lint-checks.sh)
NOTIFY_HOOK=$(_hook_cmd notify-on-completion.sh)

_h() { jq -n --arg cmd "$1" '{"type":"command","command":$cmd}'; }

jq -n \
  --argjson session_start  "[{\"hooks\":[  $(_h "$TIMELINE_HOOK")  ]}]" \
  --argjson session_end    "[{\"hooks\":[  $(_h "$TIMELINE_HOOK"), $(_h "$NOTIFY_HOOK")  ]}]" \
  --argjson pre_edit       "{\"matcher\":\"Edit|Write|MultiEdit|NotebookEdit\",\"hooks\":[ $(_h "$GUARD_HT_HOOK"), $(_h "$GUARD_REVIEW_ROUNDS_HOOK") ]}" \
  --argjson pre_bash       "{\"matcher\":\"Bash\",\"hooks\":[ $(_h "$GUARD_GIT_HOOK"), $(_h "$GUARD_VERIFY_MODELS_HOOK") ]}" \
  --argjson pre_all        "{\"matcher\":\"\",\"hooks\":[ $(_h "$TIMELINE_HOOK") ]}" \
  --argjson post_all       "[{\"matcher\":\"\",\"hooks\":[ $(_h "$TIMELINE_HOOK") ]}]" \
  --argjson stop           "[{\"hooks\":[ $(_h "$GATE_LINT_HOOK") ]}]" \
  '{hooks:{
    SessionStart: $session_start,
    SessionEnd: $session_end,
    PreToolUse: [$pre_edit, $pre_bash, $pre_all],
    PostToolUse: $post_all,
    Stop: $stop
  }}' > "$SETTINGS_FILE"

ok "Hooks installed."

# --------------------------------------------------------------------------- #
# Pre-seed workspace trust for the worktree
#
# Claude Code shows a "Do you trust this folder?" prompt on first launch in a
# new directory. `--dangerously-skip-permissions` does NOT bypass it (only
# `-p`/print mode does). A fresh worktree is an unknown path, so without
# pre-seeding, both panes hang at the trust prompt forever and coordination
# never starts. We atomically merge a trust entry into ~/.claude.json.
# --------------------------------------------------------------------------- #
CLAUDE_JSON="$HOME/.claude.json"
if [[ -f "$CLAUDE_JSON" ]]; then
  TRUST_TMP=$(mktemp)
  jq --arg path "$WORKTREE_DIR" '
    .projects //= {} |
    .projects[$path] = ((.projects[$path] // {}) + {
      hasTrustDialogAccepted: true,
      hasClaudeMdExternalIncludesApproved: true,
      hasClaudeMdExternalIncludesWarningShown: true,
      projectOnboardingSeenCount: 1
    })
  ' "$CLAUDE_JSON" > "$TRUST_TMP" && mv "$TRUST_TMP" "$CLAUDE_JSON"
  ok "Workspace trust pre-seeded for $WORKTREE_DIR"
else
  warn "~/.claude.json not found — trust dialog may appear in tmux panes."
fi

# --------------------------------------------------------------------------- #
# Build prompts
# --------------------------------------------------------------------------- #
SEED_MODEL_BLOCK=""
if [[ -n "$SEED_MODEL" ]]; then
  SEED_MODEL_BLOCK="
Seed model: ${SEED_MODEL}
Use this model for scaffold generation and as the primary HF source reference.
Include it in your verification targets."
fi

PROGRAMMER_TASK="Create an Architecture Adapter for: ${ARCHITECTURE}
${SEED_MODEL_BLOCK}
Memory limit: ${MAX_MEMORY_GB}GB"

REVIEWER_TASK="Review the adapter for architecture: ${ARCHITECTURE}"

# Read base prompts and append session-specific instructions
SAFE_BRANCH_LOG=$(echo "$NEW_BRANCH" | tr '/' '-')
mkdir -p "$LOGS_DIR"

PROG_PROMPT_FILE="$LOGS_DIR/${SAFE_BRANCH_LOG}.programmer-prompt.txt"
REV_PROMPT_FILE="$LOGS_DIR/${SAFE_BRANCH_LOG}.reviewer-prompt.txt"

# Strip the YAML frontmatter (first `---` to next `---`) before writing the
# prompt file. The frontmatter is only meaningful when the .md is read as
# an agent definition by Claude Code's agent-teams mode. When we pass the
# file content directly as a prompt string, the leading `---` is parsed as
# a command-line flag and claude exits with "error: unknown option '---".
_strip_frontmatter() {
  # Delete lines from the first `---` through the next `---`.
  awk 'BEGIN{in_fm=0; done=0} /^---$/ && done==0 { if(in_fm==0){in_fm=1; next} else {in_fm=0; done=1; next} } in_fm==0 && done==1 {print} in_fm==0 && done==0 && !/^---$/ {print}' "$1"
}

_strip_frontmatter "$SCRIPT_DIR/solo-programmer.md" > "$PROG_PROMPT_FILE"
printf '\n---\n## Session Task\n\n%s\n' "$PROGRAMMER_TASK" >> "$PROG_PROMPT_FILE"

_strip_frontmatter "$SCRIPT_DIR/solo-reviewer.md" > "$REV_PROMPT_FILE"
printf '\n---\n## Session Task\n\n%s\n' "$REVIEWER_TASK" >> "$REV_PROMPT_FILE"

# --------------------------------------------------------------------------- #
# Launch two Claude Code sessions in tmux
# --------------------------------------------------------------------------- #
TMUX_SESSION="tl-adapter-$SAFE_BRANCH_LOG"
PID_FILE="$LOGS_DIR/${SAFE_BRANCH_LOG}.pid"
PROG_LOG="$LOGS_DIR/${SAFE_BRANCH_LOG}.programmer.raw"
REV_LOG="$LOGS_DIR/${SAFE_BRANCH_LOG}.reviewer.raw"
WATCH_LOG="$LOGS_DIR/${SAFE_BRANCH_LOG}.watch.log"

# Use a dedicated tmux socket (`-L tl-adapter`) so our sessions don't share a
# server with any tmux session the user already has running. Without this,
# if a pane crashes or emits an unexpected control sequence, it could kill
# the user's entire tmux session. TM() wraps tmux with the socket flag.
TMUX_SOCKET="tl-adapter"
TM() { tmux -L "$TMUX_SOCKET" "$@"; }

# Kill any stale session on our dedicated socket
TM kill-session -t "$TMUX_SESSION" 2>/dev/null || true

CLAUDE_FLAGS=(
  "--dangerously-skip-permissions"
  "--permission-mode" "bypassPermissions"
)
if [[ "$AUTO_APPROVE" == true ]]; then
  CLAUDE_FLAGS+=("--allowedTools" "bash" "read" "write" "edit" "glob" "grep")
fi

# Per-role model pins from the prompt files' frontmatter (the frontmatter is
# stripped before the content is passed as a prompt, so the model must be
# re-applied via --model or both roles silently run the default model).
_frontmatter_model() { sed -n 's/^model:[[:space:]]*//p' "$1" | head -1; }
PROG_MODEL="$(_frontmatter_model "$SCRIPT_DIR/solo-programmer.md")"
REV_MODEL="$(_frontmatter_model "$SCRIPT_DIR/solo-reviewer.md")"
PROG_MODEL_FLAG=""; [[ -n "$PROG_MODEL" ]] && PROG_MODEL_FLAG="--model $PROG_MODEL"
REV_MODEL_FLAG="";  [[ -n "$REV_MODEL"  ]] && REV_MODEL_FLAG="--model $REV_MODEL"

# Common environment exports.
#
# IMPORTANT: the outer shell may itself be running inside a Claude Code
# session (e.g. the user is running this launcher from within Claude Code).
# If so, CLAUDECODE, CLAUDE_CODE_ENTRYPOINT, CLAUDE_CODE_EXECPATH, and
# MCP_CONNECTION_NONBLOCKING leak into the spawned claude and confuse it —
# a nested Claude Code may try to connect to the parent's SSE port or
# behave oddly during startup. Unset them to guarantee a clean session.
EXPORT_BLOCK="
unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_EXECPATH CLAUDE_CODE_SSE_PORT CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING MCP_CONNECTION_NONBLOCKING
export TL_ADAPTER_BUILDER_ROOT='$PROJECT_ROOT'
export MAX_MEMORY_GB='$MAX_MEMORY_GB'
export NOTIFICATION_WEBHOOK_URL='${NOTIFICATION_WEBHOOK_URL:-}'
export NOTIFICATION_NUMBER='${NOTIFICATION_NUMBER:-}'
export HF_TOKEN='${HF_TOKEN:-}'
export HUGGING_FACE_HUB_TOKEN='${HF_TOKEN:-}'
cd '$WORKTREE_DIR'
"

# Build wrapper scripts for each pane
PROG_WRAPPER="$LOGS_DIR/${SAFE_BRANCH_LOG}.programmer-wrapper.sh"
cat > "$PROG_WRAPPER" <<WEOF
#!/usr/bin/env bash
$EXPORT_BLOCK
PROMPT="\$(cat '$PROG_PROMPT_FILE')"
claude ${CLAUDE_FLAGS[*]} $PROG_MODEL_FLAG "\$PROMPT"
echo ""
echo "[programmer session ended — tmux pane stays open 1 hour]"
sleep 3600
WEOF
chmod +x "$PROG_WRAPPER"

REV_WRAPPER="$LOGS_DIR/${SAFE_BRANCH_LOG}.reviewer-wrapper.sh"
cat > "$REV_WRAPPER" <<WEOF
#!/usr/bin/env bash
$EXPORT_BLOCK
PROMPT="\$(cat '$REV_PROMPT_FILE')"
claude ${CLAUDE_FLAGS[*]} $REV_MODEL_FLAG "\$PROMPT"
echo ""
echo "[reviewer session ended — tmux pane stays open 1 hour]"
sleep 3600
WEOF
chmod +x "$REV_WRAPPER"

# Create tmux session with two panes: left=programmer, right=reviewer
log "Launching solo pair in tmux session '$TMUX_SESSION'..."
log "  worktree : $WORKTREE_DIR"
log "  arch     : $ARCHITECTURE"
log "  mode     : solo (file-based coordination, no agent-teams)"

TM new-session -d -s "$TMUX_SESSION" -c "$WORKTREE_DIR" "$PROG_WRAPPER"
TM split-window -t "$TMUX_SESSION" -h -c "$WORKTREE_DIR" "$REV_WRAPPER"

# Name the panes for clarity when attached
TM select-pane -t "$TMUX_SESSION:0.0" -T "programmer"
TM select-pane -t "$TMUX_SESSION:0.1" -T "reviewer"
TM set-option -t "$TMUX_SESSION" pane-border-status top 2>/dev/null || true

# Pipe each pane's output to its own raw log
TM pipe-pane -t "$TMUX_SESSION:0.0" -O "python3 -u -c \"
import sys, os
f = open('$PROG_LOG', 'a', buffering=1)
n = 0
for line in sys.stdin:
    f.write(line)
    n += 1
    if n % 500 == 0:
        try:
            if os.path.getsize('$PROG_LOG') > 50_000_000:
                f.close()
                os.replace('$PROG_LOG', '$PROG_LOG.1')
                f = open('$PROG_LOG', 'a', buffering=1)
        except OSError:
            pass
\""

TM pipe-pane -t "$TMUX_SESSION:0.1" -O "python3 -u -c \"
import sys, os
f = open('$REV_LOG', 'a', buffering=1)
n = 0
for line in sys.stdin:
    f.write(line)
    n += 1
    if n % 500 == 0:
        try:
            if os.path.getsize('$REV_LOG') > 50_000_000:
                f.close()
                os.replace('$REV_LOG', '$REV_LOG.1')
                f = open('$REV_LOG', 'a', buffering=1)
        except OSError:
            pass
\""

# Capture PIDs
PROG_PID=$(TM list-panes -t "$TMUX_SESSION:0" -F '#{pane_pid}' | head -1)
echo "${PROG_PID:-0}" > "$PID_FILE"

# Start the signal coordinator (routes signals between panes; also handles
# completion — it subsumes watch-completion.sh in solo mode)
"$SCRIPT_DIR/solo-coordinator.sh" "$WORKTREE_DIR" "$TMUX_SESSION" "$ARCHITECTURE" \
  >> "$WATCH_LOG" 2>&1 &
COORDINATOR_PID=$!
echo "$COORDINATOR_PID" > "$LOGS_DIR/${SAFE_BRANCH_LOG}.coordinator.pid"

ok "Solo pair launched in tmux session '$TMUX_SESSION' (dedicated socket '$TMUX_SOCKET')"
ok "  Left pane:  programmer (PID: ${PROG_PID:-?}, model: ${PROG_MODEL:-default})"
ok "  Right pane: reviewer (model: ${REV_MODEL:-default})"
ok "  Coordinator: PID $COORDINATOR_PID → $WATCH_LOG"
ok "  Attach:     tmux -L $TMUX_SOCKET attach -t $TMUX_SESSION"
ok "  Status:     ./agents/launch-agent-pair.sh status"
ok "  Timeline:   ./agents/launch-agent-pair.sh logs $SAFE_BRANCH_LOG"
ok "  Signals:    ls $WORKTREE_DIR/.adapter-workspace/signals/"
