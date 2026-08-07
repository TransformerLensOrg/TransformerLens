#!/usr/bin/env bash
# =============================================================================
# launch.sh
#
# Creates a git worktree on a new branch in the TransformerLens repo and kicks
# off a Programmer/Reviewer Claude Code agent pair inside it. This is the
# launch path only; operational subcommands (status, logs, attach, send, stop,
# clean) live in ops.sh. The user-facing entry point is launch-agent-pair.sh.
#
# Usage:
#   ./launch.sh --architecture <class> [options]
#
# Options:
#     [--seed-model <hf-id>]       Known model for scaffolding (e.g. Qwen/Qwen3.5-9B)
#     [--target-repo <path>]       Path to TransformerLens repo (default from .env)
#     [--base-branch <branch>]     Base branch (default: DEFAULT_BASE_BRANCH from .env)
#     [--new-branch  <branch>]     Feature branch (default: feature/<architecture>-adapter)
#     [--max-memory <gb>]          Memory limit in GB (default: DEFAULT_MAX_MEMORY_GB from .env)
#     [--programmer-prompt <text>] Override the auto-generated programmer prompt
#     [--reviewer-prompt   <text>] Extra review criteria/focus for the Reviewer
#     [--worktree-dir <path>]      Where to create the worktree (default: $WORKTREE_BASE/<new-branch>)
#     [--background]               Run detached inside a tmux session
#     [--retry]                    Resume from saved progress (safe for planning/programming)
#     [--auto-approve]             Run agents in auto-approve mode (no permission prompts)
#     [--skip-arch-check]          Skip the pre-flight architecture existence check
#
# Example:
#   # Interactive (foreground)
#   ./launch.sh --architecture CohereForCausalLM
#
#   # Background (detached) — run multiple in parallel
#   ./launch.sh --architecture CohereForCausalLM --background
#   ./launch.sh --architecture CodeGenForCausalLM --background
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/.logs"

# Shared helpers (log/ok/warn/err/require_cmd/iso_now)
TL_LOG_TAG="launch"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

# Friendly command name for user-facing hints.
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
PROGRAMMER_PROMPT=""
REVIEWER_PROMPT=""
WORKTREE_DIR=""
AUTO_APPROVE=false
BACKGROUND=false
RETRY=false
SKIP_ARCH_CHECK=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-repo)        TARGET_REPO="$2";        shift 2 ;;
    --architecture)       ARCHITECTURE="$2";       shift 2 ;;
    --seed-model)         SEED_MODEL="$2";         shift 2 ;;
    --base-branch)        BASE_BRANCH="$2";        shift 2 ;;
    --new-branch)         NEW_BRANCH="$2";         shift 2 ;;
    --max-memory)         MAX_MEMORY_GB="$2";      shift 2 ;;
    --programmer-prompt)  PROGRAMMER_PROMPT="$2";  shift 2 ;;
    --reviewer-prompt)    REVIEWER_PROMPT="$2";    shift 2 ;;
    --worktree-dir)       WORKTREE_DIR="$2";       shift 2 ;;
    --background)         BACKGROUND=true;          shift   ;;
    --retry)              RETRY=true;               shift   ;;
    --auto-approve)       AUTO_APPROVE=true;        shift   ;;
    --skip-arch-check)    SKIP_ARCH_CHECK=true;     shift   ;;
    -h|--help)            usage ;;
    *) err "Unknown argument: $1" ;;
  esac
done

# Load defaults from .env if present
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
[[ -z "$ARCHITECTURE" ]] && err "--architecture is required (e.g., CohereForCausalLM)"

# Apply defaults: base branch from .env, new branch from architecture name
BASE_BRANCH="${BASE_BRANCH:-${DEFAULT_BASE_BRANCH:-dev-4.x}}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-${DEFAULT_MAX_MEMORY_GB:-96}}"

if [[ -z "$NEW_BRANCH" ]]; then
  # Convert architecture class to lowercase branch name.
  # e.g. CohereForCausalLM -> cohere, Gemma3ForConditionalGeneration -> gemma3
  ARCH_SHORT=$(echo "$ARCHITECTURE" | sed -E 's/(For(Causal|Conditional|Masked).*|LMHead.*)$//' | tr '[:upper:]' '[:lower:]')
  NEW_BRANCH="feature/${ARCH_SHORT}-adapter"
fi

# Auto-generate programmer prompt from architecture if not explicitly provided
if [[ -z "$PROGRAMMER_PROMPT" ]]; then
  SEED_MODEL_BLOCK=""
  if [[ -n "$SEED_MODEL" ]]; then
    SEED_MODEL_BLOCK="
Seed model: ${SEED_MODEL}
Use this model for scaffold generation (analyze-hf-model.py --scaffold) and
as the primary source for reading modeling_*.py / configuration_*.py. Include
it in your verification targets."
  fi

  PROGRAMMER_PROMPT="Create an Architecture Adapter for the HuggingFace architecture: ${ARCHITECTURE}
${SEED_MODEL_BLOCK}
Follow your 3-step lifecycle (Plan, Program, Verify):

Step 1 — Planning:
1. Find models using this architecture on HuggingFace
2. Read the HF model source (modeling_*.py, configuration_*.py) to understand the architecture
3. Use \$TL_ADAPTER_BUILDER_ROOT/scripts/analyze-hf-model.py on a representative model to extract config details
4. Read docs/adapter-specification.md for the full specification
5. Find the closest existing adapter in transformer_lens/model_bridge/supported_architectures/
6. Write a phased plan and submit for review

Step 2 — Programming:
7. Implement each phase, getting reviewer approval before moving to the next

Step 3 — Verification:
8. Run verify_models on the top 5 models of this architecture that fit in ${MAX_MEMORY_GB}GB
9. If failures, return to planning and fix

Architecture class: ${ARCHITECTURE}"
fi

REVIEWER_PROMPT="${REVIEWER_PROMPT:-Review the programmer changes thoroughly for correctness, edge cases, code style, and maintainability.}"

# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
require_cmd git
require_cmd claude

# Use --target-repo as the repo root (the TransformerLens repo)
REPO_ROOT="$(cd "$TARGET_REPO" && git rev-parse --show-toplevel 2>/dev/null)" \
  || err "--target-repo '$TARGET_REPO' is not a git repository."

# The repo-level .env may hold HF_TOKEN (repo convention for HF-Hub-hitting
# commands); the builder's own .env wins if both define it.
if [[ -z "${HF_TOKEN:-}" && -f "$REPO_ROOT/.env" ]]; then
  HF_TOKEN="$({ . "$REPO_ROOT/.env" >/dev/null 2>&1 || true; printf '%s' "${HF_TOKEN:-}"; })"
fi

# Where to create worktrees. WORKTREE_BASE may be set in .env; otherwise
# fall back to "<parent-of-repo>/worktrees" (matches the original layout).
WORKTREE_BASE="${WORKTREE_BASE:-$(dirname "$REPO_ROOT")/worktrees}"

# --------------------------------------------------------------------------- #
# Early opt-out: check if architecture is already supported
# --------------------------------------------------------------------------- #
FACTORY_FILE="$REPO_ROOT/transformer_lens/factories/architecture_adapter_factory.py"
if [[ -f "$FACTORY_FILE" ]] && grep -q "\"${ARCHITECTURE}\"" "$FACTORY_FILE"; then
  ok "Architecture '${ARCHITECTURE}' already has an adapter in TransformerLens."
  log "Found in: $FACTORY_FILE"
  log "No work needed. Exiting."
  exit 0
fi

# --------------------------------------------------------------------------- #
# Pre-flight: does the architecture actually exist?
#
# Checks the installed transformers package first (fast, local), then falls
# back to a bounded HuggingFace Hub scan. Aborts on typos before any worktree
# is created. Runs in the target repo's uv env so it sees the project's
# transformers and huggingface_hub installs.
#
# Bypass with --skip-arch-check for genuinely new architectures that haven't
# made it into any of the above yet.
# --------------------------------------------------------------------------- #
# --seed-model implies --skip-arch-check (the user knows the arch exists)
if [[ -n "$SEED_MODEL" && "$SKIP_ARCH_CHECK" == false ]]; then
  log "Seed model '${SEED_MODEL}' provided — skipping architecture existence check."
  SKIP_ARCH_CHECK=true
fi

if [[ "$SKIP_ARCH_CHECK" == true ]]; then
  warn "Skipping architecture existence check."
else
  log "Validating '${ARCHITECTURE}' exists in transformers or on HuggingFace Hub..."
  set +e
  # Unset VIRTUAL_ENV so uv doesn't warn when the user has a different venv
  # active in their shell — the validator needs the TransformerLens project
  # env, not whatever happens to be on PATH.
  env -u VIRTUAL_ENV uv run --project "$REPO_ROOT" \
    python "$PROJECT_ROOT/scripts/validate-architecture.py" "$ARCHITECTURE"
  validate_exit=$?
  set -e
  case "$validate_exit" in
    0)
      ok "Architecture validated."
      ;;
    1)
      err "Architecture '${ARCHITECTURE}' does not exist in the installed transformers package or on HuggingFace Hub. Check the spelling (class names are case-sensitive, e.g. 'CohereForCausalLM'). If this is a genuinely new architecture not yet on HF, rerun with --skip-arch-check."
      ;;
    2)
      warn "Could not fully verify '${ARCHITECTURE}' (missing deps or network error). Proceeding anyway — rerun with --skip-arch-check to suppress this warning."
      ;;
    *)
      err "validate-architecture.py returned unexpected exit code: $validate_exit"
      ;;
  esac
fi

if [[ -z "$WORKTREE_DIR" ]]; then
  SAFE_BRANCH=$(echo "$NEW_BRANCH" | tr '/' '-')
  WORKTREE_DIR="${WORKTREE_BASE}/${SAFE_BRANCH}"
fi

git -C "$REPO_ROOT" rev-parse --verify "$BASE_BRANCH" &>/dev/null \
  || err "Base branch '$BASE_BRANCH' does not exist in this repository."

# Source progress tracking
# shellcheck disable=SC1091
source "$SCRIPT_DIR/progress.sh"

# --------------------------------------------------------------------------- #
# Create or reuse the worktree — atomic with rollback on partial failure.
#
# Instead of check-then-create (TOCTOU), we attempt creation directly and
# interpret the exit code:
#   - Success → new worktree, proceed with init
#   - "already exists" → resume path
#   - Other failure → abort
#
# A cleanup trap removes a partially-created worktree + branch if anything
# between creation and the first successful progress_init fails.
# --------------------------------------------------------------------------- #
RESUME_CONTEXT=""
REUSE_WORKTREE=false

if [[ -d "$WORKTREE_DIR" ]] && git -C "$REPO_ROOT" rev-parse --verify "$NEW_BRANCH" &>/dev/null; then
  # Both directory and branch exist — this is a resume, not a race.
  ok "Existing worktree found at $WORKTREE_DIR (branch: $NEW_BRANCH) — resuming."
  REUSE_WORKTREE=true
else
  # Attempt atomic creation. git-worktree-add will fail if the branch or
  # directory already exists (concurrent launch lost the race).
  log "Creating worktree at: $WORKTREE_DIR"
  log "  base branch : $BASE_BRANCH"
  log "  new branch  : $NEW_BRANCH"

  _cleanup_partial_worktree() {
    warn "Cleaning up partial worktree after failure..."
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
    git -C "$REPO_ROOT" branch -D "$NEW_BRANCH" 2>/dev/null || true
  }

  if ! git -C "$REPO_ROOT" worktree add -b "$NEW_BRANCH" "$WORKTREE_DIR" "$BASE_BRANCH" 2>/tmp/tl-worktree-err.$$; then
    wt_err=$(cat /tmp/tl-worktree-err.$$ 2>/dev/null)
    rm -f /tmp/tl-worktree-err.$$
    if echo "$wt_err" | grep -qiE "already exists|already checked out"; then
      err "Worktree or branch already exists (concurrent launch?): $wt_err"
    else
      err "Failed to create worktree: $wt_err"
    fi
  fi
  rm -f /tmp/tl-worktree-err.$$

  # Trap: if anything between here and the end of init fails, roll back.
  trap '_cleanup_partial_worktree' ERR

  ok "Worktree created."

  # Initialize fresh progress file
  progress_init "$WORKTREE_DIR" "$ARCHITECTURE"
  ok "Progress tracking initialized."

  # Clear the ERR trap — init succeeded, worktree is fully set up.
  trap - ERR
fi

if [[ "$REUSE_WORKTREE" == true ]]; then
  # Handle --retry: resume planning/programming, but NOT verification
  if [[ "$RETRY" == true ]]; then
    progress_read "$WORKTREE_DIR"
    if [[ "$PROGRESS_EXISTS" == "true" && "$PROGRESS_STEP" == "verification" ]]; then
      warn "Last session crashed during verification."
      "$PROJECT_ROOT/scripts/notify.sh" "Crash occurred for ${ARCHITECTURE} during verification, please manually resume"
      log "Verification requires manual resume. Exiting."
      exit 1
    elif [[ "$PROGRESS_EXISTS" == "true" ]]; then
      ok "Progress is at '${PROGRESS_STEP}' — retrying from there."
    fi
  fi

  # Generate resume context from saved progress
  RESUME_CONTEXT=$(generate_resume_context "$WORKTREE_DIR")
  if [[ -n "$RESUME_CONTEXT" ]]; then
    progress_read "$WORKTREE_DIR"
    ok "Resuming from: step=${PROGRESS_STEP}, completed_phases=${PROGRESS_COMPLETED_PHASES:-none}"
  fi
fi

# --------------------------------------------------------------------------- #
# Inject agent definitions into the worktree (.claude/agents/ is tool config,
# required by Claude Code agent teams — no repo code is copied)
# --------------------------------------------------------------------------- #
AGENTS_DIR="$WORKTREE_DIR/.claude/agents"
mkdir -p "$AGENTS_DIR"

# Create .adapter-workspace/ for brief/plan/scaffold/timeline/reviews — NOT in
# .claude/ because Claude Code treats .claude/ paths specially and prompts on
# writes even with --dangerously-skip-permissions.
mkdir -p "$WORKTREE_DIR/.adapter-workspace/reviews"
mkdir -p "$WORKTREE_DIR/.adapter-workspace/phase-reports"
touch "$WORKTREE_DIR/.adapter-workspace/adapter-brief.md" 2>/dev/null || true
touch "$WORKTREE_DIR/.adapter-workspace/adapter-plan.md" 2>/dev/null || true
touch "$WORKTREE_DIR/.adapter-workspace/adapter-scaffold.py" 2>/dev/null || true
touch "$WORKTREE_DIR/.adapter-workspace/timeline.jsonl" 2>/dev/null || true

# Install Claude Code hooks in the worktree so every tool call, subagent spawn,
# and session event is automatically logged to the timeline. No agent
# cooperation needed — hooks fire at the framework level.
TIMELINE_PATH="$WORKTREE_DIR/.adapter-workspace/timeline.jsonl"
SETTINGS_FILE="$WORKTREE_DIR/.claude/settings.json"
mkdir -p "$WORKTREE_DIR/.claude"

# Absolute paths to the hook scripts. Claude Code invokes hook commands via
# /bin/sh -c "<cmd>", so paths with spaces need shell quoting. We build the
# command strings as '<path>' (single-quoted) and let jq handle all JSON
# escaping — this prevents broken JSON if paths contain backslashes,
# double-quotes, or other special characters.
_hook_cmd() { echo "'$SCRIPT_DIR/hooks/$1'"; }

TIMELINE_HOOK=$(_hook_cmd timeline-capture.sh)
GUARD_HT_HOOK=$(_hook_cmd guard-hooked-transformer.sh)
GUARD_GIT_HOOK=$(_hook_cmd guard-git.sh)
GUARD_REVIEW_ROUNDS_HOOK=$(_hook_cmd guard-review-rounds.sh)
GUARD_VERIFY_MODELS_HOOK=$(_hook_cmd guard-verify-models.sh)
GATE_REVIEWER_WRITES_HOOK=$(_hook_cmd gate-reviewer-writes-file.sh)
GATE_LINT_HOOK=$(_hook_cmd gate-lint-checks.sh)
NOTIFY_HOOK=$(_hook_cmd notify-on-completion.sh)

# Build settings.json via jq so all hook paths are properly JSON-escaped.
# This is the only reliable way to handle paths with spaces, backslashes,
# or other characters that would break a naive heredoc interpolation.
_h() { jq -n --arg cmd "$1" '{"type":"command","command":$cmd}'; }

jq -n \
  --argjson session_start  "[{\"hooks\":[  $(_h "$TIMELINE_HOOK")  ]}]" \
  --argjson session_end    "[{\"hooks\":[  $(_h "$TIMELINE_HOOK"), $(_h "$NOTIFY_HOOK")  ]}]" \
  --argjson subagent_start "[{\"hooks\":[  $(_h "$TIMELINE_HOOK")  ]}]" \
  --argjson subagent_stop  "[{\"hooks\":[  $(_h "$TIMELINE_HOOK"), $(_h "$GATE_REVIEWER_WRITES_HOOK")  ]}]" \
  --argjson pre_edit       "{\"matcher\":\"Edit|Write|MultiEdit|NotebookEdit\",\"hooks\":[ $(_h "$GUARD_HT_HOOK"), $(_h "$GUARD_REVIEW_ROUNDS_HOOK") ]}" \
  --argjson pre_bash       "{\"matcher\":\"Bash\",\"hooks\":[ $(_h "$GUARD_GIT_HOOK"), $(_h "$GUARD_VERIFY_MODELS_HOOK") ]}" \
  --argjson pre_all        "{\"matcher\":\"\",\"hooks\":[ $(_h "$TIMELINE_HOOK") ]}" \
  --argjson post_all       "[{\"matcher\":\"\",\"hooks\":[ $(_h "$TIMELINE_HOOK") ]}]" \
  --argjson stop           "[{\"hooks\":[ $(_h "$GATE_LINT_HOOK") ]}]" \
  '{hooks:{
    SessionStart: $session_start,
    SessionEnd: $session_end,
    SubagentStart: $subagent_start,
    SubagentStop: $subagent_stop,
    PreToolUse: [$pre_edit, $pre_bash, $pre_all],
    PostToolUse: $post_all,
    Stop: $stop
  }}' > "$SETTINGS_FILE"
ok "Hooks installed: timeline, HookedTransformer guardrail, git guardrail, review-rounds limit, verify-models gate, reviewer-writes-file gate, lint gate, completion notifier"

# Pre-seed workspace trust for the worktree. Without this, a first launch in
# a fresh worktree hangs at Claude Code's "Do you trust this folder?" prompt
# because --dangerously-skip-permissions does NOT bypass the trust dialog
# (only -p/print mode does). We atomically merge a trust entry into
# ~/.claude.json so the spawned claude sees the worktree as pre-approved.
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
fi

MAIN_AGENTS_DIR="$SCRIPT_DIR"

# Memory lock reference (injected into both agents)
memory_lock_block() {
  cat <<BLOCK

---
## Memory Lock

Before running \`verify_models\` or loading a full HuggingFace model, read
\`\$TL_ADAPTER_BUILDER_ROOT/docs/memory-lock.md\` for the lock protocol.
Always use \`overlord-request.sh run\` to wrap the command — do NOT source.
BLOCK
}

inject_agent() {
  local role="$1"
  local extension="$2"
  local dest="$AGENTS_DIR/${role}.md"

  if [[ -f "$MAIN_AGENTS_DIR/${role}.md" ]]; then
    cp "$MAIN_AGENTS_DIR/${role}.md" "$dest"
    log "Copied base ${role}.md template."
  else
    log "No base ${role}.md found — generating minimal default."
    cat > "$dest" <<MDEOF
---
name: ${role}
description: |
  $([ "$role" = "programmer" ] \
    && echo "Implements coding tasks assigned by the orchestrator." \
    || echo "Reviews code written by the Programmer and provides structured feedback.")
model: claude-sonnet-4-6
isolation: worktree
---

$([ "$role" = "programmer" ] \
    && echo "You are a skilled software engineer. Implement tasks carefully, write clean code, and respond to reviewer feedback until your changes are approved." \
    || echo "You are a rigorous code reviewer. Analyse the programmer's changes and provide structured, actionable feedback. Approve only when you are satisfied with correctness, style, and robustness.")
MDEOF
  fi

  # Append session-specific prompt
  printf '\n---\n## Session-Specific Instructions\n\n%s\n' "$extension" >> "$dest"

  # Append memory lock reference
  memory_lock_block >> "$dest"

  ok "Wrote ${role}.md to worktree."
}

inject_agent "programmer" "$PROGRAMMER_PROMPT"
inject_agent "reviewer"   "$REVIEWER_PROMPT"

# --------------------------------------------------------------------------- #
# Build the orchestration prompt from template + signals
# --------------------------------------------------------------------------- #
# shellcheck disable=SC1091
source "$SCRIPT_DIR/signals.sh"

PROTOCOL_BLOCK=$(generate_protocol_block)

# Render orchestrator.md template with placeholder substitution
ORCHESTRATOR_TEMPLATE="$SCRIPT_DIR/orchestrator.md"
[[ -f "$ORCHESTRATOR_TEMPLATE" ]] || err "orchestrator.md not found at $ORCHESTRATOR_TEMPLATE"

ORCHESTRATION_PROMPT=$(sed \
  -e "s|{{ARCHITECTURE}}|${ARCHITECTURE}|g" \
  -e "s|{{NEW_BRANCH}}|${NEW_BRANCH}|g" \
  -e "s|{{BASE_BRANCH}}|${BASE_BRANCH}|g" \
  -e "s|{{MAX_MEMORY_GB}}|${MAX_MEMORY_GB}|g" \
  -e "s|{{PROJECT_ROOT}}|${PROJECT_ROOT}|g" \
  -e "s|{{SIG_CODE_APPROVED}}|${SIG_CODE_APPROVED}|g" \
  "$ORCHESTRATOR_TEMPLATE")

# Replace multi-line placeholders that sed can't handle
ORCHESTRATION_PROMPT="${ORCHESTRATION_PROMPT//\{\{PROTOCOL_BLOCK\}\}/$PROTOCOL_BLOCK}"
ORCHESTRATION_PROMPT="${ORCHESTRATION_PROMPT//\{\{PROGRAMMER_PROMPT\}\}/$PROGRAMMER_PROMPT}"
ORCHESTRATION_PROMPT="${ORCHESTRATION_PROMPT//\{\{REVIEWER_PROMPT\}\}/$REVIEWER_PROMPT}"
ORCHESTRATION_PROMPT="${ORCHESTRATION_PROMPT//\{\{RESUME_CONTEXT\}\}/$RESUME_CONTEXT}"

# --------------------------------------------------------------------------- #
# Launch Claude Code inside the worktree
# --------------------------------------------------------------------------- #
SAFE_BRANCH_LOG=$(echo "$NEW_BRANCH" | tr '/' '-')
mkdir -p "$LOGS_DIR"

CLAUDE_FLAGS=(
  "--dangerously-skip-permissions"
  "--permission-mode" "bypassPermissions"
  "--debug-file" "$LOGS_DIR/${SAFE_BRANCH_LOG}.debug.log"
  "--debug" "hooks"
)
if [[ "$AUTO_APPROVE" == true ]]; then
  CLAUDE_FLAGS+=("--allowedTools" "bash" "read" "write" "edit" "glob" "grep")
fi

MODE="interactive"
[[ "$BACKGROUND" == true ]] && MODE="background"

log "Launching agent pair..."
log "  worktree    : $WORKTREE_DIR"
log "  project root: $PROJECT_ROOT"
log "  memory lock : /tmp/tl-adapter-builder.lock (flock)"
log "  hooks debug : $LOGS_DIR/${SAFE_BRANCH_LOG}.debug.log"
log "  mode        : $MODE"

cd "$WORKTREE_DIR"

if [[ "$BACKGROUND" == true ]]; then
  require_cmd tmux

  LOG_FILE="$LOGS_DIR/${SAFE_BRANCH_LOG}.log"
  PID_FILE="$LOGS_DIR/${SAFE_BRANCH_LOG}.pid"
  RAW_LOG="${LOG_FILE%.log}.raw"
  TMUX_SESSION="tl-adapter-$SAFE_BRANCH_LOG"
  WRAPPER="$LOGS_DIR/${SAFE_BRANCH_LOG}.wrapper.sh"
  PROMPT_FILE="$LOGS_DIR/${SAFE_BRANCH_LOG}.prompt.txt"
  WATCH_LOG="$LOGS_DIR/${SAFE_BRANCH_LOG}.watch.log"

  log "  tmux session: $TMUX_SESSION"
  log "  raw log     : $RAW_LOG"
  log "  watch log   : $WATCH_LOG"
  echo ""

  # Kill any stale session with the same name
  # Use a dedicated tmux socket so our sessions don't share a server with
  # any tmux session the user already has running. If a pane crashes or
  # emits an unexpected control sequence, it can't kill the user's tmux.
  TMUX_SOCKET="tl-adapter"
  TM() { tmux -L "$TMUX_SOCKET" "$@"; }

  TM kill-session -t "$TMUX_SESSION" 2>/dev/null || true

  # Write the orchestration prompt to a file (too big for shell quoting)
  printf '%s' "$ORCHESTRATION_PROMPT" > "$PROMPT_FILE"

  # Build the wrapper script tmux will execute. This avoids shell quoting
  # nightmares inside `tmux new-session -d`'s command string.
  cat > "$WRAPPER" <<WRAPPER_EOF
#!/usr/bin/env bash
# When this launcher runs inside an outer Claude Code session, the outer
# session's env vars leak into the spawned claude and confuse it (the
# nested claude may try to connect to the parent's SSE port). Unset them
# before starting the inner session. CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS
# is re-exported below because this is exactly the mode we want.
unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_EXECPATH CLAUDE_CODE_SSE_PORT CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING MCP_CONNECTION_NONBLOCKING
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
export TL_ADAPTER_BUILDER_ROOT="$PROJECT_ROOT"
export MAX_MEMORY_GB="$MAX_MEMORY_GB"
export NOTIFICATION_WEBHOOK_URL="${NOTIFICATION_WEBHOOK_URL:-}"
export NOTIFICATION_NUMBER="${NOTIFICATION_NUMBER:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
cd "$WORKTREE_DIR"

# Start the completion watcher in the background so a successful adapter
# build auto-terminates the session and fires the SessionEnd hook chain
# (notify-on-completion.sh → notify.sh → Slack). The watcher polls
# .adapter-progress.json and sends /exit to the tmux pane when the adapter
# is genuinely done. See agents/watch-completion.sh for details.
"$SCRIPT_DIR/watch-completion.sh" "$WORKTREE_DIR" "$TMUX_SESSION" "$ARCHITECTURE" \\
  >> "$WATCH_LOG" 2>&1 &
WATCHER_PID=\$!
trap 'kill \$WATCHER_PID 2>/dev/null || true' EXIT

# Read the orchestration prompt from the file
PROMPT="\$(cat "$PROMPT_FILE")"
claude "\$@" "\$PROMPT"
CLAUDE_EXIT=\$?

# Claude has exited (either via /exit from the watcher or a manual exit).
# Kill the watcher if it's still running; the trap will handle the cleanup
# in the abnormal-exit case.
kill \$WATCHER_PID 2>/dev/null || true

echo ""
echo "[session ended with exit code \$CLAUDE_EXIT — tmux pane will stay open for 1 hour]"
echo "Press Ctrl-b then d to detach, or Ctrl-b then x to close."
sleep 3600
WRAPPER_EOF
  chmod +x "$WRAPPER"

  # Launch Claude inside a detached tmux session. tmux provides a real pty,
  # so claude's interactive mode works as expected. The session persists in
  # the background and can be attached to interactively at any time.
  TM new-session -d -s "$TMUX_SESSION" -c "$WORKTREE_DIR" \
    "$WRAPPER" "${CLAUDE_FLAGS[@]}"

  # Pipe all pane output to the raw log via a python rotator that caps
  # the file at ~50MB. When exceeded, the current log is moved to .1
  # (one backup kept). This prevents unbounded growth on long sessions.
  TM pipe-pane -t "$TMUX_SESSION" -O \
    "python3 -u -c \"
import sys, os
f = open('$RAW_LOG', 'a', buffering=1)
n = 0
for line in sys.stdin:
    f.write(line)
    n += 1
    if n % 500 == 0:
        try:
            if os.path.getsize('$RAW_LOG') > 50_000_000:
                f.close()
                os.replace('$RAW_LOG', '$RAW_LOG.1')
                f = open('$RAW_LOG', 'a', buffering=1)
        except OSError:
            pass
\""

  # Capture the tmux session's top-level pane PID so stop/status works
  CLAUDE_PID=$(TM list-panes -t "$TMUX_SESSION" -F '#{pane_pid}' 2>/dev/null | head -1)
  echo "${CLAUDE_PID:-0}" > "$PID_FILE"

  ok "Agent pair launched in tmux session '$TMUX_SESSION' (PID: ${CLAUDE_PID:-?})"
  ok "Raw log:  $RAW_LOG"
  ok "Attach:   $TL_CMD attach $SAFE_BRANCH_LOG"
  ok "Send msg: $TL_CMD send $SAFE_BRANCH_LOG \"<message>\""
  ok "Detach from tmux: Ctrl-b d"
  ok "Status:   $TL_CMD status"
  ok "Timeline: $TL_CMD logs $SAFE_BRANCH_LOG"
else
  echo ""

  # Interactive foreground mode
  CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 \
    TL_ADAPTER_BUILDER_ROOT="$PROJECT_ROOT" \
    MAX_MEMORY_GB="$MAX_MEMORY_GB" \
    NOTIFICATION_WEBHOOK_URL="${NOTIFICATION_WEBHOOK_URL:-}" \
    NOTIFICATION_NUMBER="${NOTIFICATION_NUMBER:-}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
    claude "${CLAUDE_FLAGS[@]}" \
    "$ORCHESTRATION_PROMPT"

  CLAUDE_EXIT=$?
fi

# --------------------------------------------------------------------------- #
# Post-session cleanup (foreground only — background cleans up via `clean`)
# --------------------------------------------------------------------------- #
if [[ "$BACKGROUND" == false && -f "$WORKTREE_DIR/.adapter-progress.json" ]]; then
  VERIFIED=$(python3 -c "import json; print(json.load(open('$WORKTREE_DIR/.adapter-progress.json')).get('verification_passed', False))" 2>/dev/null || echo "False")
  if [[ "$VERIFIED" == "True" ]]; then
    ok "Adapter completed successfully."
    log "Worktree preserved for manual commit: $WORKTREE_DIR"
    log "  Clean up when done: $TL_CMD clean $ARCHITECTURE"
  else
    log "Session ended but adapter not verified. Worktree preserved for resume."
    log "  Resume: $TL_CMD --architecture $ARCHITECTURE"
    log "  Clean:  $TL_CMD clean $ARCHITECTURE"
  fi
fi
