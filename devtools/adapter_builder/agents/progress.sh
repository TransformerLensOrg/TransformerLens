#!/usr/bin/env bash
# =============================================================================
# progress.sh — Lightweight adapter build progress tracking
#
# Writes/reads .adapter-progress.json in the worktree to track lifecycle state.
# Used by launch-agent-pair.sh to construct resume prompts after crashes.
#
# All writes use Python's fcntl.flock to prevent concurrent
# orchestrator/programmer/hook updates from clobbering each other. This
# works on both Linux and macOS (unlike bash flock which isn't on macOS).
#
# Usage (sourced by launch-agent-pair.sh):
#   source agents/progress.sh
#   progress_init "$WORKTREE_DIR" "$ARCHITECTURE"
#   progress_update "$WORKTREE_DIR" "step" "planning"
#   progress_update "$WORKTREE_DIR" "current_phase" "B"
#   progress_add_completed_phase "$WORKTREE_DIR" "A"
#   progress_read "$WORKTREE_DIR"  # sets PROGRESS_* vars
# =============================================================================

PROGRESS_FILE=".adapter-progress.json"

# --------------------------------------------------------------------------- #
# Locked JSON write via Python fcntl.flock — works on Linux and macOS.
# The lock file is co-located with the progress file as .adapter-progress.lock
# so it's per-worktree, not global. The Python script is fed via heredoc to
# avoid bash variable expansion mangling multiline strings.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Initialize a new progress file
# --------------------------------------------------------------------------- #
progress_init() {
  local worktree="$1"
  local architecture="$2"
  local filepath="$worktree/$PROGRESS_FILE"
  local lockfile="${filepath%.json}.lock"
  local now
  now="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  python3 - "$filepath" "$lockfile" "$architecture" "$now" <<'PYEOF'
import json, fcntl, sys
filepath, lockfile, arch, now = sys.argv[1:5]
lockfd = open(lockfile, 'w')
try:
    fcntl.flock(lockfd, fcntl.LOCK_EX)
    json.dump({
        "architecture": arch,
        "step": "planning",
        "plan_approved": False,
        "current_phase": None,
        "completed_phases": [],
        "verification_attempts": 0,
        "verified_models": [],
        "verification_passed": False,
        "final_review_passed": False,
        "last_updated": now,
        "started_at": now,
    }, open(filepath, "w"), indent=2)
finally:
    fcntl.flock(lockfd, fcntl.LOCK_UN)
    lockfd.close()
PYEOF
}

# --------------------------------------------------------------------------- #
# Update a field in the progress file
# --------------------------------------------------------------------------- #
progress_update() {
  local worktree="$1"
  local key="$2"
  local value="$3"
  local filepath="$worktree/$PROGRESS_FILE"
  local lockfile="${filepath%.json}.lock"

  [[ ! -f "$filepath" ]] && return 1

  python3 - "$filepath" "$lockfile" "$key" "$value" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" <<'PYEOF'
import json, fcntl, sys
filepath, lockfile, key, value, now = sys.argv[1:6]
lockfd = open(lockfile, 'w')
try:
    fcntl.flock(lockfd, fcntl.LOCK_EX)
    with open(filepath) as f:
        p = json.load(f)
    if value == 'true': value = True
    elif value == 'false': value = False
    elif value == 'null': value = None
    elif value.isdigit(): value = int(value)
    p[key] = value
    p['last_updated'] = now
    with open(filepath, 'w') as f:
        json.dump(p, f, indent=2)
finally:
    fcntl.flock(lockfd, fcntl.LOCK_UN)
    lockfd.close()
PYEOF
}

# --------------------------------------------------------------------------- #
# Add a completed phase
# --------------------------------------------------------------------------- #
progress_add_completed_phase() {
  local worktree="$1"
  local phase="$2"
  local filepath="$worktree/$PROGRESS_FILE"
  local lockfile="${filepath%.json}.lock"

  [[ ! -f "$filepath" ]] && return 1

  python3 - "$filepath" "$lockfile" "$phase" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" <<'PYEOF'
import json, fcntl, sys
filepath, lockfile, phase, now = sys.argv[1:5]
lockfd = open(lockfile, 'w')
try:
    fcntl.flock(lockfd, fcntl.LOCK_EX)
    with open(filepath) as f:
        p = json.load(f)
    if phase not in p['completed_phases']:
        p['completed_phases'].append(phase)
    p['last_updated'] = now
    with open(filepath, 'w') as f:
        json.dump(p, f, indent=2)
finally:
    fcntl.flock(lockfd, fcntl.LOCK_UN)
    lockfd.close()
PYEOF
}

# --------------------------------------------------------------------------- #
# Read progress into shell variables (no lock needed — read-only)
# --------------------------------------------------------------------------- #
progress_read() {
  local worktree="$1"
  local filepath="$worktree/$PROGRESS_FILE"

  if [[ ! -f "$filepath" ]]; then
    PROGRESS_EXISTS=false
    return 1
  fi

  PROGRESS_EXISTS=true
  PROGRESS_STEP=$(python3 -c "import json; print(json.load(open('$filepath')).get('step','unknown'))" 2>/dev/null)
  PROGRESS_PLAN_APPROVED=$(python3 -c "import json; print(str(json.load(open('$filepath')).get('plan_approved',False)).lower())" 2>/dev/null)
  PROGRESS_CURRENT_PHASE=$(python3 -c "import json; print(json.load(open('$filepath')).get('current_phase','none'))" 2>/dev/null)
  PROGRESS_COMPLETED_PHASES=$(python3 -c "import json; print(','.join(json.load(open('$filepath')).get('completed_phases',[])))" 2>/dev/null)
  PROGRESS_VERIFICATION_ATTEMPTS=$(python3 -c "import json; print(json.load(open('$filepath')).get('verification_attempts',0))" 2>/dev/null)
  PROGRESS_VERIFIED_MODELS=$(python3 -c "import json; print(','.join(json.load(open('$filepath')).get('verified_models',[])))" 2>/dev/null)
  PROGRESS_LAST_UPDATED=$(python3 -c "import json; print(json.load(open('$filepath')).get('last_updated','unknown'))" 2>/dev/null)
}

# --------------------------------------------------------------------------- #
# Generate a resume prompt from progress state
# --------------------------------------------------------------------------- #
generate_resume_context() {
  local worktree="$1"
  progress_read "$worktree"

  if [[ "$PROGRESS_EXISTS" != "true" ]]; then
    echo ""
    return
  fi

  cat <<RESUME

## RESUMING FROM CRASH

This session is resuming from a previous crash. Here is the saved state:

- **Step:** ${PROGRESS_STEP}
- **Plan approved:** ${PROGRESS_PLAN_APPROVED}
- **Current phase:** ${PROGRESS_CURRENT_PHASE}
- **Completed phases:** ${PROGRESS_COMPLETED_PHASES:-none}
- **Verification attempts:** ${PROGRESS_VERIFICATION_ATTEMPTS}
- **Last updated:** ${PROGRESS_LAST_UPDATED}

### What to do:
RESUME

  case "$PROGRESS_STEP" in
    planning)
      if [[ "$PROGRESS_PLAN_APPROVED" == "true" ]]; then
        echo "The plan was approved. Begin implementation at Phase A (or the first incomplete phase)."
      else
        echo "The plan was not yet approved. Check if a plan file exists and resume the planning/review cycle."
      fi
      ;;
    programming)
      echo "Implementation was in progress."
      if [[ -n "$PROGRESS_COMPLETED_PHASES" ]]; then
        echo "Phases already completed and approved: ${PROGRESS_COMPLETED_PHASES}"
      fi
      if [[ "$PROGRESS_CURRENT_PHASE" != "none" && "$PROGRESS_CURRENT_PHASE" != "None" ]]; then
        echo "Resume from phase: ${PROGRESS_CURRENT_PHASE}. Check git diff to see what was already written for this phase."
      else
        echo "Check git diff and the plan file to determine which phase to start next."
      fi
      ;;
    verification)
      echo "Verification was in progress (attempt #${PROGRESS_VERIFICATION_ATTEMPTS})."
      if [[ -n "$PROGRESS_VERIFIED_MODELS" ]]; then
        echo "Models already verified: ${PROGRESS_VERIFIED_MODELS}. Skip these and continue with remaining models."
      fi
      echo "Re-run verify_models on unverified models."
      ;;
    *)
      echo "Unknown state. Read the plan file and check git diff to determine where to resume."
      ;;
  esac
}
