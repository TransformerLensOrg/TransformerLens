# Memory Lock

A flock-based lock at `/tmp/tl-adapter-builder.lock` prevents concurrent
memory-intensive operations across all agent pairs.

## When to use the lock

**REQUIRES the lock:**
- Running `verify_models` or `run_benchmark_suite`
- Loading a full HuggingFace model (e.g., `AutoModelForCausalLM.from_pretrained`)

**Does NOT require the lock (everything else):**
- Reading/writing/editing source files, git, unit tests, scripts, installs, etc.

## Usage — ALWAYS use the `run` subcommand

The `run` subcommand acquires the lock, executes the wrapped command while
holding the lock, and releases the lock when the wrapped command exits — all
in **one** shell process. This is the ONLY pattern that works from the Bash
tool:

```bash
"$TL_ADAPTER_BUILDER_ROOT/agents/overlord-request.sh" run "verify_models: <model_id>" -- \
  uv run python -m transformer_lens.tools.model_registry.verify_models \
    --model <model_id> \
    --max-memory $MAX_MEMORY_GB \
    --device cpu --dtype float32
```

The `--` separator terminates lock arguments; everything after it is the
wrapped command. Exit code of the wrapped command propagates out.

## Why NOT `source … && overlord_acquire … ; <cmd> ; overlord_release`

That pattern looks reasonable but is **broken** in this environment. Each
Bash tool call runs in a fresh shell — functions sourced in one call are
not defined in the next, and the lock fd opened by `overlord_acquire` is
held by the current shell, so it's released the instant that shell exits.
If you split acquire / work / release across separate Bash tool calls,
each call acquires and immediately releases the lock, providing **zero**
protection.

```bash
# DO NOT do this — each line runs in its own shell, the lock is gone by
# the time you reach the work command:
source "$TL_ADAPTER_BUILDER_ROOT/agents/overlord-request.sh"   # call 1 (lock acquired + released)
overlord_acquire "verify_models"                               # call 2 (overlord_acquire not defined)
uv run python -m transformer_lens.tools.model_registry.verify_models ...   # call 3 (unprotected)
overlord_release                                               # call 4 (no-op)
```

**Always use `overlord-request.sh run` to wrap the command. Do NOT source.**

## Check lock status

```bash
"$TL_ADAPTER_BUILDER_ROOT/agents/overlord-request.sh" status
```
