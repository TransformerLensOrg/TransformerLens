---
description: Guided workflow for adding a new architecture adapter to TransformerBridge.
argument-hint: <hf_repo>
---

You are adding TransformerBridge support for the HuggingFace model `$ARGUMENTS`. If `$ARGUMENTS` is empty, ask the user for the HF repo path before continuing.

**Read first:**

- [transformer_lens/model_bridge/supported_architectures/AGENTS.md](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md) — the adapter contract, starter-adapter table, **4-place registration steps**, config-attr propagation rules, the `default_prepend_bos` trap, common gotchas, and how to add the model entry to `supported_models.json`.
- [transformer_lens/tools/model_registry/AGENTS.md](../../transformer_lens/tools/model_registry/AGENTS.md) — the verification workflow and the `verify_models` vs `main_benchmark` trap.
- [docs/source/content/adapter_development/adapter-creation-guide.md](../../docs/source/content/adapter_development/adapter-creation-guide.md) and [docs/source/content/adapter_development/hf-model-analysis-guide.md](../../docs/source/content/adapter_development/hf-model-analysis-guide.md) — authoritative step-by-step references.

Execute this checklist, stopping at each step until it is genuinely done:

1. **Check registry state and decide whether to verify.**

   Determine the current state:
   - **Architecture supported?** Check `SUPPORTED_ARCHITECTURES` in [`transformer_lens/factories/architecture_adapter_factory.py`](../../transformer_lens/factories/architecture_adapter_factory.py).
   - **Model in registry?** Check whether `$ARGUMENTS` appears in [`transformer_lens/tools/model_registry/data/supported_models.json`](../../transformer_lens/tools/model_registry/data/supported_models.json), and if so what `status` it has (0=unverified, 1=verified, 2=skipped, 3=failed).

   Then branch:

   - **Architecture supported AND model already in registry with `status == 1` (verified)** → the model is already verified end-to-end. Ask the user what symptom they're seeing — this is most likely a bug-report path, not an add-support path. Stop the workflow here.
   - **Architecture supported, model in registry with `status != 1`** (0 unverified, 2 skipped, 3 failed) → proceed to **Confirm before verification** below. If `status == 3`, read the existing `note` field first to see what failure mode you may be about to re-trigger.
   - **Architecture supported, model NOT in registry** → add a registry entry per [supported_architectures/AGENTS.md §"Adding the HF repo to the registry"](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#adding-the-hf-repo-to-the-registry) with `status: 0` and null phase scores, then proceed to **Confirm before verification**.
   - **Architecture NOT supported** → skip to step 2 below (the full adapter-authoring workflow).

   ### Confirm before verification

   **Always** ask the user before running verification — even for small models. Verification is non-trivial and resource-bound. To give them an informed decision:

   1. **Get a resource projection** without loading the model:
      ```
      set -a; source .env; set +a
      uv run python -m transformer_lens.tools.model_registry.verify_models --model "$ARGUMENTS" --dry-run
      ```
      Capture the projected memory and parameter-count estimate from the dry-run output.
   2. **Present the user with the cost of verification**:
      - **Model**: `<model_id>` (HF repo: `https://huggingface.co/<model_id>`)
      - **Architecture**: `<HFArchClass>` (from `config.architectures[0]`)
      - **Estimated parameters**: `<N>` (from the dry-run)
      - **Projected memory**: `<X> GB` (from the dry-run) — they need at least this much free on the target device
      - **Expected runtime**: typically 30 s – 2 min for sub-1B models, 2–15 min for 1B–7B, 15+ min for 7B+ or multimodal
      - **What verification does**: loads the model, runs Phases 1–4 (forward correctness vs HF, hook firing + gradients, weight processing, generation quality), and on success updates `supported_models.json` via `update_model_status()`
      - **Token requirement**: whether `HF_TOKEN` is needed (gated repos: Llama, Mistral, Gemma, gated Qwen variants)
   3. **Ask explicitly**: "Do you want me to run verification on this machine? (Y/N)"

   **If the user confirms** → run `/verify-model $ARGUMENTS`. On pass, the registry now reflects verified status and you're done — no new code needed. On fail, the failure mode tells you whether to fix the existing adapter (per-sibling regression — see [debugging_numerical_divergence.md](../../docs/source/content/debugging_numerical_divergence.md)) or escalate.

   **If the user rejects** → direct them to file a tracking issue so a maintainer with appropriate hardware can run the verification:
   ```
   gh issue create \
     --title "Verify model support: <model_id>" \
     --body "$(cat <<'EOF'
   ## Model
   - Repo: https://huggingface.co/<model_id>
   - Architecture: <HFArchClass>
   - Estimated parameters: <N>
   - Projected memory: <X> GB
   - Gated: yes / no (HF_TOKEN required: yes / no)

   ## Registry state
   - Architecture adapter exists: yes
   - In supported_models.json: yes (status: <N>) / no — entry added in this issue's PR

   ## Motivation
   <what the user is trying to do; any symptom they've observed>

   ## Next step
   Run on appropriate hardware and report results:
   \`\`\`
   set -a; source .env; set +a
   uv run python -m transformer_lens.tools.model_registry.verify_models --model <model_id>
   \`\`\`
   EOF
   )"
   ```
   If `gh` isn't installed, link them to `https://github.com/TransformerLensOrg/TransformerLens/issues/new` with the same body content. Then stop the workflow — the registry-entry add (if you made one above) can either land in its own small PR or be folded into the maintainer's verification PR; either is fine.

2. **Analyze the HF model.** Read its `config.json` and source. Identify embedding, attention, MLP, normalization, and output-head layouts. Run through the [Config-attr propagation table in supported_architectures/AGENTS.md](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#config-attr-propagation) for non-standard attributes (`final_logit_softcapping`, `sliding_window`, etc.) and decide which need to surface on `self.cfg`.
3. **Pick a starting adapter** using the [starter-adapter table](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#starter-adapter-table). Copy it into [`transformer_lens/model_bridge/supported_architectures/`](../../transformer_lens/model_bridge/supported_architectures/) as `<arch>.py`. **Watch tokenizer-policy flags** — see [Tokenizer policy in supported_architectures/AGENTS.md](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#tokenizer-policy) (`default_prepend_bos`, padding side, EOS handling, chat-template wiring are all per-model, not per-architecture).
4. **Fill in `self.component_mapping`** so each HF module path resolves to a canonical Bridge name. Hook names are Bridge-native.
5. **Register in all four sites** per the [registration steps in supported_architectures/AGENTS.md](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#registration-steps).
6. **Add the HF repo entry** to [`data/supported_models.json`](../../transformer_lens/tools/model_registry/data/supported_models.json) per the [Adding the HF repo to the registry section](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#adding-the-hf-repo-to-the-registry). Then ask the user whether to also add the canonical sibling variants from `CANONICAL_AUTHORS_BY_ARCH[<HFArchClass>]`.
7. **Verify** end-to-end: run `/verify-model $ARGUMENTS` (one model only — do not parallelize).
8. **Write an integration test** under [tests/integration/](../../tests/integration/) that asserts logit parity with HuggingFace. Use fp32 + eager attention to match `boot_transformers`' load configuration. Gate probes for optional structural features (`resid_mid` etc.) rather than assuming they exist.
9. **Run `/task-complete`** — handles comment cleanup, `/format`, and the standard test tiers, looping until clean.
