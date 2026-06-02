---
description: Guided workflow for adding a new architecture adapter to TransformerBridge.
argument-hint: <hf_repo>
---

You are adding TransformerBridge support for the HuggingFace model `$ARGUMENTS`. If `$ARGUMENTS` is empty, ask the user for the HF repo path before continuing.

Each step below names the doc to read **only when you reach that step** — do not load all of them up front.

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

   Always ask the user before running verification, even for small models. To give them an informed decision:

   1. Run dry-run to project cost:
      ```
      set -a; source .env; set +a
      uv run python -m transformer_lens.tools.model_registry.verify_models --model "$ARGUMENTS" --dry-run
      ```
   2. Show the user: model ID, architecture class, estimated parameters, projected memory (GB), whether `HF_TOKEN` is needed, runtime expectation (30 s – 2 min sub-1B, 2–15 min 1B–7B, 15+ min 7B+ or multimodal), and what verification does (Phases 1–4; updates `supported_models.json` on success).
   3. Ask: "Run verification on this machine? (Y/N)"

   **Confirm** → `/verify-model $ARGUMENTS`. On pass, registry updated, done. On fail, see [debugging_numerical_divergence.md](../../docs/source/content/debugging_numerical_divergence.md) (per-sibling adapter bug, not a new adapter).

   **Reject** → `gh issue create --template verify-model.md` (fill from the dry-run output you already collected). If `gh` isn't installed: <https://github.com/TransformerLensOrg/TransformerLens/issues/new?template=verify-model.md>. Stop the workflow.

2. **Analyze the HF model.** Read its `config.json` and source. Identify embedding, attention, MLP, normalization, and output-head layouts. Now read [Config-attr propagation in supported_architectures/AGENTS.md](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#config-attr-propagation) and run through the non-standard attributes table (`final_logit_softcapping`, `sliding_window`, etc.); decide which need to surface on `self.cfg`.

3. **Pick a starting adapter.** Read the [starter-adapter table in supported_architectures/AGENTS.md](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#starter-adapter-table). Copy the closest match into [`transformer_lens/model_bridge/supported_architectures/`](../../transformer_lens/model_bridge/supported_architectures/) as `<arch>.py`. **Watch tokenizer-policy flags** — see [Tokenizer policy in supported_architectures/AGENTS.md](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#tokenizer-policy) (`default_prepend_bos`, padding side, EOS handling, chat-template wiring are all per-model, not per-architecture).

4. **Fill in `self.component_mapping`** so each HF module path resolves to a canonical Bridge name. Hook names are Bridge-native. The [Minimal contract](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#minimal-contract) and [Common gotchas](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#common-gotchas) tables in supported_architectures/AGENTS.md are the references for this step.

5. **Register in all four sites** per the [registration steps in supported_architectures/AGENTS.md](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#registration-steps). After registering, run the invariant test to confirm the four-place wiring: `uv run pytest tests/unit/tools/test_model_registry.py -k TestRegistrySyncedWithFactory`.

6. **Add the HF repo entry** to [`data/supported_models.json`](../../transformer_lens/tools/model_registry/data/supported_models.json) per [supported_architectures/AGENTS.md §"Adding the HF repo to the registry"](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#adding-the-hf-repo-to-the-registry). Then ask the user whether to also add the canonical sibling variants from `CANONICAL_AUTHORS_BY_ARCH[<HFArchClass>]`.

7. **Verify** end-to-end: run `/verify-model $ARGUMENTS`. (`/verify-model` enforces dry-run-first and one-at-a-time.)

8. **Write tests.** Read [Required tests in supported_architectures/AGENTS.md](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#required-tests) for the two-layer pattern (unit adapter test + integration parity test) and copy the closest sibling.

9. **Run `/task-complete`** — handles comment cleanup, `/format`, and the standard test tiers, looping until clean.
