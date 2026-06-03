---
description: Guided workflow for adding a new architecture adapter to TransformerBridge.
argument-hint: <hf_repo>
---

Adding TransformerBridge support for HF model `$ARGUMENTS`. If empty, ask the user for the HF repo path first.

Each step names the doc to read **when you reach that step** — don't load all up front.

1. **Check registry state and decide whether to verify.**

   State:
   - Architecture supported? Check `SUPPORTED_ARCHITECTURES` in [`architecture_adapter_factory.py`](../../transformer_lens/factories/architecture_adapter_factory.py).
   - Model in registry? Check [`supported_models.json`](../../transformer_lens/tools/model_registry/data/supported_models.json); note `status` (0=unverified, 1=verified, 2=skipped, 3=failed).

   Branch:

   - **Supported AND `status==1`** → already verified. Ask the user the symptom (bug-report path, not add-support). Stop.
   - **Supported, `status != 1`** → proceed to **Confirm before verification**. If `status==3`, read existing `note` for the prior failure mode.
   - **Supported, not in registry** → add an entry per [§Adding the HF repo to the registry](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#adding-the-hf-repo-to-the-registry) with `status: 0` and null scores, then proceed.
   - **Not supported** → skip to step 2.

   ### Confirm before verification

   Always ask the user first, even for small models:

   1. Dry-run to project cost:
      ```
      set -a; source .env; set +a
      uv run python -m transformer_lens.tools.model_registry.verify_models --model "$ARGUMENTS" --dry-run
      ```
   2. Show: model ID, architecture class, estimated parameters, projected memory (GB), HF_TOKEN needed?, runtime (30 s–2 min sub-1B, 2–15 min 1B–7B, 15+ min 7B+/multimodal), what verification does (Phases 1–4; updates `supported_models.json` on success).
   3. Ask: "Run verification on this machine? (Y/N)"

   **Confirm** → `/verify-model $ARGUMENTS`. On pass, done. On fail, see [debugging_numerical_divergence.md](../../docs/source/content/debugging_numerical_divergence.md) (per-sibling adapter bug).

   **Reject** → `gh issue create --template verify-model.md` (fill from dry-run output). No `gh`? <https://github.com/TransformerLensOrg/TransformerLens/issues/new?template=verify-model.md>. Stop.

2. **Analyze the HF model.** Read `config.json` and source — identify embedding, attention, MLP, normalization, output-head layouts. Read [§Config-attr propagation](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#config-attr-propagation) and decide which non-standard attrs (`final_logit_softcapping`, `sliding_window`, etc.) need surfacing on `self.cfg`.

3. **Pick a starting adapter.** See [§Starter-adapter table](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#starter-adapter-table). Copy into [`supported_architectures/`](../../transformer_lens/model_bridge/supported_architectures/) as `<arch>.py`. **Tokenizer-policy flags are per-model** — see [§Tokenizer policy](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#tokenizer-policy).

4. **Fill `self.component_mapping`.** Bridge-native hook names. Reference: [§Minimal contract](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#minimal-contract), [§Common gotchas](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#common-gotchas).

5. **Register in all four sites** per [§Registration steps](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#registration-steps). Then run the invariant test: `uv run pytest tests/unit/tools/test_model_registry.py -k TestRegistrySyncedWithFactory`.

6. **Add the HF repo entry** to [`data/supported_models.json`](../../transformer_lens/tools/model_registry/data/supported_models.json) per [§Adding the HF repo to the registry](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#adding-the-hf-repo-to-the-registry). Ask the user about adding canonical sibling variants from `CANONICAL_AUTHORS_BY_ARCH[<HFArchClass>]`.

7. **Verify** end-to-end: `/verify-model $ARGUMENTS`. Read both `status` AND per-phase scores. `STATUS_VERIFIED` means hard gates passed (see [§Phase-score thresholds](../../transformer_lens/tools/model_registry/AGENTS.md#phase-score-thresholds)) — but P4's 50% bar is intentionally lenient. P4 well below 100% on a small parity-test model + `status==1` → suspect missing `preprocess_weights` fold or wrong `default_prepend_bos`; investigate before step 8.

8. **Write tests** per [§Required tests](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md#required-tests) (unit + integration). Copy the closest sibling.

9. **`/task-complete`** — comment cleanup, `/format`, standard test tiers, loop until clean.
