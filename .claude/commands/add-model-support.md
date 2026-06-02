---
description: Guided workflow for adding a new architecture adapter to TransformerBridge.
argument-hint: <hf_repo>
---

You are adding TransformerBridge support for the HuggingFace model `$ARGUMENTS`. If `$ARGUMENTS` is empty, ask the user for the HF repo path before continuing.

Read these first:

- [transformer_lens/model_bridge/supported_architectures/AGENTS.md](../../transformer_lens/model_bridge/supported_architectures/AGENTS.md) — adapter contract, starter-adapter table, registration steps, common gotchas
- [docs/source/content/adapter_development/adapter-creation-guide.md](../../docs/source/content/adapter_development/adapter-creation-guide.md) and [docs/source/content/adapter_development/hf-model-analysis-guide.md](../../docs/source/content/adapter_development/hf-model-analysis-guide.md) — authoritative step-by-step references

Execute this checklist, stopping at each step until it is genuinely done:

1. Confirm the model is not already supported by an existing architecture adapter. Short circuit and direct the user to the adapter if it already exists.
2. **Analyze the HF model**: read its `config.json` and source. Identify embedding, attention, MLP, normalization, and output head layouts. Note any quirks (rotary embeddings, GQA/MQA, RMSNorm vs LayerNorm, biases, attention sinks, MoE routing). Then list every non-standard HF config attribute that the base adapter machinery doesn't already map — e.g. `final_logit_softcapping` / `attn_logit_softcapping`, `query_pre_attn_scalar`, `sliding_window`, `layer_types`, custom `eps_attr` names — and decide for each whether it needs to be surfaced on `self.cfg`. HF-raw attrs are invisible to TL-side consumers unless explicitly propagated.
3. **Pick a starting adapter** from [transformer_lens/model_bridge/supported_architectures/](../../transformer_lens/model_bridge/supported_architectures/) closest to the target architecture. Copy it (or [docs/source/_static/adapter-template.py](../../docs/source/_static/adapter-template.py)) into a new file named after the architecture. **Do NOT inherit `default_prepend_bos` from the starter** — that flag reflects the target's tokenizer policy, not its architecture, and same-family models often differ. Check the target's tokenizer config or chat-template behavior before setting it; if unsure, leave it unset and let the framework default apply.
4. **Fill in component mappings** so each HF module path resolves to a canonical TransformerLens Bridge name. Hook names should be Bridge-native (`blocks.{i}.hook_out` etc.), not HookedTransformer aliases.
5. **Register the adapter** in ALL of:
   - [transformer_lens/model_bridge/supported_architectures/__init__.py](../../transformer_lens/model_bridge/supported_architectures/__init__.py) — import + `__all__` entry
   - [transformer_lens/factories/architecture_adapter_factory.py](../../transformer_lens/factories/architecture_adapter_factory.py) — import + `SUPPORTED_ARCHITECTURES` entry (key must match `config.architectures[0]` exactly)
   - [transformer_lens/tools/model_registry/\_\_init\_\_.py](../../transformer_lens/tools/model_registry/__init__.py) — `HF_SUPPORTED_ARCHITECTURES` set entry + `CANONICAL_AUTHORS_BY_ARCH` map entry
   - [transformer_lens/tools/model_registry/generate_report.py](../../transformer_lens/tools/model_registry/generate_report.py) — one-line entry in `ARCHITECTURE_DESCRIPTIONS` so the generated docs table covers the new architecture
6. **Add the HF repo path** to [transformer_lens/tools/model_registry/data/supported_models.json](../../transformer_lens/tools/model_registry/data/supported_models.json) so `/verify-model` can resolve it. Do NOT add it to [transformer_lens/supported_models.py](../../transformer_lens/supported_models.py) — that file is HookedTransformer-only. After adding the user-provided repo, ask whether to also add the canonical sibling variants under the same org (the values from `CANONICAL_AUTHORS_BY_ARCH[<HFArchClass>]`) — e.g. when adding `google/gemma-2-2b`, prompt about `google/gemma-2-2b-it`, `google/gemma-2-9b`, `google/gemma-2-9b-it`, `google/gemma-2-27b`, `google/gemma-2-27b-it`. The HF scraper picks these up eventually, but explicit entries unblock `/verify-model` against the siblings now.
7. **Verify** end-to-end: run `/verify-model $ARGUMENTS` (one model only — do not parallelize).
8. **Write an integration test** under [tests/integration/](../../tests/integration/) that asserts logit parity with HuggingFace. Use fp32 + eager attention to match `boot_transformers`' load configuration. Gate probes for optional structural features (`resid_mid` etc.) rather than assuming they exist.
9. **Run `/task-complete`** — handles comment cleanup, `/format`, and the standard test tiers (unit + docstring + acceptance + integration), looping until clean.
