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
2. **Analyze the HF model**: read its `config.json` and source. Identify embedding, attention, MLP, normalization, and output head layouts. Note any quirks (rotary embeddings, GQA/MQA, RMSNorm vs LayerNorm, biases, attention sinks, MoE routing).
3. **Pick a starting adapter** from [transformer_lens/model_bridge/supported_architectures/](../../transformer_lens/model_bridge/supported_architectures/) closest to the target architecture. Copy it (or [docs/source/_static/adapter-template.py](../../docs/source/_static/adapter-template.py)) into a new file named after the architecture.
4. **Fill in component mappings** so each HF module path resolves to a canonical TransformerLens Bridge name. Hook names should be Bridge-native (`blocks.{i}.hook_out` etc.), not HookedTransformer aliases.
5. **Register the adapter** in BOTH:
   - [transformer_lens/model_bridge/supported_architectures/__init__.py](../../transformer_lens/model_bridge/supported_architectures/__init__.py)
   - [transformer_lens/factories/architecture_adapter_factory.py](../../transformer_lens/factories/architecture_adapter_factory.py)
6. **Add the HF repo path to the Bridge registry** under [transformer_lens/tools/model_registry/](../../transformer_lens/tools/model_registry/). Do NOT add it to [transformer_lens/supported_models.py](../../transformer_lens/supported_models.py) — that file is HookedTransformer-only.
7. **Verify** end-to-end: run `/verify-model $ARGUMENTS` (one model only — do not parallelize).
8. **Write an integration test** under [tests/integration/](../../tests/integration/) that asserts logit parity with HuggingFace. Use fp32 + eager attention to match `boot_transformers`' load configuration. Gate probes for optional structural features (`resid_mid` etc.) rather than assuming they exist.
9. **Run `/task-complete`** — handles comment cleanup, `/format`, and the standard test tiers (unit + docstring + acceptance + integration), looping until clean.
