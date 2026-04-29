# Architecture Adapter Creation Guide

A walkthrough for developers writing a new Architecture Adapter for the TransformerLens `TransformerBridge` system. This guide distills the process of developing an adapter into a set of steps that can be followed start-to-finish.

If you just want the API reference, jump to [adapter-specification.md](adapter-specification.md). If you have a specific HF model in hand and want a config-extraction cookbook, see [hf-model-analysis-guide.md](hf-model-analysis-guide.md). This document ties those together with workflow and review practice.

## What an adapter is

An **Architecture Adapter** is a Python class that extends `ArchitectureAdapter` and tells `TransformerBridge` three things about a HuggingFace model:

1. **Config attributes** — set on `self.cfg` in `__init__` (normalization type, positional embedding type, GQA params, etc.)
2. **Component mapping** — `self.component_mapping`, a dict mapping TransformerLens canonical names (`embed`, `blocks`, `attn.q`, …) to `GeneralizedComponent` Bridge instances pointed at HF module paths.
3. **Weight processing conversions** — `self.weight_processing_conversions`, a dict of tensor-reshape rules that translate HF weight layouts to TL layouts during loading.

Once registered, users can `boot_transformers("<your-model>")` and get a fully hooked TransformerLens model with weights loaded from HF.

## Prerequisites

Before starting, make sure you can:

- Read PyTorch model code and trace a forward pass
- Run a HF model locally with `transformers`
- Use `model.named_modules()` and `model.state_dict()` to inspect structure
- Identify whether a model uses RoPE vs learned positional embeddings, RMSNorm vs LayerNorm, gated vs standard MLP, separate vs joint QKV, MHA vs GQA vs MQA. ([hf-model-analysis-guide.md](hf-model-analysis-guide.md) has a decision tree.)

You do **not** need to memorize every Bridge component — the existing adapters in `transformer_lens/model_bridge/supported_architectures/` are your reference library.

## Analyze the architecture

### Read the HF source

Open the two files that define the architecture in `transformers`:

- `models/<arch>/modeling_<arch>.py` — the model code
- `models/<arch>/configuration_<arch>.py` — the config class

Read every `__init__` and every `forward`. You are looking for:

- Module hierarchy: what's nested in what, named how
- Forward pass order: norm before/after attention? residual where?
- Bias presence on each linear layer
- Normalization type and the *exact* attribute name of its epsilon (`variance_epsilon`, `rms_norm_eps`, `layer_norm_eps`, `eps`, …)
- Attention type (MHA / GQA / MQA) and whether QKV are separate or joint
- MLP type (gated / standard) and projection names
- Anything that looks weird (special scaling, conditional padding, dtype upcasts in softmax, …)

Also extract the standard config-to-TL field mapping (see [hf-model-analysis-guide.md](hf-model-analysis-guide.md) for the table).

### Find the closest reference adapter

Almost every new model is a variant of an existing pattern. Pick the nearest match from `supported_architectures/`:

| If your model is like…                | Start from…                          |
|---------------------------------------|--------------------------------------|
| Llama, Mistral, Qwen2, Gemma, OLMo    | `llama.py`                           |
| Qwen2/Qwen3 (gated config, MLPBridge) | `qwen2.py`                           |
| GPT-2, GPT-J, GPT-Neo                 | `gpt2.py`                            |
| BLOOM, Falcon                         | `bloom.py` or `falcon.py`            |
| T5 / encoder-decoder                  | `t5.py`                              |
| MoE                                   | `mixtral.py` or `granite_moe.py`     |
| Multimodal (vision+text)              | `llava.py` or `gemma3_multimodal.py` |

### Write down what you found

Before writing any adapter code, take notes on the architecture. This is for your own use, it does not need to be formally documented. It will help inform your decisions going forward.

At minimum, capture:

- **Source files** — exact paths in `transformers`
- **Module hierarchy** — every HF module path you'll need, with line numbers in the source where it's defined
- **Config fields** — the HF names and their TL equivalents
- **Architectural properties** — normalization, position embeddings, attention type, MLP type, biases
- **Forward pass flow** — order of operations in the block, attention, and MLP
- **Reference adapter** — closest existing adapter, and a list of every way your target differs from it
- **Representative models** — small variants (≤7B parameters) you'll use for verification

## Implement the adapter

### File layout

- **Adapter file:** `transformer_lens/model_bridge/supported_architectures/<model_name>.py`
- **Class name:** `<ModelName>ArchitectureAdapter` (e.g. `LlamaArchitectureAdapter`)
- **Module name:** lowercase + underscores (`llama.py`, `qwen2.py`, `granite_moe.py`)

Start from [adapter-template.py](../../_static/adapter-template.py). It's a Llama-pattern skeleton with TODOs at every decision point.

A reasonable order for filling it in:

1. Config attributes (drives everything else)
2. Weight processing conversions
3. Component mapping
4. Optional overrides (only the ones you actually need)
5. Registration

### Config attributes

Set these on `self.cfg` in `__init__` *before* building the component mapping (the bridges read from `self.cfg`):

| Attribute                  | Type   | Purpose                                       |
|----------------------------|--------|-----------------------------------------------|
| `normalization_type`       | `str`  | `"RMS"` or `"LN"`                             |
| `positional_embedding_type`| `str`  | `"rotary"` or `"standard"`                    |
| `final_rms`                | `bool` | Final norm is RMSNorm                         |
| `gated_mlp`                | `bool` | MLP has gate projection (SwiGLU)              |
| `attn_only`                | `bool` | Model has no MLP layers (rare)                |
| `uses_rms_norm`            | `bool` | Should match `normalization_type == "RMS"`    |
| `eps_attr`                 | `str`  | HF attribute name for norm epsilon            |

For GQA models, also forward `n_key_value_heads`:

```python
if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
    self.cfg.n_key_value_heads = cfg.n_key_value_heads
```

### Component mapping

For each TL canonical name, instantiate the right Bridge component and point its `name=` parameter at the HF module path (relative to the model root for top-level entries, relative to the block for block submodules).

A standard Llama-style mapping:

```python
self.component_mapping = {
    "embed":       EmbeddingBridge(name="model.embed_tokens"),
    "rotary_emb":  RotaryEmbeddingBridge(name="model.rotary_emb"),
    "blocks": BlockBridge(
        name="model.layers",
        submodules={
            "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
            "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
            "attn": PositionEmbeddingsAttentionBridge(
                name="self_attn",
                config=self.cfg,
                submodules={
                    "q": LinearBridge(name="q_proj"),
                    "k": LinearBridge(name="k_proj"),
                    "v": LinearBridge(name="v_proj"),
                    "o": LinearBridge(name="o_proj"),
                },
                requires_attention_mask=True,
                requires_position_embeddings=True,
            ),
            "mlp": GatedMLPBridge(
                name="mlp",
                config=self.cfg,
                submodules={
                    "gate": LinearBridge(name="gate_proj"),
                    "in":   LinearBridge(name="up_proj"),
                    "out":  LinearBridge(name="down_proj"),
                },
            ),
        },
    ),
    "ln_final":  RMSNormalizationBridge(name="model.norm", config=self.cfg),
    "unembed":   UnembeddingBridge(name="lm_head", config=self.cfg),
}
```

The full bridge component catalog (attention variants, MLP variants, specialized bridges for BLOOM/CLIP/Siglip/T5/MoE/etc.) is in [adapter-specification.md](adapter-specification.md) under "Available Bridge Components."

### Weight processing conversions

For models with separate Q/K/V/O projections, use the built-in helper:

```python
self.weight_processing_conversions = {
    **self._qkvo_weight_conversions(),
}
```

It generates the standard `(n h) m -> n m h` rearrangements with the right head/kv-head counts.

For combined-QKV models (GPT-2 style), see `gpt2.py`'s `QKVSplitRearrangeConversion` for the pattern. For other oddball layouts, define custom `ParamProcessingConversion` or `RearrangeTensorConversion` instances.

### Optional overrides

Implement only the ones you need:

- **`setup_component_testing(hf_model, bridge_model=None)`** — required for RoPE models, to wire the rotary embedding instance through to the attention bridges. Skip for models with standard positional embeddings.
- **`preprocess_weights(state_dict)`** — for arch-specific weight transforms before standard processing (e.g., Gemma scales embeddings by `sqrt(d_model)`).
- **`prepare_loading(model_name, model_kwargs)`** — patch HF model classes before `from_pretrained()`.
- **`prepare_model(hf_model)`** — post-load fixups before bridge creation.

### Registration

Two files to update:

1. `transformer_lens/model_bridge/supported_architectures/__init__.py` — add the import and append to `__all__`.
2. `transformer_lens/factories/architecture_adapter_factory.py` — add to the import block and to `SUPPORTED_ARCHITECTURES`:

   ```python
   "<HFArchitectureClass>": <YourAdapterClass>,
   ```

Forgetting registration is the most common silent failure — the adapter exists but `boot_transformers` can't find it.

### Tests

Write tests that exercise actual behavior:

- Hook names resolve correctly
- Weight shapes match expectations after loading
- Forward pass produces sensible output for a tiny variant

### New bridge components

Don't add a new bridge unless the existing ones can't express your model. The bar is: the `forward()` must be fundamentally different from any existing bridge. If you do add one:

- Place it in `transformer_lens/model_bridge/generalized_components/`
- Export it from the package `__init__`
- Write tests covering its forward pass and any state it carries

## Verify the adapter

The `verify_models` tool runs a real HF model side-by-side with your bridge and compares activations across four phases. Each phase produces a numeric score; the model passes if all phase scores meet their thresholds.

### Pick models

From your representative-models list, take the smallest variants (prefer ≤7B parameters), up to 5, sorted by HuggingFace download count. Verifying multiple sizes catches scaling bugs that single-model verification misses.

### Run verification

One model at a time, with float32 by default:

```bash
uv run python -m transformer_lens.tools.model_registry.verify_models \
  --model <model-id> \
  --max-memory <GB> \
  --device cpu \
  --dtype float32 \
  --no-ht-reference
```

If a model OOMs with float32, retry that single model with `--dtype bfloat16`. Set `--max-memory` to roughly 75-85% of your device memory, to ensure adequate space for running the benchmarks.

### Read the status

Each model gets a status:

- **status=1** — passed, move to the next model
- **status=2** — skipped by `verify_models` (e.g., exceeded the memory pre-check). Note it and move on; not an adapter bug.
- **status=3** — phase score failure. Stop and fix. Read the `note` and the per-phase scores, find the root cause, fix the adapter, re-verify.

### Lint

After all chosen models pass:

```bash
uv run mypy .
make check-format
```

Both must be clean. Don't paper over mypy errors with `# type: ignore` — fix the underlying type. If mypy is wrong about something, that's a real issue worth investigating, not silencing.

## Before you open a PR

`verify_models` will catch most numerical bugs, but a few things are worth a once-over by eye.

**Sanity-check against the HF source.** Skim your adapter with the HF `modeling_<arch>.py` open alongside it. Module paths, config attribute names, and bias presence are the usual suspects — easy to get wrong from memory and easy to spot when you look directly.

**Watch for the subtle stuff.** When the adapter reimplements a computation or defines weight conversions, the things that bite are operation order (split before or after the layernorm?), dtype upcasting in softmax, and conditional logic that only fires under certain conditions in HF (e.g., flash-attention paths). If something in your code looks like it "probably matches" HF, that's a good place to stop and check.

**Don't reach for abstraction prematurely.** If you've added a base class or protocol with only one or two concrete uses, you're probably better off without it. The same goes for config knobs that don't have a current consumer.

**Confirm the boring stuff is done.** Both registration sites (`__init__.py` and `architecture_adapter_factory.py`), `mypy` and format checks clean, tests doing real work rather than asserting mocks return their mock values.

## Common pitfalls

- **Wrong `eps_attr` name.** Models that look identical use different attribute names (`variance_epsilon`, `rms_norm_eps`, `eps`). Read the norm class.
- **Forgetting `n_key_value_heads`.** Without it, GQA models silently reshape weights as if they were MHA — verification fails with cryptic shape errors.
- **Missing registration.** Adapter exists but the factory can't find it. Update both `__init__.py` and `architecture_adapter_factory.py`.
- **Skipping `setup_component_testing` for RoPE.** Rotary embeddings need to be wired through to each attention bridge or component testing produces nonsense.
- **Reusing `model.norm` when the path is `model.final_layernorm`.** Module paths look similar across architectures but rarely match exactly — always verify against the actual HF source.
- **Tautological tests.** "Test that mock returns mock_value" is not a test. Tests should exercise real shapes, real forward passes, real hook resolution.
- **`# type: ignore` on mypy errors.** Find the root cause; the type error is usually telling you something real about the bridge config.
- **Coding before the architecture is understood.** The single biggest time-waster. Five pages of code based on a wrong assumption about module paths is worse than no code.
