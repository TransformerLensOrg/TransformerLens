---
orphan: true
---

# Architecture Adapter Specification

This document is the primary reference for building Architecture Adapters for the TransformerLens TransformerBridge system.

## What Is an Architecture Adapter?

An Architecture Adapter is a Python class that extends `ArchitectureAdapter` (from `transformer_lens.model_bridge.architecture_adapter`). It maps between a HuggingFace model's internal structure and TransformerLens's canonical component names. Every adapter must define three things:

1. **Config attributes** — set on `self.cfg` in `__init__`
2. **Component mapping** — `self.component_mapping` dict mapping TL names to Bridge instances
3. **Weight processing conversions** — `self.weight_processing_conversions` dict for tensor reshaping

## File Location and Naming

- **Adapter file:** `transformer_lens/model_bridge/supported_architectures/<model_name>.py`
- **Class name:** `<ModelName>ArchitectureAdapter` (e.g., `LlamaArchitectureAdapter`)
- **Module name:** lowercase, underscores (e.g., `llama.py`, `qwen2.py`, `granite_moe.py`)

## Registration Checklist

After creating the adapter, register it in these files:

1. **`transformer_lens/model_bridge/supported_architectures/__init__.py`**
   - Add import: `from transformer_lens.model_bridge.supported_architectures.<module> import <ClassName>`
   - Add to `__all__` list

2. **`transformer_lens/factories/architecture_adapter_factory.py`**
   - Add import (in the existing import block from `supported_architectures`)
   - Add entry to `SUPPORTED_ARCHITECTURES` dict: `"<HFArchitectureClass>": <AdapterClass>`

## Config Attributes

Set these on `self.cfg` in `__init__` before building the component mapping:

| Attribute | Type | Description | Examples |
|-----------|------|-------------|----------|
| `normalization_type` | `str` | `"RMS"` or `"LN"` | Llama="RMS", GPT2="LN" |
| `positional_embedding_type` | `str` | `"rotary"` or `"standard"` | Llama="rotary", GPT2="standard" |
| `final_rms` | `bool` | Whether final layer norm is RMS | Llama=True, GPT2=False |
| `gated_mlp` | `bool` | Whether MLP uses gate projection | Llama=True, GPT2=False |
| `attn_only` | `bool` | Whether model has no MLP layers | Usually False |
| `uses_rms_norm` | `bool` | Redundant with normalization_type but needed | Match normalization_type |
| `eps_attr` | `str` | Attribute name for norm epsilon | `"variance_epsilon"`, `"layer_norm_eps"` |

### GQA (Grouped Query Attention)

If the model uses GQA (n_key_value_heads < n_heads), set:
```python
if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
    self.cfg.n_key_value_heads = cfg.n_key_value_heads
```

## Component Mapping

`self.component_mapping` is a `dict[str, GeneralizedComponent]` mapping TransformerLens canonical names to Bridge instances. The Bridge `name=` parameter is the HuggingFace module path.

### Standard Mapping (Llama-style decoder-only)

```python
self.component_mapping = {
    "embed": EmbeddingBridge(name="model.embed_tokens"),
    "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
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
                    "in": LinearBridge(name="up_proj"),
                    "out": LinearBridge(name="down_proj"),
                },
            ),
        },
    ),
    "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
    "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
}
```

### GPT2-style Mapping (standard positional embeddings, combined QKV)

```python
self.component_mapping = {
    "embed": EmbeddingBridge(name="transformer.wte"),
    "pos_embed": PosEmbedBridge(name="transformer.wpe"),
    "blocks": BlockBridge(
        name="transformer.h",
        config=self.cfg,
        submodules={
            "ln1": NormalizationBridge(name="ln_1", config=self.cfg),
            "attn": JointQKVAttentionBridge(
                name="attn",
                config=self.cfg,
                submodules={
                    "qkv": LinearBridge(name="c_attn"),
                    "o": LinearBridge(name="c_proj"),
                },
            ),
            "ln2": NormalizationBridge(name="ln_2", config=self.cfg),
            "mlp": MLPBridge(
                name="mlp",
                submodules={
                    "in": LinearBridge(name="c_fc"),
                    "out": LinearBridge(name="c_proj"),
                },
            ),
        },
    ),
    "ln_final": NormalizationBridge(name="transformer.ln_f", config=self.cfg),
    "unembed": UnembeddingBridge(name="lm_head"),
}
```

> **Note:** GPT2's `MLPBridge` and `UnembeddingBridge` do not pass `config=`. The `config` parameter is optional on these bridges — match the existing adapter's pattern.

## Weight Processing Conversions

`self.weight_processing_conversions` maps TransformerLens weight paths to `ParamProcessingConversion` instances that handle tensor reshaping during weight loading.

### Standard QKVO Conversions (most models)

For models with separate Q/K/V/O projections, use the built-in helper:

```python
self.weight_processing_conversions = {
    **self._qkvo_weight_conversions(),
}
```

This generates rearrangement rules for:
- `blocks.{i}.attn.q.weight` — `(n h) m -> n m h` with `n=n_heads`
- `blocks.{i}.attn.k.weight` — `(n h) m -> n m h` with `n=n_kv_heads`
- `blocks.{i}.attn.v.weight` — `(n h) m -> n m h` with `n=n_kv_heads`
- `blocks.{i}.attn.o.weight` — `m (n h) -> n h m` with `n=n_heads`

### Custom Conversions

For models with non-standard weight layouts (e.g., combined QKV), define custom `ParamProcessingConversion` or `RearrangeTensorConversion` instances. See `gpt2.py` for the `QKVSplitRearrangeConversion` example.

## Available Bridge Components

### Core Components

| Component | Use When |
|-----------|----------|
| `EmbeddingBridge` | Token embeddings |
| `UnembeddingBridge` | Output head (lm_head) |
| `BlockBridge` | Transformer block container (always named "blocks") |
| `LinearBridge` | Any linear/projection layer |

### Normalization

| Component | Use When |
|-----------|----------|
| `NormalizationBridge` | LayerNorm |
| `RMSNormalizationBridge` | RMSNorm |

### Attention

| Component | Use When |
|-----------|----------|
| `AttentionBridge` | Basic attention (no positional embeddings passed) |
| `PositionEmbeddingsAttentionBridge` | Attention that receives position embeddings (RoPE models) |
| `JointQKVAttentionBridge` | Combined QKV single linear layer (GPT-2 style) |
| `JointQKVPositionEmbeddingsAttentionBridge` | Combined QKV with position embeddings |

### MLP

| Component | Use When |
|-----------|----------|
| `MLPBridge` | Standard 2-layer MLP (in/out) or with separate gate |
| `GatedMLPBridge` | Gated MLP with gate/up/down projections (SwiGLU) |
| `JointGateUpMLPBridge` | MLP where gate and up projections are fused |

### Position Embeddings

| Component | Use When |
|-----------|----------|
| `PosEmbedBridge` | Learned positional embeddings (GPT-2 style) |
| `RotaryEmbeddingBridge` | Rotary position embeddings (RoPE) |

### Specialized

| Component | Use When |
|-----------|----------|
| `MoEBridge` | Mixture of Experts routing |
| `SymbolicBridge` | Placeholder/container with no direct HF module |
| `Conv1DBridge` | 1D convolution layers |
| `T5BlockBridge` | T5-specific block structure |
| `CLIPVisionEncoderBridge` | CLIP vision encoder (multimodal) |
| `CLIPVisionEncoderLayerBridge` | Individual CLIP vision encoder layer |
| `SiglipVisionEncoderBridge` | Siglip vision encoder (multimodal) |
| `SiglipVisionEncoderLayerBridge` | Individual Siglip vision encoder layer |
| `VisionProjectionBridge` | Vision-to-text projection (multimodal) |

### Architecture-Specific (Bloom/Falcon)

These exist for architectures with non-standard internal structures. Discover them by reading the reference adapter.

| Component | Use When |
|-----------|----------|
| `BloomBlockBridge` | BLOOM transformer blocks |
| `BloomAttentionBridge` | BLOOM attention mechanism |
| `BloomMLPBridge` | BLOOM MLP |
| `AudioFeatureExtractorBridge` | Audio feature extraction (HuBERT) |
| `ConvPosEmbedBridge` | Convolutional positional embeddings (HuBERT) |

## Optional Overrides

### `setup_component_testing(hf_model, bridge_model=None)`

Called after adapter creation. Use to set up model-specific references for component testing. Required for RoPE models to set rotary embedding references:

```python
def setup_component_testing(self, hf_model, bridge_model=None):
    rotary_emb = hf_model.model.rotary_emb
    if bridge_model is not None and hasattr(bridge_model, "blocks"):
        for block in bridge_model.blocks:
            if hasattr(block, "attn"):
                block.attn.set_rotary_emb(rotary_emb)
    attn_bridge = self.get_generalized_component("blocks.0.attn")
    attn_bridge.set_rotary_emb(rotary_emb)
```

### `preprocess_weights(state_dict)`

Apply architecture-specific weight transformations before standard processing. Example: Gemma scales embeddings by `sqrt(d_model)`.

### `prepare_loading(model_name, model_kwargs)`

Called before `from_pretrained()`. Use to patch HF model classes.

### `prepare_model(hf_model)`

Called after model loading but before bridge creation. Use for post-load fixups.

## Common Architecture Patterns

### Pattern 1: Llama-like (most modern models)

RoPE + RMSNorm + GatedMLP + separate Q/K/V/O. Uses `GatedMLPBridge`. Used by: Llama, Mistral, Gemma, OLMo, Granite, StableLM.

**Qwen2 variant:** Nearly identical to Llama but uses `MLPBridge` instead of `GatedMLPBridge` (while still setting `gated_mlp = True` and having gate/in/out submodules). Used by: Qwen2, Qwen3.

### Pattern 2: GPT2-like

Standard positional embeddings + LayerNorm + standard MLP + combined QKV. Used by: GPT-2, GPT-J, GPT-Neo/NeoX.

### Pattern 3: MoE (Mixture of Experts)

Similar to Llama-like but with `MoEBridge` replacing the MLP. Used by: Mixtral, GraniteMoE, OLMoE.

### Pattern 4: Multimodal

Extends a text-only pattern with vision encoder and projection bridges. Used by: LLaVA, LLaVA-Next, Gemma3 Multimodal.

## Imports Template

```python
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,           # or MLPBridge for non-gated
    LinearBridge,
    PositionEmbeddingsAttentionBridge,  # or JointQKVAttentionBridge
    RMSNormalizationBridge,   # or NormalizationBridge for LayerNorm
    RotaryEmbeddingBridge,    # only for RoPE models
    UnembeddingBridge,
)
```

## Testing

After creating an adapter, verify it by:

1. Running the adapter-specific unit tests
2. Loading a small model variant with `boot_transformers(model_name)`
3. Verifying hook names resolve correctly
4. Checking that weight shapes match expectations
