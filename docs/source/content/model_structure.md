# TransformerBridge Model Structure

This page describes the structure exposed by TransformerBridge, the canonical hook names to use, and the expected tensor shapes at each hook point.

## Overview

TransformerBridge wraps a Hugging Face model behind a consistent TransformerLens interface. It relies on:
- An ArchitectureAdapter that understands the HF module graph and provides a mapping to bridge components
- Generalized components (Embedding, Attention, MLP, Normalization, Block) exposing uniform hook points
- A light aliasing layer for backwards compatibility with legacy TransformerLens hook names

Construct a bridge from a HF model id:

```python
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources import transformers as bridge_sources  # registers boot

bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
```

You can then call the familiar APIs: `to_tokens`, `to_string`, `generate`, `run_with_hooks`, `run_with_cache`.

## Top-Level Components

Typical decoder-only models expose these top-level components (names vary by architecture):
- `embed`: token embedding
- `pos_embed` (if applicable) or rotary embeddings inside attention
- `blocks`: list-like container of transformer blocks
- `ln_final` (if applicable): final normalization
- `unembed`: output projection to vocabulary logits

Each `blocks.{i}` is a `BlockBridge` with subcomponents:
- `ln1`: normalization before attention
- `attn`: attention module
- `ln2`: normalization before MLP
- `mlp`: MLP module

## Canonical Hook Names

Use these canonical (non-aliased) names when adding hooks or reading from the cache.

### Embedding
- `embed.hook_in`: token ids (batch, pos)
- `embed.hook_out`: embeddings (batch, pos, d_model)
  - *Legacy alias: `hook_embed`*
- `pos_embed.hook_in` / `pos_embed.hook_out`: same shapes as above
  - *Legacy alias: `hook_pos_embed`*

### Residual stream
- `blocks.{i}.hook_in`: residual stream into block (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_resid_pre`*
- `blocks.{i}.hook_out`: residual stream out of block (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_resid_post`*
- `blocks.{i}.attn.hook_out`: residual stream after attention (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_resid_mid`*

### Attention
- `blocks.{i}.attn.hook_in`: (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_attn_in`*
- `blocks.{i}.attn.hook_out`: (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_attn_out`*
- `blocks.{i}.attn.hook_hidden_states`: primary output for caching (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.attn.hook_result`*
- `blocks.{i}.attn.hook_attn_scores`: raw attention scores before softmax (batch, n_heads, pos, pos)
- `blocks.{i}.attn.hook_pattern`: attention pattern after softmax and NaN handling (n_heads, pos, pos)
  - *Legacy alias: `blocks.{i}.attn.hook_attention_weights`*
- When present, sub-projections: `blocks.{i}.attn.q/k/v/o.hook_in` / `.hook_out` (commonly (batch, pos, d_model))
  - *Legacy aliases: `blocks.{i}.hook_q_input`, `blocks.{i}.hook_k_input`, `blocks.{i}.hook_v_input`, `blocks.{i}.hook_q`, `blocks.{i}.hook_k`, `blocks.{i}.hook_v`*

#### Individual Q/K/V Hooks
All attention bridges provide access to individual Q, K, V activations through `HookPointWrapper` properties:

- `blocks.{i}.attn.q.hook_in` / `blocks.{i}.attn.q.hook_out`: Q projection hooks (batch, pos, n_heads, d_head)
- `blocks.{i}.attn.k.hook_in` / `blocks.{i}.attn.k.hook_out`: K projection hooks (batch, pos, n_heads, d_head)
- `blocks.{i}.attn.v.hook_in` / `blocks.{i}.attn.v.hook_out`: V projection hooks (batch, pos, n_heads, d_head)

#### Joint QKV Attention (GPT-2 style)
For models using fused QKV projections (like GPT-2), the `JointQKVAttentionBridge` provides additional hooks:

- `blocks.{i}.attn.qkv.hook_in`: input to QKV projection (batch, pos, d_model)
- `blocks.{i}.attn.qkv.hook_out`: output from QKV projection (batch, pos, 3*d_model)
- `blocks.{i}.attn.qkv.q_hook_in`: input to Q projection (batch, pos, d_model)
- `blocks.{i}.attn.qkv.q_hook_out`: output from Q projection (batch, pos, n_heads, d_head)
- `blocks.{i}.attn.qkv.k_hook_in`: input to K projection (batch, pos, d_model)
- `blocks.{i}.attn.qkv.k_hook_out`: output from K projection (batch, pos, n_heads, d_head)
- `blocks.{i}.attn.qkv.v_hook_in`: input to V projection (batch, pos, d_model)
- `blocks.{i}.attn.qkv.v_hook_out`: output from V projection (batch, pos, n_heads, d_head)

### MLP
- `blocks.{i}.mlp.hook_in`: (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_mlp_in`*
- `blocks.{i}.mlp.hook_pre`: (batch, pos, d_mlp)
  - *Legacy alias: `blocks.{i}.hook_mlp_in` (via `mlp.in.hook_out`)*
- `blocks.{i}.mlp.hook_out`: (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_mlp_out`*

### Normalization
- `blocks.{i}.ln1.hook_in` / `.hook_out`: (batch, pos, d_model)
  - *Legacy aliases for `.hook_out`: `blocks.{i}.ln1.hook_normalized`, `blocks.{i}.ln1.hook_scale`*
- Similarly for `ln2`
  - *Legacy aliases for `.hook_out`: `blocks.{i}.ln2.hook_normalized`, `blocks.{i}.ln2.hook_scale`*

### Unembedding / Logits
- `unembed.hook_in`: (batch, pos, d_model)
- `unembed.hook_out`: (batch, pos, d_vocab)

## Shapes at a Glance

- Residual stream and hidden states: (batch, pos, d_model)
- Attention scores: (batch, n_heads, pos, pos)
- Attention patterns: (n_heads, pos, pos) - after batch dimension removal
- QKV projections: (batch, pos, n_heads, d_head)
- MLP pre-activation: (batch, pos, d_mlp)
- Embeddings: (batch, pos, d_model)
- Unembedding logits: (batch, pos, d_vocab)
- LayerNorm normalized / scale: (batch, pos, d_model)

These shapes are exercised in the multi-model shape test: `tests/integration/test_hook_shape_compatibility.py`.

## Booting from Hugging Face

`TransformerBridge.boot_transformers(model_id, ...)`:
- Loads the HF config/model/tokenizer
- Selects the appropriate ArchitectureAdapter
- Maps HF config fields to TransformerLens config (e.g., `d_model`, `n_heads`, `n_layers`, `d_mlp`, `d_vocab`, `n_ctx`, ...)
- Constructs the bridge and registers all hook points

## Fused QKV Attention

Some architectures use a fused QKV projection (like GPT-2). The bridge's `JointQKVAttentionBridge` provides access to individual Q, K, V activations through the `QKVBridge` submodule. This allows for:

1. **Individual Q/K/V hooking**: You can hook into `blocks.{i}.attn.qkv.q_hook_out`, `k_hook_out`, or `v_hook_out` to modify individual attention heads
2. **Attention pattern creation**: The bridge automatically creates attention patterns from the attention scores and applies them through `hook_pattern`
3. **Compatibility with legacy code**: Legacy hook names like `blocks.{i}.hook_v` are aliased to the appropriate QKV hooks

The canonical attention hooks (`attn.hook_in/out`, `attn.hook_pattern`, etc.) retain the shapes listed above, while the QKV-specific hooks provide access to the individual attention components.

## Aliases and Backwards Compatibility

A minimal alias layer exists to ease migration from older TransformerLens names (e.g., `blocks.{i}.hook_resid_pre` → `blocks.{i}.hook_in`). New code should prefer the canonical names documented here.

## Example: Caching and Inspecting Hooks

```python
prompt = "Hello world"
logits, cache = bridge.run_with_cache(prompt)

# List some attention-related hooks on the first block
for k in cache.keys():
    if k.startswith("blocks.0.attn"):
        print(k, cache[k].shape)
```

For larger examples and a multi-model shape check, see `tests/integration/test_hook_shape_compatibility.py`.
