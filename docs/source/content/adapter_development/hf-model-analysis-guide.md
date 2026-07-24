# HuggingFace Model Analysis Guide

This guide explains how to analyze a HuggingFace model to extract the information needed to build a TransformerLens Architecture Adapter.

## Read the model's config.json

Every HF model has a `config.json` that contains architecture details. You can access it via:

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("model-name-or-path")
print(config)
```

Or via the HuggingFace API:
```bash
curl -s "https://huggingface.co/model-name/resolve/main/config.json" | python -m json.tool
```

### Key config fields to extract

| HF Config Field | TL Config Field | Description |
|-----------------|-----------------|-------------|
| `hidden_size` | `d_model` | Model dimension |
| `num_attention_heads` | `n_heads` | Number of attention heads |
| `num_key_value_heads` | `n_key_value_heads` | KV heads (for GQA; if absent or equal to n_heads, not GQA) |
| `intermediate_size` | `d_mlp` | MLP intermediate dimension |
| `num_hidden_layers` | `n_layers` | Number of transformer blocks |
| `vocab_size` | `d_vocab` | Vocabulary size |
| `max_position_embeddings` | `n_ctx` | Maximum sequence length |
| `rms_norm_eps` | `eps` | Normalization epsilon |
| `model_type` | — | Architecture family (e.g., "llama", "gpt2", "mistral") |
| `architectures` | `architecture` | HF class name (e.g., `["LlamaForCausalLM"]`) |

## Determine architecture characteristics

### Normalization type

Check the model code or config:
- **RMSNorm** → `normalization_type = "RMS"` — Look for `RMSNorm` in the model code, or `rms_norm_eps` in config
- **LayerNorm** → `normalization_type = "LN"` — Look for `LayerNorm`, or `layer_norm_eps` / `layer_norm_epsilon` in config

Also identify the epsilon attribute name:
- `"variance_epsilon"` (Llama)
- `"rms_norm_eps"` (some models expose this directly)
- `"layer_norm_eps"` (GPT-2, BERT)
- `"eps"` (generic)

### Positional embedding type

- **Rotary (RoPE)** → `positional_embedding_type = "rotary"` — Most modern models (Llama, Mistral, Qwen, Gemma)
- **Learned/Standard** → `positional_embedding_type = "standard"` — GPT-2, OPT
- Check for `RotaryEmbedding` class in the model code

### Attention type

- **Multi-Head Attention (MHA)** — `n_key_value_heads == n_heads` or field absent
- **Grouped Query Attention (GQA)** — `n_key_value_heads < n_heads` (e.g., Llama 3, Mistral)
- **Multi-Query Attention (MQA)** — `n_key_value_heads == 1` (e.g., Falcon)

### MLP type

- **Gated MLP (SwiGLU)** → `gated_mlp = True` — Has gate/up/down projections (Llama, Qwen, Gemma)
- **Standard MLP** → `gated_mlp = False` — Has fc1/fc2 or c_fc/c_proj (GPT-2)

### QKV layout

- **Separate Q/K/V** — Most models: `q_proj`, `k_proj`, `v_proj`
- **Combined QKV** — GPT-2 style: single `c_attn` or `query_key_value` linear layer

## Inspect module names

To find the exact HuggingFace module paths for the component mapping:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("model-name", torch_dtype="auto")

# Print all named modules
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")
```

### What to look for

Map these HF module paths to TL component mapping entries:

| TL Name | Look for in HF | Common HF Paths |
|---------|----------------|-----------------|
| `embed` | Token embedding | `model.embed_tokens`, `transformer.wte` |
| `pos_embed` | Position embedding (if standard) | `transformer.wpe` |
| `rotary_emb` | Rotary embedding (if RoPE) | `model.rotary_emb`, `model.layers.0.self_attn.rotary_emb` |
| `blocks` | Layer list | `model.layers`, `transformer.h`, `model.decoder.layers` |
| `ln1` | Pre-attention norm | `input_layernorm`, `ln_1` |
| `ln2` | Post-attention norm | `post_attention_layernorm`, `ln_2` |
| `attn` | Self-attention module | `self_attn`, `attn` |
| `attn.q` | Query projection | `q_proj`, `query` |
| `attn.k` | Key projection | `k_proj`, `key` |
| `attn.v` | Value projection | `v_proj`, `value` |
| `attn.o` | Output projection | `o_proj`, `out_proj`, `dense`, `c_proj` |
| `attn.qkv` | Combined QKV (if used) | `c_attn`, `query_key_value` |
| `mlp` | MLP module | `mlp`, `feed_forward` |
| `mlp.gate` | Gate projection (if gated) | `gate_proj`, `w1` |
| `mlp.in` | Up/input projection | `up_proj`, `c_fc`, `fc1`, `w3` |
| `mlp.out` | Down/output projection | `down_proj`, `c_proj`, `fc2`, `w2` |
| `ln_final` | Final layer norm | `model.norm`, `transformer.ln_f`, `model.final_layernorm` |
| `unembed` | LM head | `lm_head`, `embed_out` |

## Check for biases

```python
# Check if a specific layer has bias
layer = model.model.layers[0]
print(f"Q bias: {layer.self_attn.q_proj.bias is not None}")
print(f"MLP in bias: {layer.mlp.up_proj.bias is not None}")
```

Document which layers lack biases — this affects the "Optional Parameters" section of the adapter docstring.

## Examine state dict keys

```python
# Print all parameter names and shapes
for key, param in model.state_dict().items():
    print(f"{key}: {param.shape}")
```

This helps verify:
- Weight naming patterns match your component mapping
- Tensor shapes match expected dimensions
- No unexpected parameters that need special handling

## Find an existing similar adapter

Check if a similar architecture already has an adapter. Most new models are variants of existing patterns:

| If your model is like... | Start from adapter... |
|--------------------------|----------------------|
| Llama, Mistral, Qwen2, Gemma | `llama.py` |
| GPT-2, GPT-J | `gpt2.py` |
| BLOOM, Falcon | `bloom.py` or `falcon.py` |
| T5, encoder-decoder | `t5.py` |
| MoE model | `mixtral.py` or `granite_moe.py` |
| Multimodal (vision+text) | `llava.py` or `gemma3_multimodal.py` |

## Quick reference: decision tree

```
1. Does the model use RMSNorm or LayerNorm?
   → RMSNorm: normalization_type="RMS", use RMSNormalizationBridge
   → LayerNorm: normalization_type="LN", use NormalizationBridge

2. Does the model use RoPE or learned positional embeddings?
   → RoPE: positional_embedding_type="rotary", add RotaryEmbeddingBridge, use PositionEmbeddingsAttentionBridge
   → Learned: positional_embedding_type="standard", add PosEmbedBridge

3. Are Q/K/V separate or combined?
   → Separate: use PositionEmbeddingsAttentionBridge with q/k/v/o submodules
   → Combined: use JointQKVAttentionBridge with qkv/o submodules

4. Does the MLP have a gate projection?
   → Yes (gate+up+down): gated_mlp=True, use GatedMLPBridge
   → No (in+out): gated_mlp=False, use MLPBridge

5. Is n_key_value_heads < n_heads?
   → Yes: GQA — set n_key_value_heads on cfg
   → No: standard MHA — no special handling needed
```
