# Supported Architectures — AGENTS.md

Read [the root AGENTS.md](../../../AGENTS.md) for project-wide rules. This file covers conventions specific to writing TransformerBridge architecture adapters.

For the **verification** workflow after writing an adapter, see [tools/model_registry/AGENTS.md](../../tools/model_registry/AGENTS.md). Do not duplicate verification content here.

---

## TL;DR

- **One file per architecture family**, not per model. `llama.py` covers Llama 1 / 2 / 3 / 3.1 / 3.2; do NOT add `llama7b.py` or `llama2.py`. Family splits happen only when internal structure changes (`gemma1.py` / `gemma2.py` / `gemma3.py`, `qwen.py` / `qwen2.py` / `qwen3.py`, `mistral.py` / `mixtral.py`).
- **Two-place registration is mandatory.** Re-export in [`__init__.py`](__init__.py) AND add to `SUPPORTED_ARCHITECTURES` in [`../../factories/architecture_adapter_factory.py`](../../factories/architecture_adapter_factory.py). Missing either causes silent runtime failure at boot.
- **`self.component_mapping: ComponentMapping`** is the load-bearing contract. Maps canonical TransformerLens paths → Bridge components wrapping HF module paths.
- **Hook names are Bridge-native** (`blocks.{i}.hook_out`, `blocks.{i}.attn.q.hook_out`). Don't invent HT-style aliases here — those are registered separately in [`bridge.py`](../bridge.py).
- **Copy from the closest existing adapter** before writing from scratch. See the starter table below.

---

## Minimal contract

Every adapter inherits from `ArchitectureAdapter` ([`../architecture_adapter.py`](../architecture_adapter.py)) and declares in `__init__`:

```python
class MyArchitectureAdapter(ArchitectureAdapter):
    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # Required config flags — set on self.cfg, not just default_config
        self.cfg.normalization_type = "RMS"           # "RMS" or "LN"
        self.cfg.positional_embedding_type = "rotary" # "rotary" | "standard" | "relative_positional_bias"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False

        # Architecture-conditional flags
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads  # GQA / MQA

        # Weight reshape rules where Bridge layout differs from HF
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        # The load-bearing mapping
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "attn": PositionEmbeddingsAttentionBridge(...),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "mlp": GatedMLPBridge(...),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
```

Optional overrides on the class:

- `default_cfg` — class-level defaults merged into the runtime config
- `preprocess_weights()` — pre-load weight transforms (e.g. Gemma embedding scaling)
- `applicable_phases` — subset of `[1, 2, 3, 4, 7, 8]` (default `[1, 2, 3, 4]`); see [tools/model_registry/AGENTS.md](../../tools/model_registry/AGENTS.md) for what each phase tests
- `supports_generation` — set `False` for encoder-only models (BERT, HuBERT)

---

## Starter-adapter table

| Target shape | Start from | Why |
|---|---|---|
| Decoder-only causal LM, RMSNorm + RoPE | [`llama.py`](llama.py) | Canonical modern shape; already handles GQA via `n_key_value_heads` |
| Decoder-only causal LM, LayerNorm + combined QKV | [`gpt2.py`](gpt2.py) | Older GPT-style; demonstrates QKV split/rearrange |
| GQA / MQA | [`llama.py`](llama.py) | Pass `n_kv_heads=` to `_qkvo_weight_conversions()` |
| RMSNorm with offset | [`gemma1.py`](gemma1.py) | Uses `rmsnorm_uses_offset = True` + `ArithmeticTensorConversion` |
| Mixture-of-experts | [`mixtral.py`](mixtral.py) | Uses `MoEBridge` with batched experts |
| Vision-language | [`llava.py`](llava.py) | Dual encoder + projection pathways |
| Encoder-decoder (T5-style) | [`t5.py`](t5.py) | Sets `supports_generation = False`, separate encoder/decoder block lists |
| State-space model | [`mamba.py`](mamba.py), [`mamba2.py`](mamba2.py) | Off the transformer path entirely |
| Encoder-only with CTC head | [`bert.py`](bert.py), [`hubert.py`](hubert.py) | `supports_generation = False` |

---

## Registration steps

After writing `myarch.py` with `MyArchitectureAdapter`:

1. **Re-export in [`__init__.py`](__init__.py):**
   ```python
   from transformer_lens.model_bridge.supported_architectures.myarch import (
       MyArchitectureAdapter,
   )
   ```
2. **Register in [`../../factories/architecture_adapter_factory.py`](../../factories/architecture_adapter_factory.py)** — add to the imports block AND to `SUPPORTED_ARCHITECTURES`:
   ```python
   from transformer_lens.model_bridge.supported_architectures import (
       ...,
       MyArchitectureAdapter,
   )

   SUPPORTED_ARCHITECTURES = {
       ...,
       "MyArchForCausalLM": MyArchitectureAdapter,  # HF class name from config.architectures[]
   }
   ```
   The key must match `config.architectures[0]` from the HF model's `config.json` exactly.
3. **Add to the registry constants** in [`../../tools/model_registry/__init__.py`](../../tools/model_registry/__init__.py):
   - `HF_SUPPORTED_ARCHITECTURES` set
   - `CANONICAL_AUTHORS_BY_ARCH` map (foundation-trained orgs for the scraper's download-threshold bypass)

Missing step 1 → import error at boot. Missing step 2 → `boot_transformers` raises "unsupported architecture." Missing step 3 → scraper misses canonical models.

---

## Common gotchas

| HF convention | Handled in | How |
|---|---|---|
| RoPE (rotary positional embeddings) | `llama.py`, `mistral.py`, `qwen2.py`+ | `RotaryEmbeddingBridge(name="model.rotary_emb")` + `cfg.positional_embedding_type = "rotary"` |
| GQA / MQA (`n_key_value_heads < n_heads`) | `llama.py`, `mistral.py`, `falcon.py`, `cohere.py` | Set `cfg.n_key_value_heads`; pass `n_kv_heads=` to `_qkvo_weight_conversions()` |
| RMSNorm with offset | `gemma1.py`, `gemma2.py`, `gemma3.py` | `cfg.rmsnorm_uses_offset = True` + `ArithmeticTensorConversion(ADDITION, 1.0)` |
| Custom RMSNorm eps attribute | `llama.py` | `cfg.eps_attr = "variance_epsilon"` (Llama uses this instead of `eps`) |
| Standard LayerNorm | `gpt2.py`, `bloom.py` | `cfg.normalization_type = "LN"` |
| Gated MLP (`gate_proj`, `up_proj`, `down_proj`) | `llama.py`, `mistral.py`, `gemma1.py`, `qwen2.py`+ | `GatedMLPBridge` with submodules `{gate, in, out}` |
| Combined QKV (`c_attn`) | `gpt2.py`, `bloom.py` | `QKVSplitRearrangeConversion` to split + rearrange |
| Split Q/K/V (standard) | `llama.py`, `mistral.py`, most modern | `self._qkvo_weight_conversions()` helper |
| MoE routing | `mixtral.py`, `deepseek_v3.py`, `qwen3_moe.py`, `granite_moe.py` | `MoEBridge` with `gate` + batched expert submodules |
| Missing biases (RMSNorm has no `b`; Llama has no attn/MLP biases) | `llama.py` (documented in docstring) | Weight processing handles `None` via `ProcessWeights._safe_get_tensor()` |
| KV cache layout | All (implicit) | Adapter delegates; HF module manages internally |

---

## Verification

After registering the adapter, validate end-to-end via the model registry — see [tools/model_registry/AGENTS.md](../../tools/model_registry/AGENTS.md). The minimum check is:

```bash
set -a; source .env; set +a
python -m transformer_lens.tools.model_registry.verify_models --architectures MyArchForCausalLM --per-arch 1
```

And an integration test under [`tests/integration/`](../../../tests/integration/) asserting logit parity with HuggingFace at fp32 + eager attention.

---

## Hard "don'ts"

- **No single-model files.** One file per architecture family.
- **No `# type: ignore` on `ComponentMapping` types.** Use `isinstance` / `typing.cast` if the type system disagrees. See [AGENTS.md §10](../../../AGENTS.md#10-hard-rules).
- **No skipping the factory registration.** The adapter file existing is not enough — without the entry in `SUPPORTED_ARCHITECTURES`, `boot_transformers` cannot find it and fails at runtime.
- **No HT-style hook aliases inside adapters.** Aliases live in [`../bridge.py`](../bridge.py) via `build_alias_to_canonical_map()`; adapters declare Bridge-native names only.
