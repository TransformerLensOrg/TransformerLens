# Supported Architectures — AGENTS.md

Read [the root AGENTS.md](../../../AGENTS.md) for project-wide rules. This file covers conventions specific to writing TransformerBridge architecture adapters.

For the **verification** workflow after writing an adapter, see [tools/model_registry/AGENTS.md](../../tools/model_registry/AGENTS.md). Do not duplicate verification content here.

---

## TL;DR

- **One file per architecture family**, not per model. `llama.py` covers Llama 1 / 2 / 3 / 3.1 / 3.2; do NOT add `llama7b.py` or `llama2.py`. Family splits happen only when internal structure changes (`gemma1.py` / `gemma2.py` / `gemma3.py`, `qwen.py` / `qwen2.py` / `qwen3.py`, `mistral.py` / `mixtral.py`).
- **Four-place registration is mandatory.** See [Registration steps](#registration-steps) — missing any one causes silent runtime failure or stale generated docs.
- **`self.component_mapping: ComponentMapping`** is the load-bearing contract. Maps canonical TransformerLens paths → Bridge components wrapping HF module paths.
- **Hook names are Bridge-native** (`blocks.{i}.hook_out`, `blocks.{i}.attn.q.hook_out`). Don't invent HT-style aliases here — those are registered separately in [`bridge.py`](../bridge.py).
- **Copy from the closest existing adapter** before writing from scratch. See the [starter table](#starter-adapter-table).
- **Surface non-standard HF config attrs explicitly** — they are invisible to TL-side consumers unless propagated to `self.cfg`. See [Config-attr propagation](#config-attr-propagation).
- **Tokenizer policy is per-model, not per-architecture.** Never inherit `default_prepend_bos`, `default_padding_side`, EOS handling, or chat-template wiring from the starter adapter without checking the target's `tokenizer_config.json`. See [Tokenizer policy](#tokenizer-policy).

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
- `preprocess_weights()` — pre-load weight transforms (e.g. Gemma embedding scaling, Cohere `logit_scale` fold). See [When to override `preprocess_weights`](#when-to-override-preprocess_weights).
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

## Config-attr propagation

HF raw config attributes are invisible to TL-side consumers unless explicitly propagated to `self.cfg`. After analysing the HF model, walk the `config.json` and list every non-standard attribute. For each, decide whether the base adapter machinery already handles it; if not, mirror it onto `self.cfg`.

**Surface-on-cfg attributes** (set on `self.cfg.<name>`; the rest of the bridge reads them from there):

| HF attribute | Surface as | Used by |
|---|---|---|
| `final_logit_softcapping` | `self.cfg.final_logit_softcapping` | Gemma2/3 — final-layer logit clip |
| `attn_logit_softcapping` | `self.cfg.attn_logit_softcapping` | Gemma2/3 — attention-score clip |
| `query_pre_attn_scalar` | `self.cfg.query_pre_attn_scalar` | Gemma2/3 — query scaling override |
| `sliding_window` | `self.cfg.sliding_window` | Mistral, Qwen2, Gemma2 — local-attention layers |
| `layer_types` | `self.cfg.layer_types` | Hybrid models with per-layer attention type lists |
| Non-standard RMSNorm eps key | `self.cfg.eps_attr = "<attribute_name>"` | Llama uses `"variance_epsilon"` instead of `"eps"` |

**Weight-fold attributes** (need BOTH surface-on-cfg AND fold-into-weight via `preprocess_weights` — see [the next section](#when-to-override-preprocess_weights)):

| HF attribute | Fold target | Used by |
|---|---|---|
| `logit_scale` | `unembed.weight` (multiply in fp32 then cast back) | Cohere — final logits scaled by `1/16` |
| `embedding_multiplier` / embed-scale flags | `embed.weight` (multiply) | Gemma — embeddings scaled by `√d_model` |
| Tied unembed with extra scale | `unembed.weight` | T5-family tied projection variants |

Rule of thumb: if the model card or HF source mentions a numerical knob, assume it needs to land on `self.cfg`. If that knob changes weights or final outputs and HF's forward applies it natively, you ALSO need a `preprocess_weights` override or compatibility mode will diverge.

---

## When to override `preprocess_weights`

The framework default for `preprocess_weights()` is a no-op pass-through. Override it when **a numerical operation that HF's forward applies natively must also be baked into the raw weights** — otherwise `bridge.enable_compatibility_mode()` (which calls `process_weights` on the raw weights, expecting them to already encode all the math) produces wrong results.

The trigger rule:

> If a config attr changes the math of the forward pass AND would not be re-applied during compatibility-mode weight processing, fold it into the relevant weight inside `preprocess_weights()`.

Concrete examples in-tree:

- **Cohere** — `cfg.logit_scale` (default `0.0625`) must be folded into `unembed.weight`. HF forward multiplies logits by `logit_scale`; compat-mode `process_weights` does not, so without the fold, compat-mode logits are off by `1/16`.
- **Gemma1 / Gemma2 / Gemma3** — embedding scale (`√d_model`) must be folded into `embed.weight`. HF's `GemmaTextScaledWordEmbedding` scales internally on forward; compat-mode reads embeddings raw.

Skeleton:

```python
import torch

def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold <attr> into <weight> before ProcessWeights runs.

    bridge.py clones unembed.weight before calling this, so the scale does not
    leak into the tied embed.weight.
    """
    scale: float = getattr(self.cfg, "<attr>")  # set in __init__
    if scale != 1.0:  # no-op when scale is identity
        key = "unembed.weight"
        if key in state_dict:
            orig_dtype = state_dict[key].dtype
            state_dict[key] = (state_dict[key].float() * scale).to(orig_dtype)
    return state_dict
```

**When you do NOT need `preprocess_weights`:** if the numerical operation lives entirely inside an HF submodule that the bridge delegates to (e.g. RoPE inside `CohereRotaryEmbedding`), HF forward applies it on both paths and you inherit parity for free.

**Verification**: a missing `preprocess_weights` fold most often degrades Phase 4 (text-generation quality), and can show up as Phase 3 (weight processing) below 75% if severe enough to trip the strict gate. P4's 50% bar is intentionally lenient, so a sub-100% P4 on a small parity-test model is worth investigating even when the system reports `STATUS_VERIFIED`. See [tools/model_registry/AGENTS.md §Phase-score thresholds](../../tools/model_registry/AGENTS.md#phase-score-thresholds).

---

## Tokenizer policy

Tokenizer behaviour is **per-model**, not per-architecture. Sibling models in the same family routinely differ — the chat-instruct variant may prepend BOS where the base does not, the multilingual fork may pad on the left where the original pads on the right, the new-generation tokenizer may have a different EOS. **Do not copy these values from your starter adapter blindly; check the target's [tokenizer_config.json] and chat template first.**

Tokenizer-policy fields the adapter / load path is responsible for:

| Flag | Meaning | Default behaviour | What goes wrong if wrong |
|---|---|---|---|
| `default_prepend_bos` | Prepend BOS to every input by default | Framework default unless set | Off-by-one in logit positions; generation starting from wrong context |
| `default_padding_side` | `"left"` or `"right"` | HF tokenizer default | Generation produces garbage for batched inputs of unequal length |
| EOS handling in generation | Stop on `tokenizer.eos_token_id` (single or list) | Single token from `cfg.eos_token_id` | Generation runs past the natural stop or stops prematurely |
| Chat-template wiring | Whether the bridge auto-applies `tokenizer.apply_chat_template` | Off — user calls it explicitly | Instruct-tuned models produce base-model-style continuations |
| Tokenizer class mismatch (HF reports `<X>Tokenizer`, model card uses `<Y>Tokenizer`) | n/a | Whatever HF resolves | Subtle BPE/SentencePiece divergence; rare but real for forks |

**How to verify the right values:**

1. **Run the tokenizer.** `tokenizer_config.json` LIES for some architectures — Cohere is the canonical example: it declares `add_bos_token=False` but HF's `__call__` prepends BOS anyway via `add_special_tokens=True`. The only reliable check is to actually invoke it:

   ```python
   from transformers import AutoTokenizer
   t = AutoTokenizer.from_pretrained("<hf_repo>")
   print("encode:", t.encode("hello"))         # raw encode
   print("__call__:", t("hello").input_ids)    # what generation actually uses
   print("bos_token_id:", t.bos_token_id)
   ```

   If `t("hello").input_ids[0] == t.bos_token_id`, set `cfg.default_prepend_bos = True`. If the first token is NOT BOS, leave the flag unset.

2. **Cross-check against `tokenizer_config.json`** for `padding_side`, `bos_token`, `eos_token`, `chat_template`. These tend to be honest; the runtime override mostly affects BOS-prepending.

3. **Compare to the closest sibling already in the registry.** If they differ, treat that as a deliberate choice and propagate it.

4. **When unsure, leave the framework default.** Explicit wrongness is worse than implicit defaulting — the wrong explicit value hides under "I configured it" assumptions.

`default_prepend_bos` is the most common trap (Cohere's tokenizer-config-vs-runtime mismatch, instruct/base divergence within Llama / Mistral / Gemma families). The same recipe applies to verifying every flag above.

---

## Registration steps

After writing `myarch.py` with `MyArchitectureAdapter`, register in **all four** sites. Missing any one breaks something:

1. **[`__init__.py`](__init__.py)** — import + add to `__all__`:
   ```python
   from transformer_lens.model_bridge.supported_architectures.myarch import (
       MyArchitectureAdapter,
   )
   ```
   Missing → import error at boot.

2. **[`../../factories/architecture_adapter_factory.py`](../../factories/architecture_adapter_factory.py)** — import + `SUPPORTED_ARCHITECTURES` entry:
   ```python
   from transformer_lens.model_bridge.supported_architectures import (
       ...,
       MyArchitectureAdapter,
   )

   SUPPORTED_ARCHITECTURES = {
       ...,
       "MyArchForCausalLM": MyArchitectureAdapter,  # key must match config.architectures[0] exactly
   }
   ```
   Missing → `boot_transformers` raises "unsupported architecture."

3. **[`../../tools/model_registry/__init__.py`](../../tools/model_registry/__init__.py)** — two updates in this file:
   - Add `"MyArchForCausalLM"` to `HF_SUPPORTED_ARCHITECTURES`
   - Add `"MyArchForCausalLM": ["foundation-org-1", "foundation-org-2"]` to `CANONICAL_AUTHORS_BY_ARCH`

   Missing → HF scraper misses canonical models for this architecture (download-threshold bypass).

4. **[`../../tools/model_registry/generate_report.py`](../../tools/model_registry/generate_report.py)** — one-line entry in `ARCHITECTURE_DESCRIPTIONS`:
   ```python
   "MyArchForCausalLM": "Short human-readable description.",
   ```
   Missing → generated docs table omits the new architecture.

**Verify the four-place wiring**: [`tests/unit/tools/test_model_registry.py`](../../../tests/unit/tools/test_model_registry.py) has a `TestRegistrySyncedWithFactory` class with four bidirectional invariants over `SUPPORTED_ARCHITECTURES`, `HF_SUPPORTED_ARCHITECTURES`, and `CANONICAL_AUTHORS_BY_ARCH`. Run `uv run pytest tests/unit/tools/test_model_registry.py -k TestRegistrySyncedWithFactory` after registering — the failure message tells you which set is missing your new key. (There's an `INTENTIONAL_EXCLUDES` carve-out in the same class for internal-only architectures and HF-emits-different-casing aliases; almost no new adapters need to add themselves there.)

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

## Adding the HF repo to the registry

After registration, add the model ID to [`../../tools/model_registry/data/supported_models.json`](../../tools/model_registry/data/supported_models.json) so `verify_models` can resolve it. The entry shape is:

```json
{
  "architecture_id": "MyArchForCausalLM",
  "model_id": "org/repo-name",
  "status": 0,
  "verified_date": null,
  "metadata": null,
  "note": null,
  "phase1_score": null,
  "phase2_score": null,
  "phase3_score": null,
  "phase4_score": null,
  "phase7_score": null,
  "phase8_score": null
}
```

**Hand-edits are allowed only for adding new model-ID entries.** The `status`, `verified_date`, `note`, and `phaseN_score` fields are written exclusively by `update_model_status()` — never set them manually. See [tools/model_registry/AGENTS.md](../../tools/model_registry/AGENTS.md).

When adding the user-provided model, prompt the user whether to also add the canonical sibling variants from `CANONICAL_AUTHORS_BY_ARCH[<HFArchClass>]` — e.g. when adding `google/gemma-2-2b`, ask about `google/gemma-2-2b-it`, `google/gemma-2-9b`, `google/gemma-2-9b-it`, `google/gemma-2-27b`, `google/gemma-2-27b-it`. The HF scraper picks these up eventually, but explicit entries unblock `verify_models` against the siblings now.

Do NOT add the model to [`transformer_lens/supported_models.py`](../../supported_models.py) — that file is HookedTransformer-only.

---

## Model source paths: `boot_transformers` vs `boot_native`

TransformerBridge has **two parallel load paths**, both routed through the same adapter system:

| Source | Entry | Use case |
|---|---|---|
| `boot_transformers` | `TransformerBridge.boot_transformers(model_name, ...)` | Default: wraps a HuggingFace model. Adapter maps HF module paths to canonical names. Every adapter in this directory supports this path. |
| `boot_native` | `TransformerBridge.boot_native(cfg, init_mode=...)` | TL-native transformer built from scratch, no HF dependency. Used by [`Realtime_Training_Telemetry_Demo.ipynb`](../../../demos/Realtime_Training_Telemetry_Demo.ipynb) and research that needs cfg-driven small models. |

Native models are built by [`transformer_lens/model_bridge/sources/native/model.py`](../sources/native/model.py) and routed through [`native.py`](native.py) — the adapter whose `component_mapping` adapts to `cfg` (rotary drops `pos_embed`, RMS norm picks `RMSNormalizationBridge`, gated MLP picks `GatedMLPBridge`, `attn_only` drops MLP). Init policies live in [`sources/native/init.py`](../sources/native/init.py): `gpt2` (default, Normal with 1/√(2·n_layers) residual scaling), `xavier_uniform` / `xavier_normal`, `kaiming_uniform` / `kaiming_normal`. Determinism uses a scoped `torch.Generator` so seeded init does not perturb the caller's global RNG.

Cfg-driven Native features: `normalization_type` (LN / RMS / RMSPre), `final_rms`, `gated_mlp`, `attn_only`, `n_key_value_heads` (GQA), `attn_scores_soft_cap`, `output_logits_soft_cap`, `positional_embedding_type` (standard / rotary), `rotary_dim` / `rotary_base` / `rope_scaling` (linear PI, dynamic/NTK, llama3 by-parts).

**Sister source paths** under [`sources/`](../sources/) — `transformers/`, `vllm/`, `inspect/` — are different backends; their existence is mostly an internal detail for new adapter authors. Almost all new work goes through `transformers/` (HF backend) and only writes a `supported_architectures/<arch>.py` file.

When **does a new adapter need to support both paths?** Almost never — if you're adding support for an HF-released architecture, you're writing a `boot_transformers` adapter and you're done. `boot_native` is for research-grade from-scratch models and uses the single `native.py` adapter generically. Touch `native.py` only if you're adding a primitive (e.g. a new norm type) that the cfg-driven dispatch doesn't already cover.

---

## Required tests

Two test layers per architecture, both required for review:

### 1. Unit adapter test — `tests/unit/model_bridge/supported_architectures/test_<arch>_adapter.py`

26 of these exist; they all follow the same shape. Copy from a sibling that matches your architecture's quirks:

- **Standard decoder LM** template: [`test_gemma1_adapter.py`](../../../tests/unit/model_bridge/supported_architectures/test_gemma1_adapter.py) or [`test_gpt2_adapter.py`](../../../tests/unit/model_bridge/supported_architectures/test_gpt2_adapter.py)
- **GQA / modern RMSNorm+RoPE** template: [`test_qwen3_adapter.py`](../../../tests/unit/model_bridge/supported_architectures/test_qwen3_adapter.py)
- **Multimodal** template: [`test_llava_adapter.py`](../../../tests/unit/model_bridge/supported_architectures/test_llava_adapter.py), [`test_gemma3_multimodal_adapter.py`](../../../tests/unit/model_bridge/supported_architectures/test_gemma3_multimodal_adapter.py)

The minimal shape:

```python
def _make_cfg(d_model: int = 32) -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=d_model, d_head=d_model // 4, n_layers=2, n_ctx=128,
        n_heads=4, d_vocab=256, d_mlp=64, architecture="MyArchForCausalLM",
    )

@pytest.fixture(scope="module")
def adapter() -> MyArchitectureAdapter:
    return MyArchitectureAdapter(_make_cfg())

class TestMyArchHookCompatibility:
    def test_<arch_quirk>(self, adapter): ...
```

Unit adapter tests don't load weights — they instantiate the adapter from a synthetic cfg and assert structural / hook-compatibility properties (component_mapping shape, whether `setup_hook_compatibility` is overridden, conversion shapes, etc.). They run in the default `make unit-test` tier with no HF Hub access.

Add a dedicated test for **each architecture quirk** your adapter handles — softcaps, RMSNorm offsets, sliding window, custom eps attr, MoE routing. The Gemma1 test is a great example of a one-quirk-one-test file ("the adapter must NOT override setup_hook_compatibility because the HF embedding scales internally").

### 2. Integration parity test — `tests/integration/model_bridge/test_<arch>_adapter.py` (or `test_<arch>_bridge.py`)

Loads a real cached HF model, asserts logit parity vs HuggingFace at fp32 + eager attention. Templates: [`test_deepseek_adapter.py`](../../../tests/integration/model_bridge/test_deepseek_adapter.py), [`test_falcon_adapter.py`](../../../tests/integration/model_bridge/test_falcon_adapter.py), [`test_mamba_adapter.py`](../../../tests/integration/model_bridge/test_mamba_adapter.py).

If your architecture's reference model is large enough to OOM CI, gate it with `@pytest.mark.skipif(bool(os.getenv("CI")), reason="...")` and add a row to [`tests/QUARANTINES.md`](../../../tests/QUARANTINES.md) under "CI cost / network budget."

### End-to-end registry verification

After both tests pass:

```bash
set -a; source .env; set +a
uv run python -m transformer_lens.tools.model_registry.verify_models --architectures MyArchForCausalLM --per-arch 1
```

This is what [`/verify-model`](../../../.claude/commands/verify-model.md) wraps. See [tools/model_registry/AGENTS.md](../../tools/model_registry/AGENTS.md) for the canonical workflow and the [`verify_models` vs `main_benchmark` trap](../../tools/model_registry/AGENTS.md#tldr).

If logit parity fails, see [docs/source/content/debugging_numerical_divergence.md](../../../docs/source/content/debugging_numerical_divergence.md) for the bisection workflow.

---

## Hard "don'ts"

- **No single-model files.** One file per architecture family.
- **No `# type: ignore` on `ComponentMapping` types.** Use `isinstance` / `typing.cast` if the type system disagrees. See [AGENTS.md §10](../../../AGENTS.md#10-hard-rules).
- **No skipping any of the four registration sites.** Each one breaks something different (import, factory, scraper, generated docs). The invariant test in [`test_model_registry.py`](../../../tests/unit/tools/test_model_registry.py) will catch most of these.
- **No HT-style hook aliases inside adapters.** Aliases live in [`../bridge.py`](../bridge.py) via `build_alias_to_canonical_map()`; adapters declare Bridge-native names only.
- **No inheriting tokenizer-policy flags from the starter adapter** (`default_prepend_bos`, padding side, EOS). Tokenizer policy is per-model; see [Tokenizer policy](#tokenizer-policy).
- **No manual edits to existing `supported_models.json` entries' status / phase fields.** Only `update_model_status()` writes those.
- **No skipping the unit adapter test.** Every architecture has one; an adapter PR without one is incomplete.
