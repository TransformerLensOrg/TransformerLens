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

HF raw config attributes are invisible to TL-side consumers unless propagated to `self.cfg`. Walk the `config.json` for non-standard attributes; mirror anything the base machinery doesn't already handle.

**Surface-on-cfg attributes** (set on `self.cfg.<name>`; bridge reads from there):

| HF attribute | Surface as | Used by |
|---|---|---|
| `final_logit_softcapping` | `self.cfg.final_logit_softcapping` | Gemma2/3 — final-layer logit clip |
| `attn_logit_softcapping` | `self.cfg.attn_logit_softcapping` | Gemma2/3 — attention-score clip |
| `query_pre_attn_scalar` | `self.cfg.query_pre_attn_scalar` | Gemma2/3 — query scaling override |
| `sliding_window` | `self.cfg.sliding_window` | Mistral, Qwen2, Gemma2 — local-attention layers |
| `layer_types` | `self.cfg.layer_types` | Hybrid models with per-layer attention type lists |

**Weight-fold attributes** (need BOTH surface-on-cfg AND fold-into-weight via `preprocess_weights` — see [the next section](#when-to-override-preprocess_weights)):

| HF attribute | Fold target | Used by |
|---|---|---|
| `logit_scale` | `unembed.weight` (multiply in fp32 then cast back) | Cohere — final logits scaled by `1/16` |
| `embedding_multiplier` / embed-scale flags | `embed.weight` (multiply) | Gemma — embeddings scaled by `√d_model` |
| Tied unembed with extra scale | `unembed.weight` | T5-family tied projection variants |

Rule of thumb: if the model card or HF source mentions a numerical knob, assume it needs to land on `self.cfg`. If that knob changes weights or final outputs and HF's forward applies it natively, you ALSO need a `preprocess_weights` override or compatibility mode will diverge.

**Passthrough gotcha:** if your attr is NOT a declared `TransformerBridgeConfig` field, `TransformerBridgeConfig.from_dict(hf_config)` silently filters it out — `getattr(cfg, "<attr>", None)` in your adapter will return `None` and your fallback default fires regardless of the model's actual value. To propagate, add the attr name to `_HF_PASSTHROUGH_ATTRS` in [`sources/transformers.py`](../sources/transformers.py) (and the duplicate list in [`sources/_bridge_builder.py`](../sources/_bridge_builder.py)). Verify with an integration-test assertion: `assert bridge.cfg.<attr> == hf_model.config.<attr>`.

---

## When to override `preprocess_weights`

Default is a no-op pass-through. Override when a numerical op HF applies natively in forward must also be baked into raw weights — otherwise `bridge.enable_compatibility_mode()` (which calls `process_weights` expecting the math to already be in the weights) diverges.

> **Trigger:** if a config attr changes the forward-pass math AND isn't re-applied by compat-mode weight processing, fold it into the relevant weight in `preprocess_weights()`.

Examples:

- **Cohere** — `cfg.logit_scale` (default `0.0625`) folds into `unembed.weight`. HF forward multiplies logits by `logit_scale`; compat-mode doesn't.
- **Gemma1/2/3** — embedding scale (`√d_model`) folds into `embed.weight`. HF's `GemmaTextScaledWordEmbedding` scales on forward; compat-mode reads raw.

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

**Skip the override when:** the op lives inside an HF submodule the bridge delegates to (e.g. RoPE inside `CohereRotaryEmbedding`) — HF forward applies it on both paths, parity is free.

**Verification:** a missing fold degrades Phase 4 (text-generation), and Phase 3 if severe enough to trip the strict gate. P4's 50% bar is intentionally lenient, so sub-100% P4 on a small parity-test model is worth investigating even with `STATUS_VERIFIED`. See [tools/model_registry/AGENTS.md §Phase-score thresholds](../../tools/model_registry/AGENTS.md#phase-score-thresholds).

---

## Tokenizer policy

Tokenizer behaviour is **per-model**, not per-architecture — siblings routinely differ (instruct vs base BOS, padding side, EOS, chat template). **Don't copy these from your starter adapter blindly.**

| Flag | Default | Failure mode |
|---|---|---|
| `default_prepend_bos` | Framework default | Off-by-one logits; generation starts from wrong context |
| `default_padding_side` (`"left"` / `"right"`) | HF tokenizer default | Batched-uneven-length generation produces garbage |
| EOS handling (single or list) | `cfg.eos_token_id` | Generation runs past or stops short |
| Chat-template auto-apply | Off (user calls explicitly) | Instruct models produce base-style continuations |
| Tokenizer class mismatch | HF resolves | Subtle BPE/SentencePiece divergence (rare, real for forks) |

**How to verify:**

1. **Run the tokenizer.** `tokenizer_config.json` lies for some architectures — Cohere declares `add_bos_token=False` but `__call__` prepends BOS via `add_special_tokens=True`. Only reliable check is to invoke it:

   ```python
   from transformers import AutoTokenizer
   t = AutoTokenizer.from_pretrained("<hf_repo>")
   print(t("hello").input_ids, t.bos_token_id)
   ```

   First-token == BOS → set `cfg.default_prepend_bos = True`. Otherwise leave unset.

2. **Cross-check `tokenizer_config.json`** for `padding_side`, `bos_token`, `eos_token`, `chat_template` (these tend to be honest; runtime mostly overrides BOS-prepending).
3. **Compare to closest sibling** in registry; differences are usually deliberate.
4. **When unsure, leave the framework default** — explicit wrongness hides under "I configured it."

`default_prepend_bos` is the most common trap (Cohere config-vs-runtime mismatch, instruct/base divergence in Llama/Mistral/Gemma).

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

**Verify the four-place wiring**: [`TestRegistrySyncedWithFactory`](../../../tests/unit/tools/test_model_registry.py) asserts four bidirectional invariants across `SUPPORTED_ARCHITECTURES`, `HF_SUPPORTED_ARCHITECTURES`, `CANONICAL_AUTHORS_BY_ARCH`. Run:

```bash
uv run pytest tests/unit/tools/test_model_registry.py -k TestRegistrySyncedWithFactory
```

Failure message names the missing set. (`INTENTIONAL_EXCLUDES` in the test handles internal-only architectures and HF-emits-different-casing aliases; new adapters rarely belong there.)

---

## Common gotchas

| HF convention | Handled in | How |
|---|---|---|
| RoPE (rotary positional embeddings) | `llama.py`, `mistral.py`, `qwen2.py`+ | `RotaryEmbeddingBridge(name="model.rotary_emb")` + `cfg.positional_embedding_type = "rotary"` |
| GQA / MQA (`n_key_value_heads < n_heads`) | `llama.py`, `mistral.py`, `falcon.py`, `cohere.py` | Set `cfg.n_key_value_heads`; pass `n_kv_heads=` to `_qkvo_weight_conversions()` |
| RMSNorm with offset | `gemma1.py`, `gemma2.py`, `gemma3.py` | `cfg.rmsnorm_uses_offset = True` + `ArithmeticTensorConversion(ADDITION, 1.0)` |
| Standard LayerNorm | `gpt2.py`, `bloom.py` | `cfg.normalization_type = "LN"` |
| Gated MLP (`gate_proj`, `up_proj`, `down_proj`) | `llama.py`, `mistral.py`, `gemma1.py`, `qwen2.py`+ | `GatedMLPBridge` with submodules `{gate, in, out}` |
| Combined QKV (`c_attn`) | `gpt2.py`, `bloom.py` | `QKVSplitRearrangeConversion` to split + rearrange |
| Split Q/K/V (standard) | `llama.py`, `mistral.py`, most modern | `self._qkvo_weight_conversions()` helper |
| MoE routing | `mixtral.py`, `deepseek_v3.py`, `qwen3_moe.py`, `granite_moe.py` | `MoEBridge` with `gate` + batched expert submodules |
| Missing biases (RMSNorm has no `b`; Llama has no attn/MLP biases) | `llama.py` (documented in docstring) | Weight processing handles `None` via `ProcessWeights._safe_get_tensor()` |
| KV cache layout | All (implicit) | Adapter delegates; HF module manages internally |

---

## Adding the HF repo to the registry

After registration, add the model ID to [`data/supported_models.json`](../../tools/model_registry/data/supported_models.json) so `verify_models` can resolve it. Entry shape:

```json
{
  "architecture_id": "MyArchForCausalLM",
  "model_id": "org/repo-name",
  "status": 0,
  "verified_date": null,
  "metadata": null,
  "note": null,
  "phase1_score": null, "phase2_score": null, "phase3_score": null,
  "phase4_score": null, "phase7_score": null, "phase8_score": null
}
```

**Hand-edits add new entries only.** `status`, `verified_date`, `note`, `phaseN_score` are written by `update_model_status()`; never set manually. See [tools/model_registry/AGENTS.md](../../tools/model_registry/AGENTS.md).

Prompt the user about canonical sibling variants from `CANONICAL_AUTHORS_BY_ARCH[<HFArchClass>]` — e.g. adding `google/gemma-2-2b`, ask about `-2b-it`, `-9b`, `-9b-it`, `-27b`, `-27b-it`. The HF scraper picks these up eventually; explicit entries unblock `verify_models` immediately.

Do NOT add the model to [`supported_models.py`](../../supported_models.py) — HookedTransformer-only.

---

## Model source paths: `boot_transformers` vs `boot_native`

Two parallel load paths, same adapter system:

| Source | Entry | Use case |
|---|---|---|
| `boot_transformers` | `TransformerBridge.boot_transformers(model_name, ...)` | Default — wraps an HF model. Every adapter here supports it. |
| `boot_native` | `TransformerBridge.boot_native(cfg, init_mode=...)` | TL-native, no HF dependency. Used by [`Realtime_Training_Telemetry_Demo.ipynb`](../../../demos/Realtime_Training_Telemetry_Demo.ipynb) and cfg-driven small models. |

Native models live in [`sources/native/model.py`](../sources/native/model.py); routed through [`native.py`](native.py) whose `component_mapping` adapts to `cfg` (rotary drops `pos_embed`, RMS norm → `RMSNormalizationBridge`, gated → `GatedMLPBridge`, `attn_only` drops MLP). Init policies in [`sources/native/init.py`](../sources/native/init.py): `gpt2` (default, Normal + 1/√(2·n_layers) residual scaling), `xavier_uniform/normal`, `kaiming_uniform/normal`. Scoped `torch.Generator` keeps seeded init from perturbing global RNG.

Cfg-driven Native features: `normalization_type` (LN / RMS / RMSPre), `final_rms`, `gated_mlp`, `attn_only`, `n_key_value_heads`, `attn_scores_soft_cap`, `output_logits_soft_cap`, `positional_embedding_type`, `rotary_dim`, `rotary_base`, `rope_scaling` (linear PI, dynamic/NTK, llama3 by-parts).

Sister backends under [`sources/`](../sources/) — `transformers/`, `vllm/`, `inspect/` — are internal. Almost all new work writes a `supported_architectures/<arch>.py` file targeting the HF backend.

**Need both paths?** Almost never. Touch `native.py` only when adding a primitive (e.g. new norm type) the cfg-driven dispatch doesn't cover.

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

No weight load, no HF Hub access — synthetic cfg + structural assertions only. Runs in default `make unit-test`.

Add one test per architecture quirk (softcaps, RMSNorm offsets, sliding window, MoE routing). Gemma1's "must NOT override `setup_hook_compatibility`" is a good one-quirk-one-test example.

### 2. Integration parity test — `tests/integration/model_bridge/test_<arch>_adapter.py`

Loads a real cached HF model; asserts logit parity at fp32 + eager attention. Templates: [`test_deepseek_adapter.py`](../../../tests/integration/model_bridge/test_deepseek_adapter.py), [`test_falcon_adapter.py`](../../../tests/integration/model_bridge/test_falcon_adapter.py), [`test_mamba_adapter.py`](../../../tests/integration/model_bridge/test_mamba_adapter.py).

Required test classes (one per concern; copy-rename from a sibling):

| Class | Asserts |
|---|---|
| `Test<Arch>BridgeCreation` | Boot succeeds, expected components present, `cfg.<flags>` set correctly |
| `Test<Arch>ForwardEquivalence` | Logit parity vs HF at fp32 + eager attention |
| `Test<Arch>HFDelegation` | Bridge submodules hold live HF objects (e.g. `bridge.blocks[0].attn.q.original_component is hf_model.model.layers[0].self_attn.q_proj`) |
| `Test<Arch>HookShapes` | Hook fires; output shape matches expectation (replace with `ParallelHooks` / `MoEHooks` etc. for architecture-specific block shapes) |
| `Test<Arch><Quirk>` | One class per architectural quirk: `LogitScale`, `RMSNormOffset`, `Softcap`, `TiedEmbedding`, etc. — each propagated config attr should have a test asserting `bridge.cfg.<attr> == hf_model.config.<attr>` |

The boot fixture uses `dtype=` (Bridge's API), NOT `torch_dtype=` (HF's):

```python
@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(
        "<hf_repo>", device="cpu", dtype=torch.float32, attn_implementation="eager",
    )
```

If the reference model OOMs CI, gate with `@pytest.mark.skipif(bool(os.getenv("CI")), ...)` and add a row to [`QUARANTINES.md`](../../../tests/QUARANTINES.md) under "CI cost / network budget."

### End-to-end registry verification

```bash
set -a; source .env; set +a
uv run python -m transformer_lens.tools.model_registry.verify_models --architectures MyArchForCausalLM --per-arch 1
```

Wrapped by [`/verify-model`](../../../.claude/commands/verify-model.md). See [tools/model_registry/AGENTS.md](../../tools/model_registry/AGENTS.md) for the contract and the [`verify_models` vs `main_benchmark` trap](../../tools/model_registry/AGENTS.md#tldr). For parity failures: [debugging_numerical_divergence.md](../../../docs/source/content/debugging_numerical_divergence.md).

---

## Hard "don'ts"

- **No single-model files** — one file per architecture family.
- **No `# type: ignore` on `ComponentMapping`** — `isinstance` / `cast` ([AGENTS.md §10](../../../AGENTS.md#10-hard-rules)).
- **No skipping any of the four registration sites** — each breaks something different. [Invariant test](../../../tests/unit/tools/test_model_registry.py) catches most.
- **No HT-style hook aliases inside adapters** — aliases live in [`bridge.py`](../bridge.py); adapters declare Bridge-native names only.
- **No inheriting tokenizer-policy flags from the starter** — see [Tokenizer policy](#tokenizer-policy).
- **No manual edits to existing `supported_models.json` entries' status/phase fields** — only `update_model_status()`.
- **No skipping the unit adapter test** — required for every architecture.
