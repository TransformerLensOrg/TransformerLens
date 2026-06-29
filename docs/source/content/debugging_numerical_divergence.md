# Debugging Numerical Divergence

When a Bridge adapter's integration test fails by `~1e-3` (or any larger delta) against the HuggingFace reference, the failure mode is almost always one of a small set of recurring bugs. This page walks the bisection workflow.

> A note before you start: it's tempting to attribute small drift to "floating-point noise" and move on, but genuine bugs and accumulated rounding error are indistinguishable at small magnitudes until you measure. The [numerical-work conventions in contributing.md](contributing.md#numerical-work) describe the cheap fp64 check that disambiguates the two.

---

## 0. Setup checklist

- **fp32 + eager attention on both sides.** `dtype=torch.float32, attn_implementation="eager"`. `sdpa` / `flash_attention_2` mask bugs.
- **`enable_compatibility_mode(no_processing=True)`** for the first pass — isolates forward-pass bugs from weight-processing bugs. See [compatibility_mode.md](compatibility_mode.md).
- **Single-token first**, then 5–10, then longer. Most adapter bugs surface single-token.
- **Same seed / no dropout.** A stray `nn.Dropout(p=0.1)` in a generalized component silently de-correlates runs.

## 1. Bisect by component

Walk Bridge hooks vs HF `output_hidden_states=True` / `output_attentions=True` and find the first layer where they diverge:

| Stage | Bridge hook | HF output |
|---|---|---|
| Embedding | `embed.hook_out` | `outputs.hidden_states[0]` |
| Block i pre-attn-norm | `blocks.{i}.ln1.hook_out` | (HF doesn't expose; compute from `hidden_states[i]`) |
| Block i Q / K / V | `blocks.{i}.attn.q.hook_out`, `.k.hook_out`, `.v.hook_out` | (HF doesn't expose; instrument `model.layers[i].self_attn` directly) |
| Block i attention output | `blocks.{i}.attn.hook_out` | `outputs.attentions[i]` (pattern), then hidden_state delta |
| Block i MLP output | `blocks.{i}.mlp.hook_out` | (HF doesn't expose; hidden_state delta) |
| Block i residual out | `blocks.{i}.hook_resid_post` | `outputs.hidden_states[i+1]` |
| Final norm | `ln_final.hook_out` | (HF inlines into lm_head) |
| Logits | `unembed.hook_out` | `outputs.logits` |

The first hop where they disagree localizes the bug.

## 2. Common root causes, in order of frequency

| Symptom | Likely cause | Where to look |
|---|---|---|
| Logits off everywhere but Q/K/V close | RoPE base / scaling mismatch | Adapter's `RotaryEmbeddingBridge` setup; check `cfg.rotary_base`, `cfg.rope_scaling` |
| Attention output drifts; Q / K / V match | Wrong `n_key_value_heads`, wrong head reshape | `_qkvo_weight_conversions(n_kv_heads=...)`; GQA-aware split |
| First-layer outputs off; embeddings off | Embedding scaling missing (Gemma, T5) | `preprocess_weights()` override; `cfg.scale_embeddings` |
| Off by a constant scale in residual | Final-RMS-norm offset missing | `cfg.rmsnorm_uses_offset = True` + `ArithmeticTensorConversion(ADDITION, 1.0)` |
| Logits flat / saturated at extremes | Missing logit softcap | `cfg.output_logits_soft_cap` from HF's `final_logit_softcapping` |
| Attention pattern collapses to argmax | Missing attention-score softcap | `cfg.attn_scores_soft_cap` from HF's `attn_logit_softcapping` |
| First MLP off; gate matches | Forgot gated-MLP wiring | `GatedMLPBridge` with `{gate, in, out}` submodules — not `MLPBridge` |
| Bias-related drift | Adapter assumes biases that don't exist (Llama / RMSNorm) | `ProcessWeights._safe_get_tensor` handles `None`; check the weight-processing conversions are bias-aware |
| Drift only in compatibility mode | Hook semantic carve-out missing for post-norm or MLA | See [compatibility_mode.md](compatibility_mode.md) §"Hook semantic parity" |

## 3. Isolating weight-processing bugs

If `no_processing=True` matches HF but `enable_compatibility_mode()` (default) drifts:

- The bug is in weight processing, not the forward pass.
- Bisect by toggling individual flags: `fold_ln`, `center_writing_weights`, `center_unembed`, `fold_value_biases`. The first one that introduces drift is the culprit.
- See [compatibility_mode.md §"What each flag does"](compatibility_mode.md#what-each-flag-does).

## 4. Comparing against `boot_transformers` for the same model

If Bridge ≠ HF on a model that already passes `verify_models`, your adapter likely diverges from the canonical Bridge load configuration. Quick sanity check:

```python
import torch
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformers import AutoModelForCausalLM

ref = TransformerBridge.boot_transformers(model_name, device="cpu", dtype=torch.float32)
hf = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, attn_implementation="eager"
)

ids = torch.tensor([[hf.config.bos_token_id or 1]])
ref_logits = ref(ids)
hf_logits = hf(ids).logits

print((ref_logits - hf_logits).abs().max())  # should be < 1e-4 in fp32
```

If `boot_transformers` itself disagrees with HF on the same model, the issue is upstream of your adapter (probably a `_HF_PASSTHROUGH_ATTRS` gap in `transformer_lens/model_bridge/sources/_bridge_builder.py`, or a non-standard HF config attribute that the adapter never propagated onto `self.cfg`). HF raw config attributes are invisible to TL-side consumers unless explicitly mirrored. Common attributes that need propagation: `final_logit_softcapping` (Gemma2/3), `attn_logit_softcapping` (Gemma2/3), `query_pre_attn_scalar` (Gemma2/3), `sliding_window` (Mistral, Qwen2, Gemma2), `layer_types` (hybrid models), and non-standard RMSNorm eps attribute names (Llama uses `variance_epsilon`).

## 5. Bisecting `verify_models` phase failures

`verify_models` reports phase-by-phase. Map the failing phase to the bisection focus:

| Phase | What failed | Start here |
|---|---|---|
| 1 | Forward correctness vs HF | Steps 1–4 above; this is the standard parity workflow |
| 2 | Hook firing / gradient flow | The hook isn't registered, or it's firing on a tensor that's been replaced (in-place op). Grep adapter for in-place ops on hookable tensors. |
| 3 | Weight processing | Run with `no_processing=True` to isolate. Then bisect compat-mode flags per §3 above. |
| 4 | Text-generation quality | Usually tokenizer policy: `default_prepend_bos`, padding side, EOS handling, chat-template wiring. Tokenizer behaviour is per-model, not per-architecture — check the target's `tokenizer_config.json`, don't inherit from a sibling. Less often, a generation-loop divergence; rerun with `--no-ht-reference` to skip HT comparison. |
| 7 | Multimodal alignment | Vision encoder output drift or projection mismatch. Llava / Gemma3-multimodal only. |
| 8 | Audio | HuBERT only; check CTC head and audio-feature alignment. |

## 6. What "fp noise" actually looks like

Empirically, in this codebase:

- **fp32, eager attention, single forward**: HF vs Bridge max-abs diff is typically `< 5e-5`. Anything ≥ `1e-4` is suspicious.
- **bf16, eager**: `< 1e-2` is the noise floor.
- **fp32, sdpa**: `< 5e-4` due to sdpa's internal reductions. Use eager for parity tests.

If you suspect noise, the cheap proof is to run **fp64**: `dtype=torch.float64` on both sides. If the diff stays the same magnitude, it's a bug. If it drops by ~8 orders of magnitude, it was noise. See the [numerical-work conventions in contributing.md](contributing.md#numerical-work) for more context on why this check is worth running.

## 7. Tooling

- `make integration-test PYTEST_ADDOPTS="-k <arch> -s"` — focused run with stdout.
- `transformer_lens/scratch.py` (gitignored) — drop one-off bisection scripts here without polluting `git status`.
- `.adapter-workspace/` (gitignored) — sibling directory for WIP adapter notes / repros.
- `bridge.run_with_cache(ids)` — returns `(logits, cache)`; `cache["blocks.{i}.hook_resid_post"]` is the easiest path to per-layer diffs.

---

If you exhaust this guide and still can't localize the bug, the failure pattern is worth adding to §2 above so the next contributor doesn't repeat the bisection.
