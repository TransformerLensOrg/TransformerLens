# TransformerBridge Compatibility Mode

`TransformerBridge.boot_transformers(...)` returns a bridge whose **default numerics match HuggingFace** — raw weights, no folding, no centering. Calling `bridge.enable_compatibility_mode()` afterwards puts the bridge into **HookedTransformer-equivalent numerics** — weights folded, centered, and the legacy hook aliases registered.

Most research code that was written against `HookedTransformer.from_pretrained(...)` assumes compatibility mode. Most new code that needs HF-faithful logits does not.

> Source: [`transformer_lens/model_bridge/bridge.py:enable_compatibility_mode`](../../../transformer_lens/model_bridge/bridge.py).

---

## When to enable it

| Use case | Compatibility mode? | Why |
|---|---|---|
| Logit lens / direct logit attribution | **Yes** | These analyses reason in the post-fold-LN coordinate system; raw HF weights produce different (wrong) attributions. |
| Residual-stream norm analysis | **Yes** | Centered weights give the residual a meaningful zero. |
| Circuit analysis using HT-style hook names (`blocks.{i}.attn.hook_q`, `hook_resid_pre`, etc.) | **Yes** | Legacy aliases register only after compat mode. |
| Logit parity against HuggingFace | **No** | Folding changes weights; logits will not match HF. |
| Generation / inference vs HF baseline | **No** | Same reason. |
| Verifying a new adapter's forward pass | **No (initially)** | Use `enable_compatibility_mode(no_processing=True)` to get hook aliases without weight processing — isolates forward-pass bugs from weight-processing bugs. |

## What each flag does

```python
bridge.enable_compatibility_mode(
    disable_warnings: bool = False,
    no_processing: bool = False,
    fold_ln: bool = True,
    center_writing_weights: bool = True,
    center_unembed: bool = True,
    fold_value_biases: bool = True,
    refactor_factored_attn_matrices: bool = False,
)
```

| Flag | Default | Effect |
|---|---|---|
| `no_processing` | `False` | If `True`, **overrides all other processing flags to False** — registers the legacy hook aliases only, leaves weights raw. The "I want HT hook names but HF numerics" mode. |
| `fold_ln` | `True` | Folds LayerNorm scale + bias into the subsequent linear weights so the LayerNorm modules become pure normalization. Changes weights; mathematically equivalent. |
| `center_writing_weights` | `True` | Subtracts the mean from each "writing" weight (`W_out` in attention, MLP-down). Makes residual contributions sum to zero per layer, which makes residual-stream norms interpretable. |
| `center_unembed` | `True` | Subtracts the mean from the unembedding matrix. Logits become mean-zero — affects logit-lens output but not argmax. |
| `fold_value_biases` | `True` | Folds attention value biases into the output bias. Same numerics, fewer parameters. |
| `refactor_factored_attn_matrices` | `False` | Refactors `W_Q @ W_K.T` and `W_V @ W_O` for analysis. Off by default because it's slow and only matters for specific factored-matrix research. |
| `disable_warnings` | `False` | Suppresses warnings emitted by legacy component aliases when accessed. |

After processing, the bridge **also**:

- Re-initializes the hook registry.
- Calls `_setup_hook_compatibility()` on every component (installs HT-style hook conversions like reshaping `hook_z` from `[batch, seq, d_model]` to `[batch, seq, n_heads, d_head]`).
- Registers HT-style hook aliases recursively across blocks.

`compatibility_mode` is then `True` on the bridge and on every component, so subsequent operations behave as if the bridge were loaded by `HookedTransformer.from_pretrained()`.

## Hook semantic parity

After `enable_compatibility_mode()`, these HT hook names fire on the **pre-norm residual** (matching HookedTransformer semantics):

- `blocks.{i}.attn.hook_q_input`, `hook_k_input`, `hook_v_input`
- `blocks.{i}.hook_attn_in`
- `blocks.{i}.hook_mlp_in` (gated on `cfg.use_hook_mlp_in`; toggle via `bridge.set_use_hook_mlp_in(True)`)

**Carve-outs** ([issue #1317](https://github.com/TransformerLensOrg/TransformerLens/issues/1317)):

- **Post-norm architectures** (OLMo 2, BERT-style) read the **post-attention residual** instead, because the norm semantically lives elsewhere in the block.
- **MLA blocks** (DeepSeek V2 / V3 / R1) do **not** expose the split-qkv aliases — MLA's compressed K/V doesn't have a clean split.

An adapter author for a new post-norm or MLA-style architecture must handle these carve-outs in `setup_hook_compatibility`. The Gemma1/Gemma2 adapters are exemplars of when **not** to override `setup_hook_compatibility` — `GemmaTextScaledWordEmbedding` already scales internally, so any added `hook_conversion` would double-scale `embed.hook_out`.

## The four-quadrant test matrix

The integration conftest at [`tests/integration/model_bridge/conftest.py`](../../../tests/integration/model_bridge/conftest.py) provides four bridge variants for every test model:

| Variant | `compatibility_mode` | `no_processing` | Tests… |
|---|---|---|---|
| `gpt2_bridge` | off | n/a | HF-faithful numerics |
| `gpt2_bridge_compat` | on | `False` | HT-equivalent numerics |
| `gpt2_bridge_compat_no_processing` | on | `True` | Hook aliases without weight processing — used to bisect numerical bugs |
| (HT side) `gpt2_hooked_processed`, `gpt2_hooked_unprocessed` | n/a | n/a | Reference HookedTransformer with/without weight processing |

New integration tests should use the variant that matches the property they're testing. Tests of HF parity → `gpt2_bridge`. Tests of HT-API behaviour → `gpt2_bridge_compat`. Tests of hook semantics regardless of weights → `gpt2_bridge_compat_no_processing`.

## Cost

`enable_compatibility_mode()` mutates the bridge's weights in-place. It is:

- **One-shot**: calling it twice on the same bridge is safe but pointless; the second call re-folds already-folded weights into the new (already-folded) LayerNorm modules — which is a no-op for `fold_ln` semantically, but it does re-run the centering subtractions. Don't.
- **Not reversible** from within the bridge: if you need raw weights again, re-boot the bridge.
- **Idempotent in `_setup_hook_compatibility`**: that method can be called multiple times safely; only `process_weights` mutates weights.

## See also

- [supported_architectures/AGENTS.md](../../../transformer_lens/model_bridge/supported_architectures/AGENTS.md) — adapter contract; the `setup_hook_compatibility` override hook is documented there.
- [debugging_numerical_divergence.md](debugging_numerical_divergence.md) — uses `no_processing=True` as a key bisection tool.
- [migrating_to_v3.md](migrating_to_v3.md) — when porting HT code, you almost always want `enable_compatibility_mode()`.
