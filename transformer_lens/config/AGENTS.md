# Config — AGENTS.md

The config dataclasses that drive both `HookedTransformer` and `TransformerBridge`. Read [the root AGENTS.md](../../AGENTS.md) for project-wide rules.

## File map

| File | Class | Used by |
|---|---|---|
| [`transformer_lens_config.py`](transformer_lens_config.py) | `TransformerLensConfig` | Minimal base — only fields actually used by the system |
| [`transformer_bridge_config.py`](transformer_bridge_config.py) | `TransformerBridgeConfig(TransformerLensConfig)` | The Bridge config; what every Bridge adapter receives as `cfg` |
| [`hooked_transformer_config.py`](hooked_transformer_config.py) | `HookedTransformerConfig` | Legacy HT-only config (deprecated; see [AGENTS.md §2](../../AGENTS.md#2-two-systems-live-in-this-repo)) |

## Adding a new HF-config attr to `TransformerBridgeConfig` — decision tree

The `logit_scale` bug existed because the rules below weren't documented anywhere. **Pick exactly one of the four paths**; doing more than one creates silent-override risk.

| Use case | Path |
|---|---|
| First-class TL field that adapters / hooks / weight processing read | **Declare as a dataclass parameter** on `TransformerBridgeConfig`. Set a sensible default. Update `map_default_transformer_lens_config` in [`sources/transformers.py`](../model_bridge/sources/transformers.py) to translate the HF-config attr name to your field name. |
| HF attr the adapter reads at runtime, no semantic translation needed | **Add to `_HF_PASSTHROUGH_ATTRS`** in BOTH [`sources/transformers.py:481`](../model_bridge/sources/transformers.py) AND [`sources/_bridge_builder.py:18`](../model_bridge/sources/_bridge_builder.py). The trap: adding to only one half-fixes. See [sources/AGENTS.md](../model_bridge/sources/AGENTS.md). |
| HF attr name differs from existing TL field | **Add an explicit handler** in `map_default_transformer_lens_config` (e.g. Gemma2's `final_logit_softcapping` → `output_logits_soft_cap`). Don't also add to PASSTHROUGH. |
| Just a derived view of an existing field | **Add a `@property`** on `TransformerBridgeConfig` (e.g. `head_dim` aliases `d_head`). |

**Don't** declare the same attr both as a dataclass field AND a PASSTHROUGH entry — PASSTHROUGH writes happen AFTER `from_dict`, so the runtime value will silently overwrite whatever defaults / explicit handlers set.

## Existing properties that may surprise you

- **`head_dim`** is a read-only `@property` aliasing `d_head` (line 212). `setattr(cfg, "head_dim", val)` raises `AttributeError: property 'head_dim' has no setter`. Don't add it to PASSTHROUGH.
- **`n_heads`** has `= -1` as its placeholder in the constructor signature, deliberately. Comment in the source: *"Add n_heads to signature so it's not filtered out by from_dict"*. Don't "fix" this default.

## Verifying a new field propagates

Integration test pattern (the regression test that would have caught the `logit_scale` bug):

```python
def test_cfg_<attr>_matches_hf(bridge: TransformerBridge, hf_model: Any) -> None:
    """Regression: <attr> must propagate from HF (not silently fall back)."""
    bridge_val = getattr(bridge.cfg, "<attr>")
    assert bridge_val == hf_model.config.<attr>
```

The `getattr` form sidesteps mypy's "undeclared field" complaint without `# type: ignore` (see [AGENTS.md §10](../../AGENTS.md#10-hard-rules)).

Pick a test model whose `<attr>` value differs from the adapter's hardcoded fallback (if any) — otherwise the assertion passes by tautology.
