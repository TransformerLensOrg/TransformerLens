# Architecture Unit Test Suite Guide

This is the companion to the [Architecture Adapter Creation Guide](adapter-creation-guide.md)'s "Tests" step — read that first for where the file goes and how to boot the adapter.

A new adapter needs two test layers:

- **Unit adapter test** — `tests/unit/model_bridge/supported_architectures/test_<arch>_adapter.py`. Instantiates the adapter from a *synthetic* `TransformerBridgeConfig` and asserts **structural** properties. No weight load, no HF Hub, runs in `make unit-test`. **This guide is about this layer.**
- **Integration parity test** — `tests/integration/model_bridge/test_<arch>_adapter.py`. Loads a real cached HF model — or a tiny random fixture when the real model OOMs CI — and asserts logit parity at fp32 + eager attention. Required; covered in [contributing.md](../contributing.md#required-tests-for-a-new-adapter).

## The one rule

> A unit test earns its place if a realistic bug in **this adapter** would fail it, **and no other test would catch that bug**.

Everything below is a corollary. When you copy a sibling test file as a starting point, don't keep a test just because the sibling has it. Keep it only if it guards something *your* adapter does that isn't already covered.

One caveat to *"no other test would catch it"*: that other test has to actually run. If your model is too large for a real integration parity test in CI and you have no tiny fixture for it, the unit suite is the sole guard — keep the behavioral coverage (hook shapes, fold values, end-to-end override effects) you'd otherwise lean on integration to provide.

## What to test

Organize around the three things an adapter decides (config, component mapping, weight conversions) plus its overrides. For each, assert the **arch-specific** choice — the thing you'd get wrong porting from a sibling.

| Area | Worth asserting | Skip |
| --- | --- | --- |
| **Component mapping** | The HF module paths and bridge **types** for this arch — especially non-standard ones (`transformer.wte`, `model.tok_embeddings`, `out_proj`, `fc_in`, `EncDecAttention`); the distinctive bridge (`JointQKVAttentionBridge`, `ParallelBlockBridge`, `SymbolicBridge`, `MoEBridge`, `SigLIP`); the exact submodule **set** (e.g. attention has `q_norm`/`k_norm`, or block has no `ln2`). | — |
| **Config quirks** | Propagation that drives *behavior*: `n_key_value_heads` (GQA) through the adapter's own branch, custom `eps_attr` value, softcap / `logit_scale` coercion + `None`-fallback, `rmsnorm_uses_offset`, `parallel_attn_mlp`, `uses_combined_qkv`, `supports_fold_ln=False` when a fused projection forces it, multimodal/`gated_q_proj` flags. | A flag whose only effect is the literal you set (see "config-literal" below). |
| **Weight conversions** | Logic the **adapter** implements: a fused-QKV split's numerical partition (which rows are Q vs K vs V — e.g. GPT-2 thirds, CodeGen's `[Q,V,K]` `mp_num` ordering, Baichuan/InternLM2 interleaved layouts), a manual LayerNorm fold (values folded, weight reset to ones, dtype preserved), the exact conversion **key set** (no stray norm/bias entries). | The einops rearrange itself (see "dependency test"). |
| **Overrides** | Each branch of `setup_component_testing` / `preprocess_weights` / `prepare_model` / `prepare_loading` you wrote — the happy path *and* the defensive `hasattr`/`None` guards, the no-op-when-absent path, the rejection guard. | Overrides you didn't write. |
| **Behavioral hook shapes** | Where the adapter's config drives reshaping: GQA `hook_k`/`hook_v` at `n_key_value_heads`, MQA single KV head, hybrid layers where attn hooks are **absent** on linear-attention layers. | Generic `(batch, seq, d_model)` output shape (it's the shared bridge's contract, not yours). |

If your model invents a mechanism (AltUp's stacked residual, T5's relative-position bias, NoPE layers), test the **observable consequence** of it — the active-stream hook shape, the `is_cross_attention`/`requires_relative_position_bias` flags, a NoPE layer ignoring position embeddings end-to-end.

The behavioral rows (hook shapes, end-to-end effects) need a forward pass, but the unit layer has no weights — so you wire one by hand. The pattern (see `test_mixtral_adapter.py::TestMixtralGQAHookShapes` for a worked example): build a small fake attention `nn.Module`, attach it with `set_original_component`, wire the child q/k/v/o bridges, call `setup_hook_compatibility()`, then run with identity rotary `(cos, sin)` inputs and read the hook. It's more scaffolding than a structural assertion — reserve it for reshaping logic a structural check can't reach.

## What not to test

These five anti-patterns make up the bulk of potential test bloat. Each "test" below changes only when the line it mirrors changes, or when a third party / the base class changes — never when *your adapter* regresses.

### 1. Config-literal restatements

```python
# The adapter sets self.cfg.normalization_type = "RMS" one line away.
def test_normalization_type(self, adapter):
    assert adapter.cfg.normalization_type == "RMS"     # tautological
```

Asserting `cfg.<flag> == <the literal you assigned>` is a change-detector, not a behavior guard. Test what the flag *does* (the bridge type it selects, the fold it enables) — and delete the literal restatement only once that effect is covered (e.g. a component-mapping test already asserts the resulting `RMSNormalizationBridge` / `NormalizationBridge`). If the effect isn't tested anywhere, test the effect, not the flag. Exception: a flag whose value is a deliberate **anti-drift** choice against a sibling (e.g. OLMoE `final_rms=False` vs the family default, GPT-BigCode `n_key_value_heads=1` for MQA) is worth one test with a comment saying why.

### 2. Factory / registration duplicates

```python
def test_factory_returns_my_adapter(self):
    assert isinstance(factory.select(cfg), MyAdapter)   # already covered globally
```

`tests/unit/tools/test_model_registry.py::TestRegistrySyncedWithFactory` bidirectionally checks that every adapter is registered and synced across all four sites. Per-adapter `test_factory_*` / `test_in_supported_architectures` / `test_import_from_init` just re-run that global invariant.

### 3. Dependency tests

```python
def test_q_weight_splits_into_n_heads(self, adapter):
    out = conversion.convert(w); assert torch.equal(conversion.revert(out), w)  # tests einops
```

A convert↔revert round-trip over a `RearrangeTensorConversion` tests that einops permutations are lossless — a property of the conversion engine, not your adapter. Likewise, running a split `nn.Linear` forward to check its output shape tests `torch.nn.Linear`. Assert the conversion's **pattern/axis metadata** (that's your decision) and the **numerical partition** (which input rows land where); leave the rearrange's correctness to the conversion library's own tests.

### 4. Base-class retests

If the adapter doesn't override it, don't test it here — the base class's own tests cover it. Two common forms:

- **Base-helper conversions.** When you build conversions with `self._qkvo_weight_conversions()` unchanged, the rearrange patterns and the GQA `n_kv_heads` axis come from the base helper. Testing them tests the base. (If you define conversions *inline*, they're yours — test them.)
- **Inherited bridge defaults.** `assert attn.optional is False` when the adapter never sets `optional`, or asserting the base `get_random_inputs` shapes — these assert defaults you inherited, not choices you made.

### 5. Subsumed and duplicate assertions

```python
def test_q_weight_key_present(self): assert "q.weight" in conv      # subsumed
def test_exactly_four_keys(self):     assert len(conv) == 4         # subsumed
def test_only_qkvo_keys(self):        assert set(conv) == {...}     # keep this one
```

A single exact-set assertion subsumes the per-key membership checks and the count check. Keep the strongest one. The same goes for the same assertion copy-pasted into two classes, and for a pure-`pass` subclass adapter — if the class body overrides nothing, the parent's test file already covers every config/mapping/conversion; the subclass needs exactly one test, that it *is* a subclass.

## A litmus before you commit a test

Ask which of these would make the test fail. If it's only the first, delete it:

1. Editing the exact line in the adapter the test mirrors → **tautological** (anti-pattern 1, 2, 5).
2. Upgrading einops / torch / transformers → **dependency** (anti-pattern 3).
3. Editing the **base** `ArchitectureAdapter` / a generalized component → belongs in **base-class tests** (anti-pattern 4).
4. Introducing a realistic bug in **this adapter's** mapping, conversion, or override → **keep it.**

## Common pitfalls when writing the suite

- **Asserting the fixture's own input.** `_make_cfg(default_prepend_bos=True)` then `assert adapter.cfg.default_prepend_bos is True` tests your test, not the adapter. (Tokenizer-policy flags the adapter *sets itself* are fair game.)
- **Mirroring the assignment instead of the effect.** `cfg.parallel_attn_mlp = True` → don't assert the flag; assert the block is a `ParallelBlockBridge` with no `ln2`.
- **Treating base-helper output as adapter logic.** Only test QKVO patterns/axes when the adapter customizes them.
- **Anchoring on a sibling's checklist.** Copy a sibling for *structure*, then add tests for *your* arch's uncovered quirks and drop the ones that don't apply. The goal is to provide new coverage for unique adapter quirks.
- **Leaving orphans after trimming.** When you remove the last consumer of a fixture, helper, import, or test class, remove it too — `make check-format` (pycln + isort) will flag unused imports, but not unused fixtures.
- **`# type: ignore` on `ComponentMapping`.** Prefer `isinstance` narrowing or `typing.cast`; the project avoids `# type: ignore`.

## Checklist

- Every test maps to a config quirk, a mapping decision, a conversion, or an override **this adapter** owns.
- No config-literal restatements; no per-adapter factory/registration tests; no einops/`nn.Linear` round-trips; no base-helper or inherited-default retests.
- Exact-set assertions instead of per-key membership + count.
- Override branches (happy path + defensive guards) covered.
- Behavioral hook-shape tests for GQA/MQA/hybrid where config drives reshaping.
- No orphaned fixtures/imports/empty classes; `make check-format` and `uv run mypy .` clean.
