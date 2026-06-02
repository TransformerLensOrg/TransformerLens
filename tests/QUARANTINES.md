# Test Quarantines

Inventory of every `@pytest.mark.skip`, `skipif`, and `xfail` in [`tests/`](.). Use this to triage failures: a test on this list with the matching reason is **not your bug** and shouldn't be debugged blindly. A test that fails but **isn't** on this list is something to investigate.

The repo-wide rule (see [AGENTS.md §10](../AGENTS.md#10-hard-rules)): **never add `xfail` / `skipif` to dodge a failing CI**. Every quarantine below has a category and a "When to un-skip" line. If you add a new skip, add it here too.

---

## Permanent — optional dependency

These tests require an optional install group and are skipped on the default install. Will never enable in CI without infrastructure changes.

| Path | Marker | Trigger |
|---|---|---|
| [`tests/unit/test_lit.py`](unit/test_lit.py) (×18) | `skipif(not LIT_AVAILABLE)` | `pip install lit-nlp` (the `lit` optional-dep group) |
| [`tests/unit/components/test_attention.py`](unit/components/test_attention.py) (line 48) | `skipif(not is_bitsandbytes_available())` | `uv sync --group quantization` |
| [`tests/unit/test_weight_processing.py`](unit/test_weight_processing.py) (line 477) | `skipif(not is_bitsandbytes_available())` | Same as above |
| [`tests/unit/factories/test_mlp_factory.py`](unit/factories/test_mlp_factory.py) (line 40) | `skipif(not is_bitsandbytes_available())` | Same as above |

**Un-skip:** never (these are correct behaviour). To run locally, install the optional group.

---

## Permanent — hardware requirement

| Path | Marker | Required hardware |
|---|---|---|
| [`tests/unit/test_next_sentence_prediction.py`](unit/test_next_sentence_prediction.py) (line 131) | `skipif(not torch.cuda.is_available())` | Any CUDA device |
| [`tests/unit/model_bridge/compatibility/test_next_sentence_prediction.py`](unit/model_bridge/compatibility/test_next_sentence_prediction.py) (line 95) | `skipif(not torch.cuda.is_available())` | Any CUDA device |
| [`tests/unit/components/test_attention.py`](unit/components/test_attention.py) (line 83) | `skipif(not torch.cuda.is_available())` (`reason="CUDA required for half/bfloat16 tests"`) | Any CUDA device |
| [`tests/acceptance/test_hooked_encoder.py`](acceptance/test_hooked_encoder.py) (line 227) | `skipif(not torch.cuda.is_available())` | Any CUDA device |
| [`tests/acceptance/test_hooked_encoder_decoder.py`](acceptance/test_hooked_encoder_decoder.py) (line 421) | `skipif(not torch.cuda.is_available())` | Any CUDA device |
| [`tests/acceptance/test_multi_gpu.py`](acceptance/test_multi_gpu.py) (line 91, 105) | `skipif(torch.cuda.device_count() < 2)` | 2+ CUDA devices |
| [`tests/acceptance/test_multi_gpu.py`](acceptance/test_multi_gpu.py) (line 22) | `skipif(torch.cuda.device_count() < 4)` | 4+ CUDA devices |
| [`tests/acceptance/model_bridge/test_multi_gpu_bridge.py`](acceptance/model_bridge/test_multi_gpu_bridge.py) (line 257) | `skipif(torch.cuda.device_count() < 2)` | 2+ CUDA devices |
| [`tests/mps/test_mps_basic.py`](mps/test_mps_basic.py) (module-level `pytestmark`) | `skipif(not torch.backends.mps.is_available())` | Apple Silicon |

**Un-skip:** never. CI provides each hardware tier via its own dedicated job (CUDA via compatibility-checks → in practice CPU-only; MPS via `mps-checks`; multi-GPU is local-only). See [tests/AGENTS.md §MPS rules](AGENTS.md#mps-rules) and the carve-out list in [`.github/workflows/checks.yml`](../.github/workflows/checks.yml).

---

## Intentional — CI cost / network budget

These tests gate on `os.getenv("CI")` to avoid expensive HF Hub fetches or model loads that don't fit the runner.

| Path | Marker |
|---|---|
| [`tests/unit/model_bridge/supported_architectures/test_gemma2_adapter.py`](unit/model_bridge/supported_architectures/test_gemma2_adapter.py) (line 49) | `skipif(CI, reason="Network/disk fetch of tiny Gemma2 — skip in CI")` |
| [`tests/integration/model_bridge/test_bridge_integration.py`](integration/model_bridge/test_bridge_integration.py) (line 801) | `skipif(CI, reason="Skip Gemma2 test in CI to avoid timeout")` |
| [`tests/acceptance/model_bridge/compatibility/test_hook_completeness.py`](acceptance/model_bridge/compatibility/test_hook_completeness.py) (line 156) | `skipif(CI, reason="Gemma2 is too large for CI")` |

**Un-skip:** when local. These should run cleanly on a dev machine with `HF_TOKEN` sourced.

---

## Intentional — manual verification only

Skipped in pytest because they need real (large) weights, run as part of `verify_models`.

| Path | Marker |
|---|---|
| [`tests/integration/model_bridge/test_qwen3_moe_bridge.py`](integration/model_bridge/test_qwen3_moe_bridge.py) (lines 155, 166) | `skip(reason="Requires real weights — run manually during verification")` |

**Un-skip:** run via `/verify-model Qwen/Qwen3-MoE-...` and the canonical [`verify_models`](../transformer_lens/tools/model_registry/AGENTS.md) workflow.

---

## Upstream / platform bug — wait for fix

| Path | Marker | Underlying issue |
|---|---|---|
| [`tests/unit/model_bridge/test_bridge_generate_no_tokenizer.py`](unit/model_bridge/test_bridge_generate_no_tokenizer.py) (lines 30, 128) | `skipif(_MACOS_ARM64, reason="Upstream macOS-arm64 KV-cache NaN; see linked issue.")` | Upstream PyTorch / HF KV-cache NaN on M-series Macs |

**Un-skip:** when the upstream issue resolves. Don't bypass — the test will produce NaN logits.

---

## ⚠️ Technical debt — actively quarantined whole-file

These are **entire test modules** skipped via module-level `pytestmark`. They represent significant coverage gaps. Treat them as priority work to re-enable, not as permanent quarantine.

| Path | Reason |
|---|---|
| [`tests/acceptance/test_hooked_transformer.py`](acceptance/test_hooked_transformer.py) (line 19) | `"Temporarily skipped due to CI test pollution issues"` |
| [`tests/acceptance/test_hooked_encoder.py`](acceptance/test_hooked_encoder.py) (line 13) | `"Temporarily skipped due to CI test pollution issues"` |
| [`tests/acceptance/test_hooked_encoder_decoder.py`](acceptance/test_hooked_encoder_decoder.py) (line 10) | `"Temporarily skipped due to CI test pollution issues"` |

**Un-skip:** root-cause the "test pollution issues" — likely a fixture-scope or import-ordering bug that leaks state across tests. Until then, these major acceptance tiers are dark.

---

## Technical debt — individual

Tests quarantined individually pending fixes. Each should have a tracking issue or a clear "fix me" note in the surrounding code.

| Path | Marker | What it covers |
|---|---|---|
| [`tests/unit/factored_matrix/test_constructor.py`](unit/factored_matrix/test_constructor.py) (line 54) | `skip(...)` | FactoredMatrix constructor edge case |
| [`tests/unit/model_bridge/test_architecture_adapter.py`](unit/model_bridge/test_architecture_adapter.py) (line 436) | `skip(...)` | Adapter behaviour |
| [`tests/unit/model_bridge/test_bridge_vs_hooked_transformer_patching.py`](unit/model_bridge/test_bridge_vs_hooked_transformer_patching.py) (lines 138, 142) | `skipif(...) / xfail(...)` | Bridge↔HT patching parity |
| [`tests/unit/model_bridge/test_hook_alias_resolution.py`](unit/model_bridge/test_hook_alias_resolution.py) (line 89) | `xfail(strict=True, reason=...)` per-architecture | Hook-alias gaps |
| [`tests/unit/model_bridge/supported_architectures/test_qwen3_5_adapter.py`](unit/model_bridge/supported_architectures/test_qwen3_5_adapter.py) (lines 574, 609, 660, 680, 771) | `skipif(...)` × 5 | Specific Qwen3.5 adapter behaviours |
| [`tests/unit/model_bridge/supported_architectures/test_qwen3_next_adapter.py`](unit/model_bridge/supported_architectures/test_qwen3_next_adapter.py) (line 531) | `skipif(...)` | Qwen3-Next adapter behaviour |
| [`tests/integration/test_weight_processing_integration.py`](integration/test_weight_processing_integration.py) (line 238) | `skip(...)` | Weight-processing edge case |
| [`tests/integration/test_tensor_extraction_consistency.py`](integration/test_tensor_extraction_consistency.py) (line 33) | `skip(...)` | Tensor extraction consistency |
| [`tests/integration/test_tokenization_methods.py`](integration/test_tokenization_methods.py) (line 53) | `skipif(...)` | Tokenization method coverage |
| [`tests/integration/test_hooked_encoder_properties.py`](integration/test_hooked_encoder_properties.py) (line 71) | `xfail(...)` | HookedEncoder properties |
| [`tests/acceptance/model_bridge/compatibility/test_backward_hooks.py`](acceptance/model_bridge/compatibility/test_backward_hooks.py) (line 11) | `skip(...)` | Backward-hook compatibility |
| [`tests/acceptance/test_hooked_transformer.py`](acceptance/test_hooked_transformer.py) (lines 551, 560) | `skipif(...)` × 2 (inside the module-level skip) | Additional `from_pretrained_no_processing` cases |

**Un-skip:** debug the underlying issue and remove the marker. Each removed marker should land in a focused PR with a regression test.

---

## Adding a new quarantine

If you're tempted to add a `skip` / `skipif` / `xfail`, **read [AGENTS.md §10](../AGENTS.md#10-hard-rules) first**. The default answer is "fix the bug instead."

If a quarantine really is the right call:

1. Pick the right marker — `skipif(condition)` for environment gates, `skip(reason=)` for known-bad code paths, `xfail(strict=True, reason=)` when you expect failure and want CI to alert you if it starts passing.
2. Use a `reason=` string descriptive enough to look up later — not `"flaky"`, not `"broken"`.
3. **Add a row to the right section above**, with the path, marker, and an "un-skip when…" line.
4. If you're skipping a whole module via `pytestmark = pytest.mark.skip(...)`, flag it in the "actively quarantined whole-file" section so it gets visibility.
