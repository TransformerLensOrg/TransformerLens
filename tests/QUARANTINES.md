# Test Quarantines

Inventory of every `skip` / `skipif` / `xfail` in [`tests/`](.). A test on this list with the matching reason is **not your bug** — don't debug blindly. A failure NOT on this list is real.

Rule ([AGENTS.md §10](../AGENTS.md#10-hard-rules)): **never add `xfail` / `skipif` to dodge a failing CI.** New skips need a row here.

---

## Permanent — optional dependency

| Path | Marker | Trigger |
|---|---|---|
| [`unit/test_lit.py`](unit/test_lit.py) (×18) | `skipif(not LIT_AVAILABLE)` | `pip install lit-nlp` (`lit` group) |
| [`unit/components/test_attention.py`:48](unit/components/test_attention.py) | `skipif(not is_bitsandbytes_available())` | `uv sync --group quantization` |
| [`unit/test_weight_processing.py`:477](unit/test_weight_processing.py) | same | same |
| [`unit/factories/test_mlp_factory.py`:40](unit/factories/test_mlp_factory.py) | same | same |

**Un-skip:** never. Install the optional group to run locally.

---

## Permanent — hardware requirement

| Path | Marker | Required |
|---|---|---|
| [`unit/test_next_sentence_prediction.py`:131](unit/test_next_sentence_prediction.py) | `skipif(not cuda)` | Any CUDA |
| [`unit/model_bridge/compatibility/test_next_sentence_prediction.py`:95](unit/model_bridge/compatibility/test_next_sentence_prediction.py) | `skipif(not cuda)` | Any CUDA |
| [`unit/components/test_attention.py`:83](unit/components/test_attention.py) | `skipif(not cuda)` (half/bfloat16) | Any CUDA |
| [`acceptance/test_hooked_encoder.py`:227](acceptance/test_hooked_encoder.py) | `skipif(not cuda)` | Any CUDA |
| [`acceptance/test_hooked_encoder_decoder.py`:421](acceptance/test_hooked_encoder_decoder.py) | `skipif(not cuda)` | Any CUDA |
| [`acceptance/test_multi_gpu.py`:91,105](acceptance/test_multi_gpu.py) | `skipif(device_count < 2)` | 2+ CUDA |
| [`acceptance/test_multi_gpu.py`:22](acceptance/test_multi_gpu.py) | `skipif(device_count < 4)` | 4+ CUDA |
| [`acceptance/model_bridge/test_multi_gpu_bridge.py`:257](acceptance/model_bridge/test_multi_gpu_bridge.py) | `skipif(device_count < 2)` | 2+ CUDA |
| [`mps/test_mps_basic.py`](mps/test_mps_basic.py) module-level | `skipif(not mps)` | Apple Silicon |

**Un-skip:** never. CI provides each tier (CUDA via compatibility-checks → CPU-only in practice; MPS via `mps-checks`; multi-GPU local-only). See [tests/AGENTS.md §MPS rules](AGENTS.md#mps-rules) and the `--ignore=` list in [`checks.yml`](../.github/workflows/checks.yml).

---

## Intentional — CI cost / network budget

`skipif(os.getenv("CI"))` to avoid expensive HF fetches / large loads.

| Path | Reason |
|---|---|
| [`unit/model_bridge/supported_architectures/test_gemma2_adapter.py`:49](unit/model_bridge/supported_architectures/test_gemma2_adapter.py) | "Network/disk fetch of tiny Gemma2" |
| [`integration/model_bridge/test_bridge_integration.py`:801](integration/model_bridge/test_bridge_integration.py) | "Skip Gemma2 in CI to avoid timeout" |
| [`acceptance/model_bridge/compatibility/test_hook_completeness.py`:156](acceptance/model_bridge/compatibility/test_hook_completeness.py) | "Gemma2 too large for CI" |

**Un-skip:** locally with `HF_TOKEN` sourced.

---

## Intentional — manual verification only

| Path | Reason |
|---|---|
| [`integration/model_bridge/test_qwen3_moe_bridge.py`:155,166](integration/model_bridge/test_qwen3_moe_bridge.py) | "Requires real weights — run via `verify_models`" |

**Un-skip:** `/verify-model Qwen/Qwen3-MoE-...` ([tools/model_registry/AGENTS.md](../transformer_lens/tools/model_registry/AGENTS.md)).

---

## Upstream / platform bug

| Path | Reason | Issue |
|---|---|---|
| [`unit/model_bridge/test_bridge_generate_no_tokenizer.py`:30,128](unit/model_bridge/test_bridge_generate_no_tokenizer.py) | `skipif(_MACOS_ARM64)` — KV-cache NaN | Upstream PyTorch/HF on M-series Macs |

**Un-skip:** when upstream resolves. Don't bypass — produces NaN logits.

---

## ⚠️ Technical debt — whole-file

Entire test modules quarantined via module-level `pytestmark`. Significant coverage gap — priority to re-enable.

| Path | Reason |
|---|---|
| [`acceptance/test_hooked_transformer.py`:19](acceptance/test_hooked_transformer.py) | "CI test pollution" |
| [`acceptance/test_hooked_encoder.py`:13](acceptance/test_hooked_encoder.py) | same |
| [`acceptance/test_hooked_encoder_decoder.py`:10](acceptance/test_hooked_encoder_decoder.py) | same |

**Un-skip:** root-cause the test pollution (fixture-scope or import-ordering bug). Until then, these acceptance tiers are dark.

---

## Technical debt — individual

| Path | Marker | Covers |
|---|---|---|
| [`unit/factored_matrix/test_constructor.py`:54](unit/factored_matrix/test_constructor.py) | `skip` | FactoredMatrix constructor edge case |
| [`unit/model_bridge/test_architecture_adapter.py`:436](unit/model_bridge/test_architecture_adapter.py) | `skip` | Adapter behaviour |
| [`unit/model_bridge/test_bridge_vs_hooked_transformer_patching.py`:138,142](unit/model_bridge/test_bridge_vs_hooked_transformer_patching.py) | `skipif`/`xfail` | Bridge↔HT patching parity |
| [`unit/model_bridge/test_hook_alias_resolution.py`:89](unit/model_bridge/test_hook_alias_resolution.py) | `xfail(strict=True)` per-arch | Hook-alias gaps |
| [`unit/model_bridge/supported_architectures/test_qwen3_5_adapter.py`:574,609,660,680,771](unit/model_bridge/supported_architectures/test_qwen3_5_adapter.py) | `skipif` ×5 | Qwen3.5 quirks |
| [`unit/model_bridge/supported_architectures/test_qwen3_next_adapter.py`:531](unit/model_bridge/supported_architectures/test_qwen3_next_adapter.py) | `skipif` | Qwen3-Next quirks |
| [`integration/test_weight_processing_integration.py`:238](integration/test_weight_processing_integration.py) | `skip` | Weight-processing edge case |
| [`integration/test_tensor_extraction_consistency.py`:33](integration/test_tensor_extraction_consistency.py) | `skip` | Tensor extraction |
| [`integration/test_tokenization_methods.py`:53](integration/test_tokenization_methods.py) | `skipif` | Tokenization coverage |
| [`integration/test_hooked_encoder_properties.py`:71](integration/test_hooked_encoder_properties.py) | `xfail` | HookedEncoder properties |
| [`acceptance/model_bridge/compatibility/test_backward_hooks.py`:11](acceptance/model_bridge/compatibility/test_backward_hooks.py) | `skip` | Backward-hook compatibility |
| [`acceptance/test_hooked_transformer.py`:551,560](acceptance/test_hooked_transformer.py) | `skipif` ×2 (inside module-level skip) | `from_pretrained_no_processing` |

**Un-skip:** debug the underlying issue and remove the marker. Each removal lands in a focused PR with a regression test.

---

## Adding a new quarantine

Read [AGENTS.md §10](../AGENTS.md#10-hard-rules) first — default answer is "fix the bug instead."

If a quarantine is genuinely right:

1. Pick the right marker — `skipif(condition)` for env gates; `skip(reason=)` for known-bad paths; `xfail(strict=True, reason=)` when you expect failure and want CI to alert if it passes.
2. Use a `reason=` descriptive enough to look up — not `"flaky"` or `"broken"`.
3. Add a row above with path, marker, "un-skip when" line.
4. Whole-module `pytestmark` skips go in the ⚠️ section for visibility.
