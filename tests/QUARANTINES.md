# Test Quarantines

Inventory of every `skip` / `skipif` / `xfail` in [`tests/`](.). A test on this list with the matching reason is **not your bug** — don't debug blindly. A failure NOT on this list is real.

Rule ([AGENTS.md §10](../AGENTS.md#10-hard-rules)): **never add `xfail` / `skipif` to dodge a failing CI.** New skips need a row here.

---

## Permanent — optional dependency

| Path | Marker | Trigger |
|---|---|---|
| [`unit/test_lit.py`](unit/test_lit.py) (×18) | `skipif(not LIT_AVAILABLE)` | `pip install lit-nlp` (`lit` group) |
| [`unit/components/test_attention.py`:130](unit/components/test_attention.py) | `skipif(not is_bitsandbytes_available())` | `uv sync --group quantization` |
| [`unit/test_weight_processing.py`:477](unit/test_weight_processing.py) | same | same |
| [`unit/factories/test_mlp_factory.py`:40](unit/factories/test_mlp_factory.py) | same | same |

**Un-skip:** never. Install the optional group to run locally.

---

## Permanent — hardware requirement

| Path | Marker | Required |
|---|---|---|
| [`unit/test_next_sentence_prediction.py`:131](unit/test_next_sentence_prediction.py) | `skipif(not cuda)` | Any CUDA |
| [`unit/model_bridge/compatibility/test_next_sentence_prediction.py`:95](unit/model_bridge/compatibility/test_next_sentence_prediction.py) | `skipif(not cuda)` | Any CUDA |
| [`unit/components/test_attention.py`:165](unit/components/test_attention.py) | `skipif(not cuda)` (half/bfloat16) | Any CUDA |
| [`acceptance/test_hooked_encoder.py`:227](acceptance/test_hooked_encoder.py) | `skipif(not cuda)` | Any CUDA |
| [`acceptance/test_hooked_encoder_decoder.py`:421](acceptance/test_hooked_encoder_decoder.py) | `skipif(not cuda)` | Any CUDA |
| [`acceptance/test_multi_gpu.py`:91,105](acceptance/test_multi_gpu.py) | `skipif(device_count < 2)` | 2+ CUDA |
| [`acceptance/test_multi_gpu.py`:22](acceptance/test_multi_gpu.py) | `skipif(device_count < 4)` | 4+ CUDA |
| [`acceptance/model_bridge/test_bridge_multigpu.py`](acceptance/model_bridge/test_bridge_multigpu.py) module-level | `multigpu` marker + `skipif(device_count < 2)` | 2+ CUDA |
| [`acceptance/model_bridge/test_bridge_multigpu_device_map.py`](acceptance/model_bridge/test_bridge_multigpu_device_map.py) module-level | `multigpu` marker + `skipif(device_count < 2)` | 2+ CUDA |
| [`mps/test_mps_basic.py`](mps/test_mps_basic.py) module-level | `skipif(not mps)` | Apple Silicon |

**Un-skip:** never in CI. The two `test_bridge_multigpu*` suites are the boot_transformers multi-device verification tier — run them manually on a >= 2-GPU box (`-m multigpu`, one file per pytest process) together with `scripts/bridge_multi_device_parity.py` before releases that touch device placement. **Validated 2026-07-16 on a 2-GPU box**: both suites pass, and the parity sweep (5 architectures, `n_devices=2` and `device_map=balanced`) reported bitwise-identical activations vs single-device (worst diff 0.0 across every cached hook); HookedTransformer's legacy `test_multi_gpu.py` 2-GPU subset also passed. The box debugging added two fail-loud boot guards: tied-weight pairs split across map entries, and mixed CPU+GPU maps (accelerate CPU offload, whose materialization hooks param-reading Bridge components bypass). CI provides the other tiers (CUDA via compatibility-checks → CPU-only in practice; MPS via `mps-checks`). See [tests/AGENTS.md §MPS rules](AGENTS.md#mps-rules) and the `--ignore=` list in [`checks.yml`](../.github/workflows/checks.yml).

---

## Intentional — CI cost / network budget

`skipif(os.getenv("CI"))` to avoid expensive HF fetches / large loads.

| Path | Reason |
|---|---|
| [`unit/model_bridge/supported_architectures/test_gemma2_adapter.py`:49](unit/model_bridge/supported_architectures/test_gemma2_adapter.py) | "Network/disk fetch of tiny Gemma2" |
| [`integration/model_bridge/test_bridge_integration.py`:801](integration/model_bridge/test_bridge_integration.py) | "Skip Gemma2 in CI to avoid timeout" |
| [`acceptance/model_bridge/compatibility/test_hook_completeness.py`:156](acceptance/model_bridge/compatibility/test_hook_completeness.py) | "Gemma2 too large for CI" |

Big-model adapter tests use `@pytest.mark.slow`, CI tier filters `-m "not slow"`:

| Path | Model |
|---|---|
| [`integration/model_bridge/test_exaone4_adapter.py`:17](integration/model_bridge/test_exaone4_adapter.py) | EXAONE-4.0 1.2B |
| [`integration/model_bridge/test_bitnet_adapter.py`:15](integration/model_bridge/test_bitnet_adapter.py) | BitNet 2.4B |
| [`integration/model_bridge/test_glm_adapter.py`:11](integration/model_bridge/test_glm_adapter.py) | glm-edge 1.59B |
| [`integration/model_bridge/test_glm_asr_adapter.py`:12](integration/model_bridge/test_glm_asr_adapter.py) | GLM-ASR-Nano 2.26B (fp32 double-load) |
| [`integration/model_bridge/test_pegasus_adapter.py`:16](integration/model_bridge/test_pegasus_adapter.py) | Pegasus-XSum 568M; distilled variants are asymmetric |
| [`integration/model_bridge/test_exaone_adapter.py`:18](integration/model_bridge/test_exaone_adapter.py) | EXAONE 2.4B; no working tiny mirror (hyper-accel/tiny-random-exaone ships stale remote code) |
| [`integration/model_bridge/test_ouro_adapter.py`:35](integration/model_bridge/test_ouro_adapter.py) (skipif + `slow`) | ByteDance/Ouro-1.4B: 2.8GB download + ~11GB RAM |
| [`integration/test_jacobian_lens_kurtosis.py`](integration/test_jacobian_lens_kurtosis.py) | Qwen3.5-0.8B (~1.6GB) + lens artifacts + wikitext validation split; CPU-feasible (#1539 Tier-1) |

**Un-skip:** locally with `HF_TOKEN` sourced (slow-marked files: run the file directly or `-m slow`).

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
| [`acceptance/test_hooked_transformer.py`](acceptance/test_hooked_transformer.py) | `redwood_attn_2l` (2 tests) — `ArthurConmy/redwood_tokenizer`'s merges name a token missing from its vocab (`Ġpati`), rejected by tokenizers >= 0.20 on both the fast and slow paths | Third-party repo; the weights load fine, only the tokenizer is unusable |

**Un-skip:** evaluated at collection by actually attempting the load, so it disappears on its own
if the repo is fixed or the tokenizers constraint relaxes; substituting a different tokenizer is
not a fix (it would change token ids and invalidate the pinned expected loss).

---

## ⚠️ Technical debt — whole-file

No modules are currently quarantined this way.

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
