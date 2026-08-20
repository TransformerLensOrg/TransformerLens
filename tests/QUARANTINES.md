# Test Quarantines

Inventory of every `skip` / `skipif` / `xfail` in [`tests/`](.). A test on this list with the matching reason is **not your bug** — don't debug blindly. A failure NOT on this list is real.

Rule ([AGENTS.md §10](../AGENTS.md#10-hard-rules)): **never add `xfail` / `skipif` to dodge a failing CI.** New skips need a row here.

---

## Permanent — optional dependency

| Path | Marker | Trigger |
|---|---|---|
| [`unit/test_lit.py`](unit/test_lit.py) (×20) | `skipif(not LIT_AVAILABLE)` | `pip install lit-nlp` (`lit` group) |

**Un-skip:** never. Install the optional group to run locally.

---

## Optional dependency — gated but installed in CI

`inspect_ai` lives in the `[inspect]` extra. The CI test jobs (`compatibility-checks`, `coverage-test`, `mps-checks`) install it via `uv sync --extra inspect` in [`checks.yml`](../.github/workflows/checks.yml), so these run on every CI push — before that change they silently skipped on every job.

| Path | Marker | Trigger |
|---|---|---|
| [`unit/model_bridge/test_inspect_driver.py`:19](unit/model_bridge/test_inspect_driver.py) (whole file) | `importorskip("inspect_ai")` | `uv sync --extra inspect` |
| [`unit/model_bridge/test_inspect_vllm_provider.py`:19](unit/model_bridge/test_inspect_vllm_provider.py) (whole file) | `importorskip("inspect_ai")` | same |
| [`unit/model_bridge/sources/test_inspect_provider_model_class.py`:20](unit/model_bridge/sources/test_inspect_provider_model_class.py) (whole file) | `importorskip("inspect_ai")` | same |
| [`acceptance/model_bridge/test_inspect_provider.py`:19](acceptance/model_bridge/test_inspect_provider.py) (whole file) | `pytestmark skipif(inspect_ai missing)` | same |
| [`../transformer_lens/model_bridge/sources/inspect/conftest.py`](../transformer_lens/model_bridge/sources/inspect/conftest.py) | `collect_ignore_glob` (doctest-modules of the provider files) | same |

**Un-skip:** already un-skipped in CI. A local plain `uv sync` still skips them; install the extra.

---

## ⚠️ Coverage gap — no automated lane (real vLLM)

| Path | Marker | Trigger |
|---|---|---|
| [`unit/model_bridge/test_vllm_driver.py`](unit/model_bridge/test_vllm_driver.py) (×18) | `importorskip("vllm")` per-test | `uv sync --extra vllm` on a Linux CUDA machine (validated band: `vllm 0.20.x`) |

A `[vllm]` extra exists (Linux-only marker; declared conflicting with `[lit]` in `[tool.uv]` — vllm needs `numpy>=2` via `opencv-python-headless` while `lit-nlp` caps `numpy<2`), but CI does not install it: vllm is GPU-only and its 15 real-engine tests would not pass on CPU runners (the file's other tests mock the LLM and run everywhere). Note the extra's `vllm 0.20.x` band exact-pins `torch==2.11.0`, which is what the project lockfile resolves to. The real-engine execution path is otherwise covered only by the manual GPU run of [`demos/vLLM_Bridge_Integration_Test.ipynb`](../demos/vLLM_Bridge_Integration_Test.ipynb).

**Un-skip:** a GPU CI lane that installs vllm, or locally on a CUDA machine with `vllm==0.20.2` installed alongside the project env.

## Multi-GPU tier — `-m multigpu` (no automated lane)

[`acceptance/model_bridge/test_vllm_multigpu.py`](acceptance/model_bridge/test_vllm_multigpu.py) validates the vLLM driver's `tensor_parallel_size=2` path (capture replication, sharded-unembed gather, intervention parity vs TP=1). Gated on `importorskip("vllm")` + `torch.cuda.device_count() >= 2`, so it skips everywhere except a provisioned multi-GPU box. **Validated 2026-07-15 on 2×A6000** (vllm 0.20.2, Qwen2.5-0.5B): both multigpu files pass, plus `TL_PARITY_TP=2` parity and the TP=1 parity regression.

[`acceptance/model_bridge/test_vllm_multigpu_pp.py`](acceptance/model_bridge/test_vllm_multigpu_pp.py) validates the `pipeline_parallel_size=2` path (cross-stage capture merge, layout tripwire, first/last-stage intervention parity, logit reconstruction from last-stage gathers). Same gating. **Validated 2026-07-15 on 2×A6000** (vllm 0.20.2, Qwen2.5-0.5B): 6/6 pass, plus `TL_PARITY_PP=2` parity and the single-rank parity regression.

**Un-skip:** run on a >= 2-GPU Linux box with the `vllm` extra: `uv run pytest tests/acceptance/model_bridge/test_vllm_multigpu.py -m multigpu -v` (and the `_pos` / `_pp` siblings, each in its own process), plus `TL_PARITY_TP=2` / `TL_PARITY_PP=2` runs of `scripts/vllm_parity_report.py`.

---

## Permanent — hardware requirement

| Path | Marker | Required |
|---|---|---|
| [`unit/test_next_sentence_prediction.py`:131](unit/test_next_sentence_prediction.py) | `skipif(not cuda)` | Any CUDA |
| [`unit/model_bridge/compatibility/test_next_sentence_prediction.py`:88](unit/model_bridge/compatibility/test_next_sentence_prediction.py) | `skipif(not cuda)` | Any CUDA |
| [`unit/test_generate_no_tokenizer.py`:112](unit/test_generate_no_tokenizer.py) | `skipif(not cuda)` | Any CUDA |
| [`unit/model_bridge/test_driver_protocol.py`:103](unit/model_bridge/test_driver_protocol.py) | `skipif(not cuda)` | Any CUDA |
| [`unit/test_weight_processing.py`:475](unit/test_weight_processing.py) | `skipif(not cuda and not mps)` (cross-device fold) | Any non-CPU accelerator |
| [`acceptance/test_hooked_encoder.py`:171](acceptance/test_hooked_encoder.py) | `skipif(mps or not cuda)` (bf16/fp16) | CUDA, non-MPS |
| [`acceptance/test_hooked_encoder.py`:226](acceptance/test_hooked_encoder.py) | `skipif(not cuda)` | Any CUDA |
| [`acceptance/test_hooked_encoder_decoder.py`:460](acceptance/test_hooked_encoder_decoder.py) | `skipif(not cuda)` | Any CUDA |
| [`acceptance/model_bridge/test_bridge_multigpu.py`](acceptance/model_bridge/test_bridge_multigpu.py) module-level | `multigpu` marker + `skipif(device_count < 2)` | 2+ CUDA |
| [`acceptance/model_bridge/test_bridge_multigpu_device_map.py`](acceptance/model_bridge/test_bridge_multigpu_device_map.py) module-level | `multigpu` marker + `skipif(device_count < 2)` | 2+ CUDA |
| [`mps/test_mps_basic.py`](mps/test_mps_basic.py) module-level | `skipif(not mps)` | Apple Silicon |
| [`mps/test_mps_ssm_eager_scan.py`](mps/test_mps_ssm_eager_scan.py) module-level | `skipif(not mps)` | Apple Silicon |

**Un-skip:** never in CI. The two `test_bridge_multigpu*` suites are the boot_transformers multi-device verification tier — run them manually on a >= 2-GPU box (`-m multigpu`, one file per pytest process) together with `scripts/bridge_multi_device_parity.py` before releases that touch device placement. **Validated 2026-07-16 on a 2-GPU box**: both suites pass, and the parity sweep (5 architectures, `n_devices=2` and `device_map=balanced`) reported bitwise-identical activations vs single-device (worst diff 0.0 across every cached hook); HookedTransformer's legacy `test_multi_gpu.py` 2-GPU subset also passed (that file has since been removed by #1603). The box debugging added two fail-loud boot guards: tied-weight pairs split across map entries, and mixed CPU+GPU maps (accelerate CPU offload, whose materialization hooks param-reading Bridge components bypass). CI provides the other tiers (CUDA via compatibility-checks → CPU-only in practice; MPS via `mps-checks`). See [tests/AGENTS.md §MPS rules](AGENTS.md#mps-rules) and the `--ignore=` list in [`checks.yml`](../.github/workflows/checks.yml).

---

## Intentional — CI cost / network budget

`skipif(os.getenv("CI"))` to avoid expensive HF fetches / large loads.

| Path | Reason |
|---|---|
| [`unit/model_bridge/supported_architectures/test_gemma2_adapter.py`:50](unit/model_bridge/supported_architectures/test_gemma2_adapter.py) | "Network/disk fetch of tiny Gemma2" |
| [`integration/model_bridge/test_bridge_integration.py`:750](integration/model_bridge/test_bridge_integration.py) | "Skip Gemma2 in CI to avoid timeout" |
| [`acceptance/model_bridge/compatibility/test_hook_completeness.py`:156](acceptance/model_bridge/compatibility/test_hook_completeness.py) | "Gemma2 too large for CI" (whole file also `slow`-marked at :18 — hook-parity matrix loads real models) |

Big-model adapter tests use `@pytest.mark.slow`, CI tier filters `-m "not slow"`:

| Path | Model |
|---|---|
| [`integration/model_bridge/test_exaone4_adapter.py`:17](integration/model_bridge/test_exaone4_adapter.py) | EXAONE-4.0 1.2B |
| [`integration/model_bridge/test_bd3lm_adapter.py`:20](integration/model_bridge/test_bd3lm_adapter.py) | BD3LM-OWT (remote code; slow to avoid HF fetch on standard runs) |
| [`integration/model_bridge/test_bitnet_adapter.py`:15](integration/model_bridge/test_bitnet_adapter.py) | BitNet 2.4B |
| [`integration/model_bridge/test_cohere2_adapter.py`:18](integration/model_bridge/test_cohere2_adapter.py) | tiny-Cohere2ForCausalLM (CPU-safe, but HF fetch) |
| [`integration/model_bridge/test_falcon_h1_adapter.py`:37](integration/model_bridge/test_falcon_h1_adapter.py) | Falcon-H1 Tiny-90M + 0.5B parity suite |
| [`integration/model_bridge/test_glm_adapter.py`:11](integration/model_bridge/test_glm_adapter.py) | glm-edge 1.59B |
| [`integration/model_bridge/test_glm_asr_adapter.py`:12](integration/model_bridge/test_glm_asr_adapter.py) | GLM-ASR-Nano 2.26B (fp32 double-load) |
| [`integration/model_bridge/test_lfm2_adapter.py`:21](integration/model_bridge/test_lfm2_adapter.py) | LFM2.5-230M |
| [`integration/model_bridge/test_nemotron_h_adapter.py`:29](integration/model_bridge/test_nemotron_h_adapter.py) | Nemotron-Nano-9B-v2 (~36 GB RAM fp32) |
| [`integration/model_bridge/test_pegasus_adapter.py`:16](integration/model_bridge/test_pegasus_adapter.py) | Pegasus-XSum 568M; distilled variants are asymmetric |
| [`integration/model_bridge/test_ssm_real_weights.py`:27](integration/model_bridge/test_ssm_real_weights.py) | real SSM/recurrent-family checkpoints; fixture skips (never fails) when a checkpoint is unavailable |
| [`integration/model_bridge/test_zamba2_adapter.py`:33](integration/model_bridge/test_zamba2_adapter.py) | Zamba2-1.2B |
| [`integration/model_bridge/test_exaone_adapter.py`:18](integration/model_bridge/test_exaone_adapter.py) | EXAONE 2.4B; no working tiny mirror (hyper-accel/tiny-random-exaone ships stale remote code) |
| [`integration/model_bridge/test_ouro_adapter.py`:35](integration/model_bridge/test_ouro_adapter.py) (skipif + `slow`) | ByteDance/Ouro-1.4B: 2.8GB download + ~11GB RAM |
| [`integration/model_bridge/test_raven_adapter.py`:42](integration/model_bridge/test_raven_adapter.py) (skipif + `slow`) | huginn-0125: ~14GB download + ~28GB RAM |
| [`integration/model_bridge/test_rwkv7_adapter.py`:36](integration/model_bridge/test_rwkv7_adapter.py) (skipif + `slow`; `importorskip("fla")` at :30) | rwkv7-0.1B-g1: remote code + flash-linear-attention dep |
| [`integration/test_jacobian_lens_kurtosis.py`](integration/test_jacobian_lens_kurtosis.py) | Qwen3.5-0.8B (~1.6GB) + lens artifacts + wikitext validation split; CPU-feasible (#1539 Tier-1) |

**Un-skip:** locally with `HF_TOKEN` sourced (slow-marked files: run the file directly or `-m slow`).

---

## Intentional — manual verification only

| Path | Reason |
|---|---|
| [`integration/model_bridge/test_qwen3_moe_bridge.py`:137,148](integration/model_bridge/test_qwen3_moe_bridge.py) | "Requires real weights — run manually during verification" |

**Un-skip:** `/verify-model Qwen/Qwen3-MoE-...` ([tools/model_registry/AGENTS.md](../transformer_lens/tools/model_registry/AGENTS.md)).

---

## Upstream / platform bug

| Path | Reason | Issue |
|---|---|---|

**Un-skip:** when upstream resolves. Don't bypass — produces NaN logits.

---

## ⚠️ Technical debt — whole-file

No modules are currently quarantined this way.

**Resolved 2026-08-17.** `acceptance/test_hooked_encoder.py` and `test_hooked_encoder_decoder.py`
had carried module-level `pytest.mark.skip(reason="CI test pollution")` since #1129, removed on
this branch by #1606; `test_hooked_transformer.py` carried the same skip and was deleted outright
by #1603, which re-anchored its coverage. Re-running them found no pollution: each passes alone
and the acceptance tier is green. What the skips were hiding was four genuine failures:

| Was failing | Actual cause |
|---|---|
| `test_bert_block` | transformers 5.x returns a tensor from `BertLayer.forward`, so the test's `[0]` took batch element 0 instead of tuple element 0 |
| `test_bloom_similarity_*` (×2) | the HF fixture loaded bloom at its checkpoint dtype (fp16) while TL loads fp32 — the comparison measured HF's own fp16 error (0.259 log-softmax against *itself*), not TL. Went away with `test_hooked_transformer.py` |
| `test_model[redwood_attn_2l]`, `test_from_pretrained_no_processing[redwood_attn_2l]` | `ArthurConmy/redwood_tokenizer` has merges referencing a token absent from its vocab (`Ġpati`), which tokenizers >= 0.20 rejects on both the fast and slow paths. Went away with `test_hooked_transformer.py` |

Two silent TransformerLens bugs also lived in this blind spot the whole time: T5's decoder
self-attention was never causally masked, and its relative-position bias used the encoder's
bucketing. Both are fixed, and bound by
[`acceptance/test_hooked_encoder_decoder.py`](acceptance/test_hooked_encoder_decoder.py)'s
`test_full_model_multi_token_decoder` plus
[`unit/model_bridge/test_t5_block_parity.py`](unit/model_bridge/test_t5_block_parity.py). Keep
the encoder modules enabled.

---

## Technical debt — individual

| Path | Marker | Covers |
|---|---|---|
| [`unit/factored_matrix/test_constructor.py`:54](unit/factored_matrix/test_constructor.py) | `skip` | FactoredMatrix constructor edge case |
| [`unit/model_bridge/test_architecture_adapter.py`:453](unit/model_bridge/test_architecture_adapter.py) | `skip` | SoLU-style weight-processing paths (adapter under test is Gemma3, which has no `mlp.ln`) |
| [`unit/model_bridge/test_bridge_vs_hooked_transformer_patching.py`:138,142](unit/model_bridge/test_bridge_vs_hooked_transformer_patching.py) | `skipif`/`xfail` | Bridge↔HT patching parity |
| [`unit/model_bridge/test_hook_alias_resolution.py`:90](unit/model_bridge/test_hook_alias_resolution.py) | `xfail(strict=True)` per-arch | Hook-alias gaps |
| [`unit/model_bridge/supported_architectures/test_qwen3_5_adapter.py`:448,464,494,514,605,700,805,947,1133](unit/model_bridge/supported_architectures/test_qwen3_5_adapter.py) | `skipif` ×9 | Qwen3_5 classes absent from installed transformers |
| [`unit/model_bridge/supported_architectures/test_qwen3_next_adapter.py`:397](unit/model_bridge/supported_architectures/test_qwen3_next_adapter.py) | `skipif` | Qwen3NextForCausalLM absent from installed transformers |
| [`integration/test_weight_processing_integration.py`:279](integration/test_weight_processing_integration.py) | `skip` | Weight-processing edge case |
| [`integration/test_hooked_encoder_properties.py`:71](integration/test_hooked_encoder_properties.py) | `xfail` | HookedEncoder properties |
| [`acceptance/model_bridge/compatibility/test_backward_hooks.py`:11](acceptance/model_bridge/compatibility/test_backward_hooks.py) | `skip` | Backward-hook compatibility |

**Un-skip:** debug the underlying issue and remove the marker. Each removal lands in a focused PR with a regression test.

---

## Adding a new quarantine

Read [AGENTS.md §10](../AGENTS.md#10-hard-rules) first — default answer is "fix the bug instead."

If a quarantine is genuinely right:

1. Pick the right marker — `skipif(condition)` for env gates; `skip(reason=)` for known-bad paths; `xfail(strict=True, reason=)` when you expect failure and want CI to alert if it passes.
2. Use a `reason=` descriptive enough to look up — not `"flaky"` or `"broken"`.
3. Add a row above with path, marker, "un-skip when" line.
4. Whole-module `pytestmark` skips go in the ⚠️ section for visibility.
