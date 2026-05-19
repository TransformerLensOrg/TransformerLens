# Open Issues Triage (v3)

**Generated:** 2026-05-08 (v3 — re-verified against current code)
**Repo:** TransformerLensOrg/TransformerLens
**Open issues:** 48 (44 re-verified from v2 + 4 opened since)
**v2 archived at:** [OPEN_ISSUES_TRIAGE_OLD.md](OPEN_ISSUES_TRIAGE_OLD.md)

## What changed since v2

- **37 issues closed** during the v2 cycle (archived in [OPEN_ISSUES_TRIAGE_OLD.md](OPEN_ISSUES_TRIAGE_OLD.md))
- **44 entries re-verified** against current code; ~9 had material updates from PRs landed in the v2 cycle
- **4 new entries** added with stub triage (#1263, #1275, #1280, #1291)

### Newly closeable based on v3 re-verification

- **#729** — adapter-creation guide landed (PR #1274)
- **#846** — bridge `hf_model.config` priority fixed (PR #1279)
- **#912** — mT5 wired through bridge with full verification (PR #1289)
- **#950** — SimpleStories family verified end-to-end (PR #1292)
- **#1133** — already covered-close in v2; v3 refreshes citation

### Material code-state changes since v2 (still open, but verdict updated)

- **#290** — empty-name circular reference confirmed fixed at `hook_points.py:420-421`
- **#483** — HT side fixed by PR #1267; bridge mirror still needed
- **#569 / #684** — bridge quantization had a real bug (uint8 cast) fixed by PR #1276; multi-device by PR #1270
- **#615** — PR #1276 dtype-cast fix benefits non-quantized too via shared `GeneralizedComponent`
- **#661** — bridge now exposes `set_use_split_qkv_input`
- **#837 / #911 / #968** — multi-device bridge (PR #1270) now merged on main
- **#1148** — reporter committed to building tutorial

The v2 methodology section (HT-side / Bridge-side / Replication / Next step) still applies — see [OPEN_ISSUES_TRIAGE_OLD.md](OPEN_ISSUES_TRIAGE_OLD.md#methodology-per-issue).

## Summary table (sorted by issue number)

| Issue | Title | Bucket |
|---|---|---|
| #111 | [Demo of direct path patching](#issue-111) | `not-addressed-difficult` |
| #112 | [Helper to display vectors of logits nicely](#issue-112) | `not-addressed-simple` |
| #210 | [`get_full_resid_decomposition` accept tensor argument](#issue-210) | `not-addressed-simple` |
| #290 | [GPU memory leak when HookedTransformer goes out of scope](#issue-290) | `partial-leave-open` |
| #297 | [Better print-outs for currently attached hooks](#issue-297) | `not-addressed-simple` |
| #341 | [Update FactoredMatrix.svd() (uses deprecated `torch.svd`, returns V not Vh)](#issue-341) | `not-addressed-simple` |
| #385 | [Pythia / Rotary Embeddings don't match HuggingFace](#issue-385) | `bug-still-reproduces` |
| #453 | [`from_pretrained()` always downloads same weights with `checkpoint_label`](#issue-453) | `bug-likely-fixed-needs-verification` |
| #462 | [Add support for Mamba](#issue-462) | `fixed-on-transformerbridge` |
| #479 | [Memory efficient causal mask implementation](#issue-479) | `partial-leave-open` |
| #481 | [Tracr to TransformerLens demo broken](#issue-481) | `bug-still-reproduces` |
| #483 | [`HookedTransformer.generate()` `pad_token_id` error when tokenizer unset](#issue-483) | `partial-leave-open` |
| #509 | [LayerNorm folding not implemented for BertBlock](#issue-509) | `not-addressed-difficult` |
| #543 | [Grokking demo broken in Colab](#issue-543) | `bug-likely-fixed-needs-verification` |
| #569 | [Cannot load Llama 3 70B on multigpu in 4bit](#issue-569) | `fixed-on-transformerbridge` |
| #588 | [Setup unit tests to cover model configurations](#issue-588) | `partial-leave-open` |
| #595 | [Add Stopping Criteria support](#issue-595) | `not-addressed-simple` |
| #615 | [HookedTransformer output not identical to HuggingFace for Llama 3](#issue-615) | `fixed-on-transformerbridge` |
| #644 | [Documentation: Map the Act Names to the Transformer](#issue-644) | `not-addressed-simple` |
| #661 | [Pythia output inconsistent across batch sizes with `use_split_qkv_input=True`](#issue-661) | `bug-still-reproduces` |
| #684 | [Expand quantization model support beyond Llama](#issue-684) | `fixed-on-transformerbridge` |
| #697 | [Activation cache during generate](#issue-697) | `not-addressed-simple` |
| #704 | [Add support for TracrBench](#issue-704) | `not-relevant-close` |
| #710 | [MVP Support For 1-2 Models Per-Modality](#issue-710) | `not-addressed-difficult` |
| #720 | [Review current matmul function usages](#issue-720) | `partial-leave-open` |
| #729 | [Guide to adding new models](#issue-729) | `covered-close` |
| #737 | [Q reshape with model loaded in 4bit](#issue-737) | `partial-leave-open` |
| #773 | [TransformerLens on models with different layernorm placement (BioGPT)](#issue-773) | `not-addressed-difficult` |
| #796 | [`FactoredMatrix.svd()` `lru_cache` prevents GC](#issue-796) | `not-addressed-simple` |
| #798 | [Remove `model_args` (use only `model_kwargs`)](#issue-798) | `not-addressed-simple` |
| #830 | [Type hint support for `self.model` in `ActivationCache`](#issue-830) | `not-addressed-simple` |
| #837 | [Multi-GPU device ordinal issue (`n_devices=3` for llama2-7b)](#issue-837) | `fixed-on-transformerbridge` |
| #846 | [Prioritize local `hf_model.config` for Qwen models](#issue-846) | `fixed-on-transformerbridge` |
| #867 | [Does TransformerLens support LVLM like Qwen2-VL?](#issue-867) | `not-addressed-difficult` |
| #869 | [Custom generative video transformer](#issue-869) | `not-addressed-difficult` |
| #888 | [Adapt HookedTransformer to a non-supported model (CLIP language model)](#issue-888) | `not-addressed-difficult` |
| #911 | [PosEmbed device error with `accelerate`](#issue-911) | `fixed-on-transformerbridge` |
| #912 | [Support mT5 models](#issue-912) | `covered-close` |
| #950 | [Support SimpleStories models](#issue-950) | `covered-close` |
| #953 | [Add basic support for Gemma 3n (E2B & E4B)](#issue-953) | `not-addressed-difficult` |
| #968 | [`unsloth/llama-3.2-3b-instruct` with 2× 3060 device-mismatch](#issue-968) | `bug-likely-fixed-needs-verification` |
| #1080 | [Import fails by default in Colab (numpy ABI mismatch)](#issue-1080) | `bug-likely-fixed-needs-verification` |
| #1133 | [`tokenize_and_concatenate` cuts tokens mid-document](#issue-1133) | `covered-close` |
| #1148 | [Tutorial for "Real-Time Training Dynamics" (VSM Telemetry)](#issue-1148) | `not-addressed-simple` |
| #1263 | [Direct Logit Attribution Tool](#issue-1263) | `needs-triage` (new) |
| #1275 | [Update Benchmarks & Verify Models to support Quantized models](#issue-1275) | `needs-triage` (new) |
| #1280 | [Add support for `cpu`, `meta`, and `disk` to TransformerBridge `device_map`](#issue-1280) | `needs-triage` (new) |
| #1291 | [CI HuggingFace Call Reduction](#issue-1291) | `needs-triage` (new) |

## Per-issue entries

<a id="issue-111"></a>

#### #111 — Demo of direct path patching

- **Issue**: Add a section to Exploratory Analysis Demo demonstrating direct path patching for all head pairs. PR #49 was an early attempt.
- **HookedTransformer**: still no first-class path-patching helper. `demos/Activation_Patching_in_TL_Demo.ipynb` and `demos/Attribution_Patching_Demo.ipynb` exist but neither covers direct path patching.
- **TransformerBridge**: same — no path-patching primitive in either API.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: callum mcdougall pointed users at the [ARENA IOI notebook](https://colab.research.google.com/drive/1KgrEwvCKdX-8DQ1uSiIuxwIiwzJuQ3Gw). Either close with a docs pointer to ARENA, or implement a TL helper that wraps the pattern (~80 LoC).

<a id="issue-112"></a>

#### #112 — Helper to display vectors of logits nicely

- **Issue**: Neel asked for two things: **MVP** — function mapping logit vector → pandas DataFrame `(token_index, token_string, logit, log_prob, probability)`. **Bonus** — nostalgebraist-style `plot_logit_lens` heatmap.
- **HookedTransformer**: `test_prompt` in [transformer_lens/utilities/exploratory_utils.py:14](transformer_lens/utilities/exploratory_utils.py#L14) prints top-k for prompt+answer — partial spirit of the MVP but print-only, single-position. No `logits_to_df`, no `plot_logit_lens` heatmap. Unchanged since v2.
- **TransformerBridge**: same — `test_prompt` works through bridge; no separate helper.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~30 LoC for `logits_to_df(logits, tokenizer, top_k=None) -> pd.DataFrame`, ~50 LoC for matplotlib `plot_logit_lens`. Both small library additions independent of CircuitsVis.

<a id="issue-210"></a>

#### #210 — `get_full_resid_decomposition` accept tensor argument

- **Issue**: Add a `project_output_onto: [d_model]` or `[d_model, num_outputs]` argument so neuron-decomposition doesn't blow GPU memory by materializing `[batch, pos, d_mlp, d_model]`.
- **HookedTransformer**: signature at [transformer_lens/ActivationCache.py:1091](transformer_lens/ActivationCache.py#L1091) still has no `project_output_onto`. Memory-blowing path still active.
- **TransformerBridge**: same — bridge imports the same `ActivationCache` class ([transformer_lens/model_bridge/bridge.py:34](transformer_lens/model_bridge/bridge.py#L34)).
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: add `project_output_onto` kwarg + `(neurons * (W_out @ project_output_onto))` path. ~15 LoC + 1 test. Alan Cooney offered to take it; never landed.

<a id="issue-290"></a>

#### #290 — GPU memory leak when HookedTransformer goes out of scope

- **Issue**: `del model; gc.collect(); torch.cuda.empty_cache()` doesn't reclaim memory after loading multiple models in a loop.
- **HookedTransformer**: the empty-name circular reference is now fixed at [transformer_lens/hook_points.py:420-421](transformer_lens/hook_points.py#L420-L421) (`if name == "": continue`). However, the `state_dict[k] = v.to(device)` non-detach concern from the thread is not visibly addressed in current `HookedTransformer.py`.
- **TransformerBridge**: PR #1229 (`4cbb0f88`) fixed a *separate* Joint-QKV bridge memory leak (deepcopy bug) — unrelated to this issue. Bridge still delegates to HF; no TL-specific circular refs.
- **Replication**: `[unverifiable]` — needs GPU profiling tooling and ~10× model loads.
- **What changed since v2**: confirmed circular-reference fix is in place; v2 was uncertain.
- **Bucket**: `partial-leave-open`
- **Next step**: re-run `fil-profile` reproduction on current `dev`. If residual leak exists, focus on `move_model_modules_to_device` overlap with multi-GPU bug cluster (#837/#907/#911/#968).

<a id="issue-297"></a>

#### #297 — Better print-outs for currently attached hooks

- **Issue**: API for listing hooks attached to a model + HookPoint, with detail.
- **HookedTransformer**: no first-class `model.list_hooks()` or `HookPoint.describe()` API. `model.hook_dict` publicly accessible; `hp.fwd_hooks`/`bwd_hooks` inspectable. Confirmed via grep — no `list_active_hooks` in [transformer_lens/hook_points.py](transformer_lens/hook_points.py).
- **TransformerBridge**: same — uses same `hook_points` machinery.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: add `HookedRootModule.list_active_hooks()` returning `Dict[str, List[hook_repr]]`. ~15 LoC + 1 test. Abandoned PR #302 was the prior attempt.

<a id="issue-341"></a>

#### #341 — Update FactoredMatrix.svd() (uses deprecated `torch.svd`, returns V not Vh)

- **Issue**: TL uses deprecated `torch.svd` (which returns V, not Vh) inside `FactoredMatrix.svd`. Should switch to `torch.linalg.svd` and return Vh per modern convention.
- **HookedTransformer/Bridge**: confirmed at [transformer_lens/FactoredMatrix.py:230-233](transformer_lens/FactoredMatrix.py#L230-L233) — still `torch.svd(...)`. Last commit on file was `90cf7476` (eigenvalues type fix), not relevant. No fix landed since v2.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~15-line fix — switch to `torch.linalg.svd(..., full_matrices=False)`, return `Vh` directly, update docstring noting the breaking change. `diego898` offered to send PR. Land with a deprecation warning.

<a id="issue-385"></a>

#### #385 — Pythia / Rotary Embeddings don't match HuggingFace

- **Issue**: Logit drift between `HookedTransformer` and HF for Pythia models. Llama-2-7b-chat reportedly catastrophic. Llama-3.2 rotary mismatch persists per chengjiali.
- **HookedTransformer**: rotary code lives at [transformer_lens/components/abstract_attention.py:599](transformer_lens/components/abstract_attention.py#L599). Last touched by PR #1218 (`2c41b6c9` Weight processing/position embeddings attention) and PR #1231 (`524bca93` rotary_base types). No new pythia-specific fixes since v2.
- **TransformerBridge**: bridge uses HF's rotary directly via `RotaryEmbeddingBridge` delegating to `model.rotary_emb`. By construction matches HF.
- **Replication**: `[empirically replicated]` per v2 (NaN logits in fp32 baseline). Not re-run this round.
- **Bucket**: `bug-still-reproduces` + `fixed-on-transformerbridge` for bridge users
- **Next step**: investigate the v2-reported NaN regression — verify whether it persists with full `from_pretrained` on current HEAD; bisect against `2c41b6c9`. Bridge users avoid this entirely.

<a id="issue-453"></a>

#### #453 — `from_pretrained()` always downloads same weights with `checkpoint_label`

- **Issue**: Reporter passes `checkpoint_label=...` and gets identical weights regardless of label. `checkpoint_index` works.
- **HookedTransformer**: signature at [transformer_lens/HookedTransformer.py:1158-1159](transformer_lens/HookedTransformer.py#L1158-L1159) has `checkpoint_index` and `checkpoint_value` — **NOT `checkpoint_label`**. The kwarg is silently absorbed into `**from_pretrained_kwargs`. Unchanged since v2.
- **TransformerBridge**: no checkpoint feature — uses HF's native loading only.
- **Replication**: `[code-verified]`
- **Bucket**: `bug-likely-fixed-needs-verification` (effectively user-error)
- **Next step**: respond to reporter that the parameter is `checkpoint_value`. Optionally validate unknown kwargs in `from_pretrained` and raise. ~10 LoC defensive change.

<a id="issue-462"></a>

#### #462 — Add support for Mamba

- **Issue**: Add Mamba SSM architecture support.
- **HookedTransformer**: not supported (by design — Mamba is fundamentally different from attention transformers).
- **TransformerBridge**: `MambaArchitectureAdapter` and `Mamba2ArchitectureAdapter` registered at [transformer_lens/factories/architecture_adapter_factory.py:95-96](transformer_lens/factories/architecture_adapter_factory.py#L95-L96). Both `MambaForCausalLM` and `Mamba2ForCausalLM` HF model classes mapped. SSM beta support landed via PR #1246 (`7cf84596`).
- **Replication**: `[code-verified]`
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: close with comment pointing at `TransformerBridge.boot_transformers("state-spaces/mamba-130m-hf")`. Mamba support is shipped.

<a id="issue-479"></a>

#### #479 — Memory efficient causal mask implementation

- **Issue**: Each `Attention` layer registers a `(n_ctx, n_ctx)` boolean `causal_mask` buffer. ~86 GB overhead at Qwen 72B × 32K ctx.
- **HookedTransformer**: confirmed at [transformer_lens/components/abstract_attention.py:120-128](transformer_lens/components/abstract_attention.py#L120-L128) — `causal_mask = torch.tril(torch.ones((self.cfg.n_ctx, self.cfg.n_ctx)).bool())` and `register_buffer("mask", causal_mask)`. Bug as reported still present for ALL HT architectures.
- **TransformerBridge**: architecture-dependent (per v2). GPT2-family inherits HF's static `(max_pos, max_pos)` buffer. Modern HF impls (GPTNeoX/Pythia/Llama/Qwen/Mistral/Gemma) use `_update_causal_mask` per forward — zero overhead. The motivating Qwen 72B case is fixed on bridge.
- **Replication**: `[empirically replicated]` per v2.
- **Bucket**: `partial-leave-open`
- **Next step**: bridge users on modern architectures already have the desired memory profile. HT-side fix (~30 LoC: replace pre-allocated buffer with on-the-fly construction in `apply_causal_mask`) closes it for the legacy path and GPT2-family use cases.

<a id="issue-481"></a>

#### #481 — Tracr to TransformerLens demo broken

- **Issue**: Demo notebook assumes "the unembed is a projection onto the first few elements of the residual stream" — wrong because Tracr re-orders the residual stream alphabetically. Needs Tracr upstream PR to expose the unembed matrix.
- **HookedTransformer**: confirmed at [demos/Tracr_to_Transformer_Lens_Demo.ipynb:233](demos/Tracr_to_Transformer_Lens_Demo.ipynb) — `sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)` line still there. Demo NOT ported to TransformerBridge. No commits on the notebook since `7784be1c` (IPython magic deprecation, unrelated).
- **TransformerBridge**: same — Tracr-specific issue applies regardless of API; bug is in unembed-matrix derivation, not in TL's hook system.
- **Replication**: `[code-verified]`
- **Bucket**: `bug-still-reproduces`
- **Next step**: needs Tracr upstream PR to expose `unembed_matrix` in `tracr.params`. FlyingPumba said they'd attempt the upstream change. Without that, demo is fundamentally limited.

<a id="issue-483"></a>

#### #483 — `HookedTransformer.generate()` `pad_token_id` error when tokenizer unset

- **Issue**: `model.generate()` on a `HookedTransformer` with no tokenizer raises `AttributeError: 'NoneType' object has no attribute 'pad_token_id'`. Use case: training models on tokenizer-less domains (e.g., character-level integer addition).
- **HookedTransformer**: ✅ fixed by PR #1267 (commit `b1cc8c80`, "Fix generate() when tokenizer is unset and add regression tests"). The `assert self.tokenizer is not None` was removed from the top of both `generate()` and `generate_stream()`; logic now branches on `tokenizer_has_eos_token` and falls back to user-supplied `eos_token_id`. See [transformer_lens/HookedTransformer.py:2068-2089](transformer_lens/HookedTransformer.py#L2068-L2089). Regression test at [tests/unit/test_generate_no_tokenizer.py](tests/unit/test_generate_no_tokenizer.py).
- **TransformerBridge**: ❌ not fixed. [transformer_lens/model_bridge/bridge.py:2550-2566](transformer_lens/model_bridge/bridge.py#L2550-L2566) and the parallel block at L2826-L2839 still dereference `self.tokenizer.eos_token_id` unguarded; same gap as v2. The mirror-to-bridge expectation was not met when #1267 landed.
- **Replication**: `[code-verified]`
- **What changed since v2**: PR #1267 landed on HT only; bridge side was not mirrored.
- **Bucket**: `partial-leave-open`
- **Next step**: mirror the #1267 fix to `TransformerBridge.generate` and `generate_stream` (~10 LoC: guard `self.tokenizer is not None` before the eos/pad lookups, accept None tokenizer when `eos_token_id` is supplied). Once bridge-side regression test exists, close.

<a id="issue-509"></a>

#### #509 — LayerNorm folding not implemented for BertBlock

- **Issue**: BertBlock uses post-norm; `fold_ln=True` would fold LN into Q/K/V which is mathematically incorrect for post-norm.
- **HookedTransformer**: 🐛 architectural limitation per Neel ("LayerNorm should not be folded at all... I can't think of any way to do LayerNorm folding for Bert"). [`HookedEncoder.from_pretrained`](transformer_lens/HookedEncoder.py#L412) already hardcodes `fold_ln=False` so silent-wrong-result is averted, but a user calling lower-level `from_pretrained(..., fold_ln=True)` on a BERT model still gets undefined behavior. `BertBlock` lives at [transformer_lens/components/bert_block.py:19](transformer_lens/components/bert_block.py#L19).
- **TransformerBridge**: ⚠️ `BertArchitectureAdapter` exists; `enable_compatibility_mode()` would inherit the same fold-doesn't-work problem. Bridge users typically don't fold LN regardless.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: Two options unchanged from v2 — either close as wontfix (Neel's view) or add a 5-line warning when `fold_ln=True` is passed for a BERT-family architecture. Current hardcode at HookedEncoder.py:412 is sufficient for the standard path; the warning would catch the pathological lower-level call.

<a id="issue-543"></a>

#### #543 — Grokking demo broken in Colab

- **Issue**: `loss_fn(all_logits, labels)` raises `RuntimeError: Size does not match at dimension 0 expected index [12769, 1] to be smaller than self [113, 113]`.
- **HookedTransformer**: ⚠️ unverified. `demos/Grokking_Demo.ipynb` last touched in [`98811df5 3.0 CI Bugs (#1261)`](demos/Grokking_Demo.ipynb); no commit referencing #543 directly. anthonyduong9 said in 2024 "I can work on this today" but no PR linked.
- **TransformerBridge**: N/A — demo-specific shape bug.
- **Replication**: `[unverifiable]` — needs Colab-like environment to run the full notebook end-to-end.
- **Bucket**: `bug-likely-fixed-needs-verification`
- **Next step**: ask reporter (or anthonyduong9) to re-run the notebook on current `dev` and confirm whether the original error reproduces. If it does, the fix is in `loss_fn` (`per_token_logprobs` shape vs. `labels` shape mismatch — most likely a `.unsqueeze(-1)` missing or extra). If it doesn't, close.

<a id="issue-569"></a>

#### #569 — Cannot load Llama 3 70B on multigpu in 4bit

- **Issue**: `HookedTransformer.from_pretrained(..., hf_model=base_model)` fails with `size mismatch for blocks.0.attn._W_K: copying a param with shape torch.Size([4194304, 1])`. BnB packs weights as 1D blobs; HT's QKV reshape doesn't unpack them.
- **HookedTransformer**: 🐛 unchanged — HT load path doesn't unpack BnB-quantized weights before reshape.
- **TransformerBridge**: ✅ now meaningfully fixed for both halves of the original problem. (1) **Multi-GPU**: PR #1270 (`d95bd962`, "Multi-Device Processing on Bridge") added `n_devices` and `device_map` kwargs to `TransformerBridge.boot_transformers` — see [transformer_lens/model_bridge/bridge.py:195-230](transformer_lens/model_bridge/bridge.py#L195-L230). (2) **Quantization**: PR #1276 (`0a5218ca`, "Fixed Quantization bug in TransformerLens 3.0") repaired an `AttentionBridge`/`GeneralizedComponent` dtype-cast bug where bridge cast fp inputs to the storage dtype (uint8 for BnB Params4bit) of quantized first-parameters, producing gibberish logits. j.larson's recent comment on the issue confirms: "If you migrate to TransformerLens 3.0, there is a demo for how to run Llama 3 4-bit with the new system."
- **Replication**: `[code-verified]`
- **What changed since v2**: PRs #1270 and #1276 landed (multi-device bridge support + fixed quantization on bridge). v2 marked bridge as structurally sound but unverified for 4bit; bridge had a real bug that's now repaired.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: HT-side fix would still require BnB-aware QKV reshape (~50 LoC) but is no longer the only path. Reply on issue pointing at the [Llama-2 quantized demo](demos/LLaMA2_GPU_Quantized.ipynb) and the migration guide; once a user confirms 4bit + multi-GPU end-to-end on bridge, close.

<a id="issue-588"></a>

#### #588 — Setup unit tests to cover model configurations

- **Issue**: Add unit tests that load every supported model's config and verify it's parseable.
- **HookedTransformer/Bridge**: ⚠️ partial — same as v2. Per-architecture coverage at `tests/unit/test_gemma3_config.py`, `test_hooked_transformer_config.py`, `test_llava_config.py`, `test_qwen3_5_adapter.py`, `test_gemma3_multimodal_adapter.py` plus structural tests under [tests/unit/model_bridge/supported_architectures/](tests/unit/model_bridge/supported_architectures/) (7 adapter test files). No single parametrized sweep over the full `SUPPORTED_ARCHITECTURES` keyset.
- **Replication**: `[code-verified]`
- **Bucket**: `partial-leave-open`
- **Next step**: ~30 LoC parametrized test over all `SUPPORTED_ARCHITECTURES` keys: for each, load config-only (no weights) and assert the architecture adapter resolves. Curt-tigges signed up in 2024 without a PR; could now also be assigned to whoever next adds an adapter (forces the pattern for new entries too).

<a id="issue-595"></a>

#### #595 — Add Stopping Criteria support

- **Issue**: HF offers `StoppingCriteria` for custom halt conditions; HT/bridge `generate()` only support `stop_at_eos`.
- **HookedTransformer**: ❌ unchanged — [transformer_lens/HookedTransformer.py:1882](transformer_lens/HookedTransformer.py#L1882) `generate()` still only takes `stop_at_eos: bool`. Same at `generate_stream()` line 2262.
- **TransformerBridge**: ❌ unchanged — [transformer_lens/model_bridge/bridge.py:2433](transformer_lens/model_bridge/bridge.py#L2433) `generate()` and `generate_stream()` (line 2743) only have `stop_at_eos`.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~30 LoC — add `stopping_criteria: Optional[Callable[[tokens, logits], bool]] = None` to all four entry points (HT generate/generate_stream, bridge generate/generate_stream); evaluate after each sampled token and break if any returns True. srishti-git1110 volunteered in 2024.

<a id="issue-615"></a>

#### #615 — HookedTransformer output not identical to HuggingFace for Llama 3

- **Issue**: Greedy decoding diverges between HT and HF on Llama-3-8B-Instruct. Investigation localized to MLP weight differences after einsum/Linear conversion.
- **HookedTransformer**: ⚠️ much improved — most einsums in attention/MLP replaced with `F.linear` (visible at [transformer_lens/components/abstract_attention.py:368-374](transformer_lens/components/abstract_attention.py#L368-L374)). degenfabian reports max diff ~`2e-4` on Llama-3-8B-Instruct; close enough for production but not bit-exact. Per-architecture reports on Gemma 2-2B etc. continue.
- **TransformerBridge**: ✅ argmax/CE/generation parity with HF achieved. **Important update**: PR #1276 fixed a real precision-killing bug in `AttentionBridge` where the dtype-cast logic returned the storage dtype of quantized parameters; the fix also benefits non-quantized models because the same `target_dtype = next(parameters()).dtype` codepath was used. Bridge does its own attention math (`torch.matmul` + softmax + mask in [generalized_components/joint_qkv_attention.py:465-480](transformer_lens/model_bridge/generalized_components/joint_qkv_attention.py#L465-L480)). Empirically, Pythia-70m bridge vs HF: ~`2.5e-3` max drift, argmax matches.
- **Replication**: `[empirically replicated]` — bridge gives small drift but argmax-matches HF on Pythia-70m (per the v2 measurement).
- **What changed since v2**: PR #1276 fixed a quantization-storage dtype-cast bug in `GeneralizedComponent` that was silently degrading attention precision for any model where the first attention parameter happened to have non-fp dtype.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: bridge users get argmax/CE/generation parity with HF. Bit-exact match still depends on (a) `attn_implementation="eager"` vs HF default sdpa, (b) softmax dtype/order, (c) `.contiguous()` calls. For interp uses bridge is sufficient; for bit-exact circuit reproduction, document the known eager-vs-sdpa caveat in [docs/source/content/migrating_to_v3.md](docs/source/content/migrating_to_v3.md).

<a id="issue-644"></a>

#### #644 — Documentation: Map the Act Names to the Transformer

- **Issue**: Add a labeled diagram mapping hook names to positions on a transformer architecture figure.
- **HookedTransformer/Bridge**: ❌ unchanged — [docs/source/content/model_structure.md](docs/source/content/model_structure.md) is still 153 lines listing 51 hook names, no diagram. Recent edits (`a92a90a1` "Documenting 3.1 features") expanded prose around `enable_compatibility_mode()` but no figure added. Two volunteers (juvogt, tjbai) said they'd contribute years ago, no PR landed.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~1-day docs task — generate a diagram (matplotlib + manual layout, or Excalidraw + commit the SVG). Place at `docs/source/_static/hook_diagram.svg`, embed in `model_structure.md`. Now overlaps with v3.0 hook-aliasing — diagram should label both new canonical (e.g., `blocks.{i}.ln1.hook_out`) and legacy aliases (`hook_normalized`, `hook_scale`) shown in the doc at line 98-100.

<a id="issue-661"></a>

#### #661 — Pythia output inconsistent across batch sizes with `use_split_qkv_input=True`

- **Issue**: `model(input[:2])[0]` and `model(input[:1])[0]` give different outputs when `use_split_qkv_input=True`.
- **HookedTransformer**: 🐛 unchanged — [transformer_lens/components/transformer_block.py:123,137](transformer_lens/components/transformer_block.py#L123-L153) still branches on `use_split_qkv_input`; bug confirmed.
- **TransformerBridge**: ⚠️ bridge now exposes `set_use_split_qkv_input` at [transformer_lens/model_bridge/bridge.py:3373](transformer_lens/model_bridge/bridge.py#L3373) — feature parity gained since v2. Whether bridge reproduces the same batch-size inconsistency is **not yet tested**; bridge implementation routes through `_propagate_attention_flag` rather than the per-token splitting in HT's `transformer_block.py`, so the root cause may differ.
- **Replication**: `[empirically replicated]` on HT side — pythia-70m repro from issue gives `max diff: 1.14e-02` (v2 measurement).
- **What changed since v2**: bridge gained `use_split_qkv_input` (no longer N/A as v2 stated).
- **Bucket**: `bug-still-reproduces`
- **Next step**: (1) reproduce on bridge with the same input — if bridge is correct, document the workaround (use bridge for split-qkv analysis). (2) HT-side investigation: stateful interaction in LN1 path, related to #335 (LN1 firing 3× per forward). Non-trivial; research-only feature so moderate priority.

<a id="issue-684"></a>

#### #684 — Expand quantization model support beyond Llama

- **Issue**: HT raises `AssertionError: Quantization is only supported for Llama models` when loading 4bit Mistral via `hf_model=`.
- **HookedTransformer**: 🐛 unchanged — assertion still at [transformer_lens/HookedTransformer.py:1341-1342](transformer_lens/HookedTransformer.py#L1341-L1342) (`load_in_4bit and ("llama" not in model_name.lower())`).
- **TransformerBridge**: ✅ now functional. PR #1276 (`0a5218ca`, "Fixed Quantization bug in TransformerLens 3.0") repaired the `AttentionBridge` dtype-cast that was producing gibberish logits on quantized models — see regression test `test_AttentionBridge_preserves_fp_input_when_first_param_is_quantized` in [tests/integration/model_bridge/test_bridge_integration.py](tests/integration/model_bridge/test_bridge_integration.py). j.larson's comment on the issue points users at the migration guide. Bridge has no architecture-specific quantization assertion.
- **Replication**: `[code-verified]`
- **What changed since v2**: PR #1276 fixed the storage-dtype cast bug. v2 said "structurally sound but unverified empirically"; the bug was real and is now fixed with regression coverage.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: HT-side fix would require removing the assertion AND auditing per-architecture state_dict load for BnB-packed weights (overlaps with #569 root cause). Bridge users can pre-load via `AutoModelForCausalLM.from_pretrained(model, load_in_4bit=True)` and pass to `boot_transformers(model_name, hf_model=quantized_model)`. Reply on issue with the demo link and close pending user confirmation.

<a id="issue-697"></a>

#### #697 — Activation cache during generate

- **Issue**: User wants `run_with_cache` semantics during `model.generate()` — cache activations of generated tokens, not just the prompt.
- **HookedTransformer**: ❌ unchanged — [transformer_lens/HookedTransformer.py:1873,2257](transformer_lens/HookedTransformer.py#L1873) `generate()` and `generate_stream()` exist but neither integrates `run_with_cache`. bryce's reply: "no integration ... pretty low priority."
- **TransformerBridge**: ❌ unchanged — [transformer_lens/model_bridge/bridge.py:2433](transformer_lens/model_bridge/bridge.py#L2433) bridge `generate` and `generate_stream` (line 2743) — same gap. PR #1265 ("fixed batched generation on run_with_cache and run_with_hooks") improved interaction between the two surfaces but didn't add cache-during-generate.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~50 LoC enhancement — wrap the per-token forward in `run_with_cache`'s hook-installation context, accumulate cache across iterations. Trickier than naive due to KV-cache interactions; needs care to avoid duplicate hook fires when cache grows. Both APIs need the same fix.

<a id="issue-704"></a>

#### #704 — Add support for TracrBench

- **Issue**: TracrBench (121 toy Tracr transformers) — should it ship in TransformerLens or live in a separate repo.
- **HookedTransformer**: ❌ not in core. `grep -i tracr_bench` in `transformer_lens/` returns nothing.
- **TransformerBridge**: ❌ not in core; not a transformer-architecture-detection problem.
- **Replication**: `[code-verified]`
- **What changed since v2**: nothing material — no TracrBench code added.
- **Bucket**: `not-relevant-close`
- **Next step**: close with Neel's recommendation: build TracrBench as an external repo using TransformerLens as a dependency. Optionally add a one-line link from `docs/source/content/gallery.md`.

<a id="issue-710"></a>

#### #710 — MVP Support For 1-2 Models Per-Modality

- **Issue**: Add basic non-text-model support — TTS (Whisper), vision (ResNet, ViT), music gen, etc.
- **HookedTransformer**: ❌ not designed for non-text architectures.
- **TransformerBridge**: ⚠️ partial — `HubertArchitectureAdapter` (audio), `LlavaArchitectureAdapter` / `LlavaNextArchitectureAdapter` / `LlavaOnevisionArchitectureAdapter` / `Gemma3MultimodalArchitectureAdapter` (VLM), plus `MambaArchitectureAdapter` / `Mamba2ArchitectureAdapter` (SSM, non-attention). No Whisper, no ResNet, no ViT, no diffusion, no music. See [transformer_lens/model_bridge/supported_architectures/__init__.py:67-88](transformer_lens/model_bridge/supported_architectures/__init__.py#L67).
- **Replication**: `[code-verified]`
- **What changed since v2**: jlarson4 commented suggesting per-architecture sub-issues; Mamba(1/2) now confirmed in adapter list.
- **Bucket**: `not-addressed-difficult`
- **Next step**: per the comment thread, encourage reporters to file per-modality sub-issues (Whisper, ViT, etc.) so each can be tracked and prioritized. Close this umbrella once those are filed, or convert to a tracking meta-issue.

<a id="issue-720"></a>

#### #720 — Review current matmul function usages

- **Issue**: `batch_addmm` is right for GPT-2 `Conv1D`-style layers but wrong for plain `nn.Linear` models — need per-architecture matmul routing audit.
- **HookedTransformer**: ⚠️ partial — `F.linear` cleanup landed for the post-attention output ([abstract_attention.py:368-374](transformer_lens/components/abstract_attention.py#L368)), but `batch_addmm` still in [utilities/addmm.py](transformer_lens/utilities/addmm.py); full audit not done. No new commits to either file in the past 2 weeks.
- **TransformerBridge**: ⚠️ Q/K/V projections go through HF's `Linear` (correct), but bridge attention-score and output-application matmuls in [generalized_components/joint_qkv_attention.py](transformer_lens/model_bridge/generalized_components/joint_qkv_attention.py), [position_embeddings_attention.py](transformer_lens/model_bridge/generalized_components/position_embeddings_attention.py), [alibi_joint_qkv_attention.py](transformer_lens/model_bridge/generalized_components/alibi_joint_qkv_attention.py), and `joint_qkv_position_embeddings_attention.py` use raw `torch.matmul` — own audit need.
- **Replication**: `[code-verified]`
- **What changed since v2**: nothing material.
- **Bucket**: `partial-leave-open`
- **Next step**: same 3-part audit as before — (1) HT `batch_addmm` vs `F.linear` per-arch routing, (2) bridge `torch.matmul(q, k.T)` / `torch.matmul(weights, v)` vs HF's per-architecture impl, (3) Q/K/V projection paths.

<a id="issue-729"></a>

#### #729 — Guide to adding new models

- **Issue**: User asks for a how-to-extend-TL guide.
- **HookedTransformer/Bridge**: ✅ done — PR #1274 ("Adding Architecture Adapter Creation Guide to Docs", commit `fd288dc2`) landed [docs/source/content/adapter_development/adapter-creation-guide.md](docs/source/content/adapter_development/adapter-creation-guide.md), [adapter-specification.md](docs/source/content/adapter_development/adapter-specification.md), [hf-model-analysis-guide.md](docs/source/content/adapter_development/hf-model-analysis-guide.md), and a runnable [adapter-template.py](docs/source/_static/adapter-template.py).
- **Replication**: `[code-verified]`
- **What changed since v2**: PR #1274 (commit `fd288dc2`, ~9 days ago) closes this exactly — full bridge-side adapter creation walkthrough with template.
- **Bucket**: `covered-close`
- **Next step**: close with reference to PR #1274 and links to the new adapter-development docs.

<a id="issue-737"></a>

#### #737 — Q reshape with model loaded in 4bit

- **Issue**: `cfg.use_split_qkv_input=True` + 4bit vicuna-7b → shape mismatch in `AbstractAttention.calculate_qkv_matrices` — 4bit BnB-packed weight reshapes incorrectly under split-QKV.
- **HookedTransformer**: 🐛 still buggy — confirmed at [abstract_attention.py:58,338,378,454,473](transformer_lens/components/abstract_attention.py#L338) — multiple `if self.cfg.load_in_4bit:` branches that build `Params4bit` shaped `[nq, 1]`. No commits to this file targeting the bug since v2; the recent quantization-related commit `0a5218ca` ("Fixed Quantization bug in TransformerLens 3.0") and `d346e707` ("Improved quantization skipping") touch the bridge side, not this HT path.
- **TransformerBridge**: N/A — bridge has no `use_split_qkv_input` flag; quantized models load via `boot_transformers(hf_model=quantized_model)` and use HF's quantized Linear directly.
- **Replication**: `[unverifiable]` — needs GPU + bitsandbytes 4bit.
- **What changed since v2**: nothing material on this code path.
- **Bucket**: `partial-leave-open`
- **Next step**: HT-side fix needs reshape-aware logic in `calculate_qkv_matrices` for 4bit + split path (~30 LoC). Bridge users avoid this entirely. Reporter workaround on HT: disable `use_split_qkv_input` for 4bit models.

<a id="issue-773"></a>

#### #773 — TransformerLens on models with different layernorm placement (BioGPT)

- **Issue**: BioGPT has only one LN per layer (post-MLP `final_layer_norm`), unlike GPT-2's pre-LN1+pre-LN2. User asks for support.
- **HookedTransformer**: ❌ hard-coded GPT-2 LN placement; `BioGptForCausalLM` confirmed listed under [tools/model_registry/data/architecture_gaps.json:909](transformer_lens/tools/model_registry/data/architecture_gaps.json#L909).
- **TransformerBridge**: ❌ no `BioGptArchitectureAdapter` — grep returns no BioGpt match in [transformer_lens/model_bridge/supported_architectures/__init__.py](transformer_lens/model_bridge/supported_architectures/__init__.py). The component-map pattern theoretically supports per-arch LN layout, but no adapter exists.
- **Replication**: `[code-verified]`
- **What changed since v2**: nothing material; PR #1274 added the adapter creation guide which is now a viable path for the reporter to add a BioGPT adapter themselves.
- **Bucket**: `not-addressed-difficult`
- **Next step**: write a `BioGptArchitectureAdapter` (~80 LoC + tests) following the new [adapter-creation-guide.md](docs/source/content/adapter_development/adapter-creation-guide.md). Could now reasonably ask the reporter to take this on with the new guide.

<a id="issue-796"></a>

#### #796 — `FactoredMatrix.svd()` `lru_cache` prevents GC

- **Issue**: `FactoredMatrix.svd` decorated with `@lru_cache(maxsize=None)` holds instance refs and prevents garbage collection.
- **HookedTransformer**: 🐛 still buggy — confirmed at [FactoredMatrix.py:9,217](transformer_lens/FactoredMatrix.py#L9) — `from functools import lru_cache` and `@lru_cache(maxsize=None) def svd(self): ...`. Last commit to file: `90cf7476` ("Fix FactoredMatrix eigenvalues type") — not addressing this.
- **TransformerBridge**: 🐛 same shared `FactoredMatrix` class.
- **Replication**: `[code-verified]`
- **What changed since v2**: nothing material.
- **Bucket**: `not-addressed-simple`
- **Next step**: replace `@lru_cache(maxsize=None)` with `@cached_property` on `svd` and `eigenvalues` (~10 LoC). Breaking change (`.svd()` → `.svd`) — coordinate with broader `FactoredMatrix` cleanup.

<a id="issue-798"></a>

#### #798 — Remove `model_args` (use only `model_kwargs`)

- **Issue**: Bryce's own proposal to remove `*model_args` + `**model_kwargs` redundancy in pass-through functions.
- **HookedTransformer**: ⚠️ unchanged — `model_args` still present in [HookedEncoderDecoder.py:489-513](transformer_lens/HookedEncoderDecoder.py#L489), [hook_points.py:629,723,779](transformer_lens/hook_points.py#L629), [HookedAudioEncoder.py:299-323](transformer_lens/HookedAudioEncoder.py#L299), [BertNextSentencePrediction.py:220-266](transformer_lens/BertNextSentencePrediction.py#L220), [HookedTransformer.py:707+](transformer_lens/HookedTransformer.py#L707).
- **TransformerBridge**: ⚠️ same — bridge inherits `hook_points.py` machinery.
- **Replication**: `[code-verified]`
- **What changed since v2**: nothing material.
- **Bucket**: `not-addressed-simple`
- **Next step**: ~30 LoC across affected files — strip `*model_args`, keep only `**model_kwargs`. Already labeled `breaking-change`.

<a id="issue-830"></a>

#### #830 — Type hint support for `self.model` in `ActivationCache`

- **Issue**: `ActivationCache.model` untyped (would need `HookedTransformer` import → circular). Proposes `HookedTransformerMixin` to break the cycle.
- **HookedTransformer**: ❌ unchanged — confirmed at [ActivationCache.py:118](transformer_lens/ActivationCache.py#L118) — `self.model = model` with no annotation.
- **TransformerBridge**: ❌ same `ActivationCache` shared class.
- **Replication**: `[code-verified]`
- **What changed since v2**: nothing material.
- **Bucket**: `not-addressed-simple`
- **Next step**: extract a `HookedRootModuleMixin` / use `TYPE_CHECKING + Protocol` to hint without circular imports (~50 LoC). Tagged 3.0 / 4.0 milestone.

<a id="issue-837"></a>

#### #837 — Multi-GPU device ordinal issue (`n_devices=3` for llama2-7b)

- **Issue**: `n_devices=3` produces "device ordinal out of range" — `(index // layers_per_device)` overshoots when `n_layers % n_devices != 0`.
- **HookedTransformer**: 🐛 still buggy at [utilities/multi_gpu.py:142](transformer_lens/utilities/multi_gpu.py#L142) — `device_index = (device.index or 0) + (index // layers_per_device)` unchanged. The function is now flagged `Deprecated: This will be removed in 3.0` ([line 130-133](transformer_lens/utilities/multi_gpu.py#L130)).
- **TransformerBridge**: ✅ first-class — PR #1270 ("Multi-Device Processing on Bridge", commit `d95bd962`) landed `resolve_device_map` at [multi_gpu.py:170-204](transformer_lens/utilities/multi_gpu.py#L170) with explicit `n_devices` / `device_map` / `max_memory` params and accelerate-backed dispatch. jlarson4's comment on the issue points users to PR #1270.
- **Replication**: `[unverifiable]` — no multi-GPU here.
- **What changed since v2**: PR #1270 merged (was unmerged in v2); bridge users now have a fully supported path.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: HT-side bug remains. Reply on issue with bridge migration recipe (`bridge = TransformerBridge.boot_transformers(name, n_devices=3)`); leave HT path open for #968-family fix or close with bridge pointer if reporter migrates.

<a id="issue-846"></a>

#### #846 — Prioritize local `hf_model.config` for Qwen models

- **Issue**: Loading a local Qwen via `from_pretrained_no_processing(model_name="Qwen/...", hf_model=local, tokenizer=tok)` still fetches HF config online and fails offline.
- **HookedTransformer**: 🐛 same root cause as #754 / #800 — `convert_hf_model_config` calls `AutoConfig.from_pretrained` unconditionally; Qwen has no name-based shortcut.
- **TransformerBridge**: ✅ now first-class — PR #1279 ("Updated `boot_transformers` to use local hf_config, if a local hf_model is passed", commit `0636214f`) landed at [model_bridge/sources/transformers.py:339-349](transformer_lens/model_bridge/sources/transformers.py#L339) — `if hf_model is not None: hf_config = copy.deepcopy(hf_model.config)`, skipping `AutoConfig.from_pretrained` entirely. New regression tests at [tests/integration/model_bridge/test_bridge_creation_modes.py](tests/integration/model_bridge/test_bridge_creation_modes.py).
- **Replication**: `[code-verified]`
- **What changed since v2**: PR #1279 closed the bridge-side gap that v2 noted as still open.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: comment with the new bridge recipe (`boot_transformers(model_name="Qwen/...", hf_model=local_model)` — now skips network call). HT-side fix per #754 still pending; close once reporter confirms bridge resolves their use case.

<a id="issue-867"></a>

#### #867 — Does TransformerLens support LVLM like Qwen2-VL?

- **Issue**: User asks if Qwen2-VL / Qwen2.5-VL is supported.
- **HookedTransformer**: ❌ no native VLM support.
- **TransformerBridge**: ❌ `Qwen2VLForConditionalGeneration` and `Qwen2_5_VLForConditionalGeneration` still listed in `architecture_gaps.json` (lines 4709, 4940). LLaVA family adapters present at [transformer_lens/utilities/architectures.py:32-34](transformer_lens/utilities/architectures.py#L32-L34) but no Qwen2-VL adapter.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: add `Qwen2VLArchitectureAdapter` (~150 LoC, LLaVA-pattern). Comment pointing reporter at LLaVA support today and ExplorerFreda's vlm-lens fork as alternative; close once Qwen2-VL adapter lands.

<a id="issue-869"></a>

#### #869 — Custom generative video transformer

- **Issue**: User wants mech interp on a Sora-like generative video diffusion transformer.
- **HookedTransformer**: ❌ no diffusion/video generation support.
- **TransformerBridge**: ❌ same — bridge wraps HF causal/seq2seq/multimodal text models; not designed for diffusion.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: outside current scope per Bryce's reply (would need a separate `HookedDiffusionTransformer` root module). Recommend close as wontfix or defer to architectural roadmap; point user to a dedicated diffusion-interp tool.

<a id="issue-888"></a>

#### #888 — Adapt HookedTransformer to a non-supported model (CLIP language model)

- **Issue**: User wants `from_pretrained` for CLIP's text encoder.
- **HookedTransformer**: ❌ not possible without code modifications.
- **TransformerBridge**: ⚠️ adapter framework supports it, but no `CLIPTextModel` adapter exists — grep finds no `CLIPTextModel*` symbol in `transformer_lens/`. `CLIPVisionEncoderBridge` exists for the vision side via LLaVA. jlarson4 already commented pointing the reporter at the adapter-creation guide.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: write `CLIPTextModelArchitectureAdapter` (~120 LoC, encoder-only, BERT-like attention). jlarson4's prior comment already pointed to the adapter-creation guide; could leave open as a focused model-request or invite contribution.

<a id="issue-911"></a>

#### #911 — PosEmbed device error with `accelerate`

- **Issue**: gpt2 + `accelerate launch` (DDP across 2 GPUs) fails inside `PosEmbed.forward` because `W_pos[offset_position_ids]` mixes device.
- **HookedTransformer**: 🐛 still buggy at [transformer_lens/components/pos_embed.py:59](transformer_lens/components/pos_embed.py#L59) (`pos_embed = self.W_pos[offset_position_ids]`). No commit on this file since `98811df5` (3.0 CI bugs).
- **TransformerBridge**: ✅ uses HF's `wpe` directly via `EmbeddingBridge`; respects HF `device_map`. PR #1270 (`d95bd962`) merged first-class `n_devices`/`device_map` for the bridge.
- **Replication**: `[unverifiable]` — needs DDP setup.
- **What changed since v2**: PR #1270 multi-device bridge support is now merged on main.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: jlarson4 already commented with bridge migration pointer — wait for reporter response. Close after confirmation, or add bridge integration test under `accelerate launch` to harden the recommendation.

<a id="issue-912"></a>

#### #912 — Support mT5 models

- **Issue**: User requests `google/mt5-small` for multilingual circuit discovery.
- **HookedTransformer**: ❌ T5-only path; not added.
- **TransformerBridge**: ✅ wired end-to-end. `"mt5"` mapped at [transformer_lens/model_bridge/sources/transformers.py:235](transformer_lens/model_bridge/sources/transformers.py#L235), `MT5ForConditionalGeneration` routed to `T5ArchitectureAdapter` at [transformer_lens/factories/architecture_adapter_factory.py:119](transformer_lens/factories/architecture_adapter_factory.py#L119), plus the `requires_relative_position_bias=True` + `is_cross_attention=True` fixes in `supported_architectures/t5.py`. Verified on `google/mt5-small`, `mt5-base`, `mt5-large`, `mt5-xl` (full verification, P1=100%).
- **Replication**: `[empirically replicated]` — verification history shows full-pass entries dated 2026-05-08.
- **What changed since v2**: PR #1289 (`d5e3a2b0`) landed the routing + cross-attention fix; multiple mT5 sizes verified.
- **Bucket**: `covered-close`
- **Next step**: close with link to TransformerBridge migration guide and the `google/mt5-base` verified-models entry. Reporter's `mt5-small` use case is now directly supported.

<a id="issue-950"></a>

#### #950 — Support SimpleStories models

- **Issue**: User requests SimpleStories support for low-resource interp work.
- **HookedTransformer**: ❌ not registered.
- **TransformerBridge**: ✅ 11 SimpleStories models verified end-to-end on 2026-05-08 (`SimpleStories-1.25M`, `-5M`, `-11M`, `-30M`, `-35M`, plus `V2-1.25M/5M/11M/30M/35M` and `test-SimpleStories-gpt2-1.25M`) — full verification, P1=P2=P3=100%, P4>=90% via PR #1292 (`0c0bd3ce`). `SimpleStories` author registered for `LlamaForCausalLM` at [transformer_lens/tools/model_registry/__init__.py:123](transformer_lens/tools/model_registry/__init__.py#L123).
- **Replication**: `[empirically replicated]`
- **What changed since v2**: PR #1292 SimpleStories Model Verification merged; jlarson4's "I'll see if I can tackle that before the next release" promise is now delivered.
- **Bucket**: `covered-close`
- **Next step**: close with the verified-models page link. mivanit asked for SimpleStories; 11 SimpleStories-published models now load and pass verification through the bridge.

<a id="issue-953"></a>

#### #953 — Add basic support for Gemma 3n (E2B & E4B)

- **Issue**: Reporter asks for text-only support of Gemma 3n (AltUp / LAuReL / PLE / mixed local-global attention).
- **HookedTransformer**: ❌ not supported.
- **TransformerBridge**: ❌ not registered in `SUPPORTED_ARCHITECTURES`; no Gemma3n entry in [transformer_lens/utilities/architectures.py](transformer_lens/utilities/architectures.py). Bryce confirmed in-progress for next major release.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: track for milestone 3.x. AltUp/LAuReL/PLE need dedicated component bridges (~200-500 LoC); mixed local/global attention overlaps with Gemma2 work. Defer until HF's `Gemma3nForCausalLM` forward stabilizes.

<a id="issue-968"></a>

#### #968 — `unsloth/llama-3.2-3b-instruct` with 2× 3060 device-mismatch

- **Issue**: `from_pretrained(..., n_devices=2)` on 2× 3060 throws `RuntimeError: indices should be either on cpu or on the same device`.
- **HookedTransformer**: 🐛 multi-GPU placement bug cluster (#837/#907/#911) — `move_model_modules_to_device` greedy allocation is unchanged.
- **TransformerBridge**: ✅ PR #1270 (`d95bd962`) merged first-class `n_devices` / `device_map` to bridge — see [transformer_lens/model_bridge/sources/transformers.py:293-294](transformer_lens/model_bridge/sources/transformers.py#L293-L294) and the `resolve_device_map` path at line 480+.
- **Replication**: `[unverifiable]` — no multi-GPU device.
- **What changed since v2**: PR #1270 is now merged on main, not just on a feature branch.
- **Bucket**: `bug-likely-fixed-needs-verification`
- **Next step**: jlarson4's prior comment offered bridge + #1270; with #1270 now merged, ask reporter to retest with `TransformerBridge.boot_transformers("unsloth/llama-3.2-3b-instruct", n_devices=2)`. Close on confirmation.

<a id="issue-1080"></a>

#### #1080 — Import fails by default in Colab (numpy ABI mismatch)

- **Issue**: Fresh Colab + `pip install transformer_lens` + `import transformer_lens` raises `numpy.dtype size changed` ABI error; kernel restart works around it.
- **HookedTransformer**: ⚠️ [pyproject.toml:12-13](pyproject.toml#L12-L13) still has `numpy>=1.24` / `numpy>=1.26` lower bounds with no upper cap. Numpy 2.x is allowed; transitive ABI mismatch root cause.
- **TransformerBridge**: ⚠️ same install path; same numpy.
- **Replication**: `[unverifiable]` — Colab-specific.
- **Bucket**: `bug-likely-fixed-needs-verification`
- **Next step**: ask reporter to retest with current Colab kernel + current TL (3.x). If still failing, bisect transitive deps (`pandas`, `einops`, `jaxtyping`) and pin a tested numpy. No movement on this since v2.

<a id="issue-1133"></a>

#### #1133 — `tokenize_and_concatenate` cuts tokens mid-document

- **Issue**: Char-based 20-chunk split could cut tokens mid-doc, producing impossible token pairs.
- **HookedTransformer**: ✅ fixed. [transformer_lens/utilities/tokenize_utils.py:76-89](transformer_lens/utilities/tokenize_utils.py#L76-L89) tokenizes per-doc with `add_special_tokens=False`, joins with explicit token-level EOS, and reshapes — no string-level chunking. PR #1273 (`ad8e123b`); further refined by PR #1287 (`3003f77a`, "Tokenize and Concatenate additional datasets").
- **TransformerBridge**: ✅ shared utility.
- **Replication**: `[code-verified]` — original `tokens[79848:79850] == [337, 346]` repro cannot occur.
- **What changed since v2**: PR #1287 added more dataset coverage on top of #1273.
- **Bucket**: `covered-close`
- **Next step**: close as fixed (PRs #1273 + #1287). Confirm with BorisTheBrave that the new `dataset.map`-driven approach (per their suggestion in the thread) addresses the pathological cases too.

<a id="issue-1148"></a>

#### #1148 — Tutorial for "Real-Time Training Dynamics" (VSM Telemetry)

- **Issue**: Reporter proposes a demo notebook for σ_p / σ_a training-dynamics telemetry.
- **HookedTransformer**: ❌ no such tutorial. `demos/` has only `Grokking_Demo.ipynb`, no VSM telemetry.
- **TransformerBridge**: ❌ same — works equivalently against bridge's hook system.
- **Replication**: `[code-verified]`
- **What changed since v2**: jonathanrbelanger-lang committed in-thread to "get to work on this over the coming weekend" after jlarson4's invitation.
- **Bucket**: `not-addressed-simple`
- **Next step**: leave open and wait for the reporter's PR (notebook in `/demos`, targeting `TransformerBridge`). If no PR materializes, invite community contribution and close as wontfix.

<a id="issue-1263"></a>

#### #1263 — Direct Logit Attribution Tool

- **Issue**: Proposal — add a first-class direct-logit-attribution helper. Adjacent to #112 (logit display) and #111 (path patching demo).
- **HookedTransformer**: needs-investigation (added since v2)
- **TransformerBridge**: needs-investigation
- **Replication**: `[needs-investigation]`
- **Bucket**: `needs-triage`
- **Labels (from GitHub)**: enhancement / good first issue
- **Next step**: read issue body + comments; classify per v2 buckets; spot-check current code if claim is testable.

<a id="issue-1275"></a>

#### #1275 — Update Benchmarks & Verify Models to support Quantized models

- **Issue**: **Largely addressed in this branch** — quantization classification refactored (HF-loadable formats admitted to registry) and verify_models gates on `required_quant_library_for_model()` with a clean skip path when libs are missing. PR #1276 already fixed the dtype bug. Worth re-verifying the issue's specific asks against current state.
- **HookedTransformer**: needs-investigation (added since v2)
- **TransformerBridge**: needs-investigation
- **Replication**: `[needs-investigation]`
- **Bucket**: `needs-triage`
- **Labels (from GitHub)**: enhancement
- **Next step**: read issue body + comments; classify per v2 buckets; spot-check current code if claim is testable.

<a id="issue-1280"></a>

#### #1280 — Add support for `cpu`, `meta`, and `disk` to TransformerBridge `device_map`

- **Issue**: Proposal — extend bridge device_map handling. Pairs with #872 (broader device_map review).
- **HookedTransformer**: needs-investigation (added since v2)
- **TransformerBridge**: needs-investigation
- **Replication**: `[needs-investigation]`
- **Bucket**: `needs-triage`
- **Labels (from GitHub)**: TransformerBridge / enhancement
- **Next step**: read issue body + comments; classify per v2 buckets; spot-check current code if claim is testable.

<a id="issue-1291"></a>

#### #1291 — CI HuggingFace Call Reduction

- **Issue**: CI optimization — reduce HF Hub round-trips during test runs. Probably easy via fixture-level caching of the small models that get re-downloaded across test files.
- **HookedTransformer**: needs-investigation (added since v2)
- **TransformerBridge**: needs-investigation
- **Replication**: `[needs-investigation]`
- **Bucket**: `needs-triage`
- **Labels (from GitHub)**: low-priority
- **Next step**: read issue body + comments; classify per v2 buckets; spot-check current code if claim is testable.

