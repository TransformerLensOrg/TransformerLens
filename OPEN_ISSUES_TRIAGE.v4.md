# Open Issues Triage (v4)

**Generated:** 2026-05-11 (v4 — re-verified after another sprint of closures)
**Repo:** TransformerLensOrg/TransformerLens
**Open issues:** 38 (36 re-verified from v3 + 2 opened since)
**Previous archive:** [OPEN_ISSUES_TRIAGE.v3.md](OPEN_ISSUES_TRIAGE.v3.md), [OPEN_ISSUES_TRIAGE_OLD.md](OPEN_ISSUES_TRIAGE_OLD.md) (v2)

## What changed since v3

- **12 issues closed** during the v3 cycle: #290, #569, #661, #684, #729, #846, #911, #912, #950, #968, #1133, #1275
- **36 entries re-verified** against current code; 1 newly-closeable, 1 verdict refined, 3 stubs properly triaged
- **2 new entries**: #1297 (Gemma4 adapter), #1298 (external arch registration) — both `not-addressed-simple` bridge work

### Newly closeable based on v4 re-verification

- **#483** — bridge `generate()` no-tokenizer fix landed (commit `513d157b`, May 11) with regression test. Both HT and bridge sides now guard `self.tokenizer is not None`.

### Verdict refinements (still open, evidence stronger or context corrected)

- **#462** — jlarson4 commented confirming Mamba + Granite-MoE-Hybrid shipped on bridge 3.0; close-action is now well-supported.
- **#644** — discovered an existing `TransformerLens_Diagram.svg` referenced from `index.md` but not embedded in `model_structure.md`. Bucket moved from `not-addressed-simple` to `partial-leave-open` — next step is a 1-line embed.
- **#1263** — `cache.logit_attrs` exists in [ActivationCache.py:488-606](transformer_lens/ActivationCache.py#L488-L606) but no standalone wrapper in `tools/analysis/`. Now `not-addressed-simple` (was `needs-triage`).
- **#1280** — exact blocker found at [multi_gpu.py:146-167](transformer_lens/utilities/multi_gpu.py#L146) (`_UNSUPPORTED_DEVICE_MAP_VALUES`); reporter-assigned with PR in flight. Now `partial-leave-open`.
- **#1291** — CI already caches HF model dirs; missing `concurrency` group is the targetable next step. Now `partial-leave-open`.

The v2 methodology section (HT-side / Bridge-side / Replication / Next step) still applies — see [OPEN_ISSUES_TRIAGE_OLD.md](OPEN_ISSUES_TRIAGE_OLD.md#methodology-per-issue).

## Summary table (sorted by issue number)

| Issue | Title | Bucket |
|---|---|---|
| #111 | [Demo of direct path patching](#issue-111) | `not-addressed-difficult` |
| #112 | [Helper to display vectors of logits nicely](#issue-112) | `not-addressed-simple` |
| #210 | [`get_full_resid_decomposition` accept tensor argument](#issue-210) | `not-addressed-simple` |
| #297 | [Better print-outs for currently attached hooks](#issue-297) | `not-addressed-simple` |
| #341 | [Update FactoredMatrix.svd() (uses deprecated `torch.svd`, returns V not Vh)](#issue-341) | `not-addressed-simple` |
| #385 | [Pythia / Rotary Embeddings don't match HuggingFace](#issue-385) | `bug-still-reproduces` |
| #453 | [`from_pretrained()` always downloads same weights with `checkpoint_label`](#issue-453) | `bug-likely-fixed-needs-verification` |
| #462 | [Add support for Mamba](#issue-462) | `fixed-on-transformerbridge` |
| #479 | [Memory efficient causal mask implementation](#issue-479) | `partial-leave-open` |
| #481 | [Tracr to TransformerLens demo broken](#issue-481) | `bug-still-reproduces` |
| #483 | [`HookedTransformer.generate()` `pad_token_id` error when tokenizer unset](#issue-483) | `covered-close` |
| #509 | [LayerNorm folding not implemented for BertBlock](#issue-509) | `not-addressed-difficult` |
| #543 | [Grokking demo broken in Colab](#issue-543) | `bug-likely-fixed-needs-verification` |
| #588 | [Setup unit tests to cover model configurations](#issue-588) | `partial-leave-open` |
| #595 | [Add Stopping Criteria support](#issue-595) | `not-addressed-simple` |
| #615 | [HookedTransformer output not identical to HuggingFace for Llama 3](#issue-615) | `fixed-on-transformerbridge` |
| #644 | [Documentation: Map the Act Names to the Transformer](#issue-644) | `partial-leave-open` |
| #697 | [Activation cache during generate](#issue-697) | `not-addressed-simple` |
| #704 | [Add support for TracrBench](#issue-704) | `not-relevant-close` |
| #710 | [MVP Support For 1-2 Models Per-Modality](#issue-710) | `not-addressed-difficult` |
| #720 | [Review current matmul function usages](#issue-720) | `partial-leave-open` |
| #737 | [Q reshape with model loaded in 4bit](#issue-737) | `partial-leave-open` |
| #773 | [TransformerLens on models with different layernorm placement (BioGPT)](#issue-773) | `not-addressed-difficult` |
| #796 | [`FactoredMatrix.svd()` `lru_cache` prevents GC](#issue-796) | `not-addressed-simple` |
| #798 | [Remove `model_args` (use only `model_kwargs`)](#issue-798) | `not-addressed-simple` |
| #830 | [Type hint support for `self.model` in `ActivationCache`](#issue-830) | `not-addressed-simple` |
| #837 | [Multi-GPU device ordinal issue (`n_devices=3` for llama2-7b)](#issue-837) | `fixed-on-transformerbridge` |
| #867 | [Does TransformerLens support LVLM like Qwen2-VL?](#issue-867) | `not-addressed-difficult` |
| #869 | [Custom generative video transformer](#issue-869) | `not-addressed-difficult` |
| #888 | [Adapt HookedTransformer to a non-supported model (CLIP language model)](#issue-888) | `not-addressed-difficult` |
| #953 | [Add basic support for Gemma 3n (E2B & E4B)](#issue-953) | `not-addressed-difficult` |
| #1080 | [Import fails by default in Colab (numpy ABI mismatch)](#issue-1080) | `bug-likely-fixed-needs-verification` |
| #1148 | [Tutorial for "Real-Time Training Dynamics" (VSM Telemetry)](#issue-1148) | `not-addressed-simple` |
| #1263 | [Direct Logit Attribution Tool](#issue-1263) | `not-addressed-simple` |
| #1280 | [Add support for `cpu`, `meta`, and `disk` to TransformerBridge `device_map`](#issue-1280) | `partial-leave-open` |
| #1291 | [CI HuggingFace Call Reduction](#issue-1291) | `partial-leave-open` |
| #1297 | [Gemma4 Architecture Adapter](#issue-1297) | `not-addressed-simple` |
| #1298 | [External Architecture Registration](#issue-1298) | `not-addressed-simple` |

## Per-issue entries

<a id="issue-111"></a>

#### #111 — Demo of direct path patching

- **Issue**: Add a section to Exploratory Analysis Demo demonstrating direct path patching for all head pairs. PR #49 was an early attempt.
- **HookedTransformer**: still no first-class path-patching helper. Verified — no `path_patch`/`direct_path` symbols exist anywhere under [transformer_lens/](transformer_lens/) or [transformer_lens/utilities/](transformer_lens/utilities/). [demos/Activation_Patching_in_TL_Demo.ipynb](demos/Activation_Patching_in_TL_Demo.ipynb) and [demos/Attribution_Patching_Demo.ipynb](demos/Attribution_Patching_Demo.ipynb) are the closest.
- **TransformerBridge**: same — no path-patching primitive in either API; bridge reuses the same `ActivationCache`.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: callum mcdougall pointed users at the [ARENA IOI notebook](https://colab.research.google.com/drive/1KgrEwvCKdX-8DQ1uSiIuxwIiwzJuQ3Gw). Either close with a docs pointer to ARENA, or implement a TL helper that wraps the pattern (~80 LoC).

<a id="issue-112"></a>

#### #112 — Helper to display vectors of logits nicely

- **Issue**: Neel asked for two things: **MVP** — function mapping logit vector → pandas DataFrame `(token_index, token_string, logit, log_prob, probability)`. **Bonus** — nostalgebraist-style `plot_logit_lens` heatmap.
- **HookedTransformer**: `test_prompt` in [transformer_lens/utilities/exploratory_utils.py:14](transformer_lens/utilities/exploratory_utils.py#L14) prints top-k for prompt+answer — partial spirit of the MVP but print-only, single-position. [transformer_lens/utilities/logits_utils.py](transformer_lens/utilities/logits_utils.py) exists but contains no `logits_to_df` or `plot_logit_lens` helper. Unchanged since v3.
- **TransformerBridge**: same — `test_prompt` works through bridge; no separate helper.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~30 LoC for `logits_to_df(logits, tokenizer, top_k=None) -> pd.DataFrame` (drop in [logits_utils.py](transformer_lens/utilities/logits_utils.py)), ~50 LoC for matplotlib `plot_logit_lens`. Both small library additions independent of CircuitsVis.

<a id="issue-210"></a>

#### #210 — `get_full_resid_decomposition` accept tensor argument

- **Issue**: Add a `project_output_onto: [d_model]` or `[d_model, num_outputs]` argument so neuron-decomposition doesn't blow GPU memory by materializing `[batch, pos, d_mlp, d_model]`.
- **HookedTransformer**: signature at [transformer_lens/ActivationCache.py:1091](transformer_lens/ActivationCache.py#L1091) — verified no `project_output_onto` kwarg added; memory-blowing path unchanged. No commits on the file in the last 5 days.
- **TransformerBridge**: same — bridge imports the same `ActivationCache` class.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: add `project_output_onto` kwarg + `(neurons * (W_out @ project_output_onto))` path. ~15 LoC + 1 test. Alan Cooney offered to take it; never landed.

<a id="issue-297"></a>

#### #297 — Better print-outs for currently attached hooks

- **Issue**: API for listing hooks attached to a model + HookPoint, with detail.
- **HookedTransformer**: no first-class `model.list_hooks()` or `HookPoint.describe()` API. Verified — `list_active_hooks`, `list_hooks`, and `describe()` symbols absent from [transformer_lens/hook_points.py](transformer_lens/hook_points.py). `model.hook_dict` and `hp.fwd_hooks`/`bwd_hooks` remain the only inspection surface.
- **TransformerBridge**: same — uses same `hook_points` machinery.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: add `HookedRootModule.list_active_hooks()` returning `Dict[str, List[hook_repr]]`. ~15 LoC + 1 test. Abandoned PR #302 was the prior attempt.

<a id="issue-341"></a>

#### #341 — Update FactoredMatrix.svd() (uses deprecated `torch.svd`, returns V not Vh)

- **Issue**: TL uses deprecated `torch.svd` (which returns V, not Vh) inside `FactoredMatrix.svd`. Should switch to `torch.linalg.svd` and return Vh per modern convention.
- **HookedTransformer/Bridge**: confirmed at [transformer_lens/FactoredMatrix.py:218-233](transformer_lens/FactoredMatrix.py#L218-L233) — three `torch.svd(...)` calls still present in `def svd`. No commits on file since v3.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~15-line fix — switch to `torch.linalg.svd(..., full_matrices=False)`, return `Vh` directly, update docstring noting the breaking change. `diego898` offered to send PR. Land with a deprecation warning.

<a id="issue-385"></a>

#### #385 — Pythia / Rotary Embeddings don't match HuggingFace

- **Issue**: Logit drift between `HookedTransformer` and HF for Pythia models. Llama-2-7b-chat reportedly catastrophic. Llama-3.2 rotary mismatch persists per chengjiali.
- **HookedTransformer**: rotary code lives at [transformer_lens/components/abstract_attention.py:599](transformer_lens/components/abstract_attention.py#L599). No new pythia-specific commits in the last 5 days.
- **TransformerBridge**: bridge uses HF's rotary directly via `RotaryEmbeddingBridge` at [transformer_lens/model_bridge/generalized_components/rotary_embedding.py:15](transformer_lens/model_bridge/generalized_components/rotary_embedding.py#L15), and joint-QKV/position-emb attention bridges call HF's `rotary_emb(seq_len, device)` directly. By construction matches HF.
- **Replication**: `[empirically replicated]` per v2/v3 (NaN logits in fp32 baseline). Not re-run this round.
- **Bucket**: `bug-still-reproduces` + `fixed-on-transformerbridge` for bridge users
- **Next step**: investigate the v2-reported NaN regression — verify whether it persists with full `from_pretrained` on current HEAD; bisect against `2c41b6c9`. Bridge users avoid this entirely.

<a id="issue-453"></a>

#### #453 — `from_pretrained()` always downloads same weights with `checkpoint_label`

- **Issue**: Reporter passes `checkpoint_label=...` and gets identical weights regardless of label. `checkpoint_index` works.
- **HookedTransformer**: signature at [transformer_lens/HookedTransformer.py:1158-1159](transformer_lens/HookedTransformer.py#L1158-L1159) has `checkpoint_index` and `checkpoint_value` — **NOT `checkpoint_label`**. The kwarg is silently absorbed into `**from_pretrained_kwargs`. Loader at [transformer_lens/loading_from_pretrained.py:1693-1707](transformer_lens/loading_from_pretrained.py#L1693-L1707) similarly only references `checkpoint_index`/`checkpoint_value`. Unchanged since v3.
- **TransformerBridge**: no checkpoint feature — uses HF's native loading only.
- **Replication**: `[code-verified]`
- **Bucket**: `bug-likely-fixed-needs-verification` (effectively user-error)
- **Next step**: respond to reporter that the parameter is `checkpoint_value`. Optionally validate unknown kwargs in `from_pretrained` and raise. ~10 LoC defensive change.

<a id="issue-462"></a>

#### #462 — Add support for Mamba

- **Issue**: Add Mamba SSM architecture support.
- **HookedTransformer**: not supported (by design — Mamba is fundamentally different from attention transformers).
- **TransformerBridge**: `MambaArchitectureAdapter` and `Mamba2ArchitectureAdapter` registered at [transformer_lens/factories/architecture_adapter_factory.py:95-96](transformer_lens/factories/architecture_adapter_factory.py#L95-L96). Both `MambaForCausalLM` and `Mamba2ForCausalLM` HF model classes mapped. Confirmed unchanged since v3.
- **Replication**: `[code-verified]`
- **What changed since v3**: jlarson4 left a comment on the issue confirming Mamba support has shipped on `TransformerBridge` 3.0 (also mentions Granite MoE hybrid).
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: close with comment pointing at `TransformerBridge.boot_transformers("state-spaces/mamba-130m-hf")`. Mamba support is shipped.

<a id="issue-479"></a>

#### #479 — Memory efficient causal mask implementation

- **Issue**: Each `Attention` layer registers a `(n_ctx, n_ctx)` boolean `causal_mask` buffer. ~86 GB overhead at Qwen 72B × 32K ctx.
- **HookedTransformer**: confirmed at [transformer_lens/components/abstract_attention.py:120-128](transformer_lens/components/abstract_attention.py#L120-L128) — `causal_mask = torch.tril(torch.ones((self.cfg.n_ctx, self.cfg.n_ctx)).bool())` and `register_buffer("mask", causal_mask)` still present (also at line 774 for resize). Bug as reported still present for ALL HT architectures.
- **TransformerBridge**: architecture-dependent. GPT2-family inherits HF's static `(max_pos, max_pos)` buffer. Modern HF impls (GPTNeoX/Pythia/Llama/Qwen/Mistral/Gemma) use `_update_causal_mask` per forward — zero overhead. The motivating Qwen 72B case is fixed on bridge.
- **Replication**: `[empirically replicated]` per v2.
- **Bucket**: `partial-leave-open`
- **Next step**: bridge users on modern architectures already have the desired memory profile. HT-side fix (~30 LoC: replace pre-allocated buffer with on-the-fly construction in `apply_causal_mask`) closes it for the legacy path and GPT2-family use cases.

<a id="issue-481"></a>

#### #481 — Tracr to TransformerLens demo broken

- **Issue**: Demo notebook assumes "the unembed is a projection onto the first few elements of the residual stream" — wrong because Tracr re-orders the residual stream alphabetically. Needs Tracr upstream PR to expose the unembed matrix.
- **HookedTransformer**: 🐛 confirmed at [demos/Tracr_to_Transformer_Lens_Demo.ipynb:233](demos/Tracr_to_Transformer_Lens_Demo.ipynb) — `sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)` line still present. No commits on the notebook since v3.
- **TransformerBridge**: ❌ N/A — Tracr-specific issue applies regardless of API; root cause is in the unembed-matrix derivation, not in TL's hook system. Demo not ported to bridge.
- **Replication**: `[code-verified]`
- **Bucket**: `bug-still-reproduces`
- **Next step**: needs Tracr upstream PR to expose `unembed_matrix` in `tracr.params`. FlyingPumba previously volunteered. Without that, demo is fundamentally limited.

<a id="issue-483"></a>

#### #483 — `HookedTransformer.generate()` `pad_token_id` error when tokenizer unset

- **Issue**: `model.generate()` with no tokenizer raises `AttributeError: 'NoneType' object has no attribute 'pad_token_id'`. Use case: training models on tokenizer-less domains (e.g., character-level integer addition).
- **HookedTransformer**: ✅ fixed by PR #1267 (commit `b1cc8c80`); see [transformer_lens/HookedTransformer.py:2068-2089](transformer_lens/HookedTransformer.py#L2068-L2089). Regression test at [tests/unit/test_generate_no_tokenizer.py](tests/unit/test_generate_no_tokenizer.py).
- **TransformerBridge**: ✅ now fixed by commit `513d157b` ("fix bridge side of generating with no tokenizer", May 11 2026). Both [bridge.py:2548-2571](transformer_lens/model_bridge/bridge.py#L2548-L2571) (`generate`) and [bridge.py:2829-2851](transformer_lens/model_bridge/bridge.py#L2829-L2851) (`generate_stream`) now guard `self.tokenizer is not None` and accept user-supplied `eos_token_id`. Regression test at [tests/unit/model_bridge/test_bridge_generate_no_tokenizer.py](tests/unit/model_bridge/test_bridge_generate_no_tokenizer.py).
- **Replication**: `[code-verified]`
- **What changed since v3**: bridge-side fix landed (`513d157b`). v3's `partial-leave-open` bucket is now resolved on both sides.
- **Bucket**: `covered-close`
- **Next step**: close with reference to PR #1267 (HT) and commit `513d157b` (bridge); both regression tests in place.

<a id="issue-509"></a>

#### #509 — LayerNorm folding not implemented for BertBlock

- **Issue**: BertBlock uses post-norm; `fold_ln=True` would fold LN into Q/K/V which is mathematically incorrect for post-norm.
- **HookedTransformer**: 🐛 architectural limitation per Neel ("LayerNorm should not be folded at all... I can't think of any way to do LayerNorm folding for Bert"). [`HookedEncoder.from_pretrained`](transformer_lens/HookedEncoder.py#L412) hardcodes `fold_ln=False`. `BertBlock` at [transformer_lens/components/bert_block.py:19](transformer_lens/components/bert_block.py#L19). No changes since v3.
- **TransformerBridge**: ⚠️ `BertArchitectureAdapter` exists; `enable_compatibility_mode()` would inherit the same fold-doesn't-work problem. Bridge users typically don't fold LN regardless.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: Two options unchanged from v3 — close as wontfix (Neel's view) or add a 5-line warning when `fold_ln=True` is passed for a BERT-family architecture.

<a id="issue-543"></a>

#### #543 — Grokking demo broken in Colab

- **Issue**: `loss_fn(all_logits, labels)` raises `RuntimeError: Size does not match at dimension 0 expected index [12769, 1] to be smaller than self [113, 113]`.
- **HookedTransformer**: ⚠️ unverified. `demos/Grokking_Demo.ipynb` last touched in `98811df5 3.0 CI Bugs (#1261)`; no commits referencing #543. No new activity since v3.
- **TransformerBridge**: N/A — demo-specific shape bug.
- **Replication**: `[unverifiable]` — needs Colab-like environment to run the full notebook end-to-end.
- **Bucket**: `bug-likely-fixed-needs-verification`
- **Next step**: ask reporter (or anthonyduong9) to re-run the notebook on current `dev` and confirm whether the original error reproduces. If yes, fix is in `loss_fn` shape mismatch; if no, close.

<a id="issue-588"></a>

#### #588 — Setup unit tests to cover model configurations

- **Issue**: Add unit tests that load every supported model's config and verify it's parseable.
- **HookedTransformer/Bridge**: ⚠️ partial — same as v3. Per-architecture coverage at `tests/unit/test_gemma3_config.py`, `test_hooked_transformer_config.py`, `test_llava_config.py`, plus 7 adapter test files under [tests/unit/model_bridge/supported_architectures/](tests/unit/model_bridge/supported_architectures/) (baichuan, codegen, cohere, gpt_bigcode, internlm2, mpt, xglm). No single parametrized sweep over the full `SUPPORTED_ARCHITECTURES` keyset. Verification system improvements landed (PR #1293) but are runtime/empirical, not config-only.
- **Replication**: `[code-verified]`
- **Bucket**: `partial-leave-open`
- **Next step**: ~30 LoC parametrized test over all `SUPPORTED_ARCHITECTURES` keys: load config-only (no weights) and assert the architecture adapter resolves.

<a id="issue-595"></a>

#### #595 — Add Stopping Criteria support

- **Issue**: HF offers `StoppingCriteria` for custom halt conditions; HT/bridge `generate()` only support `stop_at_eos`.
- **HookedTransformer**: ❌ unchanged — [transformer_lens/HookedTransformer.py:1882](transformer_lens/HookedTransformer.py#L1882) `generate()` and `generate_stream()` (line 2262) still only take `stop_at_eos: bool`.
- **TransformerBridge**: ❌ unchanged — [transformer_lens/model_bridge/bridge.py:2438](transformer_lens/model_bridge/bridge.py#L2438) `generate()` and `generate_stream()` (line 2754) only have `stop_at_eos`.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~30 LoC — add `stopping_criteria: Optional[Callable[[tokens, logits], bool]] = None` to all four entry points; evaluate after each sampled token and break if any returns True. srishti-git1110 volunteered in 2024.

<a id="issue-615"></a>

#### #615 — HookedTransformer output not identical to HuggingFace for Llama 3

- **Issue**: Greedy decoding diverges between HT and HF on Llama-3-8B-Instruct. Investigation localized to MLP weight differences after einsum/Linear conversion.
- **HookedTransformer**: ⚠️ much improved — most einsums in attention/MLP replaced with `F.linear`. degenfabian reports max diff ~`2e-4` on Llama-3-8B-Instruct; close enough for production but not bit-exact.
- **TransformerBridge**: ✅ argmax/CE/generation parity with HF achieved. PR #1276 fixed an `AttentionBridge`/`GeneralizedComponent` dtype-cast bug that was silently degrading attention precision. Bridge does its own attention math at [generalized_components/joint_qkv_attention.py:465-480](transformer_lens/model_bridge/generalized_components/joint_qkv_attention.py#L465-L480).
- **Replication**: `[empirically replicated]` — bridge gives small drift but argmax-matches HF on Pythia-70m (per the v3 measurement).
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: bridge users get argmax/CE/generation parity. Bit-exact match still depends on `attn_implementation="eager"` vs HF default sdpa, softmax dtype/order, and `.contiguous()` calls. Document the known eager-vs-sdpa caveat in [docs/source/content/migrating_to_v3.md](docs/source/content/migrating_to_v3.md).

<a id="issue-644"></a>

#### #644 — Documentation: Map the Act Names to the Transformer

- **Issue**: Add a labeled diagram mapping hook names to positions on a transformer architecture figure.
- **HookedTransformer/Bridge**: ⚠️ partial — a community diagram by akozlo exists at [docs/source/_static/TransformerLens_Diagram.svg](docs/source/_static/TransformerLens_Diagram.svg) and is linked from [docs/source/index.md:21](docs/source/index.md#L21), but **not** embedded in [docs/source/content/model_structure.md](docs/source/content/model_structure.md) (153 lines, 51 hook names listed without a figure). No changes to the doc since `a92a90a1` "Documenting 3.1 features".
- **Replication**: `[code-verified]`
- **Bucket**: `partial-leave-open`
- **Next step**: embed the existing `TransformerLens_Diagram.svg` in `model_structure.md` near the hook list, and add a v3.0 hook-aliasing legend (`hook_normalized` → `ln1.hook_out`, etc.). Or commission a fresh, hook-name-labeled diagram if the existing one omits names.

<a id="issue-697"></a>

#### #697 — Activation cache during generate

- **Issue**: User wants `run_with_cache` semantics during `model.generate()` — cache activations of generated tokens, not just the prompt.
- **HookedTransformer**: ❌ unchanged — [transformer_lens/HookedTransformer.py:1873](transformer_lens/HookedTransformer.py#L1873) `generate()` and `generate_stream()` (line 2257) still don't integrate `run_with_cache`. bryce's reply: "no integration ... pretty low priority."
- **TransformerBridge**: ❌ unchanged — [transformer_lens/model_bridge/bridge.py:2434](transformer_lens/model_bridge/bridge.py#L2434) bridge `generate` and `generate_stream` (line 2749) — same gap. PR #1265 improved `run_with_cache`/`run_with_hooks` interaction but didn't add cache-during-generate.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~50 LoC enhancement — wrap the per-token forward in `run_with_cache`'s hook-installation context, accumulate cache across iterations. Trickier than naive due to KV-cache interactions; needs care to avoid duplicate hook fires when cache grows. Both APIs need the same fix.

<a id="issue-704"></a>

#### #704 — Add support for TracrBench

- **Issue**: TracrBench (121 toy Tracr transformers) — should it ship in TransformerLens or live in a separate repo.
- **HookedTransformer**: ❌ not in core. `grep -i tracr` in `transformer_lens/` returns nothing; only the Tracr→HookedTransformer demo lives in [docs/source/content/tutorials.md:39](docs/source/content/tutorials.md#L39).
- **TransformerBridge**: ❌ not in core; not a transformer-architecture-detection problem.
- **Replication**: `[code-verified]`
- **What changed since v3**: nothing material — no TracrBench code added, no new comments.
- **Bucket**: `not-relevant-close`
- **Next step**: close with Neel's recommendation: build TracrBench as an external repo using TransformerLens as a dependency. Optionally add a one-line link from `docs/source/content/gallery.md` (currently absent).

<a id="issue-710"></a>

#### #710 — MVP Support For 1-2 Models Per-Modality

- **Issue**: Add basic non-text-model support — TTS (Whisper), vision (ResNet, ViT), music gen, etc.
- **HookedTransformer**: ❌ not designed for non-text architectures.
- **TransformerBridge**: ⚠️ partial — 56 adapters total at [transformer_lens/model_bridge/supported_architectures/](transformer_lens/model_bridge/supported_architectures); audio (`hubert.py`), VLM (`llava.py`, `llava_next.py`, `llava_onevision.py`, `gemma3_multimodal.py`), SSM (`mamba.py`, `mamba2.py`). Still no Whisper, no ViT, no ResNet, no diffusion.
- **Replication**: `[code-verified]`
- **What changed since v3**: nothing material — same adapter set; multimodal text-gen fix landed (`58330ad0`) but no new modality.
- **Bucket**: `not-addressed-difficult`
- **Next step**: per the existing comment thread, encourage reporters to file per-modality sub-issues (Whisper, ViT, etc.). Convert this to a tracking meta-issue or close once sub-issues filed.

<a id="issue-720"></a>

#### #720 — Review current matmul function usages

- **Issue**: `batch_addmm` is right for GPT-2 `Conv1D`-style layers but wrong for plain `nn.Linear` models — need per-architecture matmul routing audit.
- **HookedTransformer**: ⚠️ partial — `F.linear` cleanup landed for the post-attention output ([abstract_attention.py:368-374](transformer_lens/components/abstract_attention.py#L368)); `batch_addmm` still in [utilities/addmm.py](transformer_lens/utilities/addmm.py). No new commits to either file since v3.
- **TransformerBridge**: ⚠️ same picture — bridge attention components still use raw `torch.matmul` for QK / AV: [joint_qkv_attention.py:465,480](transformer_lens/model_bridge/generalized_components/joint_qkv_attention.py#L465), [position_embeddings_attention.py:416,452](transformer_lens/model_bridge/generalized_components/position_embeddings_attention.py#L416), [alibi_joint_qkv_attention.py:98,130](transformer_lens/model_bridge/generalized_components/alibi_joint_qkv_attention.py#L98), [mla_attention.py:216,227](transformer_lens/model_bridge/generalized_components/mla_attention.py#L216), [codegen_attention.py:336,357](transformer_lens/model_bridge/generalized_components/codegen_attention.py#L336).
- **Replication**: `[code-verified]`
- **What changed since v3**: nothing material.
- **Bucket**: `partial-leave-open`
- **Next step**: same 3-part audit as before — (1) HT `batch_addmm` vs `F.linear` per-arch routing, (2) bridge `torch.matmul(q, k.T)` / `torch.matmul(weights, v)` vs HF's per-architecture impl, (3) Q/K/V projection paths.

<a id="issue-737"></a>

#### #737 — Q reshape with model loaded in 4bit

- **Issue**: `cfg.use_split_qkv_input=True` + 4bit vicuna-7b → shape mismatch in `AbstractAttention.calculate_qkv_matrices` — 4bit BnB-packed weight reshapes incorrectly under split-QKV.
- **HookedTransformer**: 🐛 still buggy — `if self.cfg.load_in_4bit:` branches confirmed at [abstract_attention.py:58,338,378,454,473,491](transformer_lens/components/abstract_attention.py#L338). No commits to abstract_attention.py since v3 targeting this path.
- **TransformerBridge**: N/A — bridge has no `use_split_qkv_input` flag; quantized models load via `boot_transformers(hf_model=quantized_model)` and use HF's quantized Linear directly. Recent quantization work (`d346e707` "Improved quantization skipping") is bridge-side, doesn't touch this HT branch.
- **Replication**: `[unverifiable]` — needs GPU + bitsandbytes 4bit.
- **What changed since v3**: nothing material on this code path.
- **Bucket**: `partial-leave-open`
- **Next step**: HT-side fix needs reshape-aware logic in `calculate_qkv_matrices` for 4bit + split path (~30 LoC). Bridge users avoid this entirely. Reporter workaround on HT: disable `use_split_qkv_input` for 4bit models.

<a id="issue-773"></a>

#### #773 — TransformerLens on models with different layernorm placement (BioGPT)

- **Issue**: BioGPT has only one LN per layer (post-MLP `final_layer_norm`), unlike GPT-2's pre-LN1+pre-LN2. User asks for support.
- **HookedTransformer**: ❌ hard-coded GPT-2 LN placement; `BioGptForCausalLM` listed at [tools/model_registry/data/architecture_gaps.json:909](transformer_lens/tools/model_registry/data/architecture_gaps.json#L909).
- **TransformerBridge**: ❌ no `BioGptArchitectureAdapter` — not in [transformer_lens/model_bridge/supported_architectures/](transformer_lens/model_bridge/supported_architectures) (56 adapters, none for BioGPT). The component-map pattern theoretically supports per-arch LN layout, but no adapter exists.
- **Replication**: `[code-verified]`
- **What changed since v3**: nothing material; adapter creation guide at [docs/source/content/adapter_development/adapter-creation-guide.md](docs/source/content/adapter_development/adapter-creation-guide.md) is now a viable path for the reporter.
- **Bucket**: `not-addressed-difficult`
- **Next step**: write a `BioGptArchitectureAdapter` (~80 LoC + tests) following the adapter-creation-guide. Reasonable to invite reporter to take this on with the guide.

<a id="issue-796"></a>

#### #796 — `FactoredMatrix.svd()` `lru_cache` prevents GC

- **Issue**: `FactoredMatrix.svd` decorated with `@lru_cache(maxsize=None)` holds instance refs and prevents garbage collection.
- **HookedTransformer**: 🐛 still buggy — `from functools import lru_cache` at [FactoredMatrix.py:9](transformer_lens/FactoredMatrix.py#L9) and `@lru_cache(maxsize=None)` at [FactoredMatrix.py:217](transformer_lens/FactoredMatrix.py#L217). No commits to file since v3.
- **TransformerBridge**: 🐛 same shared `FactoredMatrix` class.
- **Replication**: `[code-verified]`
- **What changed since v3**: nothing material.
- **Bucket**: `not-addressed-simple`
- **Next step**: replace `@lru_cache(maxsize=None)` with `@cached_property` on `svd` and `eigenvalues` (~10 LoC). Breaking change (`.svd()` → `.svd`) — coordinate with broader `FactoredMatrix` cleanup.

<a id="issue-798"></a>

#### #798 — Remove `model_args` (use only `model_kwargs`)

- **Issue**: Bryce's own proposal to remove `*model_args` + `**model_kwargs` redundancy in pass-through functions.
- **HookedTransformer**: ⚠️ unchanged — `model_args` still present in [HookedEncoderDecoder.py:489-513](transformer_lens/HookedEncoderDecoder.py#L489), [hook_points.py:629,723,779](transformer_lens/hook_points.py#L629), [HookedAudioEncoder.py:299-323](transformer_lens/HookedAudioEncoder.py#L299), [BertNextSentencePrediction.py:220-266](transformer_lens/BertNextSentencePrediction.py#L220), [HookedTransformer.py:707-735](transformer_lens/HookedTransformer.py#L707).
- **TransformerBridge**: ⚠️ same — bridge inherits `hook_points.py` machinery.
- **Replication**: `[code-verified]`
- **What changed since v3**: nothing material; no new comments.
- **Bucket**: `not-addressed-simple`
- **Next step**: ~30 LoC across affected files — strip `*model_args`, keep only `**model_kwargs`. Already labeled `breaking-change`.

<a id="issue-830"></a>

#### #830 — Type hint support for `self.model` in `ActivationCache`

- **Issue**: `ActivationCache.model` untyped (would need `HookedTransformer` import → circular). Proposes `HookedTransformerMixin` to break the cycle.
- **HookedTransformer**: ❌ unchanged — confirmed at [ActivationCache.py:118](transformer_lens/ActivationCache.py#L118) — `self.model = model` with no annotation.
- **TransformerBridge**: ❌ same `ActivationCache` shared class.
- **Replication**: `[code-verified]`
- **What changed since v3**: nothing material.
- **Bucket**: `not-addressed-simple`
- **Next step**: extract a `HookedRootModuleMixin` / use `TYPE_CHECKING + Protocol` to hint without circular imports (~50 LoC). Tagged 3.0 / 4.0 milestone.

<a id="issue-837"></a>

#### #837 — Multi-GPU device ordinal issue (`n_devices=3` for llama2-7b)

- **Issue**: `n_devices=3` produces "device ordinal out of range" — `(index // layers_per_device)` overshoots when `n_layers % n_devices != 0`.
- **HookedTransformer**: 🐛 still buggy at [utilities/multi_gpu.py:142](transformer_lens/utilities/multi_gpu.py#L142) — `device_index = (device.index or 0) + (index // layers_per_device)` unchanged. The function is flagged `Deprecated: This will be removed in 3.0` ([line 130-133](transformer_lens/utilities/multi_gpu.py#L130)).
- **TransformerBridge**: ✅ first-class — `resolve_device_map` at [multi_gpu.py:170](transformer_lens/utilities/multi_gpu.py#L170) with explicit `n_devices` / `device_map` / `max_memory` and accelerate-backed dispatch. jlarson4's comment on the issue points users to PR #1270.
- **Replication**: `[unverifiable]` — no multi-GPU here.
- **What changed since v3**: nothing material; bridge path remains the supported route.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: HT-side bug remains. Reply on issue with bridge migration recipe (`bridge = TransformerBridge.boot_transformers(name, n_devices=3)`); leave HT path open for #968-family fix or close with bridge pointer if reporter migrates.

<a id="issue-867"></a>

#### #867 — Does TransformerLens support LVLM like Qwen2-VL?

- **Issue**: User asks if Qwen2-VL / Qwen2.5-VL is supported.
- **HookedTransformer**: ❌ no native VLM support.
- **TransformerBridge**: ❌ `Qwen2VLForConditionalGeneration` and `Qwen2_5_VLForConditionalGeneration` still listed in [transformer_lens/tools/model_registry/data/architecture_gaps.json:4709,4940](transformer_lens/tools/model_registry/data/architecture_gaps.json#L4709). Multimodal set at [transformer_lens/utilities/architectures.py:31-36](transformer_lens/utilities/architectures.py#L31-L36) covers only Llava family + Gemma3.
- **Replication**: `[code-verified]`
- **What changed since v3**: no movement on Qwen-VL adapters; no new comments since v3 either.
- **Bucket**: `not-addressed-difficult`
- **Next step**: add `Qwen2VLArchitectureAdapter` (LLaVA-pattern). Continue pointing reporter at LLaVA adapters today and ExplorerFreda's vlm-lens fork.

<a id="issue-869"></a>

#### #869 — Custom generative video transformer

- **Issue**: User wants mech interp on a Sora-like generative video diffusion transformer.
- **HookedTransformer**: ❌ no diffusion / video generation support.
- **TransformerBridge**: ❌ bridge wraps HF causal/seq2seq/multimodal text models via `original_model`; not designed for diffusion. No new diffusion entry in [transformer_lens/utilities/architectures.py](transformer_lens/utilities/architectures.py).
- **Replication**: `[code-verified]`
- **What changed since v3**: no activity on issue or relevant code.
- **Bucket**: `not-addressed-difficult`
- **Next step**: outside current scope per Bryce's reply (would need a separate `HookedDiffusionTransformer` root module). Recommend close as wontfix or defer to architectural roadmap; point reporter to a dedicated diffusion-interp tool.

<a id="issue-888"></a>

#### #888 — Adapt HookedTransformer to a non-supported model (CLIP language model)

- **Issue**: User wants `from_pretrained` for CLIP's text encoder.
- **HookedTransformer**: ❌ not possible without code modifications.
- **TransformerBridge**: ⚠️ adapter framework supports it but no `CLIPTextModel` adapter exists — no `CLIPText*` symbol anywhere under `transformer_lens/`. `CLIPVisionEncoderBridge` exists for the vision side via LLaVA. jlarson4's earlier comment already pointed reporter at the adapter-creation guide.
- **Replication**: `[code-verified]`
- **What changed since v3**: no new comments; no CLIP text adapter landed.
- **Bucket**: `not-addressed-difficult`
- **Next step**: write `CLIPTextModelArchitectureAdapter` (~120 LoC, encoder-only, BERT-like attention). Leave open as a focused model-request inviting community contribution.

<a id="issue-953"></a>

#### #953 — Add basic support for Gemma 3n (E2B & E4B)

- **Issue**: Reporter asks for text-only support of Gemma 3n (AltUp / LAuReL / PLE / mixed local-global attention).
- **HookedTransformer**: ❌ not supported.
- **TransformerBridge**: ❌ no Gemma3n entry in [transformer_lens/utilities/architectures.py](transformer_lens/utilities/architectures.py); no Gemma3n symbol anywhere under `transformer_lens/`. Bryce confirmed in-progress for next major release.
- **Replication**: `[code-verified]`
- **What changed since v3**: no movement on Gemma3n adapter.
- **Bucket**: `not-addressed-difficult`
- **Next step**: track for milestone 3.x. AltUp/LAuReL/PLE need dedicated component bridges; mixed local/global attention can share Gemma2 work. Defer until HF's `Gemma3nForCausalLM` forward stabilizes.

<a id="issue-1080"></a>

#### #1080 — Import fails by default in Colab (numpy ABI mismatch)

- **Issue**: Fresh Colab + `pip install transformer_lens` + `import transformer_lens` raises `numpy.dtype size changed` ABI error; kernel restart works around it.
- **HookedTransformer**: ⚠️ [pyproject.toml:11-12](pyproject.toml#L11-L12) still has `numpy>=1.24` / `numpy>=1.26` lower bounds with no upper cap. Numpy 2.x is allowed; transitive ABI mismatch root cause unchanged.
- **TransformerBridge**: ⚠️ same install path; same numpy.
- **Replication**: `[unverifiable]` — Colab-specific.
- **What changed since v3**: no movement on numpy pinning; no new comments.
- **Bucket**: `bug-likely-fixed-needs-verification`
- **Next step**: ask reporter to retest with current Colab kernel + current TL (3.x). If still failing, bisect transitive deps and pin a tested numpy.

<a id="issue-1148"></a>

#### #1148 — Tutorial for "Real-Time Training Dynamics" (VSM Telemetry)

- **Issue**: Reporter proposes a demo notebook for σ_p / σ_a training-dynamics telemetry.
- **HookedTransformer**: ❌ no VSM/sigma_p/sigma_a tutorial in [demos/](demos/) — no VSM symbol anywhere under `demos/` or `transformer_lens/`.
- **TransformerBridge**: ❌ same — works equivalently against bridge's hook system.
- **Replication**: `[code-verified]`
- **What changed since v3**: jonathanrbelanger-lang committed in-thread to "get to work on this over the coming weekend" but no PR yet; no new commits to `demos/` referencing VSM telemetry.
- **Bucket**: `not-addressed-simple`
- **Next step**: leave open and wait for the reporter's PR (notebook in `/demos`, targeting `TransformerBridge`). If no PR materializes within a release cycle, invite community contribution and close as wontfix.

<a id="issue-1263"></a>

#### #1263 — Direct Logit Attribution Tool

- **Issue**: Add a first-class DLA helper in `transformer_lens/tools/analysis/direct_logit_attribution.py` for the new `TransformerBridge` system. Continuation of stale PR #466 (closed 2026-04-22).
- **HookedTransformer**: ⚠️ partial — `ActivationCache.logit_attrs` exists at [transformer_lens/ActivationCache.py:488-606](transformer_lens/ActivationCache.py#L488-L606) but no standalone tool that wraps the full DLA flow (residual decomposition → scaled attribution → display).
- **TransformerBridge**: ⚠️ uses the same `ActivationCache.logit_attrs`, but no dedicated bridge-friendly tool. `transformer_lens/tools/` has only `model_registry/`; no `analysis/` subpackage exists yet.
- **Replication**: `[code-verified]`
- **What changed since v3**: PR #466 was closed (2026-04-22) the same day issue #1263 was opened — explicitly creating the issue as a replacement scope. No PR yet.
- **Bucket**: `not-addressed-simple`
- **Labels**: enhancement / good first issue / help wanted / minor / complexity-moderate
- **Next step**: create `transformer_lens/tools/analysis/direct_logit_attribution.py` wrapping `cache.logit_attrs` + residual-stack decomposition into a one-call API; ship with a demo notebook. Already labelled `good first issue` — invite contributor.

<a id="issue-1280"></a>

#### #1280 — Add support for `cpu`, `meta`, and `disk` to TransformerBridge `device_map`

- **Issue**: Extend bridge `device_map` to allow `cpu` / `meta` / `disk` values. Currently rejected. Pairs with #872 (broader review) and #1270 (initial multi-device).
- **HookedTransformer**: N/A — separate device-placement model.
- **TransformerBridge**: 🐛 still rejected by design at [transformer_lens/utilities/multi_gpu.py:146-167](transformer_lens/utilities/multi_gpu.py#L146-L167): `_UNSUPPORTED_DEVICE_MAP_VALUES = {"cpu", "disk", "meta"}` validated in `_validate_device_map_values`, also blocked post-load at [transformer_lens/model_bridge/sources/transformers.py:559-566](transformer_lens/model_bridge/sources/transformers.py#L559-L566). Reporter's identified blocker is the dtype-cast loop.
- **Replication**: `[code-verified]`
- **What changed since v3**: snakefood3232 volunteered with a 3-day PR estimate (skip meta-device params, use accelerate's `align_module_device`); jlarson4 assigned them on 2026-05-05.
- **Bucket**: `partial-leave-open`
- **Next step**: wait for snakefood3232's PR — concrete fix plan documented. Reviewer: relax `_UNSUPPORTED_DEVICE_MAP_VALUES`, gate the dtype-cast loop on `param.device.type != "meta"`, and exercise via integration test that loads a small model with `device_map={"": "cpu"}`.

<a id="issue-1291"></a>

#### #1291 — CI HuggingFace Call Reduction

- **Issue**: CI optimization — reduce HF Hub round-trips during test runs to avoid 429 rate-limit failures across concurrent CI runs.
- **HookedTransformer**: ⚠️ partial — [.github/workflows/checks.yml:65-88,246-269](.github/workflows/checks.yml#L65-L88) caches ~14 model dirs across `compatibility-checks` and `coverage-test`, but no `concurrency` group is configured anywhere in the workflow; many tests still call `from_pretrained` per-test rather than via session fixtures.
- **TransformerBridge**: ⚠️ same — bridge tests under `tests/integration/model_bridge/` and `tests/acceptance/model_bridge/` share the cache but each conftest re-loads HF models.
- **Replication**: `[code-verified]`
- **What changed since v3**: ak91456 volunteered on 2026-05-09; no PR yet. Cache key bumped to `huggingface-models-v4` recently but core fixture/concurrency work hasn't started.
- **Bucket**: `partial-leave-open`
- **Labels**: enhancement / good first issue / low-priority / complexity-moderate
- **Next step**: wait for ak91456's PR. Suggested approach: (a) add `concurrency: { group: ${{ github.workflow }}-${{ github.ref }}, cancel-in-progress: true }` to `checks.yml` to dedupe stacked runs; (b) promote per-file `from_pretrained("gpt2")` calls in conftests to session-scoped fixtures.

<a id="issue-1297"></a>

#### #1297 — Gemma4 Architecture Adapter

- **Issue**: Add a `Gemma4ArchitectureAdapter` for the new Gemma4 family. Currently surfaces in `architecture_gaps.json` with relevancy 88.0 (109 models on HF, 121k cumulative downloads).
- **HookedTransformer**: N/A — bridge-only path going forward; no HT weight conversion expected.
- **TransformerBridge**: ❌ no `Gemma4ArchitectureAdapter` in [transformer_lens/model_bridge/supported_architectures/](transformer_lens/model_bridge/supported_architectures/); `Gemma4ForConditionalGeneration` not registered in [factories/architecture_adapter_factory.py](transformer_lens/factories/architecture_adapter_factory.py) or `HF_SUPPORTED_ARCHITECTURES`.
- **Replication**: `[code-verified]` — confirmed adapter and registration entries are absent.
- **Bucket**: `not-addressed-simple`
- **Next step**: copy `gemma3.py` adapter as starting template (Gemma4 is most likely a Gemma3 superset); register in factory + `HF_SUPPORTED_ARCHITECTURES` + `CANONICAL_AUTHORS_BY_ARCH` (`google`); follow `docs/source/content/adapter_development/adapter-creation-guide.md`. Then verify on the canonical Google models.

<a id="issue-1298"></a>

#### #1298 — External Architecture Registration

- **Issue**: Let users register custom architecture adapters at runtime without modifying TransformerLens source. Currently `SUPPORTED_ARCHITECTURES` in `architecture_adapter_factory.py` is hardcoded.
- **HookedTransformer**: N/A — bridge-only concept (HT loads via `OFFICIAL_MODEL_NAMES`, no plugin hook).
- **TransformerBridge**: ❌ no public registration API. The `SUPPORTED_ARCHITECTURES` dict at [factories/architecture_adapter_factory.py:65](transformer_lens/factories/architecture_adapter_factory.py#L65) is module-level and not user-mutable through any documented mechanism.
- **Replication**: `[code-verified]` — no `register_adapter` function or plugin entry-point hook.
- **Bucket**: `not-addressed-simple`
- **Next step**: design needed first — entry-point-based discovery vs. explicit `register_adapter(arch_name, adapter_class)` function. Adapter-creation-guide already exists, so the second-half (publishing your adapter) is the remaining gap.

