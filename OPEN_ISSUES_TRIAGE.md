# Open Issues Triage (v5)

**Generated:** 2026-05-12 (v5 — refreshed after this branch's closures + a separate 2026-05-13 sprint)
**Repo:** TransformerLensOrg/TransformerLens
**Open issues:** 25 (24 surviving from v4 + 1 newly opened)
**Previous archives:** [OPEN_ISSUES_TRIAGE.v4.md](OPEN_ISSUES_TRIAGE.v4.md), [OPEN_ISSUES_TRIAGE.v3.md](OPEN_ISSUES_TRIAGE.v3.md), [OPEN_ISSUES_TRIAGE_OLD.md](OPEN_ISSUES_TRIAGE_OLD.md) (v2)

## What changed since v4

- **14 issues closed**: #112, #210, #297, #341, #385, #453, #462, #483, #588, #615, #644, #720, #796, #830
  - Closed in this branch (`issues/may-12-cleanup`): #210, #297, #341, #385, #453, #615, #644, #796
  - Closed in the 2026-05-13 sprint (other branches): #112, #720, #830
  - v4 already flagged closeable: #462, #483, #588
- **1 new entry**: #1302 (additional architecture adapter tests, opened by jlarson4) — maps to `not-addressed-simple`
- **24 entries re-verified** against current code; no verdict refinements needed (all v4 verdicts still hold for surviving issues)

### Newly closeable based on v5 re-verification

None — v4's predictions all landed; nothing new flagged as ready-to-close in this pass.

### Verdict refinements (still open, context updated)

- **#543** (Grokking demo broken in Colab) — root cause confirmed during this branch's session: `loss_fn(all_logits, labels)` uses the shape-rearranged `all_logits` (113×113×113) with the flat `labels` (12769), causing the documented gather mismatch. Fix is a one-token rename (`all_logits` → `original_logits`) plus checkpoint-CPU offload for the memory tail. Bridge migration is N/A for this demo (custom-config training from scratch — outside bridge's HF-wrapping design space). Bucket stays `bug-likely-fixed-needs-verification` pending the actual patch.

The v2 methodology section (HT-side / Bridge-side / Replication / Next step) still applies — see [OPEN_ISSUES_TRIAGE_OLD.md](OPEN_ISSUES_TRIAGE_OLD.md#methodology-per-issue).

## Summary table (sorted by issue number)

| Issue | Title | Bucket |
|---|---|---|
| #111 | [Demo of direct path patching](#issue-111) | `not-addressed-difficult` |
| #479 | [Memory efficient causal mask implementation](#issue-479) | `partial-leave-open` |
| #481 | [Tracr to TransformerLens demo broken](#issue-481) | `bug-still-reproduces` |
| #509 | [LayerNorm folding not implemented for BertBlock](#issue-509) | `not-addressed-difficult` |
| #543 | [Grokking demo broken in Colab](#issue-543) | `bug-likely-fixed-needs-verification` |
| #595 | [Add Stopping Criteria support](#issue-595) | `not-addressed-simple` |
| #697 | [Activation cache during generate](#issue-697) | `not-addressed-simple` |
| #704 | [Add support for TracrBench](#issue-704) | `not-relevant-close` |
| #710 | [MVP Support For 1-2 Models Per-Modality](#issue-710) | `not-addressed-difficult` |
| #737 | [Q reshape with model loaded in 4bit](#issue-737) | `partial-leave-open` |
| #773 | [TransformerLens on models with different layernorm placement (BioGPT)](#issue-773) | `not-addressed-difficult` |
| #798 | [Remove `model_args` (use only `model_kwargs`)](#issue-798) | `not-addressed-simple` |
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
| #1302 | [Additional Architecture Adapter tests](#issue-1302) | `not-addressed-simple` |

## Per-issue entries

<a id="issue-111"></a>

#### #111 — Demo of direct path patching

- **Issue**: Add a section to Exploratory Analysis Demo demonstrating direct path patching for all head pairs. PR #49 was an early attempt.
- **HookedTransformer**: still no first-class path-patching helper. Verified — no `path_patch`/`direct_path` symbols exist anywhere under [transformer_lens/](transformer_lens/) or [transformer_lens/utilities/](transformer_lens/utilities/). [demos/Activation_Patching_in_TL_Demo.ipynb](demos/Activation_Patching_in_TL_Demo.ipynb) and [demos/Attribution_Patching_Demo.ipynb](demos/Attribution_Patching_Demo.ipynb) are the closest.
- **TransformerBridge**: same — no path-patching primitive in either API; bridge reuses the same `ActivationCache`.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: callum mcdougall pointed users at the [ARENA IOI notebook](https://colab.research.google.com/drive/1KgrEwvCKdX-8DQ1uSiIuxwIiwzJuQ3Gw). Either close with a docs pointer to ARENA, or implement a TL helper that wraps the pattern (~80 LoC).

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

<a id="issue-595"></a>

#### #595 — Add Stopping Criteria support

- **Issue**: HF offers `StoppingCriteria` for custom halt conditions; HT/bridge `generate()` only support `stop_at_eos`.
- **HookedTransformer**: ❌ unchanged — [transformer_lens/HookedTransformer.py:1882](transformer_lens/HookedTransformer.py#L1882) `generate()` and `generate_stream()` (line 2262) still only take `stop_at_eos: bool`.
- **TransformerBridge**: ❌ unchanged — [transformer_lens/model_bridge/bridge.py:2438](transformer_lens/model_bridge/bridge.py#L2438) `generate()` and `generate_stream()` (line 2754) only have `stop_at_eos`.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~30 LoC — add `stopping_criteria: Optional[Callable[[tokens, logits], bool]] = None` to all four entry points; evaluate after each sampled token and break if any returns True. srishti-git1110 volunteered in 2024.

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

<a id="issue-798"></a>

#### #798 — Remove `model_args` (use only `model_kwargs`)

- **Issue**: Bryce's own proposal to remove `*model_args` + `**model_kwargs` redundancy in pass-through functions.
- **HookedTransformer**: ⚠️ unchanged — `model_args` still present in [HookedEncoderDecoder.py:489-513](transformer_lens/HookedEncoderDecoder.py#L489), [hook_points.py:629,723,779](transformer_lens/hook_points.py#L629), [HookedAudioEncoder.py:299-323](transformer_lens/HookedAudioEncoder.py#L299), [BertNextSentencePrediction.py:220-266](transformer_lens/BertNextSentencePrediction.py#L220), [HookedTransformer.py:707-735](transformer_lens/HookedTransformer.py#L707).
- **TransformerBridge**: ⚠️ same — bridge inherits `hook_points.py` machinery.
- **Replication**: `[code-verified]`
- **What changed since v3**: nothing material; no new comments.
- **Bucket**: `not-addressed-simple`
- **Next step**: ~30 LoC across affected files — strip `*model_args`, keep only `**model_kwargs`. Already labeled `breaking-change`.

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


<a id="issue-1302"></a>

#### #1302 — Additional Architecture Adapter tests

- **Issue**: Roughly a third of registered architecture adapters have dedicated config/component-mapping tests in [tests/unit/model_bridge/supported_architectures/](tests/unit/model_bridge/supported_architectures/); the rest lack focused coverage. Existing tests (baichuan / codegen / cohere / gemma3 / gpt_bigcode / internlm2 / llava / mpt / qwen3_5 / qwen3_moe / qwen3_next / xglm / gemma3_multimodal) serve as the pattern to mirror.
- **HookedTransformer**: N/A — bridge-only concern.
- **TransformerBridge**: ❌ partial coverage. 13 adapter test files exist (as of this branch); remaining adapters under [transformer_lens/model_bridge/supported_architectures/](transformer_lens/model_bridge/supported_architectures/) are uncovered.
- **Replication**: `[code-verified]` — counted adapters in `supported_architectures/` vs. test files; gap confirmed.
- **Bucket**: `not-addressed-simple`
- **Labels**: enhancement / good first issue / help wanted / low-priority / complexity-simple / TransformerBridge
- **Next step**: identify the uncovered adapters, then mirror the pattern from any of the existing 13 — Config / ComponentMapping / WeightConversions / ArchitectureGuards classes with class-scoped fixtures. Each adapter is independent, so this parallelizes well across contributors.
