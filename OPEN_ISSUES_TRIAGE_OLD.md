# Open Issues Triage (v2)

**Generated:** 2026-04-29 (v2 — code-level verification pass)
**Repo:** TransformerLensOrg/TransformerLens
**Branch reference:** `dev`
**Open issues at v2 start:** 83 (down from 91 at v1; 9 closed during v1 cycle)
**v1 archived at:** [OPEN_ISSUES_TRIAGE.v1.md](OPEN_ISSUES_TRIAGE.v1.md)

## Why v2

The v1 triage was based on issue text alone — bodies + comments — without verifying claims against current source. Multiple corrections during the post-v1 review (#671, #846, #867, #929, #657, #219, #264) revealed the same pattern: the issue's framing didn't match what the code actually does today, in either direction (false positives where bugs were already fixed, false negatives where the bug was still real but my reason was wrong).

v2 corrects this by treating each issue as an investigation, not a reading-comprehension exercise.

## Methodology per issue

Every entry includes:

1. **HookedTransformer side** — does the buggy/missing code path still exist? `grep`/`Read` for the actual file, function, line referenced (or implied by the issue). `git log --all -S '<key string>' -- <file>` for any commits that touched it. `git log --all --grep "Fixes #N\|Closes #N\|#N\b"` for landed fixes.
2. **TransformerBridge side** — does the bug apply to the bridge's code path? The bridge wraps HF directly via `original_model`, has its own loading path (`sources/transformers.py`), uses HF's attention/PosEmbed/RMSNorm components, and has its own hook system. Many HT-specific bugs don't apply.
3. **Replication evidence** — one of:
   - `[empirically replicated]` — ran a minimal repro on this machine; bug observed
   - `[empirically not reproduced]` — ran repro; bug does not occur
   - `[code-verified]` — read the source; the buggy code path either exists or has been fixed/removed
   - `[unverifiable on this machine]` — needs hardware/environment we don't have (multi-GPU, large models, MPS, Colab, 4bit, etc.)
4. **Next step** — concrete action: close with reference, fix with file:line, migrate to bridge with recipe, ask reporter for repro, or defer with specific blocker.

## Summary by bucket

_Filled in incrementally as batches complete. Counts are over the 83 currently-open issues._

### Batch 1 only (20 issues)

| Bucket | Count |
|---|---|
| covered-close | 1 |
| partial-leave-open | 5 |
| not-addressed-simple | 5 |
| not-addressed-difficult | 2 |
| not-relevant-close | 0 |
| bug-still-reproduces | 1 |
| fixed-on-transformerbridge | 5 |
| bug-likely-fixed-needs-verification | 1 |
| **Total batch 1** | **20** |

### Batch 2 only (20 issues)

| Bucket | Count |
|---|---|
| covered-close | 0 |
| partial-leave-open | 3 |
| not-addressed-simple | 3 |
| not-addressed-difficult | 2 |
| not-relevant-close | 1 |
| bug-still-reproduces | 4 |
| fixed-on-transformerbridge | 3 |
| bug-likely-fixed-needs-verification | 1 |
| question-not-actionable | 3 |
| **Total batch 2** | **20** |

### Batch 3 only (20 issues)

| Bucket | Count |
|---|---|
| covered-close | 0 |
| partial-leave-open | 2 |
| not-addressed-simple | 4 |
| not-addressed-difficult | 4 |
| not-relevant-close | 0 |
| bug-still-reproduces | 0 |
| fixed-on-transformerbridge | 7 |
| bug-likely-fixed-needs-verification | 2 |
| question-not-actionable | 1 |
| **Total batch 3** | **20** |

### Batch 4 only (20 issues)

| Bucket | Count |
|---|---|
| covered-close | 2 |
| partial-leave-open | 2 |
| not-addressed-simple | 1 |
| not-addressed-difficult | 1 |
| not-relevant-close | 1 |
| bug-still-reproduces | 1 |
| fixed-on-transformerbridge | 8 |
| bug-likely-fixed-needs-verification | 3 |
| question-not-actionable | 1 |
| **Total batch 4** | **20** |

### Batch 5 only (1 issue)

| Bucket | Count |
|---|---|
| question-not-actionable | 1 |
| **Total batch 5** | **1** |

### Cumulative (batches 1 + 2 + 3 + 4 + 5 = 81 issues)

| Bucket | Count | Recommended action |
|---|---|---|
| covered-close | 3 | Close (3) |
| partial-leave-open | 12 | Leave open with scope note (12) |
| not-addressed-simple | 13 | Leave open / `good-first-issue` (13) |
| not-addressed-difficult | 9 | Leave open (9) |
| not-relevant-close | 2 | Close (2) |
| bug-still-reproduces | 6 | Leave open (6) |
| fixed-on-transformerbridge | 23 | Comment with bridge migration recipe (23) |
| bug-likely-fixed-needs-verification | 7 | Ask for repro (7) |
| question-not-actionable | 6 | Close with docs pointer (6) |
| **Cumulative total** | **81 of 83** | **Close: 11 / Leave open: 70** |

_(#1263 and #1264 are tracking issues opened by the maintainer; not triaged.)_

### Per-issue summary table

Status legend: ✅ resolved · ⚠️ partial · ❌ not addressed · 🐛 bug reproduces · N/A — feature/concept doesn't apply on that side.

#### Batch 1 (20 issues)

| # | Issue | Bucket | HookedTransformer | TransformerBridge | Replication |
|---|---|---|---|---|---|
| #97 | [Better docs for model_properties_table](#issue-97) | partial-leave-open | ⚠️ auto-table covers arch cols; training metadata missing | ⚠️ same auto-table | code-verified |
| #99 | [Tests + docs for ActivationCache](#issue-99) | not-addressed-simple | ⚠️ tests exist; docstring order bug remains | ⚠️ same (shared class) | code-verified |
| #100 | [Tests + docs for tokenization](#issue-100) | partial-leave-open | ⚠️ extensive tests; prepend_bos clarity gap | ⚠️ same | code-verified |
| #104 | [Mixed precision (fp16/bf16)](#issue-104) | fixed-on-transformerbridge | ⚠️ per-arch precision quirks remain | ✅ HF-native; no TL-specific NaN paths | empirical |
| #107 | [HF evals helper](#issue-107) | not-addressed-difficult | ❌ no lm-eval-harness adapter | ⚠️ users can extract `original_model` and pass to lm-eval-harness | code-verified |
| #111 | [Direct path patching demo](#issue-111) | not-addressed-difficult | ❌ no first-class helper; ARENA notebook only | ❌ same | code-verified |
| #112 | [Logit display helper](#issue-112) | not-addressed-simple | ⚠️ `test_prompt` print only; no DataFrame, no logit-lens heatmap | ⚠️ same | code-verified |
| #207 | [Hook AssertionError in Attribution Patching demo](#issue-207) | covered-close | ⚠️ broad-pattern add_hook still asserts (separate UX issue) | ✅ demo rewritten (PR #1013) + smarter filter | empirical |
| #210 | [`get_full_resid_decomposition` tensor arg](#issue-210) | not-addressed-simple | ❌ kwarg not added | ❌ same (shared ActivationCache) | code-verified |
| #264 | [GatedMLP not in docs](#issue-264) | partial-leave-open | ❌ smoke tests only; no class docstring; no config field docstring | ⚠️ better class docs; indirect adapter tests; no parity test; no config field docstring | code-verified |
| #277 | [BERT future work](#issue-277) | partial-leave-open | ⚠️ MaskedLM + 4 model sizes shipped; NSP/training/LN-fold missing | ⚠️ same architecture coverage | code-verified |
| #290 | [GPU memory leak](#issue-290) | partial-leave-open | ⚠️ major fixes landed (PR #1229); residual retention plausible | ✅ delegates to HF; no TL-specific circular refs | unverifiable |
| #297 | [Print attached hooks](#issue-297) | not-addressed-simple | ❌ no `list_hooks()` helper; `hook_dict` raw-accessible | ❌ same hook_points machinery | code-verified |
| #335 | [LN1 hooks fire 3× per forward](#issue-335) | fixed-on-transformerbridge | 🐛 reproduces (`transformer_block.py:172,174,176`) | ✅ HF attention; fires once | empirical |
| #341 | [`FactoredMatrix.svd` deprecated `torch.svd`](#issue-341) | not-addressed-simple | ❌ uses `torch.svd`, returns V not Vh | ❌ same `FactoredMatrix` | code-verified |
| #378 | [Flash attention support](#issue-378) | fixed-on-transformerbridge | ❌ no SDPA flag; hand-rolled einsum | ✅ HF `attn_implementation="sdpa"`/`"flash_attention_2"` | code-verified |
| #385 | [Pythia rotary mismatch vs HF](#issue-385) | fixed-on-transformerbridge | 🐛 NaN logits in fp32 baseline (possible regression) | ✅ uses HF rotary directly | empirical |
| #448 | [`n_params` way off](#issue-448) | bug-still-reproduces | 🐛 gpt2-small reports 84M; actual 163M | 🐛 shares calculation path | empirical |
| #453 | [`checkpoint_label` returns same weights](#issue-453) | bug-likely-fixed-needs-verification | N/A — `checkpoint_label` is not a parameter (user error; kwarg silently swallowed) | N/A — no checkpoint feature | code-verified |
| #462 | [Mamba support](#issue-462) | fixed-on-transformerbridge | ❌ not supported (by-design) | ✅ `MambaArchitectureAdapter` + `Mamba2ArchitectureAdapter` registered | code-verified |

#### Batch 2 (20 issues)

| # | Issue | Bucket | HookedTransformer | TransformerBridge | Replication |
|---|---|---|---|---|---|
| #479 | [Memory-efficient causal mask](#issue-479) | partial-leave-open | 🐛 `(n_ctx, n_ctx)` buffer per layer; ~86 GB at Qwen 72B scale | ⚠️ architecture-dependent: GPT2 inherits same buffer; Llama/Pythia/Qwen/etc. dynamic | empirical |
| #481 | [Tracr demo broken](#issue-481) | bug-still-reproduces | 🐛 `np.eye(d_model, d_vocab_out)` unembed assumption still in demo | 🐛 same — Tracr-specific, demo not bridge-ported | code-verified |
| #483 | [`generate()` no-tokenizer fail](#issue-483) | bug-still-reproduces | 🐛 reproduces — assumes `self.tokenizer` exists | 🐛 same — bridge generate also reads `self.tokenizer` | empirical |
| #502 | [VLM support question](#issue-502) | question-not-actionable | ❌ no native VLM | ⚠️ LLaVA family supported; BLIP-VQA not | code-verified |
| #509 | [BERT LN folding](#issue-509) | not-addressed-difficult | ❌ post-LN architecture; Neel: not foldable cleanly | ❌ same architectural limit | code-verified |
| #515 | [`IOIDataset` duplicate entries](#issue-515) | bug-still-reproduces | 🐛 `random.seed(42)` at top of `get_sample` (evals.py:387) | 🐛 same module | empirical |
| #523 | [Residual stack not adding up](#issue-523) | question-not-actionable | N/A — user error (forgot LN gain/bias); resolved in thread | N/A — same | code-verified |
| #543 | [Grokking demo broken](#issue-543) | bug-likely-fixed-needs-verification | ⚠️ multiple post-issue commits; needs fresh Colab repro | ⚠️ same | unverifiable |
| #569 | [Llama-3-70B 4bit multi-GPU](#issue-569) | fixed-on-transformerbridge | 🐛 BnB-packed weights fail QKV reshape | ✅ skips state_dict reshape; structurally sound, end-to-end unverified | unverifiable |
| #588 | [Tests for model configs](#issue-588) | partial-leave-open | ⚠️ tests exist for ~3 configs; not all 185 | ⚠️ structural-mapping tests for ~15 architectures | code-verified |
| #595 | [Stopping Criteria support](#issue-595) | not-addressed-simple | ❌ only `stop_at_eos`; no callable | ❌ same | code-verified |
| #615 | [HT ≠ HF for Llama 3](#issue-615) | fixed-on-transformerbridge | ⚠️ post-einsum-cleanup max diff ~2e-4 (degenfabian); residual reports persist | ⚠️ bridge has its own attention reconstruction; ~2.5e-3 max drift vs HF (argmax matches) | empirical |
| #644 | [Map act names to transformer](#issue-644) | not-addressed-simple | ❌ no diagram in `model_structure.md` | ❌ same docs source | code-verified |
| #661 | [Pythia split_qkv batch consistency](#issue-661) | bug-still-reproduces | 🐛 max diff `1.14e-02` between batch[:1] and batch[:2] | N/A — bridge has no `use_split_qkv_input` flag | empirical |
| #684 | [Quantization beyond Llama](#issue-684) | fixed-on-transformerbridge | 🐛 hard-coded "Llama only" assertion | ✅ no architecture assertion; structurally sound for non-Llama quantized, end-to-end unverified | code-verified |
| #696 | [Cached LN scale factors meaning](#issue-696) | question-not-actionable | N/A — conceptual Q; Neel answered | N/A — same | code-verified |
| #697 | [Activation cache during generate](#issue-697) | not-addressed-simple | ❌ no `run_with_cache` integration in `generate()` | ❌ same | code-verified |
| #704 | [TracrBench support](#issue-704) | not-relevant-close | ❌ not in core | ❌ not in core | code-verified |
| #710 | [MVP per-modality support](#issue-710) | not-addressed-difficult | ❌ no non-text models | ⚠️ Hubert (audio), LLaVA family (VLM); no Whisper/ResNet/diffusion | code-verified |
| #720 | [Matmul function audit](#issue-720) | partial-leave-open | ⚠️ partial — `F.linear` cleanup landed for some paths; full audit pending | ⚠️ Q/K/V projections use HF Linear (correct); attention-score `torch.matmul` and output `torch.matmul` are bridge code (own audit needed) | code-verified |

#### Batch 3 (20 issues)

| # | Issue | Bucket | HookedTransformer | TransformerBridge | Replication |
|---|---|---|---|---|---|
| #729 | [Guide to adding new models](#issue-729) | not-addressed-simple | ❌ no how-to-extend doc | ❌ same; bridge has cleaner extension primitive but no walkthrough | code-verified |
| #737 | [Q reshape in 4bit](#issue-737) | partial-leave-open | 🐛 4bit + `use_split_qkv_input` shape mismatch | N/A — bridge has no `use_split_qkv_input` flag | unverifiable |
| #754 | [Don't load HF when config passed](#issue-754) | fixed-on-transformerbridge | 🐛 `convert_hf_model_config` calls `AutoConfig.from_pretrained` unconditionally | ✅ `boot_transformers(hf_model=...)` skips AutoConfig | code-verified |
| #773 | [BioGPT-style LN placement](#issue-773) | not-addressed-difficult | ❌ hard-coded GPT-2 LN placement | ⚠️ adapter framework supports custom LN layout; no BioGPT adapter exists | code-verified |
| #778 | [Gemma2 attn order wrong](#issue-778) | fixed-on-transformerbridge | 🐛 `[global, local, ...]` (inverted from HF's `[local, global, ...]`) | ✅ uses HF `layer_types` directly | empirical |
| #784 | [Smaller precision OOM](#issue-784) | fixed-on-transformerbridge | ⚠️ source state_dict + working copy duplicates memory at load | ✅ single allocation; bf16 fits on 6GB | unverifiable |
| #796 | [`FactoredMatrix.svd` lru_cache GC](#issue-796) | not-addressed-simple | 🐛 `@lru_cache` holds instance refs | 🐛 same `FactoredMatrix` | code-verified |
| #798 | [Remove `model_args`](#issue-798) | not-addressed-simple | ⚠️ `*model_args` + `**model_kwargs` in encoder/hook_points | ⚠️ same hook_points machinery | code-verified |
| #800 | [Offline GPT2-xl load fails](#issue-800) | fixed-on-transformerbridge | 🐛 same root cause as #754 | ✅ `boot_transformers(hf_model=...)` works offline | code-verified |
| #801 | [Padding side mismatch (Gemma 2)](#issue-801) | bug-likely-fixed-needs-verification | ⚠️ original repro was on TL 2.9.0; current dev shows `'left'` for both | ✅ inherits HF tokenizer | empirical |
| #830 | [Type hint for `ActivationCache.model`](#issue-830) | not-addressed-simple | ❌ untyped to avoid circular import | ❌ same shared class | code-verified |
| #837 | [Multi-GPU device ordinal off-by-one](#issue-837) | fixed-on-transformerbridge | 🐛 same family as #907/#911/#968 | ✅ pre-loaded `hf_model` w/ accelerate works on dev; first-class via PR #1270 | unverifiable |
| #846 | [Local `hf_model.config` priority for Qwen](#issue-846) | fixed-on-transformerbridge | 🐛 same root cause as #754 | ✅ `boot_transformers(hf_model=...)` works | code-verified |
| #858 | [gemma-7b-it OOM on 2× H100](#issue-858) | fixed-on-transformerbridge | ⚠️ duplicate-allocation pattern + multi-GPU placement bugs | ✅ pre-loaded `hf_model` with `device_map="auto"` should fit | unverifiable |
| #867 | [Qwen2-VL support](#issue-867) | not-addressed-difficult | ❌ no VLM | ❌ no `Qwen2VLForConditionalGeneration` adapter — different from LLaVA family | code-verified |
| #869 | [Custom video transformer](#issue-869) | not-addressed-difficult | ❌ not designed for diffusion/video | ❌ same | code-verified |
| #872 | [Official `device_map` support](#issue-872) | partial-leave-open | 🐛 `n_devices` has placement bugs (#837 family) | ⚠️ pre-loaded `hf_model=` works on dev; PR #1270 makes it first-class but not yet merged | unverifiable |
| #873 | [Llama2-7b-chat-hf load fail](#issue-873) | bug-likely-fixed-needs-verification | ⚠️ ambiguous error in body (only screenshots); many Llama issues fixed since | ⚠️ same — needs reporter retest | unverifiable |
| #878 | [Layer-wise caching for OOM](#issue-878) | question-not-actionable | N/A — usage Q for attribution patching memory | N/A — same | unverifiable |
| #888 | [Adapt to non-supported model (CLIP language)](#issue-888) | not-addressed-difficult | ❌ no extension mechanism | ⚠️ adapter framework supports it; no `CLIPTextModelArchitectureAdapter` exists | code-verified |

#### Batch 4 (20 issues)

| # | Issue | Bucket | HookedTransformer | TransformerBridge | Replication |
|---|---|---|---|---|---|
| #894 | [Implement LongRoPE](#issue-894) | fixed-on-transformerbridge | ❌ only `llama3`/`yarn` rope_type branches; no `longrope` | ✅ delegates rope to HF; LongRoPE works natively for Phi-3.5/Phi-4-mini | code-verified |
| #902 | [NaN weights when initializing](#issue-902) | bug-likely-fixed-needs-verification | ⚠️ original repro on TL 2.15.0; current dev not yet retested | ✅ uses HF native init, not TL's `_init_weights_*` paths | unverifiable |
| #903 | [gpt2-small `n_params` reports 85M](#issue-903) | bug-still-reproduces | 🐛 same calc as #448; embeddings/unembed excluded | 🐛 shares `HookedTransformerConfig` calc | code-verified |
| #904 | [Gemma fold_value_biases device mix](#issue-904) | fixed-on-transformerbridge | 🐛 `b_O + (b_V * W_O).sum(...)` w/o `.to(device)` | ✅ no fold_value_biases by default; HF device_map respected | unverifiable |
| #907 | [PR #864 device-selection breaks multi-GPU](#issue-907) | fixed-on-transformerbridge | 🐛 greedy memory placement scatters sequential blocks | ✅ HF accelerate via `hf_model=`; PR #1270 makes it first-class | unverifiable |
| #909 | [Documentation for hookpoints](#issue-909) | covered-close | ✅ `model_structure.md` documents legacy aliases too | ✅ same doc covers canonical names | code-verified |
| #911 | [PosEmbed device error with accelerate](#issue-911) | fixed-on-transformerbridge | 🐛 `W_pos[offset_position_ids]` cross-device under DDP | ✅ uses HF's `wpe` directly via `EmbeddingBridge` | unverifiable |
| #912 | [Support mT5](#issue-912) | partial-leave-open | ❌ T5-only path | ⚠️ `MT5ForConditionalGeneration` in SUPPORTED_ARCHITECTURES; `model_type="mt5"` not in `model_type_mappings` | code-verified |
| #923 | [Pythia missing `hook_resid_mid`](#issue-923) | not-relevant-close | N/A — by-design (parallel attn+MLP) | N/A — same | code-verified |
| #929 | [Load custom small GPT-2 with hf_model](#issue-929) | fixed-on-transformerbridge | 🐛 same root cause as #754 (AutoConfig refetch) | ✅ `boot_transformers(hf_model=...)` reads user's config | code-verified |
| #930 | [Quantized Llama 3.2 fails to load](#issue-930) | fixed-on-transformerbridge | 🐛 BnB-packed shape mismatch (same as #569) | ✅ skips state_dict reshape; structurally sound, end-to-end unverified | unverifiable |
| #950 | [Support SimpleStories](#issue-950) | partial-leave-open | ❌ not registered | ⚠️ fine-tunes registered (1.25M, 35M); base SimpleStories not yet | code-verified |
| #953 | [Gemma 3n (E2B & E4B)](#issue-953) | not-addressed-difficult | ❌ not supported | ❌ not registered; AltUp/LAuReL/PLE need dedicated component bridges | code-verified |
| #962 | [Multiple GPU support question](#issue-962) | question-not-actionable | ⚠️ `n_devices=N` works (with multi-GPU bug cluster caveats) | ⚠️ `hf_model=` w/ `device_map="auto"` works; PR #1270 first-class | code-verified |
| #968 | [unsloth/llama-3.2-3b 2× 3060 indices error](#issue-968) | bug-likely-fixed-needs-verification | 🐛 multi-GPU bug cluster (#837/#907/#911) | ⚠️ jlarson4 commented offering bridge + #1270 path | unverifiable |
| #993 | [Load compressed Llama/Qwen](#issue-993) | fixed-on-transformerbridge | 🐛 hard-coded "Llama only" + reshape assumes unpacked weights | ✅ no architecture assertion; HF parameter passthrough | code-verified |
| #1039 | [Loading models from local files](#issue-1039) | fixed-on-transformerbridge | 🐛 same root cause as #754/#800 (AutoConfig refetch) | ✅ `boot_transformers(hf_model=...)` works offline | code-verified |
| #1080 | [Colab import fails (numpy ABI)](#issue-1080) | bug-likely-fixed-needs-verification | ⚠️ TL itself doesn't pin numpy<2; transitive ABI mismatch | ⚠️ same install path | unverifiable |
| #1133 | [`tokenize_and_concatenate` cuts tokens](#issue-1133) | covered-close | ✅ PR #1273 (`ad8e123b`) replaces char-chunking with per-doc tokenization | ✅ shared utility | code-verified |
| #1148 | [VSM Telemetry tutorial](#issue-1148) | not-addressed-simple | ❌ no tutorial | ❌ same; bridge hooks would work equivalently | code-verified |

#### Batch 5 (1 issue)

| # | Issue | Bucket | HookedTransformer | TransformerBridge | Replication |
|---|---|---|---|---|---|
| #1165 | [Yoruba tokenization fragmentation](#issue-1165) | question-not-actionable | ❌ no span-pooling helper | ❌ same | code-verified |

## Investigation entries

Entries are grouped chronologically by issue number, batch by batch.

---

### Batch 1: oldest 20 still-open issues

_(complete — sign-off requested before batch 2)_

<a id="issue-97"></a>

#### #97 — Better docs for model properties

- **Issue**: Improve `model_properties_table` — better column key, parallel-attn flag, positional-embed type, training metadata (dataset, dropout, weight decay).
- **HookedTransformer**: `docs/source/generated/model_properties_table.{csv,jsonl,md,html}` exists with comprehensive auto-generated content covering `parallel_attn_mlp`, `positional_embedding_type`, `original_architecture`, `normalization_type`, `gated_mlp`, `n_params`, etc. Auto-regen is wired up via `docs/make_docs.py`.
- **TransformerBridge**: same auto-table covers bridge architectures via `transformer_lens/tools/model_registry/`.
- **Replication**: `[code-verified]` — table exists and appears regenerated periodically; covers most architectural columns Neel listed.
- **What's missing**: qualitative training metadata (dataset name, training tokens, dropout, weight decay) is not in the table. That info isn't on HF configs so it would need a hand-curated supplementary table.
- **Bucket**: `partial-leave-open`
- **Next step**: small docs task — add a `training_metadata.csv` (manually curated, ~50 rows) that the table-build joins against. Or close as practically-resolved by the auto-table and open a new issue for the training-metadata supplement.

<a id="issue-99"></a>

#### #99 — Add tests + better docs for ActivationCache

- **Issue**: Add tests for `ActivationCache` methods. Comment-thread bug from `andylolu2`: `get_full_resid_decomposition` docstring says component order is `[embed, pos_embed, heads, neurons, biases]` but actual stacking order is `[*heads, *neurons, embed, pos_embed, biases]` — leading to silent unpacking errors.
- **HookedTransformer**: `tests/acceptance/test_activation_cache.py` exists. Docstring at [ActivationCache.py:1105-1107](transformer_lens/ActivationCache.py#L1105-L1107) still says "embed, pos_embed, each head result, each neuron result, and the accumulated biases" — the misleading order persists.
- **TransformerBridge**: `tests/acceptance/model_bridge/compatibility/test_activation_cache.py` exists for bridge parity.
- **Replication**: `[code-verified]` — docstring text confirmed wrong vs. actual stacking. Empirical verification of stacking order would take ~10 lines but the docstring/code mismatch is clear from reading.
- **Bucket**: `not-addressed-simple`
- **Next step**: 3-line docstring fix at `ActivationCache.py:1105-1107` to match actual `[heads, neurons, embed, pos_embed, biases]` order. Tests already cover the main flows.

<a id="issue-100"></a>

#### #100 — Add tests + better docs for tokenization methods

- **Issue**: Add tests for `to_tokens`, `to_string`, `to_str_tokens`, `get_token_position`. Clarify `prepend_bos` documentation.
- **HookedTransformer**: Multiple test files cover this — `tests/integration/test_tokenization_methods.py`, `test_only_tokenizer.py`, `test_utils_tokens.py`, `tests/acceptance/test_hook_tokens.py`, `test_tokenizer_special_tokens.py`. Tokenize utils were extracted to `transformer_lens/utilities/tokenize_utils.py`.
- **TransformerBridge**: bridge has its own `to_tokens` etc. — uses HF tokenizer directly.
- **Replication**: `[code-verified]` — substantial test coverage already exists.
- **Bucket**: `partial-leave-open`
- **Next step**: docs side — `prepend_bos` is well-documented in current `HookedTransformerConfig` (lines 144-152) but a standalone explainer page in Sphinx docs would close the original ask. ~50-line docs task.

<a id="issue-104"></a>

#### #104 — Add mixed precision (fp16/bf16) inference incl. loading

- **Issue**: Load models in fp16/bf16, esp. for large models. Comment thread surfaced two specific patches from `slavachalnev`: keep LayerNorm in fp32; apply attention scale before computing scores (avoid -inf in fp16).
- **HookedTransformer**: dtype handling exists (dtype param on `from_pretrained`, `load_in_4bit`). Thread-described NaN issues partially fixed via PRs #366, #389. Per-architecture fp16/bf16 numerical divergence vs. HF persists (Pythia, Llama-2/3 — see #385 for one such regression I confirmed).
- **TransformerBridge**: bridge wraps HF's modules directly, including HF's LayerNorm/attention/rotary. The TL-side numerical paths that caused fp16 NaN (manual LN cast, attention scale order) don't exist on the bridge — by construction matches HF in any precision HF supports. Original "load in fp16/bf16" ask shipped via `boot_transformers(..., dtype=torch.float16)`.
- **Replication**: `[empirically replicated]` — see #385: pythia-70m via `from_pretrained_no_processing` produces NaN logits even in fp32 baseline. Bridge avoids by using HF rotary/attention directly.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: bridge users get the feature + numerical correctness today. HT users with fp16 precision concerns should be pointed at the bridge migration recipe. The HT-side per-architecture audit (overlapping with #385) becomes a lower-priority "fix the legacy path" task rather than a blocker for the original feature ask.

<a id="issue-107"></a>

#### #107 — Helper to run HuggingFace evals on HookedTransformer

- **Issue**: Run HF evals (PIQA, TriviaQA, LAMBADA) against `HookedTransformer`. Suggested pivoting to `lm-evaluation-harness`.
- **HookedTransformer**: `transformer_lens/evals.py` exists but contains the OLD pre-HF eval set (e.g. `IOIDataset`). No `lm-eval-harness` adapter. No way to feed a `HookedTransformer` instance into `lm-eval-harness`'s LM interface.
- **TransformerBridge**: bridge wraps HF model, so users can pull `bridge.original_model` and feed that to `lm-eval-harness` directly. Indirect coverage only.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: meaningful integration project — write an `LM`-conforming wrapper for `HookedTransformer` per `lm-eval-harness`'s API, expose as `pip install transformer-lens[evals]`. ~200 LoC + test infra. Or document the bridge passthrough as an interim recipe.

<a id="issue-111"></a>

#### #111 — Demo of direct path patching

- **Issue**: Add a section to Exploratory Analysis Demo demonstrating direct path patching for all head pairs. PR #49 was an early attempt.
- **HookedTransformer**: `demos/Activation_Patching_in_TL_Demo.ipynb`, `demos/Attribution_Patching_Demo.ipynb` exist. Neither covers direct path patching specifically. No first-class TL helper for direct path patching.
- **TransformerBridge**: same — no path-patching primitive in either API.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: callum mcdougall pointed users at the [ARENA IOI notebook](https://colab.research.google.com/drive/1KgrEwvCKdX-8DQ1uSiIuxwIiwzJuQ3Gw) which covers path patching. Could either close with a docs pointer to ARENA, or implement a TL helper that wraps the pattern (~80 LoC).

<a id="issue-112"></a>

#### #112 — Helper to display vectors of logits nicely

- **Issue**: Neel asked for two things: **MVP** — function mapping logit vector → pandas DataFrame `(token_index, token_string, logit, log_prob, probability)`, top-K or full vocab. **Bonus** — nostalgebraist-style `plot_logit_lens` heatmap (layer × position grid, top token per cell, colored by value). Comment thread later discussed an _additional_ interactive circular visualization (sheikheddy's proposal) that Neel + sheikheddy redirected to CircuitsVis — but that redirect is about the JS-heavy interactive vis, NOT about the original MVP/bonus asks.
- **HookedTransformer**: `test_prompt` in [`transformer_lens/utilities/exploratory_utils.py`](transformer_lens/utilities/exploratory_utils.py) prints top-k tokens with logit/prob/rank for a prompt+answer — partly satisfies the spirit of the MVP but is print-only, not structured (no DataFrame return), and single-position only. No `plot_logit_lens` heatmap helper.
- **TransformerBridge**: same — `test_prompt` works through the bridge too, no separate visualization helper.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~30 LoC for the DataFrame helper (`logits_to_df(logits, tokenizer, top_k=None) -> pd.DataFrame`), ~50 LoC for the heatmap (`plot_logit_lens` matplotlib version that takes `(n_layers, pos, d_vocab)` and renders the grid). Both are small, well-scoped library additions independent of CircuitsVis.

<a id="issue-207"></a>

#### #207 — Can't add hook to pretrained model: AssertionError on `hook_q_input`

- **Issue**: Attribution Patching Demo's `model.add_hook(lambda name: True, ...)` raised `AssertionError: Cannot add hook blocks.0.hook_q_input if use_split_qkv_input is False`. The reporter's specific complaint was that the published demo crashed.
- **HookedTransformer**: the cfg-gated assertion still exists at [HookedTransformer.py:264-266](transformer_lens/HookedTransformer.py#L264-L266) — the broader UX question Neel raised in the thread ("warning vs. assertion") was not actioned.
- **TransformerBridge**: the demo itself was rewritten in PR #1013 (commit `b4fc3754` "updated loading in attribution patching demo to use transformer bridge") to use `TransformerBridge.boot_transformers("gpt2", ...)` + `model.set_use_attn_result(True)` + a smarter filter (`lambda name: "_input" not in name` — the atlaie workaround from the thread, formalized). The reported crash scenario can no longer occur in the canonical demo.
- **Replication**: `[empirically replicated]` — the underlying broad-pattern assertion still fires on `lambda name: True` (I verified during batch 1 investigation). But the _demo workflow_ runs cleanly.
- **Bucket**: `covered-close`
- **Next step**: close, pointing reporter at PR #1013 / the current `demos/Attribution_Patching_Demo.ipynb`. The latent UX concern (broad-pattern add_hook should warn-not-assert) is a separate, unfiled issue — file a new ticket if someone wants to revisit Neel's "warnings are annoying but silent bugs are worse" question.

<a id="issue-210"></a>

#### #210 — `get_full_resid_decomposition` accept tensor argument

- **Issue**: Add a `project_output_onto: [d_model]` or `[d_model, num_outputs]` argument so neuron-decomposition doesn't blow GPU memory by materializing `[batch, pos, d_mlp, d_model]`.
- **HookedTransformer**: [`ActivationCache.py:1091`](transformer_lens/ActivationCache.py#L1091) signature has no `project_output_onto`. Memory-blowing path still active.
- **TransformerBridge**: same — bridge uses the same `ActivationCache` class.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: add `project_output_onto` kwarg + `(neurons * (W_out @ project_output_onto))` path. ~15 LoC + 1 test. Alan Cooney commented they'd take it; never landed.

<a id="issue-264"></a>

#### #264 — GatedMLP not in docs

- **Issue**: Three sub-tasks: (1) docstring for `gated_mlp` config arg; (2) tests including parity vs equivalent PyTorch impl + activation-cache verification; (3) optional tutorial.
- **HookedTransformer**: `gated_mlp` config field at [HookedTransformerConfig.py:249](transformer_lens/config/HookedTransformerConfig.py#L249) has no docstring entry. `tests/unit/components/mlps/test_gated_mlp.py` is smoke tests only (init + output shape, 41 lines) — no parity test against an equivalent `nn.Module`, no cache-correctness test.
- **TransformerBridge**: `transformer_lens/model_bridge/generalized_components/gated_mlp.py` is **substantially better documented** — class docstring spells out the gated MLP structure formula, hook semantics for compat-vs-raw modes, and method-level docstrings on `__init__`, `forward`, `set_processed_weights`. But `cfg.gated_mlp` in `TransformerBridgeConfig` is also undocumented. **Test coverage** is indirect: 4 adapter unit tests (Qwen3.5, InternLM2, Cohere, Baichuan) verify GatedMLPBridge is wired correctly into each adapter — structural mapping tests, not numerical parity. Existing `test_bridge_vs_hooked_comparison.py` uses distilgpt2 (no gated MLP); `test_weight_processing_perfect_match.py` uses gpt2 (no gated MLP). The numerical parity test `0amp` suggested doesn't exist on either side.
- **Replication**: `[code-verified]`
- **Bucket**: `partial-leave-open`
- **Next step**:
  - For sub-task 1 (config docstring): ~3-line addition in both `HookedTransformerConfig` and `TransformerBridgeConfig` for `gated_mlp` field. Trivial.
  - For sub-task 2 (parity test): ~30 LoC test that loads a small gated-MLP model (e.g., `unsloth/llama-3.2-1b` or a tiny custom config) and checks `torch.allclose(bridge_output, hf_output)` end-to-end. This would close `0amp`'s ask AND provide regression coverage for the entire gated-MLP forward path on the bridge.
  - Optional sub-task 3 (tutorial): defer to a separate scoped issue.

<a id="issue-277"></a>

#### #277 — BERT: Future work (multi-checklist tracking)

- **Issue**: Tracking issue for BERT enhancements: expand demo, more BERT models, NSP support, weight processing incl. LN folding, training/finetuning support, convenience-property tests.
- **HookedTransformer**: `BertForMaskedLM` adapter present. `BertForNextSentencePrediction` is NOT registered in `SUPPORTED_ARCHITECTURES` — only mentioned as an example for `model_class=` override.
- **Models**: `bert-base-cased`, `bert-base-uncased`, `bert-large-cased`, `bert-large-uncased` all registered in `supported_models.py`.
- **Demo**: `demos/BERT.ipynb` exists.
- **LN folding for BERT**: still hard (post-norm) — covered by separate open issue #509.
- **Training/dropout support**: not addressed.
- **TransformerBridge**: same architecture coverage as HT.
- **Replication**: `[code-verified]`
- **Bucket**: `partial-leave-open`
- **Next step**: split into separate scoped issues — close #277 as a stale tracking issue and open dedicated issues for (a) NSP adapter, (b) BERT LN folding (#509), (c) training/dropout support. The meta-checklist format obscures what's actually outstanding.

<a id="issue-290"></a>

#### #290 — GPU memory leak when HookedTransformer goes out of scope

- **Issue**: `del model; gc.collect(); torch.cuda.empty_cache()` doesn't reclaim memory after loading multiple models in a loop.
- **HookedTransformer**: substantial debugging in the thread by `rusheb`/`pranavgade20`. Identified two parts: (1) circular reference in `mod_dict` (the empty-name self-reference), (2) tensors in `state_dict[k] = v.to(device)` not detaching from compute graph. PR #1229 ("detach in load") landed at least one of the fixes.
- **TransformerBridge**: bridge wraps HF model directly; HF's own memory hygiene applies. Bridge has its own concerns but the HT-specific circular-reference / non-detach issues don't apply.
- **Replication**: `[unverifiable on this machine]` — needs GPU profiling tooling and ~10× model loads to see the leak. The line numbers referenced in the thread (HookedTransformer.py:860-870) no longer match — code has been refactored.
- **Bucket**: `partial-leave-open`
- **Next step**: re-run the `fil-profile` reproduction from the comment thread on current `dev` to confirm whether residual leak exists. If yes, the next-worst-offender per `rusheb` was `move_model_modules_to_device` — that overlaps with the multi-GPU bug cluster (#837/#907/#911/#968).

<a id="issue-297"></a>

#### #297 — Better print-outs for currently attached hooks

- **Issue**: API for listing hooks attached to a model + HookPoint, with detail.
- **HookedTransformer**: no first-class `model.list_hooks()` or `HookPoint.describe()` API. `model.hook_dict` is publicly accessible (`Dict[str, HookPoint]`); `hp.fwd_hooks` and `hp.bwd_hooks` are inspectable lists. Users can roll their own iteration but it's inconvenient.
- **TransformerBridge**: same — uses the same `hook_points` machinery.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: add `HookedRootModule.list_active_hooks()` returning `Dict[str, List[hook_repr]]`. ~15 LoC + 1 test. PR #302 mentioned in the thread for sub-task (ii) was abandoned.

<a id="issue-335"></a>

#### #335 — Improve LN1's hooks (LN1 hook fires 3 times per forward)

- **Issue**: When `use_split_qkv_input=False` (default), `transformer_block.py` still calls `self.ln1(query_input/key_input/value_input)` three times. Hooks on `ln1` get called 3× and the cached tensor gets overwritten 3×.
- **HookedTransformer**: confirmed at [transformer_block.py:172,174,176](transformer_lens/components/transformer_block.py#L172-L176) — three `self.ln1(...)` calls.
- **TransformerBridge**: bridge uses HF's attention with HF's LayerNorm — fired once per forward.
- **Replication**: `[empirically replicated]` — added a counting hook on `blocks[0].ln1.hook_normalized` in gpt2; called 3 times per forward.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: Arthur tagged this `low-priority`. Fix on HT side would either pass an extra `[3, batch, pos, d_model]` dim through `ln1` OR cache the LN1 output once and reuse. ~20 LoC. Or recommend bridge migration since the bug doesn't apply there.

<a id="issue-341"></a>

#### #341 — Update FactoredMatrix.svd() (uses deprecated `torch.svd`, returns V not Vh)

- **Issue**: TL uses deprecated `torch.svd` (which returns V, not Vh) inside `FactoredMatrix.svd`. Should switch to `torch.linalg.svd` and return Vh per modern convention.
- **HookedTransformer/Bridge**: confirmed at [FactoredMatrix.py:230-233](transformer_lens/FactoredMatrix.py#L230-L233) — still uses `torch.svd(...)` and returns Vh as variable but it's actually V.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~15-line fix — switch to `torch.linalg.svd(..., full_matrices=False)`, return `Vh` directly, update docstring noting the breaking change. `diego898` in the thread offered to send PR. Breaking change so should land with a deprecation warning + version bump.

<a id="issue-378"></a>

#### #378 — Optionally use flash attention

- **Issue**: Flash attention / SDPA flag for performance. Particularly useful for Pythia-12B and SAE training. Cost: lose intermediate attention pattern hooks.
- **HookedTransformer**: no `attn_implementation` flag, no `scaled_dot_product_attention` path in `transformer_lens/components/`. Attention is hand-rolled with `einsum`.
- **TransformerBridge**: `boot_transformers` sets `attn_implementation="eager"` by default ([sources/transformers.py:399](transformer_lens/model_bridge/sources/transformers.py#L399)) so users can override to `"sdpa"` or `"flash_attention_2"` via the adapter cfg. Bridge therefore supports flash attention naturally where HF supports it.
- **Replication**: `[code-verified]`
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: HT-side fix is real work (~100 LoC + cache surface decisions). Bridge users can already use flash via HF's native support — recommend bridge migration.

<a id="issue-385"></a>

#### #385 — Pythia / Rotary Embeddings don't match HuggingFace

- **Issue**: Logit drift between `HookedTransformer` and HF for Pythia models. Llama-2-7b-chat reportedly catastrophic. Llama-3.2 rotary mismatch persists per chengjiali.
- **HookedTransformer**: rotary code lives in `transformer_lens/components/abstract_attention.py`. Multiple fixes have landed (PRs #366, #389, #454 referenced in thread; recent rotary PR `2c41b6c9 Weight processing/position embeddings attention`).
- **TransformerBridge**: bridge uses HF's rotary implementation directly via `RotaryEmbeddingBridge` delegating to HF's `model.rotary_emb`. By construction matches HF.
- **Replication**: `[empirically replicated]` — pythia-70m via `from_pretrained_no_processing` returns NaN logits when compared to HF's `GPTNeoXForCausalLM`. argmax doesn't match. This is **worse** than the issue's original report (~1e-3 to 1e-4 drift) — current state appears to have a regression.
- **Bucket**: `bug-still-reproduces` (and possibly regressed) + `fixed-on-transformerbridge` for bridge users
- **Next step**: investigate the NaN regression — first verify whether this reproduces with full `from_pretrained` (with default processing) and current `dev` HEAD. If real regression, bisect against `2c41b6c9` and other recent rotary PRs. Bridge users avoid this entirely.

<a id="issue-448"></a>

#### #448 — `n_params` counts are wrong

- **Issue**: TL's `n_params` ignores embeddings and uses an oversimplified MLP formula (the `2x` factor is wrong for SwiGLU/gated MLPs).
- **HookedTransformer**: calculation at [HookedTransformerConfig.py:325-334](transformer_lens/config/HookedTransformerConfig.py#L325-L334) — only counts attention + MLP, ignores embed/unembed/LN/biases. The `gated_mlp` factor was added (line 329 uses `2 + self.gated_mlp`) — partial improvement.
- **TransformerBridge**: same `n_params` calculation path (shared config).
- **Replication**: `[empirically replicated]` — gpt2-small reports `n_params = 84,934,656` but actual is `163,049,041`. Embeddings alone account for ~115M (W_E + W_pos + W_U); the formula misses everything except attn+MLP weights.
- **Bucket**: `bug-still-reproduces`
- **Next step**: replace the manual formula with `sum(p.numel() for p in self.parameters() if p.requires_grad)` (post-load) for total count. ~5 LoC + maintain backward compat by keeping the old "trainable parameters in transformer blocks" interpretation under a different attr name. Discussion in thread: Neel preferred total params for alignment with Pythia suite naming.

<a id="issue-453"></a>

#### #453 — `from_pretrained()` always downloads same weights with `checkpoint_label`

- **Issue**: Reporter passes `checkpoint_label=...` and gets identical weights regardless of label. `checkpoint_index` works.
- **HookedTransformer**: [HookedTransformer.py:1158-1159](transformer_lens/HookedTransformer.py#L1158-L1159) signature has `checkpoint_index` and `checkpoint_value` — **NOT `checkpoint_label`**. `checkpoint_label` is not a valid parameter; it gets silently absorbed into `**from_pretrained_kwargs` and discarded.
- **TransformerBridge**: doesn't have a checkpoint feature — uses HF's native loading only.
- **Replication**: `[code-verified]` — confirmed via `inspect.signature(HookedTransformer.from_pretrained)`: `checkpoint_label` not in parameters; `checkpoint_value` and `checkpoint_index` are.
- **Bucket**: `bug-likely-fixed-needs-verification` (or arguably `question-not-actionable`) — the original "bug" is user error with a non-existent kwarg name. The latent issue is that **kwargs silently swallows unknown args.
- **Next step**: respond to reporter that the parameter is `checkpoint_value`, not `checkpoint_label`. Optionally improve UX by validating unknown kwargs in `from_pretrained` and raising — small change but defensive. ~10 LoC.

<a id="issue-462"></a>

#### #462 — Add support for Mamba

- **Issue**: Add Mamba SSM architecture support.
- **HookedTransformer**: not supported (Mamba is fundamentally different from attention transformers; Neel's design philosophy was to keep HT focused on attention models).
- **TransformerBridge**: `MambaArchitectureAdapter` and `Mamba2ArchitectureAdapter` both registered in [SUPPORTED_ARCHITECTURES](transformer_lens/factories/architecture_adapter_factory.py). Both `MambaForCausalLM` and `Mamba2ForCausalLM` HF model classes mapped.
- **Replication**: `[code-verified]`
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: close with comment pointing at `TransformerBridge.boot_transformers("state-spaces/mamba-130m-hf")` as the supported recipe. Bridge's Mamba support is shipped.

---

### Batch 2: next 20 issues (#479 – #720)

_(complete — sign-off requested before batch 3)_

<a id="issue-479"></a>

#### #479 — Memory efficient causal mask implementation

- **Issue**: Each `Attention` layer registers a `(n_ctx, n_ctx)` boolean `causal_mask` buffer. For models with large `n_ctx` (e.g. Qwen 72B at 32768 ctx × 80 layers), this is ~86 GB of overhead. Should compute the mask on the fly at the actual context length.
- **HookedTransformer**: confirmed at [abstract_attention.py:120,123](transformer_lens/components/abstract_attention.py#L120) — `causal_mask = torch.tril(torch.ones((self.cfg.n_ctx, self.cfg.n_ctx)).bool())` and `register_buffer("mask", causal_mask)`. Bug as reported still present for ALL architectures via HT.
- **TransformerBridge**: **architecture-dependent**. Bridge wraps HF's attention modules; HF's choice about static-vs-dynamic mask varies by architecture:
  - **GPT2-family** (HF `GPT2Attention.__init__` does `register_buffer("bias", torch.tril(...))` of shape `(1, 1, max_pos, max_pos)`): bridge inherits the same overhead. Empirically: gpt2 small bridge has 12 × 1MB = 12.5 MB of `(1024, 1024)` bool buffers — identical to HT.
  - **GPTNeoX / Pythia / Llama / Qwen / Mistral / Gemma** (modern HF attention impls use `_update_causal_mask` per forward, no static buffer in `__init__`): bridge has zero overhead. Empirically: Pythia attn `__init__` declares only Q/K/V/output linears + scaling; no buffers.
  - **The issue's motivating example (Qwen 72B at 32K ctx × 80 layers ≈ 86 GB)**: Qwen uses Llama-family attention → resolved on the bridge. The user's actual blocker is fixed.
  - **GPT2 use case**: the bridge gives no relief, but the absolute overhead at gpt2's 1024 ctx is only 12 MB — not the same severity as Qwen 72B.
- **Replication**: `[empirically replicated]` — gpt2 bridge total mask-buffer bytes: `12,582,912` (12 layers × 1MB each); gpt2 HT total: `12,582,912` (identical). HF GPT2's `register_buffer("bias", ...)` is the source.
- **Bucket**: `partial-leave-open`
- **Next step**: bridge users on modern architectures (Llama, Qwen, Pythia, etc. — covering most large-context use cases) already have the memory profile the issue asks for. For GPT2-family on the bridge, the overhead is small in absolute terms but the architectural problem exists. HT-side fix (~30 LoC: replace pre-allocated buffer with on-the-fly construction in `apply_causal_mask`) would close it for the legacy path. Worth noting that fixing it on HT alone solves it for GPT2 use cases regardless of bridge migration.

<a id="issue-481"></a>

#### #481 — Tracr to TransformerLens demo broken

- **Issue**: Demo notebook assumes "the unembed is a projection onto the first few elements of the residual stream" — wrong because Tracr re-orders the residual stream alphabetically by RASP variable name. Demo silently fails on any RASP program where the output variable doesn't sort to the top. The fix needs Tracr to expose the unembed matrix in its model params.
- **HookedTransformer**: demo at [`demos/Tracr_to_Transformer_Lens_Demo.ipynb`](demos/Tracr_to_Transformer_Lens_Demo.ipynb) — `grep` confirms `sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)` line is still in the notebook. Demo NOT ported to TransformerBridge (no `boot_transformers` reference). Bug-as-described still reproduces.
- **TransformerBridge**: same — Tracr-specific issue applies regardless of which TL API the demo uses; the bug is in the unembed-matrix derivation, not in TL's hook system.
- **Replication**: `[code-verified]`
- **Bucket**: `bug-still-reproduces`
- **Next step**: needs Tracr upstream PR to expose `unembed_matrix` in `tracr.params`. Reporter (FlyingPumba) said they'd attempt the upstream change. Without that, the demo is fundamentally limited to RASP programs whose output variable sorts to the top of the residual stream.

<a id="issue-483"></a>

#### #483 — `HookedTransformer.generate()` `pad_token_id` error when tokenizer unset

- **Issue**: `model.generate()` on a `HookedTransformer` with no tokenizer raises `AttributeError: 'NoneType' object has no attribute 'pad_token_id'`. Use case: training models on tokenizer-less domains (e.g., character-level integer addition).
- **HookedTransformer**: confirmed at [HookedTransformer.py:772-773](transformer_lens/HookedTransformer.py#L772-L773) — `if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token` — assumes tokenizer exists. No `pad_token_id` parameter on `generate()`.
- **TransformerBridge**: bridge's generate also relies on `self.tokenizer` for padding decisions (`bridge.py:2266`); same gap.
- **Replication**: `[empirically replicated]` — minimal HookedTransformerConfig + no tokenizer + `generate(input, eos_token_id=0)` raises `AssertionError` (different surface error than originally reported, but the underlying gap is the same: generate path assumes a tokenizer).
- **Bucket**: `bug-still-reproduces`
- **Next step**: ~10 LoC fix — add optional `pad_token_id: Optional[int] = None` kwarg to `generate()`, threaded through to padding logic. Reporter offered to send a PR. Same fix should land on bridge `generate` for parity.

<a id="issue-502"></a>

#### #502 — How to use TransformerLens with HF visual language models?

- **Issue**: User asks how to use TL with `Salesforce/blip-vqa-capfilt-large` and `xtuner/llava-internlm2-7b`.
- **HookedTransformer**: no native VLM support. zazamrykh's fork added LLaVA support (referenced in thread).
- **TransformerBridge**: LLaVA family natively supported — `LlavaArchitectureAdapter`, `LlavaNextArchitectureAdapter`, `LlavaOnevisionArchitectureAdapter` registered. Demo at [`demos/LLaVA.ipynb`](demos/LLaVA.ipynb). BLIP-VQA not yet supported (different VLM architecture).
- **Replication**: `[code-verified]`
- **Bucket**: `question-not-actionable`
- **Next step**: close with response — LLaVA family is now first-class on TransformerBridge (point at the demo). BLIP-VQA would need a new adapter; user can file a separate model-request issue if they want that specifically.

<a id="issue-509"></a>

#### #509 — LayerNorm folding not implemented for BertBlock

- **Issue**: BertBlock uses post-norm (LN after attention/MLP, not before). `fold_ln=True` still folds LN into Q/K/V which is mathematically incorrect for post-norm.
- **HookedTransformer**: `BertBlock` lives at [HookedEncoder.py:24,51,53](transformer_lens/HookedEncoder.py). Neel's reply in the thread is decisive: *"LayerNorm should not be folded at all. You cannot fold it into W_O, because that would change the norm of the output of the layer and thus the LayerNorm scale. I can't think of any way to do LayerNorm folding for Bert, unfortunately"*. Architectural limitation.
- **TransformerBridge**: bridge's `BertArchitectureAdapter` exists but `enable_compatibility_mode()` would inherit the same fold-doesn't-work issue. Most users don't fold LN on BERT regardless.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: this is a fundamental property of post-LN architectures, not a fixable bug. Resolution: either close as wontfix (Neel's view) or document in `enable_compatibility_mode()` that BertBlock should use `fold_ln=False`. The latter is ~5 LoC + a one-line warning when fold_ln=True is passed for BertBlock.

<a id="issue-515"></a>

#### #515 — `evals.IOIDataset` all entries identical

- **Issue**: All entries in IOIDataset are the same. Cause: `random.seed(42)` at the top of `get_sample`.
- **HookedTransformer/Bridge**: shared `transformer_lens/evals.py`. Confirmed at line 387: `def get_sample(self, symmetric=False): random.seed(42); template: str = random.choice(self.templates); ...` — re-seeding to 42 at the top of every sample gives identical samples.
- **Replication**: `[empirically replicated]` — first 3 dataset entries have identical `prompt` tensors (verified with `torch.equal`).
- **Bucket**: `bug-still-reproduces`
- **Next step**: ~1 LoC fix — remove the `random.seed(42)` line at evals.py:387 (or move it outside the loop, but the comment in the file suggests it shouldn't be there at all). Trivial PR.

<a id="issue-523"></a>

#### #523 — Residual stack not adding up (logit lens)

- **Issue**: User loads gpt2-small with `fold_ln=False`, expects `accumulated_resid[-1] @ W_U` to match logits, doesn't.
- **HookedTransformer**: Neel's reply in the thread is the answer — user wasn't applying LN gain/bias before unembedding. Correct formula: `(final_residual_post_ln * model.ln_final.w + model.ln_final.b) @ model.W_U + model.b_U`.
- **TransformerBridge**: same — applies to either API; the issue is conceptual.
- **Replication**: `[code-verified]`
- **Bucket**: `question-not-actionable`
- **Next step**: close with link to Neel's reply. The conceptual gap could be addressed by adding an `apply_ln=True` example to the `accumulated_resid` docstring or the model_structure.md page (overlaps with #644 / hook semantics).

<a id="issue-543"></a>

#### #543 — Grokking demo broken in Colab

- **Issue**: `loss_fn(all_logits, labels)` raises `RuntimeError: Size does not match at dimension 0 expected index [12769, 1] to be smaller than self [113, 113]`.
- **HookedTransformer**: demo at [`demos/Grokking_Demo.ipynb`](demos/Grokking_Demo.ipynb). Recent commits include `58b007f8 Fix type of HookedTransformerConfig.device (#1230)`, `98811df5 3.0 CI Bugs (#1261)`, `69326dad Updating notebooks` — multiple post-issue updates.
- **TransformerBridge**: not directly relevant — this is a demo-specific shape bug.
- **Replication**: `[unverifiable on this machine]` — would need to actually run the full notebook in a Colab-like environment.
- **Bucket**: `bug-likely-fixed-needs-verification`
- **Next step**: anthonyduong9 commented "I can work on this today" but no PR linked. Ask reporter (or a contributor) to re-run the notebook on current `dev` and confirm the original error still occurs.

<a id="issue-569"></a>

#### #569 — Cannot load Llama 3 70B on multigpu in 4bit

- **Issue**: `HookedTransformer.from_pretrained(..., hf_model=base_model)` fails with `size mismatch for blocks.0.attn._W_K: copying a param with shape torch.Size([4194304, 1]) from checkpoint, the shape in current model is torch.Size([8, 8192, 128])`. Multiple users report the same on Llama-3-8B and Llama-2-70B. BnB packs weights as 1D blobs; HT's QKV reshape doesn't handle this packing.
- **HookedTransformer**: HT's loading path (`load_and_process_state_dict` + `convert_llama_weights`) doesn't unpack BnB-quantized weights before reshape.
- **TransformerBridge**: bridge's load path (`sources/transformers.py`) accepts pre-loaded `hf_model` and skips state_dict conversion entirely — the BnB-pack shape-mismatch error in HT's `load_state_dict` cannot occur because that step doesn't run. Forward pass on bridge: bridge does its own attention reconstruction (see #615 / #720 entries) but the q/k/v projections go through `LinearBridge` wrapping HF's quantized Linear modules, so quantization is transparently honored at the projection step. Bridge therefore avoids the loading-time error.
- **Replication**: `[unverifiable on this machine]` — needs ≥2 GPUs + BnB. Structurally: bridge skips the broken state_dict conversion path, so the specific `RuntimeError` the issue reports cannot manifest. Whether 4bit + multigpu works end-to-end on bridge with a 70B model has not been empirically tested (no machine here has the hardware).
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: HT-side fix would require BnB-aware QKV reshape (~50 LoC, architecture-specific). Bridge users can use `TransformerBridge.boot_transformers(model_name, hf_model=quantized_hf_model)` to skip the broken path. Caveat: end-to-end 4bit+multigpu on bridge is structurally sound but unverified empirically; first user to try it on hardware should confirm and we close on that confirmation.

<a id="issue-588"></a>

#### #588 — Setup unit tests to cover model configurations

- **Issue**: Add unit tests that load every supported model's config and verify it's parseable.
- **HookedTransformer/Bridge**: per-architecture config tests now exist for several models — `tests/unit/test_gemma3_config.py`, `tests/unit/test_hooked_transformer_config.py`, `tests/unit/test_llava_config.py`. Plus structural-mapping tests for ~15 architectures under `tests/unit/model_bridge/supported_architectures/`. Not all 185 models systematically covered, but the foundation exists.
- **Replication**: `[code-verified]`
- **Bucket**: `partial-leave-open`
- **Next step**: parametrize a single test over the full `SUPPORTED_ARCHITECTURES` keys (~30 LoC) — for each, load config-only via `boot_transformers(model_name, load_weights=False)` and assert it succeeds. Curt-tigges originally signed up for this in 2024 but no PR linked.

<a id="issue-595"></a>

#### #595 — Add Stopping Criteria support

- **Issue**: HF offers `StoppingCriteria` class that can halt generation on custom conditions (regex match, max length per beam, etc.). HT's `generate()` only supports `stop_at_eos`.
- **HookedTransformer**: confirmed — [HookedTransformer.py:1882,1918,2069](transformer_lens/HookedTransformer.py#L1882) only `stop_at_eos: bool` parameter; no callable-based stopping.
- **TransformerBridge**: bridge `generate` at [bridge.py:2371](transformer_lens/model_bridge/bridge.py#L2371) — same; only `stop_at_eos`.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~30 LoC — add `stopping_criteria: Optional[Callable[[tokens, logits], bool]] = None` parameter to both `HookedTransformer.generate` and `TransformerBridge.generate`, evaluate after each sampled token, break if any returns True. srishti-git1110 volunteered in 2024.

<a id="issue-615"></a>

#### #615 — HookedTransformer output not identical to HuggingFace for Llama 3

- **Issue**: Greedy decoding diverges between HT and HF on Llama-3-8B-Instruct. Investigation in thread localized to MLP weight differences after einsum/Linear conversion.
- **HookedTransformer**: substantial post-issue cleanup happened — most einsum calls in attention/MLP replaced with `F.linear` (visible at [abstract_attention.py:368,374](transformer_lens/components/abstract_attention.py#L368)). Latest collaborator update (degenfabian) reports max diff `0.0002` on Llama-3-8B-Instruct after einsum removals — likely close enough for production, but per-user reports continue (Gemma 2-2B, etc).
- **TransformerBridge**: **NOT a passthrough to HF's attention** — `JointQKVAttentionBridge._reconstruct_attention` and `PositionEmbeddingsAttentionBridge.forward` both do their own attention math: `torch.matmul(q, k.transpose(-2,-1)) * scale`, then their own softmax+mask, then `torch.matmul(weights, v)`. They wrap HF's `q_proj`/`k_proj`/`v_proj` weights via `LinearBridge`, but the score computation, mask application, and output reshape are bridge code. Empirically (Pythia-70m fp32, see #385): bridge max diff vs HF is `2.56e-3` (mean `2.78e-4`, argmax matches across all positions). Materially better than HT's NaN, but not bit-exact.
- **Replication**: `[empirically replicated]` — Pythia-70m bridge vs HF gives 2.5e-3 max drift, argmax matches.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: bridge users get argmax/CE/generation parity with HF (the user-visible behavior the issue actually reports). Bit-exact match isn't achieved — bridge's attention reconstruction has small drift, likely from one of: (a) bridge forces `attn_implementation="eager"` while HF default may pick sdpa, (b) softmax dtype/order differences, (c) intermediate `.contiguous()` calls. For most interpretability uses this is fine; for analyses requiring strict numerical identity (e.g., bit-exact circuit reproduction), bridge is closer than HT but not perfect.

<a id="issue-644"></a>

#### #644 — Documentation: Map the Act Names to the Transformer

- **Issue**: Add a labeled diagram mapping hook names to positions on a transformer architecture figure (Vaswani-style).
- **HookedTransformer/Bridge**: [`docs/source/content/model_structure.md`](docs/source/content/model_structure.md) is 153 lines listing 51 hook names with descriptions, but no diagram. Two volunteers (juvogt, tjbai) said they'd contribute, no PR landed.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~1-day docs task — generate a diagram (matplotlib + manual layout, or Excalidraw + commit the SVG). Place at `docs/source/_static/hook_diagram.svg`, embed in `model_structure.md`. Overlaps with #657's `hook_normalized` semantics under fold_ln (good chance to add that note while editing the page).

<a id="issue-661"></a>

#### #661 — Pythia output inconsistent across batch sizes with `use_split_qkv_input=True`

- **Issue**: `model(input[:2])[0]` and `model(input[:1])[0]` give different outputs when `use_split_qkv_input=True`.
- **HookedTransformer**: `transformer_block.py:123,137` branch on `use_split_qkv_input`. Bug confirmed.
- **TransformerBridge**: bridge has no `use_split_qkv_input` flag — feature doesn't exist on the bridge, so the bug doesn't apply, but bridge users can't replicate the workflow either.
- **Replication**: `[empirically replicated]` — pythia-70m, the exact repro from the issue, gives `max diff: 1.14e-02` (over 10× the issue's 1e-3 tolerance).
- **Bucket**: `bug-still-reproduces`
- **Next step**: investigate why per-token splitting changes batch-vs-single output. Likely a stateful interaction in the LN1 path (related to #335 — LN1 firing 3× per forward). Fix is non-trivial; `use_split_qkv_input` is a research-only feature so priority is moderate.

<a id="issue-684"></a>

#### #684 — Expand quantization model support beyond Llama

- **Issue**: HT raises `AssertionError: Quantization is only supported for Llama models` when loading a 4bit Mistral via `hf_model=`.
- **HookedTransformer**: confirmed at [HookedTransformer.py:1341-1342](transformer_lens/HookedTransformer.py#L1341-L1342) — explicit hard-coded `"llama" not in model_name.lower()` assertion blocks any non-Llama 4bit model.
- **TransformerBridge**: bridge's `boot_transformers(hf_model=...)` path has no architecture-specific assertion. Bridge does its own attention reconstruction but the q/k/v projections wrap HF's Linear modules (which are the BnB-quantized linears when quantization is active), so quantization passes through transparently at the projection step.
- **Replication**: `[code-verified]` — the assertion is still in place at the cited HT line. Bridge load path was structurally verified (no architecture filter in `boot_transformers`); end-to-end forward of a 4bit-quantized Mistral via the bridge has not been tested on this machine (no GPU + BnB).
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: HT-side fix is to remove the assertion AND audit per-architecture state_dict load for BnB-packed weights (overlaps with #569 root cause). Bridge users can already pre-load via `AutoModelForCausalLM.from_pretrained(model, load_in_4bit=True)` and pass to `boot_transformers(model_name, hf_model=quantized_model)`. Caveat: bridge's manual attention reconstruction with BnB-quantized weights is structurally sound but unverified empirically; first user with hardware should confirm.

<a id="issue-696"></a>

#### #696 — About the cached layernorm scale factors

- **Issue**: Conceptual question about why `apply_ln_to_stack` uses cached scale factors instead of recomputing LN per-component.
- **HookedTransformer/Bridge**: `apply_ln_to_stack` at [ActivationCache.py:987](transformer_lens/ActivationCache.py#L987). Neel answered in thread: cached scale factors are needed because we want to apply the FINAL residual's LN to PARTIAL components, and you can't infer the final norm from partial components.
- **Replication**: `[code-verified]`
- **Bucket**: `question-not-actionable`
- **Next step**: close with link to Neel's answer. Could add a 2-line note to the docstring clarifying the design rationale (overlaps with #523 in spirit — both are LN-application confusion).

<a id="issue-697"></a>

#### #697 — Activation cache during generate

- **Issue**: User wants `run_with_cache` semantics during `model.generate()` — cache activations of generated tokens, not just the prompt.
- **HookedTransformer**: confirmed via [HookedTransformer.py:1873,2255](transformer_lens/HookedTransformer.py#L1873) — `generate()` and `generate_stream()` exist but neither integrates `run_with_cache`. bryce's reply: "no integration ... pretty low priority."
- **TransformerBridge**: bridge's `generate` at [bridge.py:2371](transformer_lens/model_bridge/bridge.py#L2371) — same gap.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~50 LoC enhancement — wrap the per-token forward in `run_with_cache`'s hook-installation context, accumulate cache across iterations. Trickier than naive due to KV-cache interactions; needs care to avoid duplicate hook fires when cache grows. Both APIs need the same fix.

<a id="issue-704"></a>

#### #704 — Add support for TracrBench

- **Issue**: TracrBench is a 121-model dataset of toy Tracr transformers for sanity-checking interp methods. Reporter has all models on HuggingFace; asks whether they should live in TransformerLens or a separate repo.
- **HookedTransformer/Bridge**: not present in either side.
- **Replication**: `[code-verified]` — `grep -i tracr_bench` returns nothing in `transformer_lens/`.
- **Bucket**: `not-relevant-close`
- **Next step**: Neel's reply was decisive: *"My personal inclination would be to just make this into another repo that builds on TransformerLens."* TracrBench should live in its own repo (HannesThurnherr's) with TL as a dependency. Close with a docs pointer to the external project, possibly link from `docs/source/content/gallery.md`.

<a id="issue-710"></a>

#### #710 — MVP Support For 1-2 Models Per-Modality

- **Issue**: Add basic support for non-text models — TTS (Whisper), Vision (ResNet, ViT), Music Generation, etc. — to avoid scattered tooling.
- **HookedTransformer**: not designed for non-text architectures.
- **TransformerBridge**: partial coverage exists for some — `HubertForCTC` / `HubertModel` (audio) registered; LLaVA / LLaVA-Next / LLaVA-Onevision / Gemma3-Multimodal (vision-language) supported; CLIPVisionEncoderBridge as a sub-component. But no Whisper, no ResNet, no diffusion, no music gen. bryce's thread response argues for a "platform" approach (programmatical hook points, plugin architecture) rather than baking each modality into core TL.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: per bryce, the long-term fix is a plugin/extension architecture rather than per-model adapters. For now, vision-language is partially covered via LLaVA family; pure-vision (ViT, ResNet) needs new adapters. Worth treating this as an umbrella tracking issue and splitting into sub-issues per modality.

<a id="issue-720"></a>

#### #720 — Review current matmul function usages

- **Issue**: `batch_addmm` is the right shape for GPT-2's `Conv1D`-style layers but inappropriate for Pythia/Llama which use plain `nn.Linear`. Need per-architecture matmul routing.
- **HookedTransformer**: progress made — `transformer_lens/components/abstract_attention.py:368-374` now uses `F.linear` for the post-attention output (comment: "F.linear is a fused matmul+bias that matches HuggingFace exactly"). `batch_addmm` still in `utilities/addmm.py:22`. Full audit not done.
- **TransformerBridge**: bridge does NOT delegate the full attention computation to HF — `JointQKVAttentionBridge._reconstruct_attention` and `PositionEmbeddingsAttentionBridge` both contain their own `torch.matmul(query_states, key_states.transpose(-2,-1))` calls for attention scores, and `torch.matmul(attn_weights, value_states)` for the output. The Q/K/V projections themselves go through `LinearBridge` which wraps HF's Linear (so projection matmul = HF's matmul = correct), but the attention-score and output-application matmuls are bridge code. The same audit concern (does our matmul match HF's, does it preserve precision under different dtypes, etc.) applies to those bridge calls.
- **Replication**: `[code-verified]`
- **Bucket**: `partial-leave-open`
- **Next step**: ~3 distinct audit needs — (1) HT's `batch_addmm` vs `F.linear` per-architecture routing, (2) bridge's `torch.matmul(q, k.T)` and `torch.matmul(weights, v)` vs HF's per-architecture attention impl (e.g., HF's `LlamaAttention.forward` may upcast or use different matmul variants under specific configs), (3) Q/K/V projections (already correct via HF Linear on bridge; per-architecture on HT). Bridge users get correct projection matmuls automatically but inherit the bridge's own attention-math audit gap. Recommend bridge migration as the strategic answer for projection-related precision issues; the bridge-side attention audit is its own future work.

---

### Batch 3: next 20 issues (#729 – #888)

_(complete — sign-off requested before batch 4)_

<a id="issue-729"></a>

#### #729 — Guide to adding new models

- **Issue**: Add a how-to-extend-TL guide for users adding new model support.
- **HookedTransformer/Bridge**: `docs/source/content/` has `migrating_to_v3.md`, `model_structure.md`, `getting_started.md`, etc. — but no dedicated "add a new architecture" guide. The 3.0 bridge architecture made this easier (write an `ArchitectureAdapter` subclass), but there's no walkthrough doc.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~1-day docs task. Walk through what an `ArchitectureAdapter` does (`supported_architectures/llama.py` is a good template), document the `component_mapping` API, demonstrate adding a new entry to `SUPPORTED_ARCHITECTURES`. Overlaps with #888 (CLIP language model adapter Q) and #710 (modality scope).

<a id="issue-737"></a>

#### #737 — Q reshape with model loaded in 4bit

- **Issue**: `model.cfg.use_split_qkv_input = True` + 4bit-loaded vicuna-7b → `RuntimeError: shape '[1, 6, 32, 128]' is invalid for input of size 786432` in `AbstractAttention.calculate_qkv_matrices`. The 4bit code path passes the BnB-packed `[d*d_head*n_heads/2, 1]` weight to `bnb.matmul_4bit`, which dequantizes incorrectly with split QKV input.
- **HookedTransformer**: confirmed at [abstract_attention.py:58-59,338,342,378,381,454,458,473](transformer_lens/components/abstract_attention.py#L58) — multiple `if self.cfg.load_in_4bit:` branches that build `Params4bit` shaped `[nq, 1]`. Path interacts poorly with `use_split_qkv_input=True`.
- **TransformerBridge**: bridge has no `use_split_qkv_input` flag. Quantized models load via `boot_transformers(hf_model=quantized_model)` — bridge's manual attention reconstruction (see #615 / #720 entries) operates on HF's quantized Linear modules, so 4bit + standard hooks should work, but split QKV is a TL-specific feature not on the bridge.
- **Replication**: `[unverifiable on this machine]` — needs GPU + bnb 4bit.
- **Bucket**: `partial-leave-open`
- **Next step**: bridge users avoid this specific bug since the feature isn't there. HT-side fix requires reshape-aware logic in `calculate_qkv_matrices` for the 4bit + split path (~30 LoC). User needs a workaround on HT — currently must disable `use_split_qkv_input` for 4bit models.

<a id="issue-754"></a>

#### #754 — Don't load from HF when config is passed in

- **Issue**: User passes a locally-loaded `hf_model` to `HookedTransformer.from_pretrained(..., hf_model=local_model)` but TL still tries to download config from HuggingFace. Same root cause as #846 and #800.
- **HookedTransformer**: confirmed at [loading_from_pretrained.py:160](transformer_lens/loading_from_pretrained.py#L160) — `convert_hf_model_config` calls `AutoConfig.from_pretrained(official_model_name)` unconditionally to determine architecture. The `hf_cfg` parameter to `get_pretrained_model_config` is only used at line 1847+ for a few specific overrides (`load_in_4bit`, `d_vocab`, `rotary_base`); architecture detection still requires HF reachability. Workaround: pass a local **path** as `model_name` (line 134 checks for local `config.json`).
- **TransformerBridge**: bridge's `boot_transformers(hf_model=...)` takes the architecture from the passed model directly via [`sources/transformers.py:512`](transformer_lens/model_bridge/sources/transformers.py#L512) — `if hf_model is not None: pass`. No AutoConfig fetch needed. Offline use works.
- **Replication**: `[code-verified]`
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: bridge users can boot offline with `hf_model=`. HT-side fix is a small but careful edit — pass `hf_cfg.architectures[0]` to `convert_hf_model_config` when available, skip the AutoConfig fetch. ~10 LoC. Same fix closes #800 and #846. Saberlve's monkey-patch in the thread is the right shape.

<a id="issue-773"></a>

#### #773 — TransformerLens on models with different layernorm placement (BioGPT)

- **Issue**: BioGPT has only ONE layernorm per layer (post-MLP), unlike GPT-2's pre-LN1+pre-LN2 pattern. User asks if TL can adopt to this.
- **HookedTransformer**: `BioGPT` not in supported models. `transformer_block.py` assumes the standard GPT-2 LN placement. Bryce's reply: this is "not possible without making modifications to the code itself" — would need an experimental branch for the user.
- **TransformerBridge**: bridge's `BlockBridge` and architecture adapter pattern theoretically supports per-architecture LN placement (custom adapter could declare `ln1=None, ln2=NormalizationBridge(...)` etc.), but no BioGPT adapter exists. Bridge offers a structural hook for this; nobody's taken it.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: write a `BioGptArchitectureAdapter` for the bridge using the existing component pattern. ~80 LoC + tests. The bridge's component-map approach is exactly the right primitive; HT's hard-coded LN placement is the legacy-side problem.

<a id="issue-778"></a>

#### #778 — Gemma2 global/local attn order wrong

- **Issue**: TL configures Gemma2 attention as `[global, local, global, local, ...]` but HF Gemma2 actually uses `[local, global, local, global, ...]` (verified via the HF source at `modeling_gemma2.py`). Sliding-window placement is inverted.
- **HookedTransformer**: confirmed at [loading_from_pretrained.py:972,999,1027](transformer_lens/loading_from_pretrained.py#L972) — multiple Gemma2 configs hardcode `"attn_types": ["global", "local"] * 21`.
- **TransformerBridge**: bridge uses HF's Gemma2 attention modules directly; HF's `attention_type = config.layer_types[layer_idx]` reads from HF config which has the correct order (`['sliding_attention', 'full_attention', ...]`).
- **Replication**: `[empirically replicated]` — HF: `['sliding_attention', 'full_attention', 'sliding_attention', 'full_attention', 'sliding_attention']` (i.e., `[local, global, local, global, local, ...]`). TL: `['global', 'local', 'global', 'local', 'global', 'local']`. Inversion confirmed.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: HT-side fix is to flip the order in the loading config (~5 lines per Gemma2 variant — multiple variants need updating). Bridge users get the correct order automatically.

<a id="issue-784"></a>

#### #784 — How to load a model in smaller precision (Gemma 2 OOM on 3060)

- **Issue**: User runs `HookedTransformer.from_pretrained_no_processing("google/gemma-2-2b-it", dtype=torch.bfloat16)` on RTX 3060 (laptop) and gets CUDA OOM, while plain HF load works.
- **HookedTransformer**: dtype handling exists; the OOM is likely from TL holding both an FP32 reference state_dict and an FP16/BF16 working copy during conversion (a known leak family — see #290). 3060 laptop typically has 6GB; gemma-2-2b in bf16 is ~5GB so very tight.
- **TransformerBridge**: bridge wraps the HF model directly; no separate FP32-ref + working-copy duplication. Should fit on the 3060.
- **Replication**: `[unverifiable on this machine]` — no GPU here. Code-level: bridge's loading at `sources/transformers.py:451` does `hf_model = model_class.from_pretrained(model_name, **model_kwargs)` then optional `.to(device)` — single allocation, dtype as requested. HT loads HF model first then re-allocates TL params, doubling peak.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: bridge users can `boot_transformers("google/gemma-2-2b-it", dtype=torch.bfloat16, device="cuda")` and not double-allocate. HT-side fix is the broader memory-leak audit (#290 family).

<a id="issue-796"></a>

#### #796 — `FactoredMatrix.svd()` `lru_cache` prevents GC

- **Issue**: `FactoredMatrix.svd` is decorated with `@lru_cache(maxsize=None)`, which holds references to instances and prevents garbage collection.
- **HookedTransformer/Bridge**: confirmed at [FactoredMatrix.py:9,217-218](transformer_lens/FactoredMatrix.py#L217) — `from functools import lru_cache` and `@lru_cache(maxsize=None) def svd(self): ...`. Instance-method `lru_cache` creates a strong ref via `self`.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~10 LoC fix per the issue's suggestion — replace `@lru_cache(maxsize=None)` with `@cached_property` (functools, stdlib). `cached_property` stores result in instance `__dict__`, no cyclic ref, GC-safe. Same fix for `eigenvalues` (the issue notes both). Breaking change since `.svd()` becomes `.svd` (property, no parens) — version-bump worthy. Worth coordinating with the broader `FactoredMatrix` cleanup (#341 also touches this file).

<a id="issue-798"></a>

#### #798 — Remove `model_args` (use only `model_kwargs`)

- **Issue**: Bryce's own proposal — clean up `model_args` + `model_kwargs` redundancy in functions that pass-through to other functions.
- **HookedTransformer**: confirmed — `model_args` still present in [`HookedEncoderDecoder.py:489,495,501,513`](transformer_lens/HookedEncoderDecoder.py#L489) and `hook_points.py:629,723,737,779`. Both `*model_args` and `**model_kwargs` accepted as positional + keyword pairs.
- **TransformerBridge**: bridge's hook_points share the same machinery. `bridge.run_with_cache` etc. inherit the same pattern.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: ~30 LoC across affected files; remove `*model_args`, keep only `**model_kwargs`. Breaking change for callers passing positional args, but Bryce filed it as a `breaking-change` already so acceptable.

<a id="issue-800"></a>

#### #800 — Load model fails (offline use, GPT2-xl local)

- **Issue**: User has GPT2-xl downloaded locally; loading via `HookedTransformer.from_pretrained` works in one Jupyter notebook but fails in another with "couldn't connect to HF" — environment-specific symptom of the deeper issue (#846 / #754) where TL fetches HF config unconditionally.
- **HookedTransformer**: same root cause as #754 / #846 — `convert_hf_model_config` does `AutoConfig.from_pretrained(...)`. If env is set up for offline (HF_HUB_OFFLINE, cached), it may work; otherwise fails.
- **TransformerBridge**: same fix as #754 — use `boot_transformers(hf_model=local_loaded_model)` or pass a local path to skip AutoConfig.
- **Replication**: `[code-verified]`
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: same as #754/#846 — HT-side fix to read architecture from `hf_cfg.architectures[0]` when available. Bridge users have the workaround today.

<a id="issue-801"></a>

#### #801 — Padding side inconsistency with HuggingFace (Gemma 2)

- **Issue**: Reporter on TL 2.9.0 found `HookedTransformer.from_pretrained('google/gemma-2-2b').tokenizer.padding_side == 'right'` while HF AutoTokenizer reports `'left'`.
- **HookedTransformer**: tested on current `dev` — both report `'left'`. Mismatch no longer reproduces. The fix likely came in via tokenizer-handling refactor between 2.9.0 and current.
- **TransformerBridge**: bridge inherits HF tokenizer settings directly; no override. Reports `'left'`.
- **Replication**: `[empirically not reproduced]` — current `dev`: TL `'left'`, HF `'left'`, no mismatch.
- **Bucket**: `bug-likely-fixed-needs-verification`
- **Next step**: comment on issue asking reporter to retest on current `dev`. If they confirm, close.

<a id="issue-830"></a>

#### #830 — Type hint support for `self.model` in `ActivationCache`

- **Issue**: `ActivationCache.model` is untyped (would need `HookedTransformer` import → circular). Proposes a `HookedTransformerMixin` to break the cycle.
- **HookedTransformer**: confirmed at [ActivationCache.py:118](transformer_lens/ActivationCache.py#L118) — `self.model = model` with no type annotation. Bryce in the thread: 4.0 work, possibly 3.0 if non-disruptive. Milestone tag: 3.0.
- **TransformerBridge**: bridge uses the same `ActivationCache`; same untyped attribute.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: extract a `HookedRootModuleMixin` (or use `TYPE_CHECKING + Protocol`) to provide type hints without circular imports. ~50 LoC + careful refactor. Tagged 3.0 milestone but not done.

<a id="issue-837"></a>

#### #837 — Multi-GPU device ordinal issue (`n_devices=3` for llama2-7b)

- **Issue**: With `n_devices=3`, `get_device_for_block_index` produces device indices that exceed the available range, throwing "device ordinal out of range." Same root cause family as #907, #911, #968.
- **HookedTransformer**: bug still in [multi_gpu.py:142](transformer_lens/utilities/multi_gpu.py#L142) — `device_index = (device.index or 0) + (index // layers_per_device)` overshoots when `n_layers % n_devices != 0` (32 layers / 3 = 10.67 → blocks 30,31 land on device 3 which doesn't exist).
- **TransformerBridge**: pre-loaded `hf_model=` with HF's `device_map="auto"` works on `dev`. The unmerged feature/multi-device-bridge PR #1270 provides first-class `n_devices=N` and `device_map=...` parameters that delegate to accelerate.
- **Replication**: `[unverifiable on this machine]` — no multi-GPU.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: HT-side fix is `device_index = (index // layers_per_device)` clamped to `n_devices - 1`. Bridge users today can use `hf_model=accelerate_dispatched_model` workaround; #1270 makes it first-class.

<a id="issue-846"></a>

#### #846 — Prioritize local `hf_model.config` for Qwen models

- **Issue**: Loading a local Qwen via `HookedTransformer.from_pretrained_no_processing(model_name="Qwen/...", hf_model=local, tokenizer=tok)` still fetches HF config online, fails offline.
- **HookedTransformer**: same root cause as #754 / #800 — `convert_hf_model_config` at [loading_from_pretrained.py:160](transformer_lens/loading_from_pretrained.py#L160) calls `AutoConfig.from_pretrained` unconditionally. `hf_cfg` only used for a few overrides at line 1847+, not architecture detection. kapedalex (contributor) commented "can not reproduce today" — likely they tested with a model that hits a name-based shortcut (Llama/Gemma branches at lines 145-157) that skips the AutoConfig fetch. Qwen has no such shortcut.
- **TransformerBridge**: bridge's `boot_transformers(hf_model=...)` skips AutoConfig entirely.
- **Replication**: `[code-verified]`
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: HT-side fix per #754. Bridge users have the workaround today via `boot_transformers(hf_model=...)`.

<a id="issue-858"></a>

#### #858 — Loading gemma-7b-it runs out of memory (2× H100)

- **Issue**: `HookedTransformer.from_pretrained_no_processing("google/gemma-7b-it", n_devices=2)` on 2× H100 fails with OOM. Bryce: multi-GPU has known issues; suggested retry after planned overhaul.
- **HookedTransformer**: gemma-7b in bf16 is ~14GB; 2× H100 (80GB each) should easily fit. OOM during loading suggests TL holds both source state_dict and target params concurrently (memory-leak family — #290). Plus the multi-GPU device-distribution bugs (#837 family) compound this.
- **TransformerBridge**: bridge has no FP32-ref + working-copy duplication. Pre-loaded `hf_model=` with `device_map="auto"` should fit easily.
- **Replication**: `[unverifiable on this machine]` — no GPU here.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: bridge migration recipe (`boot_transformers("google/gemma-7b-it", hf_model=AutoModel.from_pretrained(model, device_map="auto"))`). HT-side fix overlaps #290 (memory leak) + #837 (multi-GPU placement).

<a id="issue-867"></a>

#### #867 — Does TransformerLens support LVLM like Qwen2-VL?

- **Issue**: User asks if Qwen2-VL / Qwen2.5-VL is supported.
- **HookedTransformer**: no native VLM support.
- **TransformerBridge**: LLaVA family supported (`LlavaArchitectureAdapter`, `LlavaNextArchitectureAdapter`, `LlavaOnevisionArchitectureAdapter`); Gemma3-Multimodal supported. **Qwen2-VL specifically is NOT in `SUPPORTED_ARCHITECTURES`** — `Qwen2VLForConditionalGeneration` / `Qwen2_5_VLForConditionalGeneration` not registered. Different vision tower + projector than LLaVA, so existing adapters don't transfer.
- **Replication**: `[code-verified]` — grep confirms no Qwen2VL adapter.
- **Bucket**: `not-addressed-difficult`
- **Next step**: add `Qwen2VLArchitectureAdapter`. Per bryce thread reply: the framework supports adding it now (vision support landed); ~150 LoC for an adapter following the LLaVA pattern. Could file as a focused model-request; ExplorerFreda's vlm-lens fork in the thread offers an alternative.

<a id="issue-869"></a>

#### #869 — Custom generative video transformer

- **Issue**: User wants to do mech interp on a generative-video diffusion transformer (Sora-like, V2V).
- **HookedTransformer/Bridge**: neither supports diffusion / video generation. Bryce's reply suggests a separate root module: "this sort of model is going to be so different from what we have."
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: this is genuinely architectural — would need a `HookedDiffusionTransformer` root module separate from `HookedTransformer`/`TransformerBridge`. Outside the current scope. Recommend using a dedicated diffusion-interp tool or building a custom hook layer; close as wontfix or leave as architectural roadmap item.

<a id="issue-872"></a>

#### #872 — Add official support for `device_map`

- **Issue**: Bryce's own proposal — currently `device_map` is passed through to HF for loading but isn't a TL-supported parameter for distribution. notoookay's comment shows `n_devices=2` causes `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor` on HookedTransformer with gemma-2-2b-it.
- **HookedTransformer**: `n_devices` partially works via `move_model_modules_to_device` but has the placement bugs documented in #837/#907/#911/#968. notoookay's repro fails with the exact mid-forward device-mismatch error.
- **TransformerBridge**: on `dev`, bridge accepts a pre-loaded `hf_model=` with HF's `device_map`. On unmerged `feature/multi-device-bridge` (PR #1270 — addresses this issue directly), bridge has first-class `n_devices=N` and `device_map="auto"` parameters that delegate to accelerate.
- **Replication**: `[unverifiable on this machine]` for the multi-GPU repro; `[code-verified]` for the API surface.
- **Bucket**: `partial-leave-open`
- **Next step**: PR #1270 (currently `feature/multi-device-bridge`) brings first-class `device_map`/`n_devices` to the bridge — once merged, this issue closes for bridge users. HT-side `n_devices` rework is the multi-GPU bug cluster (#837 et al). Same remediation as #872's user-impact concern.

<a id="issue-873"></a>

#### #873 — Load Llama2-7b-chat-hf fail

- **Issue**: User screenshots show a load failure for `Llama-2-7b-chat-hf`. Body has only screenshots, no specific error text.
- **HookedTransformer**: `LlamaForCausalLM` adapter present; the model is in `supported_models.py`. Without the actual error text, hard to diagnose. Commenter sg-sy suggested specific kwargs (`n_devices`, `cache_dir`, `center_writing_weights=False`) as a workaround — implies a multi-GPU or weight-processing related issue.
- **TransformerBridge**: same architecture supported; bridge wouldn't have HT's loading-side weight-processing concerns.
- **Replication**: `[unverifiable on this machine]` — large model + ambiguous error.
- **Bucket**: `bug-likely-fixed-needs-verification`
- **Next step**: ask reporter for the actual error text + their TL version. Many Llama loading bugs (#385 rotary, #569 4bit shape, weight processing) have been fixed since this was filed.

<a id="issue-878"></a>

#### #878 — Layer-wise caching for low GPU memory (Qwen 7B Instruct)

- **Issue**: User runs attribution patching on Qwen 7B Instruct on 2× A6000 (48GB each) and gets OOM despite trying layer-wise caching. Asks for help.
- **HookedTransformer**: this is a usage question; OOM at attribution-patching scale typically requires gradient checkpointing or smaller batches/sequences.
- **TransformerBridge**: same — neither API has built-in attribution-patching memory optimization.
- **Replication**: `[unverifiable on this machine]`
- **Bucket**: `question-not-actionable`
- **Next step**: close with a docs/recipe pointer. Helpful response: gradient checkpointing via `torch.utils.checkpoint`, or per-layer hook-based caching that releases activations between layers. Recipe could fit in the "extending TL" doc that #729 calls for.

<a id="issue-888"></a>

#### #888 — Adapt HookedTransformer to a non-supported model (CLIP language model)

- **Issue**: User wants `HookedTransformer.from_pretrained` for the CLIP language model component.
- **HookedTransformer**: not possible without code modifications, per Bryce's reply.
- **TransformerBridge**: bridge has `CLIPVisionEncoderBridge` for the vision side (used by LLaVA family) but no text-side CLIP adapter. The bridge's adapter framework is the right primitive — user could write a `CLIPTextModelArchitectureAdapter` for the text encoder.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: write `CLIPTextModelArchitectureAdapter` for the bridge. The architecture is encoder-only with relatively standard transformer blocks (BERT-like attention, no causal mask). Overlaps with #729 (extending guide) — having both would let the user self-serve. ~120 LoC adapter + tests.

### Batch 4: next 20 issues (#894 – #1148)

_(complete — sign-off requested before batch 5)_

<a id="issue-894"></a>

#### #894 — Implement LongRoPE

- **Issue**: Microsoft's LongRoPE rope-scaling variant (used by Phi-4-mini and Phi-3.5-mini) requires per-segment frequency tables and short/long-factor selection based on sequence position. Without it, TL inference silently diverges from HF for long contexts.
- **HookedTransformer**: `loading_from_pretrained.py:893-906` handles `rope_type == "llama3"` and `rope_type == "yarn"` only. No `longrope` branch. LongRoPE configs are not parsed; rotary computation falls back to standard RoPE.
- **TransformerBridge**: bridge delegates rope computation to HF's `apply_rotary_pos_emb`, so LongRoPE on Phi-3.5/Phi-4-mini works natively. `phi3.py:238-240` strips `rope_scaling.rope_type == "default"` to avoid HF's strict mode rejecting the field, but LongRoPE-typed configs flow through to HF unchanged.
- **Replication**: `[code-verified]`
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: comment on the issue pointing to bridge support for Phi-3.5-mini-instruct / Phi-4-mini-instruct via `boot_transformers`. HookedTransformer-side LongRoPE would still be a non-trivial implementation (per-position frequency selection); leave the HT path as a known gap and prioritize bridge-side verification (e.g. add Phi-4-mini to integration tests).

<a id="issue-902"></a>

#### #902 — Some model weights are NaN when initializing

- **Issue**: User reports that `HookedTransformer(HookedTransformerConfig.from_dict(...))` produces NaN entries in specific weight tensors (e.g. `blocks.1.attn.W_O` with 1808 NaNs at row indices 10–11) when initialized from gpt2-small's config. Original repro on TL 2.15.0.
- **HookedTransformer**: `init_weights` at [HookedTransformer.py:1483](transformer_lens/HookedTransformer.py#L1483) iterates `named_parameters()` and runs `nn.init.normal_(param, std=initializer_range)`. With normal init from a non-degenerate `initializer_range`, NaNs would only arise from uninitialized memory or device-init bugs. Possible regression in a now-fixed init path; current dev not yet verified for this specific repro.
- **TransformerBridge**: bridge does not run TL's init paths — uses HF's native init via `from_pretrained`. Random-init via bridge (`load_weights=False`) goes through HF's `_init_weights`, not `_init_weights_gpt2`.
- **Replication**: `[unverifiable on this machine]` — would need to install TL 2.15.0 and rerun; user did not retest on later versions.
- **Bucket**: `bug-likely-fixed-needs-verification`
- **Next step**: ask reporter to retest on current dev (TL 3.x). If reproduces, add a regression test that checks `state_dict()` for NaNs after `_init_weights_*` paths. Bridge users are unaffected.

<a id="issue-903"></a>

#### #903 — gpt2-small `n_params` reports 85M (actual 124M)

- **Issue**: `model_properties_table.html` shows gpt2-small at 85M params; HF reports 124M. Reporter pinpoints `HookedTransformerConfig.n_params` calc which excludes embedding params (W_E ≈ 39M).
- **HookedTransformer**: `config/HookedTransformerConfig.py:325-334` calculates `n_params = n_layers * (d_model * d_head * n_heads * 4) + MLP terms`. Embeddings, unembedding, biases, and LN params are all excluded. Same root cause as #448.
- **TransformerBridge**: shares the same `HookedTransformerConfig` via `cfg`, so same calculation.
- **Replication**: `[code-verified]` — calculation explicitly excludes embeddings; gpt2-small W_E = 50257 × 768 ≈ 38.6M matches the gap.
- **Bucket**: `bug-still-reproduces`
- **Next step**: same fix as #448. Either add an embedding term and rename to `n_attn_mlp_params` (+ new `n_total_params`), or change docs to clarify what `n_params` measures. Behavior change has model-properties-table downstream — coordinate with #97.

<a id="issue-904"></a>

#### #904 — Gemma tensors initialized on CPU during state-dict conversion

- **Issue**: When passing a CUDA-loaded `hf_model` to `HookedTransformer.from_pretrained` with Gemma-2-2b, `fold_value_biases` raises a device-mix error because some state-dict tensors are on CPU.
- **HookedTransformer**: at [HookedTransformer.py:1875](transformer_lens/HookedTransformer.py#L1875), `fold_value_biases` does `b_O_original + (b_V[:, :, None] * W_O).sum([0, 1])` without an explicit `.to(device)`. If `convert_gemma_weights` returns a state_dict with mixed device placement (e.g. biases default-initialized on CPU when source had no biases), this fails. ZeqiangWangSurrey reports same problem on Qwen.
- **TransformerBridge**: bridge does not run `fold_value_biases` by default. With `enable_compatibility_mode(fold_value_biases=True)`, it goes through bridge's own folding path which inherits HF's device placement directly. Pre-loaded `hf_model` retains its `device_map`; no CPU/GPU mix from converter.
- **Replication**: `[unverifiable on this machine]` — would need a CUDA Gemma-2-2b load.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: HT-side fix is one line — `b_V.to(W_O.device)` before the multiply, or move the whole operation through `_to_device` in `convert_gemma_weights`. joaoncardoso offered a PR. Bridge users sidestep by skipping fold_value_biases or by relying on bridge's HF-native device handling.

<a id="issue-907"></a>

#### #907 — PR #864 device-selection refactor breaks multi-GPU

- **Issue**: PR #864 introduced greedy memory-based device allocation, replacing the previous architecture-aware sequential placement. Reporter claims `test_device_separation_and_cache` now fails. Also linked to #906 (loading on a specific device).
- **HookedTransformer**: `move_model_modules_to_device` does memory-greedy placement that can scatter sequential blocks across devices, defeating the locality optimizations relevant for transformer forward passes. Same multi-GPU bug cluster as #837/#911/#968.
- **TransformerBridge**: bridge does not use `move_model_modules_to_device`. On `dev`, accepts a pre-loaded `hf_model` with HF's `device_map="auto"` (proper architecture-aware placement via accelerate). PR #1270 (`feature/multi-device-bridge`) adds first-class `n_devices`/`device_map` parameters that delegate to accelerate.
- **Replication**: `[unverifiable on this machine]`
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: HT-side fix would require reverting or guarding the greedy-allocation path. Bridge users get correct placement via accelerate today (manual `hf_model=`) and first-class once #1270 merges. Comment on issue with bridge migration recipe.

<a id="issue-909"></a>

#### #909 — Request for documentation of hookpoints

- **Issue**: User finds it hard to map hookpoints to specific transformer architecture components; asks for a documentation enhancement explaining each hookpoint and its correspondence to architecture parts.
- **HookedTransformer**: `docs/source/content/model_structure.md` documents legacy hookpoint names (`blocks.{i}.hook_resid_pre`, `hook_attn_out`, etc.) with shapes and meaning. The doc explicitly lists legacy aliases alongside canonical bridge names, so HT users get coverage too.
- **TransformerBridge**: same doc — canonical `hook_in`/`hook_out` convention with shapes.
- **Replication**: `[code-verified]` — doc exists and is comprehensive (160 lines covering embed, residual, attention, MLP, norm, unembed; legacy aliases mapped; shapes listed).
- **Bucket**: `covered-close`
- **Next step**: comment with link to `model_structure.md` and close. If #644's diagram lands too, that further closes the gap for visual learners.

<a id="issue-911"></a>

#### #911 — PosEmbed device error with `accelerate`

- **Issue**: gpt2 + `accelerate launch` (DDP across 2 GPUs) fails inside `PosEmbed.forward` because `W_pos[offset_position_ids]` indexes a tensor that ends up on a different device than the index tensor.
- **HookedTransformer**: `components/embeddings/pos_embed.py:47,59` does `pos_embed = self.W_pos[offset_position_ids]`. Under DDP, `accelerate` broadcasts the model to each rank but TL's per-component device placement (from `move_model_modules_to_device`) doesn't track `accelerate`'s rank-local device, so index/data device-mismatch occurs.
- **TransformerBridge**: bridge does not have a `PosEmbed` component on most architectures — position embeddings are HF-native (RoPE inside attention, or HF's `Embedding` for absolute pos). For gpt2 specifically, the bridge uses HF's `GPT2Model.wpe` directly through `EmbeddingBridge`, which respects HF's device_map. No TL-specific cross-device indexing.
- **Replication**: `[unverifiable on this machine]` — needs DDP setup.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: comment with bridge migration recipe (`TransformerBridge.boot_transformers("gpt2")` plus `accelerate.prepare(bridge.original_model)` for training). HT-side fix would require all embed components to track DDP-rank device — same family as the multi-GPU cluster (#837/#907/#968).

<a id="issue-912"></a>

#### #912 — Support mT5 models

- **Issue**: User requests `google/mt5-small` support for multilingual circuit discovery (Indonesian, Malay, Javanese). T5 is supported; mT5 has the same architecture.
- **HookedTransformer**: not supported. T5-only path; no mT5 conversion logic.
- **TransformerBridge**: `MT5ForConditionalGeneration` is in `utilities/architectures.py:12` (SUPPORTED_ARCHITECTURES) but the `model_type_mappings` in `model_bridge/sources/transformers.py:234` only maps `"t5"`, not `"mt5"`. mT5 reports `model_type="mt5"` in its config, so dispatch falls through and may fail to find the adapter. The T5ArchitectureAdapter itself would likely work for mT5 since the architecture is identical, but the routing isn't wired up.
- **Replication**: `[code-verified]` — model_type mapping confirmed missing.
- **Bucket**: `partial-leave-open`
- **Next step**: add `"mt5": "MT5ForConditionalGeneration"` to `model_type_mappings` and route `MT5ForConditionalGeneration` to `T5ArchitectureAdapter`. ~5 LoC + an integration smoke test on `google/mt5-small`. Once landed, comment on issue confirming. HT side would need the full T5 conversion path extended; recommend bridge migration instead.

<a id="issue-923"></a>

#### #923 — Pythia missing `blocks.0.hook_resid_mid`

- **Issue**: User runs a cache-name assertion test on Pythia and finds no `blocks.0.hook_resid_mid`. Asks if it's a bug or alternative cache name.
- **HookedTransformer**: by-design. Pythia uses parallel attention + MLP (`parallel_attn_mlp=True`), so there is no "mid residual" — attn and MLP both read from `hook_resid_pre` and write directly into `hook_resid_post`. kapedalex confirmed this in the comment thread.
- **TransformerBridge**: same — bridge maps Pythia's GPT-NeoX architecture to a parallel block layout, no mid hook generated.
- **Replication**: `[code-verified]` — `cfg.parallel_attn_mlp` flag determines whether `hook_resid_mid` is registered.
- **Bucket**: `not-relevant-close`
- **Next step**: close with the answer kapedalex already gave. Could optionally add a friendlier error (or auto-skip mid-resid cache assertions) when the model is parallel-attn — minor UX improvement.

<a id="issue-929"></a>

#### #929 — Load custom small GPT-2 with hf_model and HF config

- **Issue**: User trained a small GPT-2 architecture model and wants to use it with HookedTransformer. Currently does it by overwriting `HookedTransformerConfig` after `get_pretrained_model_config`. Asks for a clean API.
- **HookedTransformer**: `convert_hf_model_config` calls `AutoConfig.from_pretrained` unconditionally, ignoring the user's `hf_model.config`. Same root cause as #754/#800/#846. The user's "hacky way" is the documented workaround.
- **TransformerBridge**: `boot_transformers(hf_model=user_model)` reads config from the user's pre-loaded model directly. No AutoConfig refetch. This is exactly the use case bridge solves.
- **Replication**: `[code-verified]` — same code path as #754 cluster.
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: comment with bridge migration recipe — `TransformerBridge.boot_transformers("gpt2", hf_model=user_model, tokenizer=user_tokenizer)`. Close once user confirms or after a reasonable wait.

<a id="issue-930"></a>

#### #930 — Quantized Llama 3.2 fails to load

- **Issue**: `meta-llama/Llama-3.2-3B-Instruct` with `BitsAndBytesConfig(load_in_4bit=True)` fails state_dict load — `_W_K`/`_W_V` shapes `[1572864, 1]` (BnB-packed 4bit) don't match expected `[8, 3072, 128]` (TL's 3D layout).
- **HookedTransformer**: same root cause as #569 — TL's state_dict reshape assumes unpacked weights, but BnB stores `Params4bit` as packed 1D tensors. The reshape `view(n_kv_heads, d_model, d_head)` fails on packed shapes.
- **TransformerBridge**: bridge does not reshape state_dict from HF format. Q/K/V stay in HF's `Params4bit` form inside `LinearBridge`, and `JointQKVAttentionBridge` reads them via the HF Linear's forward (which handles BnB's `bnb.matmul_4bit` natively). Structurally sound; not yet end-to-end verified on quantized Llama 3.2.
- **Replication**: `[unverifiable on this machine]`
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: comment with bridge migration recipe. End-to-end quantized verification depends on hardware availability — same gap as #569/#684.

<a id="issue-950"></a>

#### #950 — Support SimpleStories models

- **Issue**: SimpleStories family (HF: `SimpleStories/*`) is an improved successor to TinyStories — useful as small interp targets and as low-resource debugging models.
- **HookedTransformer**: not supported. No `simplestories` registry entries; no architecture mapping.
- **TransformerBridge**: SimpleStories fine-tunes (e.g. `SimpleStories/SimpleStories-1.25M`, `SimpleStories-35M`) are in `supported_models.json` via auto-discovery. Base SimpleStories model not yet registered — jlarson4's comment notes this.
- **Replication**: `[code-verified]` — registry contains SimpleStories fine-tunes; base model absent.
- **Bucket**: `partial-leave-open`
- **Next step**: add base SimpleStories to bridge's supported_models registry and verify with a smoke test. jlarson4 already volunteered to tackle before next release. HT-side support is unlikely to land — direct users to bridge.

<a id="issue-953"></a>

#### #953 — Add basic support for Gemma 3n (E2B & E4B)

- **Issue**: Gemma 3n introduces nested sub-models (Matryoshka E2B inside E4B), AltUp sparse updates, LAuReL low-rank residuals, Per-Layer Embeddings (PLE) with CPU offload, and mixed local/global attention. Reporter asks for text-only support that bypasses vision/audio.
- **HookedTransformer**: not supported. Architecture is too divergent from any existing HT path.
- **TransformerBridge**: not registered in `SUPPORTED_ARCHITECTURES`. Bryce confirmed in-progress for the next major TransformerLens release. Bridge can structurally accommodate the text-decoder portion if HF's `Gemma3nForCausalLM` exposes blocks in a familiar shape, but AltUp/LAuReL/PLE require dedicated component bridges.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-difficult`
- **Next step**: track for milestone 3.x. Per-block AltUp/LAuReL adapters + PLE handling are non-trivial — likely 200-500 LoC of adapter code plus testing. Mixed local/global attention pattern overlaps with Gemma2 work (#778). Defer until HF's Gemma3n forward is stable.

<a id="issue-962"></a>

#### #962 — Can multiple GPUs be used?

- **Issue**: User asks if `HookedTransformer.from_pretrained("Meta-Llama-3-8B-Instruct", device="auto")` works with `CUDA_VISIBLE_DEVICES=0,1`. On TL 2.11.0 it does not.
- **HookedTransformer**: `n_devices` parameter exists, but `device="auto"` is not a TL convention (it's HF's). User would need `n_devices=2`. Multi-GPU placement has the bug cluster (#837/#907/#911/#968).
- **TransformerBridge**: on `dev`, accepts `hf_model=` with HF's `device_map="auto"`. PR #1270 adds first-class `n_devices` / `device_map` parameters.
- **Replication**: `[code-verified]`
- **Bucket**: `question-not-actionable`
- **Next step**: comment with both options — for HT, use `n_devices=N`; for bridge today, pass a pre-loaded `hf_model` with `device_map="auto"`; once #1270 merges, use bridge's first-class `device_map`. Close after answer.

<a id="issue-968"></a>

#### #968 — `unsloth/llama-3.2-3b-instruct` with 2× 3060 device-mismatch

- **Issue**: `from_pretrained(..., n_devices=2)` on 2× 3060 throws `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cuda:1)`. Same multi-GPU bug cluster as #837/#907/#911.
- **HookedTransformer**: `move_model_modules_to_device` placement bug — embed/pos_embed indexed by tokens on rank-0 device while the embedding parameter ends up on rank-1.
- **TransformerBridge**: jlarson4 already commented offering bridge as the path forward. PR #1270 brings first-class multi-device.
- **Replication**: `[unverifiable on this machine]`
- **Bucket**: `bug-likely-fixed-needs-verification`
- **Next step**: jlarson4's comment already points to bridge + #1270. Wait for reporter to retest on bridge / post-#1270, then close. HT-side fix is the multi-GPU rework.

<a id="issue-993"></a>

#### #993 — Load compressed Llama/Qwen via HookedTransformer

- **Issue**: User loads `meta-llama/Llama-2-7b-chat-hf` fine but cannot load compressed (quantized/pruned) variants of the same architecture.
- **HookedTransformer**: hard-coded "Llama only" assertion in quantization path (same as #684). Pruned variants (sub-set of weights) would also fail TL's strict reshape since per-layer dims must match the registered config.
- **TransformerBridge**: structurally accepts any compressed variant that HF can load. No "Llama only" assertion. Pruned variants with sub-set state dicts work because bridge holds HF parameters by reference, not via reshape. End-to-end verification on a specific compressed checkpoint not yet done.
- **Replication**: `[code-verified]`
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: comment with bridge recipe. Verification on a known compressed checkpoint (e.g. `TheBloke/Llama-2-7B-Chat-GPTQ`) would confirm and allow closing.

<a id="issue-1039"></a>

#### #1039 — Loading models from local files in HookedTransformer

- **Issue**: User gets `LocalEntryNotFoundError`/`OSError` from HookedTransformer when trying to load a local model offline. HF `AutoModelForCausalLM` works fine for the same path.
- **HookedTransformer**: same root cause as #754/#800 — `convert_hf_model_config` calls `AutoConfig.from_pretrained` unconditionally, which tries to hit the Hub even when only a local path is provided.
- **TransformerBridge**: `boot_transformers(hf_model=...)` reads everything from the pre-loaded HF model. No Hub access required.
- **Replication**: `[code-verified]`
- **Bucket**: `fixed-on-transformerbridge`
- **Next step**: comment with bridge recipe — `TransformerBridge.boot_transformers(local_path, hf_model=hf_model, tokenizer=tokenizer)`. Close as duplicate of the #754 cluster after acknowledgment.

<a id="issue-1080"></a>

#### #1080 — Import fails by default in Colab (numpy ABI mismatch)

- **Issue**: Fresh Colab notebook + `pip install transformer_lens` + `import transformer_lens` raises `numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject`. jakobhansen-blai notes a kernel restart works around it; suggests TL pin numpy v1 implicitly.
- **HookedTransformer**: `pyproject.toml:12-13` has `numpy>=1.24` (py3.10/3.11) and `numpy>=1.26` (py3.12) — both lower bounds, no upper cap. Numpy 2.x is allowed by TL itself. The error is a transitive dep ABI mismatch (one of TL's dependencies built against numpy 1.x, runtime imported numpy 2.x or vice versa).
- **TransformerBridge**: same install path; same numpy.
- **Replication**: `[unverifiable on this machine]` — Colab-specific package versions.
- **Bucket**: `bug-likely-fixed-needs-verification`
- **Next step**: ask reporter to retest with a current Colab kernel (numpy 2.x default) and current TL. If it still fails, bisect transitive deps (likely `pandas`, `einops`, or `jaxtyping`). Pin a tested numpy if needed for Colab compat.

<a id="issue-1133"></a>

#### #1133 — `tokenize_and_concatenate` cuts tokens mid-document

- **Issue**: `tokenize_and_concatenate` slices the joined corpus into 20 character-based chunks before tokenizing — this can split mid-token, producing token pairs that would never occur naturally.
- **HookedTransformer**: PR #1273 (commit `ad8e123b`, "Improved Tokenize & Concatenate") replaces character-based chunking with per-document tokenization (`add_special_tokens=False`) and joins with token-level EOS. Tokens are no longer cut across chunk boundaries. Code comment at [tokenize_utils.py:68-70](transformer_lens/utilities/tokenize_utils.py#L68-L70) explicitly references #1133. Earlier PR #1201 (`4a5cc6f0`) also addressed the chunking issue partially.
- **TransformerBridge**: same shared utility; same fix applies.
- **Replication**: `[code-verified]` — current implementation tokenizes per-doc, concatenates with token-level EOS, then reshapes. No string-level chunking.
- **Bucket**: `covered-close`
- **Next step**: close as fixed (PR #1273). The original repro (`tokens[79848:79848+2] == [337, 346]`) cannot occur under the new implementation.

<a id="issue-1148"></a>

#### #1148 — Tutorial for "Real-Time Training Dynamics" (VSM Telemetry)

- **Issue**: Reporter proposes a new demo notebook adding `VSMTelemetry` — a ~30-line bridge class that logs Attention Coherence (σ_p) and Head Specialization (σ_a) during training, useful for studying grokking / phase transitions.
- **HookedTransformer**: no such tutorial exists. `demos/` has Grokking demo (#543, currently broken on Colab) but nothing on real-time mechanistic training telemetry.
- **TransformerBridge**: same — bridge has no training-dynamics telemetry tutorial. Would work equivalently against bridge's hook system.
- **Replication**: `[code-verified]`
- **Bucket**: `not-addressed-simple`
- **Next step**: invite contribution — reporter has a working prototype. Notebook should target `TransformerBridge` (per migration guide), use `run_with_cache` or hooks for σ_p/σ_a extraction, run on a tiny model so it executes quickly in Colab. Add to `demos/` with CI check that the notebook runs.

### Batch 5: final triage entry (#1165)

_(Issues #1263 and #1264 were opened by the maintainer for tracking and don't require triage.)_

<a id="issue-1165"></a>

#### #1165 — Strategy for high-fragmentation tokenization (Yoruba)

- **Issue**: Yoruba tonal characters (e.g. `ọ` in `Atọwọda`) trigger extreme tokenizer fragmentation under GPT-2's BPE — 9 tokens for one word, mostly byte-level fallbacks. User asks whether TransformerLens offers a recommended heuristic for pooling activations across byte-token spans for activation patching, or if the standard practice is to ignore byte-level noise.
- **HookedTransformer**: no built-in pooling utility for fragmented tokens. `to_tokens` is HF-tokenizer passthrough; cache shape is per-token. User would need to write their own span-mean / span-max reducer over the cache before patching.
- **TransformerBridge**: same — bridge inherits the HF tokenizer and offers no span-pooling helper. Activation cache shape is identical.
- **Replication**: `[code-verified]` — no `pool_token_span` / `aggregate_subword` utility exists in either codebase.
- **Bucket**: `question-not-actionable`
- **Next step**: close with a recipe-style answer pointing to two practical options: (1) use a tokenizer with better non-Latin coverage (e.g. mT5, NLLB, or a Yoruba-trained model) — which also overlaps with #912's mT5 request; (2) implement span-pooling manually with a token-to-word mapping built from the tokenizer's offset_mapping (`tokenizer(text, return_offsets_mapping=True)`), then average activations within each word's token span before patching. Could grow into a small `transformer_lens.utils.pool_token_span` helper if demand exists.

