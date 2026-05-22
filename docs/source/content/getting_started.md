# Getting Started

**Start with the [main demo](https://neelnanda.io/transformer-lens-demo) to learn how the library works, and the basic features**.

To see what using it for exploratory analysis in practice looks like, check out [my notebook analysing Indirect Object Identification](https://neelnanda.io/exploratory-analysis-demo) or [my recording of myself doing research](https://www.youtube.com/watch?v=yo4QvDn-vsU)!

Mechanistic interpretability is a very young and small field, and there are a *lot* of open problems - if you would like to help, please try working on one! For inspiration on where the field is headed and why it matters, I highly recommend reading [A Pragmatic Vision for Interpretability](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) and [How Can Interpretability Researchers Help AGI Go Well](https://www.alignmentforum.org/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well). They're a great starting point for thinking about what a useful research project looks like.

If you're new to transformers, check out my [what is a transformer tutorial](https://neelnanda.io/transformer-tutorial) and [tutorial on coding GPT-2 from scratch](https://neelnanda.io/transformer-tutorial-2) (with [an accompanying template](https://neelnanda.io/transformer-template) to write one yourself!)

## Installation

`pip install git+https://github.com/TransformerLensOrg/TransformerLens`

Import the library with `import transformer_lens`

(Note: This library used to be known as EasyTransformer, and some breaking changes have been made since the rename. If you need to use the old version with some legacy code, run `pip install git+https://github.com/TransformerLensOrg/TransformerLens@v1`.)

## Loading a Model

The canonical way to load a model is with `TransformerBridge.boot_transformers`. It figures out the architecture, picks the right adapter, and hands you back a bridge object that exposes all the familiar APIs — `to_tokens`, `to_string`, `generate`, `run_with_hooks`, and `run_with_cache`:

```python
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
logits, cache = bridge.run_with_cache("Hello world")
```

TransformerBridge wraps a HuggingFace model behind a consistent TransformerLens interface through an **architecture adapter** — each supported architecture has an adapter that maps the HF module graph onto a set of generalized components (embedding, attention, MLP, normalization, blocks) with uniform hook points. The big win is that the same interpretability code just works across arbitrary architectures, and the bridge preserves the native HF implementation rather than reimplementing it.

The bridge currently covers 50+ architectures spanning Llama, Mistral, Qwen, Gemma, OLMo, Phi, Mamba, LLaVA, and more. For the full list — including which models have been verified end-to-end — see the [TransformerBridge Models](../generated/transformer_bridge_models.md) page.

## Advice for Reading the Code

The bridge is organized around a small set of generalized components wired together by an architecture adapter, which keeps the model code much easier to navigate than the older unified implementation. For a tour of the bridge's canonical hook names, the component layout, and the expected tensor shapes at each hook point, see the [Model Structure](model_structure.md) page. A small alias layer preserves the older TransformerLens hook names (e.g. `blocks.{i}.hook_resid_pre`) so legacy notebooks keep working — but new code should prefer the canonical names.

## Environment Variables

TransformerLens reads a handful of environment variables. None are required for basic use; each enables a specific opt-in behavior.

### `HF_TOKEN`

Your [HuggingFace access token](https://huggingface.co/settings/tokens). Required for gated models (Llama, Mistral/Mixtral, Gemma families, and others) and used to authenticate any HuggingFace API call TransformerLens makes on your behalf. You will need to accept any model-specific agreements on the HuggingFace Hub before TransformerLens can load a gated model; if you skip this step, the error message will link you directly to the agreement page.

```bash
export HF_TOKEN="hf_..."
```

### `TRANSFORMERLENS_HF_RETRY`

Set to `"1"` to wrap `transformers.AutoConfig.from_pretrained`, `AutoModel.from_pretrained`, `AutoTokenizer.from_pretrained`, `AutoProcessor.from_pretrained`, and `AutoFeatureExtractor.from_pretrained` with a retry-on-429 helper. When HuggingFace returns HTTP 429 (rate-limited), the call is retried up to three times with exponential backoff, honoring numeric `Retry-After` response headers when present (HTTP-date form is not parsed; the retry falls back to exponential backoff in that case).

Intended primarily for CI environments where parallel workflow runs can trip HF's rate limits. Off by default so production callers see unmodified `transformers` behavior. The wrapping is idempotent and applied globally to the class methods; see [`enable_hf_retry`](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/utilities/hf_utils.py) for the implementation. The TransformerLens test suite enables this automatically via `tests/conftest.py`.

```bash
export TRANSFORMERLENS_HF_RETRY=1
```

### `TRANSFORMERLENS_ALLOW_MPS`

Set to `"1"` to opt in to Apple Silicon (MPS) as a target device for model inference. Off by default because not all PyTorch operations used by TransformerLens have stable MPS implementations across PyTorch versions; if you enable this and hit a backend error, the most reliable fallback is to leave the variable unset and let TransformerLens select CPU instead.

```bash
export TRANSFORMERLENS_ALLOW_MPS=1
```

## Huggingface Gated Access

For convenience, gated-model access depends only on `HF_TOKEN` above. Once you have set the token and accepted any model-specific agreements on the HuggingFace Hub, gated models load through TransformerLens with no additional configuration. The most popular gated families supported by TransformerLens are the Llama, Mistral/Mixtral, and Gemma models.
