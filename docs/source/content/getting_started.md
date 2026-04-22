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

## Huggingface Gated Access

Some of the models available in TransformerLens require gated access to be used. Luckily TransformerLens provides a way to access those models via the configuration of an environmental variable. Simply configure your [HuggingFace access token](https://huggingface.co/settings/tokens) as `HF_TOKEN` in your environment.

You will need to make sure you accept the agreements for any gated models, but once you do, the models will work with TransformerLens without issue. If you attempt to use one of these models before you have accepted any related agreements, the console output will be very helpful and point you to the URL where you need to accept an agreement. The most popular gated families supported by TransformerLens are the Llama, Mistral/Mixtral, and Gemma models.
