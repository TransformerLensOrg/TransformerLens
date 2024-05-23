# Getting Started

**Start with the [main demo](https://neelnanda.io/transformer-lens-demo) to learn how the library works, and the basic features**.

To see what using it for exploratory analysis in practice looks like, check out [my notebook analysing Indirect Objection Identification](https://neelnanda.io/exploratory-analysis-demo) or [my recording of myself doing research](https://www.youtube.com/watch?v=yo4QvDn-vsU)!

Mechanistic interpretability is a very young and small field, and there are a *lot* of open problems - if you would like to help, please try working on one! **Check out my [list of concrete open problems](https://docs.google.com/document/d/1WONBzNqfKIxERejrrPlQMyKqg7jSFW92x5UMXNrMdPo/edit) to figure out where to start.**. It begins with advice on skilling up, and key resources to check out. 

If you're new to transformers, check out my [what is a transformer tutorial](https://neelnanda.io/transformer-tutorial) and [tutorial on coding GPT-2 from scratch](https://neelnanda.io/transformer-tutorial-2) (with [an accompanying template](https://neelnanda.io/transformer-template) to write one yourself!

## Advice for Reading the Code

One significant design decision made was to have a single transformer implementation that could support a range of subtly different GPT-style models. This has the upside of interpretability code just working for arbitrary models when you change the model name in `HookedTransformer.from_pretrained`! But it has the significant downside that the code implementing the model (in `HookedTransformer.py` and `components.py`) can be difficult to read. I recommend starting with my [Clean Transformer Demo](https://neelnanda.io/transformer-solution), which is a clean, minimal implementation of GPT-2 with the same internal architecture and activation names as HookedTransformer, but is significantly clearer and better documented.

## Installation

`pip install git+https://github.com/TransformerLensOrg/TransformerLens`

Import the library with `import transformer_lens`

(Note: This library used to be known as EasyTransformer, and some breaking changes have been made since the rename. If you need to use the old version with some legacy code, run `pip install git+https://github.com/TransformerLensOrg/TransformerLens@v1`.)

## Huggingface Gated Access

Some of the models available in TransformerLens require gated access to be used. Luckily TransformerLens provides a way to access those models via the configuration of an environmental variable. Simply configure your access token found [here](https://huggingface.co/settings/tokens) as `HF_TOKEN` in your environment.

You will need to make sure you accept the agreements for any gated models, but once you do, the models will work with TransformerLens without issue. If you attempt to ues one of these models before you have accepted any related agreements, the console output will be very helpful and point you to the URL where you need to accept an agreement. As of 23/4/24, the current list of gated models supported by TransformerLens is as follows.

* https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
* https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
* https://huggingface.co/mistralai/Mistral-7B-v0.1
