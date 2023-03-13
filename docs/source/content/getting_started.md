## Getting Started

**Start with the [main demo](https://neelnanda.io/transformer-lens-demo) to learn how the library works, and the basic features**.

To see what using it for exploratory analysis in practice looks like, check out [my notebook analysing Indirect Objection Identification](https://neelnanda.io/exploratory-analysis-demo) or [my recording of myself doing research](https://www.youtube.com/watch?v=yo4QvDn-vsU)! 

Mechanistic interpretability is a very young and small field, and there are a *lot* of open problems - if you would like to help, please try working on one! **Check out my [list of concrete open problems](https://docs.google.com/document/d/1WONBzNqfKIxERejrrPlQMyKqg7jSFW92x5UMXNrMdPo/edit) to figure out where to start.**. It begins with advice on skilling up, and key resources to check out. 

If you're new to transformers, check out my [what is a transformer tutorial](https://neelnanda.io/transformer-tutorial) and [tutorial on coding GPT-2 from scratch](https://neelnanda.io/transformer-tutorial-2) (with [an accompanying template](https://neelnanda.io/transformer-template) to write one yourself!

### Advice for Reading the Code

One significant design decision made was to have a single transformer implementation that could support a range of subtly different GPT-style models. This has the upside of interpretability code just working for arbitrary models when you change the model name in `HookedTransformer.from_pretrained`! But it has the significant downside that the code implementing the model (in `HookedTransformer.py` and `components.py`) can be difficult to read. I recommend starting with my [Clean Transformer Demo](https://neelnanda.io/transformer-solution), which is a clean, minimal implementation of GPT-2 with the same internal architecture and activation names as HookedTransformer, but is significantly clearer and better documented.

### Installation

`pip install git+https://github.com/neelnanda-io/TransformerLens`

Import the library with `import transformer_lens`

(Note: This library used to be known as EasyTransformer, and some breaking changes have been made since the rename. If you need to use the old version with some legacy code, run `pip install git+https://github.com/neelnanda-io/TransformerLens@v1`.)

