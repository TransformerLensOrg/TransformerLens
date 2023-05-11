# TransformerLens

![TransformerLens](assets/rm_transformer_lens_logo.png)

[![Pypi](https://img.shields.io/pypi/v/transformer-lens)](https://pypi.org/project/transformer-lens/)

(Formerly known as EasyTransformer)

## [Read the Docs Here](https://neelnanda-io.github.io/TransformerLens/)

## [Main Tutorial Here](https://neelnanda.io/transformer-lens-demo)

See also this better and more in-depth, but still [work-in-progress tutorial from Callum McDougall](https://transformerlens-intro.streamlit.app/)

## A Library for Mechanistic Interpretability of Generative Language Models

This is a library for doing [mechanistic interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language models. The goal of mechanistic interpretability is to take a trained model and reverse engineer the algorithms the model learned during training from its weights. It is a fact about the world today that we have computer programs that can essentially speak English at a human level (GPT-3, PaLM, etc), yet we have no idea how they work nor how to write one ourselves. This offends me greatly, and I would like to solve this!

TransformerLens lets you load in an open source language model, like GPT-2, and exposes the internal activations of the model to you. You can cache any internal activation in the model, and add in functions to edit, remove or replace these activations as the model runs. The core design principle I've followed is to enable exploratory analysis. One of the most fun parts of mechanistic interpretability compared to normal ML is the extremely short feedback loops! The point of this library is to keep the gap between having an experiment idea and seeing the results as small as possible, to make it easy for **research to feel like play** and to enter a flow state. Part of what I aimed for is to make _my_ experience of doing research easier and more fun, hopefully this transfers to you!

I used to work for the [Anthropic interpretability team](transformer-circuits.pub), and I wrote this library because after I left and tried doing independent research, I got extremely frustrated by the state of open source tooling. There's a lot of excellent infrastructure like HuggingFace and DeepSpeed to _use_ or _train_ models, but very little to dig into their internals and reverse engineer how they work. **This library tries to solve that**, and to make it easy to get into the field even if you don't work at an industry org with real infrastructure! One of the great things about mechanistic interpretability is that you don't need large models or tons of compute. There are lots of important open problems that can be solved with a small model in a Colab notebook!

The core features were heavily inspired by the interface to [Anthropic's excellent Garcon tool](https://transformer-circuits.pub/2021/garcon/index.html). Credit to Nelson Elhage and Chris Olah for building Garcon and showing me the value of good infrastructure for enabling exploratory research!

## Getting Started

**Start with the [main demo](https://neelnanda.io/transformer-lens-demo) to learn how the library works, and the basic features**.

To see what using it for exploratory analysis in practice looks like, check out [my notebook analysing Indirect Objection Identification](https://neelnanda.io/exploratory-analysis-demo) or [my recording of myself doing research](https://www.youtube.com/watch?v=yo4QvDn-vsU)!

Mechanistic interpretability is a very young and small field, and there are a _lot_ of open problems - if you would like to help, please try working on one! **Check out my [list of concrete open problems](https://docs.google.com/document/d/1WONBzNqfKIxERejrrPlQMyKqg7jSFW92x5UMXNrMdPo/edit) to figure out where to start.**. It begins with advice on skilling up, and key resources to check out.

If you're new to transformers, check out my [what is a transformer tutorial](https://neelnanda.io/transformer-tutorial) and [tutorial on coding GPT-2 from scratch](https://neelnanda.io/transformer-tutorial-2) (with [an accompanying template](https://neelnanda.io/transformer-template) to write one yourself!

## Gallery

User contributed examples of the library being used in action:

- [Induction Heads Phase Change Replication](https://colab.research.google.com/github/ckkissane/induction-heads-transformer-lens/blob/main/Induction_Heads_Phase_Change.ipynb): A partial replication of [In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) from Connor Kissane
- [Decision Transformer Interpretability](https://github.com/jbloomAus/DecisionTransformerInterpretability): A set of scripts for training decision transformers which uses transformer lens to view intermediate activations, perform attribution and ablations. A write up of the initial work can be found [here](https://www.lesswrong.com/posts/bBuBDJBYHt39Q5zZy/decision-transformer-interpretability).

## Advice for Reading the Code

One significant design decision made was to have a single transformer implementation that could support a range of subtly different GPT-style models. This has the upside of interpretability code just working for arbitrary models when you change the model name in `HookedTransformer.from_pretrained`! But it has the significant downside that the code implementing the model (in `HookedTransformer.py` and `components.py`) can be difficult to read. I recommend starting with my [Clean Transformer Demo](https://neelnanda.io/transformer-solution), which is a clean, minimal implementation of GPT-2 with the same internal architecture and activation names as HookedTransformer, but is significantly clearer and better documented.

## Installation

`pip install git+https://github.com/neelnanda-io/TransformerLens`

Import the library with `import transformer_lens`

(Note: This library used to be known as EasyTransformer, and some breaking changes have been made since the rename. If you need to use the old version with some legacy code, run `pip install git+https://github.com/neelnanda-io/TransformerLens@v1`.)

## Local Development

### DevContainer

For a one-click setup of your development environment, this project includes a [DevContainer](https://containers.dev/). It can be used locally with [VS Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) or with [GitHub Codespaces](https://github.com/features/codespaces).

### Manual Setup

This project uses [Poetry](https://python-poetry.org/docs/#installation) for package management. Install as follows (this will also setup your virtual environment):

```bash
poetry config virtualenvs.in-project true
poetry install --with dev
```

Optionally, if you want Jupyter Lab you can run `poetry run pip install jupyterlab` (to install in the same virtual environment), and then run with `poetry run jupyter lab`.

Then the library can be imported as `import transformer_lens`.

### Testing

If adding a feature, please add unit tests for it to the tests folder, and check that it hasn't broken anything major using the existing tests (install pytest and run it in the root TransformerLens/ directory).

#### Running the tests

- All tests via `make test`
- Unit tests only via `make unit-test`
- Acceptance tests only via `make acceptance-test`

### Demos

If adding a feature, please add it to the demo notebook in the `demos` folder, and check that it works in the demo format. This can be tested by replacing `pip install git+https://github.com/JayBaileyCS/TransformerLens.git` with `pip install git+https://github.com/<YOUR_USERNAME_HERE>/TransformerLens.git` in the demo notebook, and running it in a fresh environment.

## Citation

Please cite this library as:

```
@misc{nandatransformerlens2022,
    title  = {TransformerLens},
    author = {Nanda, Neel},
    url    = {https://github.com/neelnanda-io/TransformerLens},
    year   = {2022}
}
```

(This is my best guess for how citing software works, feel free to send a correction!)
Also, if you're actually using this for your research, I'd love to chat! Reach out at neelnanda27@gmail.com
