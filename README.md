# TransformerLens

<!-- Status Icons -->
[![Pypi](https://img.shields.io/pypi/v/transformer-lens?color=blue)](https://pypi.org/project/transformer-lens/)
![Pepy Total Downlods](https://img.shields.io/pepy/dt/transformer_lens?color=blue) ![PyPI -
License](https://img.shields.io/pypi/l/transformer_lens?color=blue)
[![Release CD](https://github.com/neelnanda-io/TransformerLens/actions/workflows/release.yml/badge.svg)](https://github.com/neelnanda-io/TransformerLens/actions/workflows/release.yml)
[![Tests CD](https://github.com/neelnanda-io/TransformerLens/actions/workflows/checks.yml/badge.svg)](https://github.com/neelnanda-io/TransformerLens/actions/workflows/checks.yml) [![Docs CD](https://github.com/neelnanda-io/TransformerLens/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/neelnanda-io/TransformerLens/actions/workflows/gh-pages.yml)

A Library for Mechanistic Interpretability of Generative Language Models like GPT.

[![Read the Docs
Here](https://img.shields.io/badge/-Read%20the%20Docs%20Here-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white&link=https://neelnanda-io.github.io/TransformerLens/)](https://neelnanda-io.github.io/TransformerLens/)

This is a library for doing [mechanistic interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language models. The goal of mechanistic interpretability is to take a trained model and reverse engineer the algorithms the model learned during training from its weights.

TransformerLens lets you load in 50+ different open source language models, and exposes the internal activations of the model to you. You can cache any internal activation in the model, and add in functions to edit, remove or replace these activations as the model runs.

## Quick Start

### Install

```shell
pip install transformer_lens
```

### Use

```python
import transformer_lens

# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# Run the model and get logits and activations
logits, activations = model.run_with_cache("Hello World")
```

## Key Tutorials

* [Introduction to the Library and Mech Interp](https://arena-ch1-transformers.streamlit.app/[1.2]_Intro_to_Mech_Interp)
* [Demo of Main TransformerLens Features](https://neelnanda.io/transformer-lens-demo)

## Support & Community

[![Contributing Guide](https://img.shields.io/badge/-Contributing%20Guide-blue?style=for-the-badge&logo=GitHub&logoColor=white)](https://neelnanda-io.github.io/TransformerLens/content/contributing.html)

If you have issues, questions, feature requests or bug reports, please search the issues to check if it's already been answered, and if not please raise an issue!

You're also welcome to join the open source mech interp community on [Slack](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-1qosyh8g3-9bF3gamhLNJiqCL_QqLFrA). Please use issues for concrete discussions about the package, and Slack for higher bandwidth discussions about eg supporting important new use cases, or if you want to make substantial contributions to the library and want a maintainer's opinion. We'd also love for you to come and share your projects on the Slack!

## Credits

This library was created
by **[Neel Nanda](https://neelnanda.io)** and is maintained by **Joseph Bloom**.

The core features of TransformerLens were heavily inspired by the interface to [Anthropic's excellent Garcon
tool](https://transformer-circuits.pub/2021/garcon/index.html). Credit to Nelson Elhage and Chris
Olah for building Garcon and showing the value of good infrastructure for enabling exploratory
research!

### Creator's Note (Neel Nanda)

I (Neel Nanda) used to work for the [Anthropic interpretability team](transformer-circuits.pub), and I wrote this library because after I left and tried doing independent research, I got extremely frustrated by the state of open source tooling. There's a lot of excellent infrastructure like HuggingFace and DeepSpeed to _use_ or _train_ models, but very little to dig into their internals and reverse engineer how they work. **This library tries to solve that**, and to make it easy to get into the field even if you don't work at an industry org with real infrastructure! One of the great things about mechanistic interpretability is that you don't need large models or tons of compute. There are lots of important open problems that can be solved with a small model in a Colab notebook!
