---
hide-toc: true
firstpage:
lastpage:
---

# TransformerLens

(Formerly known as EasyTransformer) [![Pypi](https://img.shields.io/pypi/v/transformer-lens)](https://pypi.org/project/transformer-lens/)

## A Library for Mechanistic Interpretability of Generative Language Models

This is a library for doing [mechanistic interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language models. The goal of mechanistic interpretability is to take a trained model and reverse engineer the algorithms the model learned during training from its weights. It is a fact about the world today that we have computer programs that can essentially speak English at a human level (GPT-3, PaLM, etc), yet we have no idea how they work nor how to write one ourselves. This offends me greatly, and I would like to solve this! 

TransformerLens lets you load in an open source language model, like GPT-2, and exposes the internal activations of the model to you. You can cache any internal activation in the model, and add in functions to edit, remove or replace these activations as the model runs. The core design principle I've followed is to enable exploratory analysis. One of the most fun parts of mechanistic interpretability compared to normal ML is the extremely short feedback loops! The point of this library is to keep the gap between having an experiment idea and seeing the results as small as possible, to make it easy for **research to feel like play** and to enter a flow state. Part of what I aimed for is to make *my* experience of doing research easier and more fun, hopefully this transfers to you!

I used to work for the [Anthropic interpretability team](https://transformer-circuits.pub/), and I wrote this library because after I left and tried doing independent research, I got extremely frustrated by the state of open source tooling. There's a lot of excellent infrastructure like HuggingFace and DeepSpeed to *use* or *train* models, but very little to dig into their internals and reverse engineer how they work. **This library tries to solve that**, and to make it easy to get into the field even if you don't work at an industry org with real infrastructure! One of the great things about mechanistic interpretability is that you don't need large models or tons of compute. There are lots of important open problems that can be solved with a small model in a Colab notebook! 

The core features were heavily inspired by the interface to [Anthropic's excellent Garcon tool](https://transformer-circuits.pub/2021/garcon/index.html). Credit to Nelson Elhage and Chris Olah for building Garcon and showing me the value of good infrastructure for enabling exploratory research!

A great place to start is to take a look at a helpful diagram of [all weight matrices and activation tensors with TransformerLens notation](_static/TransformerLens_Diagram.svg) courtesy of [Austin Kozlowski](https://github.com/akozlo). Another helpful tool to help you get going as quickly as possible is our [Colab Compatability Demo](https://github.com/TransformerLensOrg/TransformerLens/tree/main/demos/Colab_Compatibility.ipynb), which will give you a good idea of what you can do in various Colab environments.

```{toctree}
:hidden:
:caption: Introduction

content/getting_started
content/getting_started_mech_interp
content/gallery
```

```{toctree}
:hidden:
:caption: Documentation

generated/code/modules
generated/model_properties_table.md
```

```{toctree}
:hidden:
:caption: Resources

content/tutorials
content/citation
content/contributing
generated/demos/Main_Demo
generated/demos/Exploratory_Analysis_Demo
content/special_cases
```

```{toctree}
:hidden:
:caption: News

content/news/release-2.0
```

```{toctree}
:hidden:
:caption: Development

content/contributing
Code Coverage <https://transformerlensorg.github.io/TransformerLens/_static/coverage/>
Github <https://github.com/TransformerLensOrg/TransformerLens>
```
