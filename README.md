# TransformerLens

<!-- Status Icons -->
[![Pypi](https://img.shields.io/pypi/v/transformer-lens?color=blue)](https://pypi.org/project/transformer-lens/)
![Pypi Total Downloads](https://img.shields.io/pepy/dt/transformer_lens?color=blue) ![PyPI -
License](https://img.shields.io/pypi/l/transformer_lens?color=blue) [![Release
CD](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/release.yml/badge.svg)](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/release.yml)
[![Tests
CD](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/checks.yml/badge.svg)](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/checks.yml)
[![Docs
CD](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/pages/pages-build-deployment)

A Library for Mechanistic Interpretability of Generative Language Models.

[![Read the Docs
Here](https://img.shields.io/badge/-Read%20the%20Docs%20Here-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white&link=https://TransformerLensOrg.github.io/TransformerLens/)](https://TransformerLensOrg.github.io/TransformerLens/)

| :exclamation:  HookedSAETransformer Removed   |
|-----------------------------------------------|

Hooked SAE has been removed from TransformerLens 2.0. The functionality is being moved to
[SAELens](http://github.com/jbloomAus/SAELens). For more information on this release, please see the
accompanying
[announcement](https://transformerlensorg.github.io/TransformerLens/content/news/release-2.0.html)
for details on what's new, and the future of TransformerLens.

This is a library for doing [mechanistic
interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language models. The
goal of mechanistic interpretability is to take a trained model and reverse engineer the algorithms
the model learned during training from its weights.

TransformerLens lets you load in 50+ different open source language models, and exposes the internal
activations of the model to you. You can cache any internal activation in the model, and add in
functions to edit, remove or replace these activations as the model runs.

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

* [Introduction to the Library and Mech
  Interp](https://arena3-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp)
* [Demo of Main TransformerLens Features](https://neelnanda.io/transformer-lens-demo)

## Gallery

Research done involving TransformerLens:

<!-- If you change this also change docs/source/content/gallery.md -->
* [Progress Measures for Grokking via Mechanistic
  Interpretability](https://arxiv.org/abs/2301.05217) (ICLR Spotlight, 2023) by Neel Nanda, Lawrence
  Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt
* [Finding Neurons in a Haystack: Case Studies with Sparse
  Probing](https://arxiv.org/abs/2305.01610) by Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine
  Harvey, Dmitrii Troitskii, Dimitris Bertsimas
* [Towards Automated Circuit Discovery for Mechanistic
  Interpretability](https://arxiv.org/abs/2304.14997) by Arthur Conmy, Augustine N. Mavor-Parker,
  Aengus Lynch, Stefan Heimersheim, Adrià Garriga-Alonso
* [Actually, Othello-GPT Has A Linear Emergent World Representation](https://neelnanda.io/othello)
  by Neel Nanda
* [A circuit for Python docstrings in a 4-layer attention-only
  transformer](https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only)
  by Stefan Heimersheim and Jett Janiak
* [A Toy Model of Universality](https://arxiv.org/abs/2302.03025) (ICML, 2023) by Bilal Chughtai,
  Lawrence Chan, Neel Nanda
* [N2G: A Scalable Approach for Quantifying Interpretable Neuron Representations in Large Language
  Models](https://openreview.net/forum?id=ZB6bK6MTYq) (2023, ICLR Workshop RTML) by Alex Foote, Neel
  Nanda, Esben Kran, Ioannis Konstas, Fazl Barez
* [Eliciting Latent Predictions from Transformers with the Tuned
  Lens](https://arxiv.org/abs/2303.08112) by Nora Belrose, Zach Furman, Logan Smith, Danny Halawi,
  Igor Ostrovsky, Lev McKinney, Stella Biderman, Jacob Steinhardt

User contributed examples of the library being used in action:

* [Induction Heads Phase Change
  Replication](https://colab.research.google.com/github/ckkissane/induction-heads-transformer-lens/blob/main/Induction_Heads_Phase_Change.ipynb):
  A partial replication of [In-Context Learning and Induction
  Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
  from Connor Kissane
* [Decision Transformer
  Interpretability](https://github.com/jbloomAus/DecisionTransformerInterpretability): A set of
  scripts for training decision transformers which uses transformer lens to view intermediate
  activations, perform attribution and ablations. A write up of the initial work can be found
  [here](https://www.lesswrong.com/posts/bBuBDJBYHt39Q5zZy/decision-transformer-interpretability).

Check out [our demos folder](https://github.com/TransformerLensOrg/TransformerLens/tree/main/demos) for
more examples of TransformerLens in practice

## Getting Started in Mechanistic Interpretability

Mechanistic interpretability is a very young and small field, and there are a _lot_ of open
problems. This means there's both a lot of low-hanging fruit, and that the bar for entry is low - if
you would like to help, please try working on one! The standard answer to "why has no one done this
yet" is just that there aren't enough people! Key resources:

* [A Guide to Getting Started in Mechanistic Interpretability](https://neelnanda.io/getting-started)
* [ARENA Mechanistic Interpretability Tutorials](https://arena3-chapter1-transformer-interp.streamlit.app/) from
  Callum McDougall. A comprehensive practical introduction to mech interp, written in
  TransformerLens - full of snippets to copy and they come with exercises and solutions! Notable
  tutorials:
  * [Coding GPT-2 from
    scratch](https://arena3-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch), with
    accompanying video tutorial from me ([1](https://neelnanda.io/transformer-tutorial)
    [2](https://neelnanda.io/transformer-tutorial-2)) - a good introduction to transformers
  * [Introduction to Mech Interp and
    TransformerLens](https://arena3-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp): An
    introduction to TransformerLens and mech interp via studying induction heads. Covers the
    foundational concepts of the library
  * [Indirect Object
    Identification](https://arena3-chapter1-transformer-interp.streamlit.app/[1.3]_Indirect_Object_Identification):
    a replication of interpretability in the wild, that covers standard techniques in mech interp
    such as [direct logit
    attribution](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=disz2gTx-jooAcR0a5r8e7LZ),
    [activation patching and path
    patching](https://www.lesswrong.com/posts/xh85KbTFhbCz7taD4/how-to-think-about-activation-patching)
* [Mech Interp Paper Reading List](https://neelnanda.io/paper-list)
* [200 Concrete Open Problems in Mechanistic
  Interpretability](https://neelnanda.io/concrete-open-problems)
* [A Comprehensive Mechanistic Interpretability Explainer](https://neelnanda.io/glossary): To look
  up all the jargon and unfamiliar terms you're going to come across!
* [Neel Nanda's Youtube channel](https://www.youtube.com/channel/UCBMJ0D-omcRay8dh4QT0doQ): A range
  of mech interp video content, including [paper
  walkthroughs](https://www.youtube.com/watch?v=KV5gbOmHbjU&list=PL7m7hLIqA0hpsJYYhlt1WbHHgdfRLM2eY&index=1),
  and [walkthroughs of doing
  research](https://www.youtube.com/watch?v=yo4QvDn-vsU&list=PL7m7hLIqA0hr4dVOgjNwP2zjQGVHKeB7T)

## Support & Community

[![Contributing
Guide](https://img.shields.io/badge/-Contributing%20Guide-blue?style=for-the-badge&logo=GitHub&logoColor=white)](https://TransformerLensOrg.github.io/TransformerLens/content/contributing.html)

If you have issues, questions, feature requests or bug reports, please search the issues to check if
it's already been answered, and if not please raise an issue!

You're also welcome to join the open source mech interp community on
[Slack](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-1qosyh8g3-9bF3gamhLNJiqCL_QqLFrA).
Please use issues for concrete discussions about the package, and Slack for higher bandwidth
discussions about eg supporting important new use cases, or if you want to make substantial
contributions to the library and want a maintainer's opinion. We'd also love for you to come and
share your projects on the Slack!

## Credits

This library was created by **[Neel Nanda](https://neelnanda.io)** and is maintained by **Joseph
Bloom**.

The core features of TransformerLens were heavily inspired by the interface to [Anthropic's
excellent Garcon tool](https://transformer-circuits.pub/2021/garcon/index.html). Credit to Nelson
Elhage and Chris Olah for building Garcon and showing the value of good infrastructure for enabling
exploratory research!

### Creator's Note (Neel Nanda)

I (Neel Nanda) used to work for the [Anthropic interpretability team](transformer-circuits.pub), and
I wrote this library because after I left and tried doing independent research, I got extremely
frustrated by the state of open source tooling. There's a lot of excellent infrastructure like
HuggingFace and DeepSpeed to _use_ or _train_ models, but very little to dig into their internals
and reverse engineer how they work. **This library tries to solve that**, and to make it easy to get
into the field even if you don't work at an industry org with real infrastructure! One of the great
things about mechanistic interpretability is that you don't need large models or tons of compute.
There are lots of important open problems that can be solved with a small model in a Colab notebook!

### Citation

Please cite this library as:

```BibTeX
@misc{nanda2022transformerlens,
    title = {TransformerLens},
    author = {Neel Nanda and Joseph Bloom},
    year = {2022},
    howpublished = {\url{https://github.com/TransformerLensOrg/TransformerLens}},
}
```
