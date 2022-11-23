This repository contains the code for all experiments in the paper "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small" (Wang et al, 2022).

<img src="https://i.imgur.com/iptFVBc.png">

This is intended as a one-time code drop. <b>The authors recommend those interested in mechanistic interpretability using the <a href="https://github.com/neelnanda-io/Easy-Transformer">Easy Transformer</a> library</b>. Contact arthur@rdwrs.com or open an issue or PR for issues with this repository.

See and run the experiments on Google Colab: https://colab.research.google.com/drive/1n4Wgulv5ev5rgRUL7ypOw0odga9LEWHA?usp=sharing .

# Setup

## Option 1) install with pip


```
pip install git+https://github.com/redwoodresearch/Easy-Transformer.git
```

## Option 2) clone repository (for development, and finer tuning)

```bash
git clone https://github.com/redwoodresearch/Easy-Transformer/
pip install -r requirements.txt
```

# In this repo

In this repo, you can find the following notebooks (some are in `easy_transformer/`):

* `experiments.py`: a notebook of several of the most interesting experiments of the IOI project.
* `completeness.py`: a notebook that generate the completeness plots in the paper, and implements the completeness functions.
* `minimality.py`: as above for minimality.
* `advex.py`: a notebook that generates adversarial examples as in the paper.
`
# Easy Transformer

## An implementation of transformers tailored for mechanistic interpretability.

It supports the importation of open sources models, a convenient handling of hooks to get access to intermediate activations and features to perform simple emperiments such as ablations and patching.

A demo notebook can be found [here](https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/main/EasyTransformer_Demo.ipynb) and a more comprehensive description of the library can be found [here](https://colab.research.google.com/drive/1_tH4PfRSPYuKGnJbhC1NqFesOYuXrir_#scrollTo=zs8juArnyuyB)