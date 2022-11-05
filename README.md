# Easy Transformer

## An implementation of transformers tailored for mechanistic interpretability.

It supports the importation of open sources models, a convenient handling of hooks 
to get access to intermediate activations and features to perform simple emperiments such as ablations and patching.

A demo notebook can be found [here](https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/main/EasyTransformer_Demo.ipynb) and a more comprehensive description of the library can be found [here](https://colab.research.google.com/drive/1_tH4PfRSPYuKGnJbhC1NqFesOYuXrir_#scrollTo=zs8juArnyuyB)


## Installation

`pip install git+https://github.com/neelnanda-io/Easy-Transformer`

## Local Development

### Setup

From the root directory of this repository: `pip install -e .`

Then the library can be imported as `import easy_transformer`.

### Testing

TODO this doesn't seem to work?

`python3 easy_transformer/tests/test_experiments.py`

When testing type annotations, check out `easy_transformer/tests/test_type_annotations.py`.