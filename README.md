# Hooked Transformer

## An implementation of transformers tailored for mechanistic interpretability.

It supports the importation of open sources models, a convenient handling of hooks 
to get access to intermediate activations and features to perform simple emperiments such as ablations and patching.

A demo notebook can be found [here](https://colab.research.google.com/github/neelnanda-io/Hooked-Transformer/blob/main/HookedTransformer_Demo.ipynb) and a more comprehensive description of the library can be found [here](https://colab.research.google.com/drive/1_tH4PfRSPYuKGnJbhC1NqFesOYuXrir_#scrollTo=zs8juArnyuyB)


## Installation

`pip install git+https://github.com/neelnanda-io/TransformerLens`

## Local Development

### Setup

This project uses [Poetry](https://python-poetry.org/docs/#installation) for package management. Install as follows (this will also setup your virtual environment):

```bash
poetry config virtualenvs.in-project true
poetry install --with dev
```

Optionally, if you want Jupyter Lab you can run `poetry run pip install jupyterlab` (to install in the same virtual environment), and then run with `poetry run jupyter lab`.

Then the library can be imported as `import transformer_lens`.

### Testing

TODO this doesn't seem to work?

`python3 transformer_lens/tests/test_experiments.py`

When testing type annotations, check out `transformer_lens/tests/test_type_annotations.py`.