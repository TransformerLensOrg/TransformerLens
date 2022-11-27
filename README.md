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

We recommend using `black` and `isort`.

### Testing

TODO this doesn't seem to work?

`python3 easy_transformer/tests/test_experiments.py`

When testing type annotations, check out `easy_transformer/tests/test_type_annotations.py`.

Another way to test is to open an `ipython` terminal and make sure that you can 
[import easy_transformer]. If there are import errors, etc., they'll show up here!
(You can make this tighter with `ipython -c 'import easy_transformer'`.)

#### Pytest

- `pytest` from the root of the project. 
- `pytest --collect-only` will fail if some of the code has type errors
- Test a particular file with e.g. `pytest easy_transformer/tests/test_bert.py`
- You might also like `pytest -q` for a quieter life. :)
- To run fewer tests, e.g. to just run one `test_model`, `pytest -k "test_model[gpt2-small-5.331855773925781]"`.
- To report progress as you go: `pytest --verbose`.
- If you're running the tests in `test_easy_transformer.py`, you might wanna run them on a `colab`.
  E.g.: https://colab.research.google.com/drive/1MKJ6nkRTNWXqXQYK524Ojjxvn7NkuDfI?usp=sharing
