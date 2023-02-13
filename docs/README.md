
# Transformer-Lens Docs


This repo contains the [NEW website]() for [TransformerLens](https://github.com/website_address_add_later.). This site is currently in Beta and we are in the process of adding/editing information. 

The documentation uses Sphinx. However, the documentation is written in regular md, NOT rst.

If you are modifying a non-environment page or an atari environment page, please PR this repo. Otherwise, follow the steps below:

## Instructions for modifying environment pages

### Editing an environment page

If you are editing an Atari environment, directly edit the md file in this repository. 

Otherwise, fork Gym and edit the docstring in the environment's Python file. Then, pip install your Gym fork and run `docs/scripts/gen_mds.py` in this repo. This will automatically generate a md documentation file for the environment.

## Build the Documentation

Install the required packages and Gym (or your fork):

```
poetry poetry install --with dev
```

To build the documentation once:

```
cd docs
make dirhtml
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b dirhtml ./source build/html
```


# Temporary

Need to use python 3.9 (3.10 has an issue with napoleon extension) and below 3.8 doesn't have sphinx suppourt.
```
poetry install --extras docs
```

Using api doc to make the rst files
```
poetry run sphinx-apidoc -f -o docs/source .
cd docs
poetry run  sphinx-autobuild -b dirhtml ./source build/html
```