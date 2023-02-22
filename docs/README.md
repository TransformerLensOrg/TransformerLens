
# Transformer-Lens Docs


This repo contains the [NEW website]() for [TransformerLens](https://github.com/website_address_add_later.). This site is currently in Beta and we are in the process of adding/editing information. 

The documentation uses Sphinx. However, the documentation is written in regular md, NOT rst.

If you are modifying a non-environment page or an atari environment page, please PR this repo. Otherwise, follow the steps below:

## Build the Documentation

Install the required packages:

Need to use python 3.9 (3.10 has an issue with napoleon extension) and below 3.8 doesn't have sphinx suppourt.
```
poetry install --extras docs
```

Using api doc to make the rst files

```bash
poetry run sphinx-apidoc -f -o docs/source .

# make the model tables file
poetry run python -m transformer_lens.make_docs
mv model_properties_table.md docs/source/
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' '1s/^/# Model Properties Table\n\n/' docs/source/model_properties_table.md
else
    sed -i '1s/^/# Model Properties Table\n\n/' docs/source/model_properties_table.md
fi
cd docs

# build the docs from source
poetry run  sphinx-autobuild -b dirhtml ./source build/html
```