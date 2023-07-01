# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TransformerLens"
copyright = "2023, Neel Nanda"
author = "Neel Nanda"
release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.githubpages",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "TransformerLens Documentation"
html_static_path = ["_static"]

html_logo = "_static/transformer_lens_logo.png"

html_favicon = "favicon.ico"


# -- Ignore some functions that are not interesting for end users ------------

functions_to_ignore = [
    # functions from load_from_pretrained.py
    "convert_hf_model_config",
    "convert_bert_weights",
    "convert_gpt2_weights",
    "convert_gptj_weights",
    "convert_llama_weights",
    "convert_mingpt_weights",
    "convert_neel_solu_old_weights",
    "convert_neo_weights",
    "convert_neox_weights",
    "convert_neel_model_config",
    "convert_opt_weights",
    "fill_missing_keys",
    "get_basic_config",
    "get_official_model_name",
    "get_pretrained_state_dict",
    "make_model_alias_map",
    # functions from make_docs.py
    "get_config",
    "get_property",
    # functions from patching.py
    "make_df_from_ranges",
    # functions from utils.py
    "check_structure",
    "clear_huggingface_cache",
    "select_compatible_kwargs",
]

autodoc_default_options = {"exclude-members": ", ".join(functions_to_ignore)}
