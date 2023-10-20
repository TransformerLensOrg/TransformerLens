"""Sphinx configuration.

https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
# pylint: disable=invalid-name
from pathlib import Path

from sphinx.ext import apidoc

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
    "nbsphinx",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]


# -- Napoleon Extension Configuration -----------------------------------------

napoleon_include_init_with_doc = True
napoleon_use_admonition_for_notes = True
napoleon_custom_sections = ["Motivation:"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "TransformerLens Documentation"
html_static_path = ["_static"]
html_logo = "_static/transformer_lens_logo.png"
html_favicon = "favicon.ico"
html_js_files = [
    "https://cdn.plot.ly/plotly-2.26.0.min.js",
]

# -- Sphinx-Apidoc Configuration ---------------------------------------------

# Functions to ignore as they're not interesting to the end user
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

# Default AutoDoc Options
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_default_options
autodoc_default_options = {
    "exclude-members": ", ".join(functions_to_ignore),
}


def run_apidoc(_):
    """Run Sphinx-Apidoc.

    Allows us to automatically generate API documentation from docstrings, every time we build the
    docs.
    """

    # Path to the package codebase
    package_path = Path(__file__).resolve().parents[2] / "transformer_lens"

    # Template directory
    template_dir = Path(__file__).resolve().parent / "apidoc_templates"

    # Output path for the generated reStructuredText files
    generated_path = Path(__file__).resolve().parent / "generated"
    output_path = generated_path / "code"
    generated_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # Arguments for sphinx-apidoc
    args = [
        "--force",  # Overwrite existing files
        "--separate",  # Put documentation for each module on its own page.
        "--templatedir=" + str(template_dir),  # Use custom templates
        "-o",
        str(output_path),
        str(package_path),
    ]

    # Call sphinx-apidoc
    apidoc.main(args)


# -- Sphinx Notebook Demo Config ---------------------------------------------

nbsphinx_execute = "never"  # Don't re-run

# -- Sphinx Setup Overrides --------------------------------------------------


def setup(app):
    """Sphinx setup overrides."""
    # Connect the run_apidoc function to the builder-inited event
    app.connect("builder-inited", run_apidoc)
