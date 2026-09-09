"""Sphinx configuration.

https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
# pylint: disable=invalid-name
from pathlib import Path
from typing import Any, Optional

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
napoleon_custom_sections = [
    "Motivation:",
    "Warning:",
    "Getting Started:",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "TransformerLens Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/transformer_lens_logo.png"
html_favicon = "favicon.ico"

# Fix to get Plotly Working
nbsphinx_prolog = r"""
.. raw:: html

    <script src="https://cdn.jsdelivr.net/npm/requirejs@2.3.6/require.min.js"></script>
    <script>
    require=requirejs;
    require.config({
        paths: {
            plotly: 'https://cdn.plot.ly/plotly-latest.min.js'
        }
    });
    </script>
"""

# -- Sphinx-Apidoc Configuration ---------------------------------------------

# Functions to ignore as they're not interesting to the end user
functions_to_ignore = [
    # functions relocated out of the deleted loading_from_pretrained.py
    "convert_neel_solu_old_weights",
    "get_official_model_name",
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
    "special-members": "__getitem__, __len__, __iter__",
}


def run_apidoc(_app: Optional[Any] = None):
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

    # Exclude modules with dedicated pages
    excluded_modules = [
        "ActivationCache.py",
        "FactoredMatrix.py",
        "SVDInterpreter.py",
    ]
    args = [
        "--force",  # Overwrite existing files
        "--separate",  # Put documentation for each module on its own page.
        "--templatedir=" + str(template_dir),  # Use custom templates
        "-o",
        str(output_path),
        str(package_path),
    ] + [str(package_path / module) for module in excluded_modules]

    # Call sphinx-apidoc
    apidoc.main(args)

    # Add exclude-members for modules with separate docs
    package_excludes = {
        "transformer_lens.rst": "ActivationCache, FactoredMatrix, SVDInterpreter, EasyTransformerConfig",
        "transformer_lens.config.rst": "TransformerBridgeConfig, TransformerLensConfig",
        "transformer_lens.conversion_utils.rst": "HookConversionSet",
    }

    for filename, excluded_members in package_excludes.items():
        rst_file = output_path / filename
        if rst_file.exists():
            content = rst_file.read_text()
            # Patch automodule directive
            package_name = filename.replace(".rst", "")
            old_directive = f".. automodule:: {package_name}\n   :members:\n   :undoc-members:\n   :show-inheritance:"
            new_directive = f"{old_directive}\n   :exclude-members: {excluded_members}"
            content = content.replace(old_directive, new_directive)
            rst_file.write_text(content)


# -- Sphinx Notebook Demo Config ---------------------------------------------

nbsphinx_execute = "never"  # Don't execute notebooks during build (avoids device/memory issues).

# -- Sphinx Setup Overrides --------------------------------------------------


def setup(app):
    """Sphinx setup overrides."""
    # Connect functions to run when watch detects a file change
    app.connect("builder-inited", run_apidoc)
