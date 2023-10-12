""" 
Generate a markdown table summarizing properties of pretrained models.

This script extracts various properties of pretrained models from the 
`easy_transformer` library, such as the number of parameters, layers, and heads, 
among others, and generates a markdown table. This table is saved to the 
docs directory.
"""
import subprocess
from functools import lru_cache
from pathlib import Path

import pandas as pd

from transformer_lens import loading

# Docs Directories
CURRENT_DIR = Path(__file__).parent
SOURCE_PATH = CURRENT_DIR / "../docs/source"
BUILD_PATH = CURRENT_DIR / "../docs/build"


@lru_cache(maxsize=None)
def get_config(model_name):
    """Retrieve the configuration of a pretrained model.

    Args:
        model_name (str): Name of the pretrained model.

    Returns:
        dict: Configuration of the pretrained model.
    """
    return loading.get_pretrained_model_config(model_name)


def get_property(name, model_name):
    """Retrieve a specific property of a pretrained model.

    Args:
        name (str): Name of the property to retrieve.
        model_name (str): Name of the pretrained model.

    Returns:
        str: Value of the specified property.
    """
    cfg = get_config(model_name)

    if name == "act_fn":
        if cfg.attn_only:
            return "attn_only"
        if cfg.act_fn == "gelu_new":
            return "gelu"
        if cfg.act_fn == "gelu_fast":
            return "gelu"
        if cfg.act_fn == "solu_ln":
            return "solu"
        return cfg.act_fn
    if name == "n_params":
        n_params = cfg.n_params
        if n_params < 1e4:
            return f"{n_params/1e3:.1f}K"
        if n_params < 1e6:
            return f"{round(n_params/1e3)}K"
        if n_params < 1e7:
            return f"{n_params/1e6:.1f}M"
        if n_params < 1e9:
            return f"{round(n_params/1e6)}M"
        if n_params < 1e10:
            return f"{n_params/1e9:.1f}B"
        if n_params < 1e12:
            return f"{round(n_params/1e9)}B"
        raise ValueError(f"Passed in {n_params} above 1T?")
    return cfg.to_dict()[name]


def generate_model_table():
    """Generate a markdown table summarizing properties of pretrained models."""

    # Create the table
    column_names = [
        "n_params",
        "n_layers",
        "d_model",
        "n_heads",
        "act_fn",
        "n_ctx",
        "d_vocab",
        "d_head",
        "d_mlp",
    ]
    df = pd.DataFrame(
        {
            name: [
                get_property(name, model_name)
                for model_name in loading.DEFAULT_MODEL_ALIASES
            ]
            for name in column_names
        },
        index=loading.DEFAULT_MODEL_ALIASES,
    )

    # Convert to markdown (with a title)
    markdown_string = df.to_markdown()
    markdown_string = "# Model Properties Table\n\n" + markdown_string

    # Save to the docs directory
    file_path = CURRENT_DIR / "../docs/source/model_properties_table.md"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(markdown_string)


def build_docs():
    """Build the docs."""
    generate_model_table()
    subprocess.run(["sphinx-build", SOURCE_PATH, BUILD_PATH], check=True)


def docs_hot_reload():
    """Hot reload the docs."""
    generate_model_table()
    subprocess.run(
        ["sphinx-autobuild", "--open-browser", SOURCE_PATH, BUILD_PATH], check=True
    )
