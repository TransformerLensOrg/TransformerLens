"""hf_utils.

This module contains utility functions related to HuggingFace
"""

from __future__ import annotations

import inspect
import json
import shutil
from typing import Any, Callable, Dict

import torch
import transformers
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from huggingface_hub import hf_hub_download

CACHE_DIR = transformers.TRANSFORMERS_CACHE


def select_compatible_kwargs(kwargs_dict: Dict[str, Any], callable: Callable) -> Dict[str, Any]:
    """Return a dict with the elements kwargs_dict that are parameters of callable"""
    return {k: v for k, v in kwargs_dict.items() if k in inspect.getfullargspec(callable).args}


def download_file_from_hf(
    repo_name,
    file_name,
    subfolder=".",
    cache_dir=CACHE_DIR,
    force_is_torch=False,
    **kwargs,
):
    """
    Helper function to download files from the HuggingFace Hub, from subfolder/file_name in repo_name, saving locally to cache_dir and returning the loaded file (if a json or Torch object) and the file path otherwise.

    If it's a Torch file without the ".pth" extension, set force_is_torch=True to load it as a Torch object.
    """
    file_path = hf_hub_download(
        repo_id=repo_name,
        filename=file_name,
        subfolder=subfolder,
        cache_dir=cache_dir,
        **select_compatible_kwargs(kwargs, hf_hub_download),
    )

    if file_path.endswith(".pth") or force_is_torch:
        return torch.load(file_path, map_location="cpu", weights_only=False)
    elif file_path.endswith(".json"):
        return json.load(open(file_path, "r"))
    else:
        print("File type not supported:", file_path.split(".")[-1])
        return file_path


def clear_huggingface_cache():
    """
    Deletes the Hugging Face cache directory and all its contents.

    This function deletes the Hugging Face cache directory, which is used to store downloaded models and their associated files. Deleting the cache directory will remove all the downloaded models and their files, so you will need to download them again if you want to use them in your code.

    Parameters:
    None

    Returns:
    None
    """
    print("Deleting Hugging Face cache directory and all its contents.")
    shutil.rmtree(CACHE_DIR)


def keep_single_column(dataset: Dataset, col_name: str):
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name - useful when we want to tokenize and mix together different strings
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


def get_dataset(dataset_name: str, **kwargs) -> Dataset:
    """
    Returns a small HuggingFace dataset, for easy testing and exploration. Accesses several convenience datasets with 10,000 elements (dealing with the enormous 100GB - 2TB datasets is a lot of effort!). Note that it returns a dataset (ie a dictionary containing all the data), *not* a DataLoader (iterator over the data + some fancy features). But you can easily convert it to a DataLoader.

    Each dataset has a 'text' field, which contains the relevant info, some also have several meta data fields

    Kwargs will be passed to the huggingface dataset loading function, e.g. "data_dir"

    Possible inputs:
    * openwebtext (approx the GPT-2 training data https://huggingface.co/datasets/openwebtext)
    * pile (The Pile, a big mess of tons of diverse data https://pile.eleuther.ai/)
    * c4 (Colossal, Cleaned, Common Crawl - basically openwebtext but bigger https://huggingface.co/datasets/c4)
    * code (Codeparrot Clean, a Python code dataset https://huggingface.co/datasets/codeparrot/codeparrot-clean )
    * c4_code (c4 + code - the 20K data points from c4-10k and code-10k. This is the mix of datasets used to train my interpretability-friendly models, though note that they are *not* in the correct ratio! There's 10K texts for each, but about 22M tokens of code and 5M tokens of C4)
    * wiki (Wikipedia, generated from the 20220301.en split of https://huggingface.co/datasets/wikipedia )
    """
    dataset_aliases = {
        "openwebtext": "stas/openwebtext-10k",
        "owt": "stas/openwebtext-10k",
        "pile": "NeelNanda/pile-10k",
        "c4": "NeelNanda/c4-10k",
        "code": "NeelNanda/code-10k",
        "python": "NeelNanda/code-10k",
        "c4_code": "NeelNanda/c4-code-20k",
        "c4-code": "NeelNanda/c4-code-20k",
        "wiki": "NeelNanda/wiki-10k",
    }
    if dataset_name in dataset_aliases:
        dataset = load_dataset(dataset_aliases[dataset_name], split="train", **kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return dataset
