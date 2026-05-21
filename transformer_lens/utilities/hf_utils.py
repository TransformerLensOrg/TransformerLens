"""hf_utils.

This module contains utility functions related to HuggingFace
"""

from __future__ import annotations

import errno
import inspect
import json
import logging
import os
import random
import shutil
import stat
import time
from typing import Any, Callable, Dict, TypeVar

import torch
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset
from datasets.load import load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE

CACHE_DIR = HF_HUB_CACHE
logger = logging.getLogger(__name__)

T = TypeVar("T")

_HF_RETRY_MAX_ATTEMPTS = 3
_HF_RETRY_BASE_DELAY_SECONDS = 10.0
_HF_RETRY_MAX_DELAY_SECONDS = 120.0


def _is_hf_rate_limit_error(exc: BaseException) -> bool:
    """Duck-typed check for HTTP 429 — covers HfHubHTTPError, requests.HTTPError, and subclasses."""
    response = getattr(exc, "response", None)
    return response is not None and getattr(response, "status_code", None) == 429


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Parse the Retry-After header from a 429 response, if present and numeric."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


_TL_RETRY_WRAPPED_ATTR = "_tl_hf_retry_wrapped"


def enable_hf_retry() -> None:
    """Globally wrap transformers ``Auto*.from_pretrained`` with retry-on-429.

    Opt-in via ``TRANSFORMERLENS_HF_RETRY=1`` or by calling this function.
    Idempotent. See :func:`call_hf_with_retry`.
    """
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoModel,
        AutoProcessor,
        AutoTokenizer,
    )

    for cls in (AutoConfig, AutoModel, AutoTokenizer, AutoProcessor, AutoFeatureExtractor):
        original = cls.from_pretrained
        if getattr(original, _TL_RETRY_WRAPPED_ATTR, False):
            continue
        underlying = original.__func__ if hasattr(original, "__func__") else original

        def _wrapped(klass, *args: Any, _orig: Any = underlying, **kwargs: Any) -> Any:
            return call_hf_with_retry(_orig, klass, *args, **kwargs)

        setattr(_wrapped, _TL_RETRY_WRAPPED_ATTR, True)
        cls.from_pretrained = classmethod(_wrapped)


def call_hf_with_retry(
    func: Callable[..., T],
    *args: Any,
    max_attempts: int = _HF_RETRY_MAX_ATTEMPTS,
    base_delay: float = _HF_RETRY_BASE_DELAY_SECONDS,
    **kwargs: Any,
) -> T:
    """Retry ``func(*args, **kwargs)`` on HTTP 429, honoring ``Retry-After``.

    Exponential backoff with ±20% jitter, capped at ``_HF_RETRY_MAX_DELAY_SECONDS``.
    Non-429 exceptions propagate immediately.
    """
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if not _is_hf_rate_limit_error(exc) or attempt == max_attempts - 1:
                raise
            wait = _retry_after_seconds(exc)
            if wait is None:
                wait = min(base_delay * (2**attempt), _HF_RETRY_MAX_DELAY_SECONDS)
                wait *= 0.8 + 0.4 * random.random()
            logger.warning(
                "HuggingFace Hub rate-limited (HTTP 429); retrying in %.1fs (attempt %d/%d)",
                wait,
                attempt + 1,
                max_attempts,
            )
            time.sleep(wait)
    raise RuntimeError("call_hf_with_retry exited loop without returning or raising")


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment. Returns None if not set."""
    return os.environ.get("HF_TOKEN", "") or None


def get_rotary_pct_from_config(config: Any) -> float:
    """Get the rotary percentage from a config object.

    In transformers v5, rotary_pct was moved to rope_parameters['partial_rotary_factor'].
    This function handles both the old and new config formats.

    Args:
        config: Config object (HuggingFace or custom)

    Returns:
        float: The rotary percentage (0.0 to 1.0)
    """
    if config is None:
        return 1.0

    # Try the old attribute first (transformers v4)
    if hasattr(config, "rotary_pct"):
        return getattr(config, "rotary_pct", 1.0)

    # Try the new rope_parameters format (transformers v5)
    if hasattr(config, "rope_parameters"):
        rope_params = getattr(config, "rope_parameters", None)
        if isinstance(rope_params, dict) and "partial_rotary_factor" in rope_params:
            return rope_params["partial_rotary_factor"]

    # Default to 1.0 (full rotary) if not found
    return 1.0


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
    file_path = call_hf_with_retry(
        hf_hub_download,
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

    This function is safe to call in parallel test execution - it will handle race
    conditions where multiple workers might try to delete the same directory.

    Parameters:
    None

    Returns:
    None
    """

    print("Deleting Hugging Face cache directory and all its contents.")

    # Check if cache directory exists
    if not os.path.exists(CACHE_DIR):
        return

    try:
        # Use a custom error handler that only ignores specific race condition errors
        def handle_remove_readonly(func, path, exc_info):
            """Error handler for Windows readonly files and race conditions."""

            excvalue = exc_info[1]
            # Ignore "directory not empty" errors (race condition - another process deleted contents)
            if isinstance(excvalue, OSError) and excvalue.errno == errno.ENOTEMPTY:
                return
            # Ignore "no such file or directory" errors (race condition - already deleted)
            if isinstance(excvalue, FileNotFoundError):
                return
            if isinstance(excvalue, OSError) and excvalue.errno == errno.ENOENT:
                return
            # For readonly files on Windows, try to make writable and retry
            if os.path.exists(path) and not os.access(path, os.W_OK):
                try:
                    os.chmod(path, stat.S_IWUSR)
                    func(path)
                except (OSError, FileNotFoundError):
                    # File disappeared or became inaccessible - race condition, ignore
                    return
            else:
                raise

        shutil.rmtree(CACHE_DIR, onerror=handle_remove_readonly)
    except FileNotFoundError:
        # Directory was deleted by another process - that's fine
        pass
    except OSError as e:
        # Only ignore "directory not empty" and "no such file" errors (race conditions)
        if e.errno not in (errno.ENOTEMPTY, errno.ENOENT):
            print(f"Warning: Could not fully clear cache: {e}")


def keep_single_column(dataset: Dataset | IterableDataset, col_name: str):
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
