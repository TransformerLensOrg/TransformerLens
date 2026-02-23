"""Build the API Documentation."""

import base64
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import warnings
from copy import deepcopy
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

import pandas as pd  # type: ignore[import-untyped]
import torch
import tqdm  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from muutils.dictmagic import TensorDictFormats, condense_tensor_dict
from muutils.misc import shorten_numerical_to_str
from transformers import AutoTokenizer  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizer

import transformer_lens  # type: ignore[import-untyped]
from transformer_lens import (
    ActivationCache,
    HookedTransformer,
    HookedTransformerConfig,
    loading,
    supported_models,
)
from transformer_lens.loading_from_pretrained import (  # type: ignore[import-untyped]
    NON_HF_HOSTED_MODEL_NAMES,
    get_pretrained_model_config,
)

DEVICE: torch.device = torch.device("meta")

# disable the symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if not HF_TOKEN.startswith("hf_"):
        raise ValueError("Invalid Hugging Face token")
except Exception as e:
    warnings.warn(
        f"Failed to get Hugging Face token -- info about certain models will be limited\n{e}"
    )

# Docs Directories
CURRENT_DIR: Path = Path(__file__).parent
SOURCE_PATH: Path = CURRENT_DIR / "../docs/source"
BUILD_PATH: Path = CURRENT_DIR / "../docs/build"
PACKAGE_DIR: Path = CURRENT_DIR.parent
DEMOS_DIR: Path = CURRENT_DIR.parent / "demos"
GENERATED_DIR: Path = CURRENT_DIR.parent / "docs/source/generated"
STATIC_DIR: Path = CURRENT_DIR.parent / "docs/source/_static"


@lru_cache(maxsize=None)
def get_config(model_name: str):
    """Retrieve the configuration of a pretrained model.

    Args:
        model_name (str): Name of the pretrained model.

    Returns:
        dict: Configuration of the pretrained model.
    """
    return loading.get_pretrained_model_config(model_name)


# manually defined known model types
KNOWN_MODEL_TYPES: Sequence[str] = (
    "gpt2",
    "distillgpt2",
    "opt",
    "gpt-neo",
    "gpt-j",
    "gpt-neox",
    "stanford-gpt2",
    "pythia",
    "solu",
    "gelu",
    "attn-only",
    "llama",
    "Llama-2",
    "bert",
    "tiny-stories",
    "stablelm",
    "bloom",
    "qwen",
    "mistral",
    "CodeLlama",
    "phi",
    "gemma",
    "yi",
    "t5",
    "mixtral",
    "Qwen2",
)

MODEL_ALIASES_MAP: dict[str, str] = transformer_lens.loading.make_model_alias_map()

# these will be copied as table columns
CONFIG_ATTRS_COPY: Sequence[str] = (
    "n_params",
    "n_layers",
    "n_heads",
    "d_model",
    "d_vocab",
    "act_fn",
    "positional_embedding_type",
    "parallel_attn_mlp",
    "original_architecture",
    "normalization_type",
)

# modify certain values when saving config
CONFIG_VALUES_PROCESS: dict[str, Callable] = {
    "initializer_range": float,
    "dtype": str,
    "device": str,
}

COLUMNS_ABRIDGED: Sequence[str] = (
    "name.default_alias",
    "name.huggingface",
    "n_params.as_str",
    "n_params.as_int",
    "cfg.n_params",
    "cfg.n_layers",
    "cfg.n_heads",
    "cfg.d_model",
    "cfg.d_vocab",
    "cfg.act_fn",
    "cfg.positional_embedding_type",
    "cfg.parallel_attn_mlp",
    "cfg.original_architecture",
    "cfg.normalization_type",
    "tokenizer.name",
    "tokenizer.class",
    "tokenizer.vocab_size",
    "tokenizer.vocab_hash",
)


def get_tensor_shapes(
    model: HookedTransformer,
    tensor_dims_fmt: TensorDictFormats = "yaml",
    except_if_forward_fails: bool = False,
) -> dict:
    """get the tensor shapes from a model"""
    model_info: dict = dict()
    # state dict
    model_info["tensor_shapes.state_dict"] = condense_tensor_dict(
        model.state_dict(), fmt=tensor_dims_fmt
    )
    model_info["tensor_shapes.state_dict.raw__"] = condense_tensor_dict(
        model.state_dict(), fmt="dict"
    )

    try:
        # input shape for activations -- "847"~="bat", subtract 7 for the context window to make it unique
        input_shape: tuple[int, int] = (847, model.cfg.n_ctx - 7)
        # why? to replace the batch and seq_len dims with "batch" and "seq_len" in the yaml
        dims_names_map: dict[int, str] = {
            input_shape[0]: "batch",
            input_shape[1]: "seq_len",
        }
        # run with cache to activation cache
        with torch.no_grad():
            cache: ActivationCache
            _, cache = model.run_with_cache(
                torch.empty(input_shape, dtype=torch.long, device=DEVICE)
            )
        # condense using muutils and store
        model_info["tensor_shapes.activation_cache"] = condense_tensor_dict(
            cache.cache_dict,
            fmt=tensor_dims_fmt,
            dims_names_map=dims_names_map,
        )
        model_info["tensor_shapes.activation_cache.raw__"] = condense_tensor_dict(
            cache.cache_dict,
            fmt="dict",
            dims_names_map=dims_names_map,
        )
    except Exception as e:
        msg: str = f"Failed to get activation cache for '{model.cfg.model_name}':\n{e}"
        if except_if_forward_fails:
            raise ValueError(msg) from e
        else:
            warnings.warn(msg)

    return model_info


def tokenizer_vocab_hash(tokenizer: PreTrainedTokenizer) -> str:
    # sort
    vocab: dict[str, int]
    try:
        vocab = tokenizer.vocab
    except Exception:
        vocab = tokenizer.get_vocab()

    vocab_hashable: list[tuple[str, int]] = list(
        sorted(
            vocab.items(),
            key=lambda x: x[1],
        )
    )
    # hash it
    hash_obj = hashlib.sha1(bytes(str(vocab_hashable), "UTF-8"))
    # convert to base64
    return base64.b64encode(
        hash_obj.digest(),
        altchars=b"-_",  # - and _ as altchars
    ).decode("UTF-8")


def get_tokenizer_info(model: HookedTransformer) -> dict:
    tokenizer: PreTrainedTokenizer = model.tokenizer
    model_info: dict = dict()
    # basic info
    model_info["tokenizer.name"] = tokenizer.name_or_path
    model_info["tokenizer.vocab_size"] = int(tokenizer.vocab_size)
    model_info["tokenizer.max_len"] = int(tokenizer.model_max_length)
    model_info["tokenizer.class"] = tokenizer.__class__.__name__

    # vocab hash
    model_info["tokenizer.vocab_hash"] = tokenizer_vocab_hash(tokenizer)
    return model_info


def get_model_info(
    model_name: str,
    include_cfg: bool = True,
    include_tensor_dims: bool = True,
    include_tokenizer_info: bool = True,
    tensor_dims_fmt: TensorDictFormats = "yaml",
    allow_warn: bool = True,
) -> tuple[str, dict]:
    """get information about the model from the default alias model name

    # Parameters:
     - `model_name : str`
        the default alias model name
     - `include_cfg : bool`
        whether to include the model config as a yaml string
       (defaults to `True`)
     - `include_tensor_dims : bool`
        whether to include the model tensor shapes
       (defaults to `True`)
     - `include_tokenizer_info : bool`
        whether to include the tokenizer info
        (defaults to `True`)
     - `tensor_dims_fmt : TensorDictFormats`
        the format of the tensor shapes. one of "yaml", "json", "dict"
       (defaults to `"yaml"`)
    """

    # assumes the input is a default alias
    if model_name not in supported_models.DEFAULT_MODEL_ALIASES:
        raise ValueError(f"Model name '{model_name}' not found in default aliases")

    # get the names and model types
    official_name: Optional[str] = MODEL_ALIASES_MAP.get(model_name, None)
    model_info: dict = {
        "name.default_alias": model_name,
        "name.huggingface": official_name,
        "name.aliases": ", ".join(
            list(supported_models.MODEL_ALIASES.get(official_name, []))  # type: ignore[arg-type]
        ),
        "model_type": None,
    }

    # Split the model name into parts
    parts: list[str] = model_name.split("-")

    # identify model type by known types
    for known_type in KNOWN_MODEL_TYPES:
        if known_type in model_name:
            model_info["model_type"] = known_type
            break

    # search for model size in name
    param_count_from_name: Optional[str] = None
    for part in parts:
        if part[-1].lower() in ["m", "b", "k"] and part[:-1].replace(".", "", 1).isdigit():
            param_count_from_name = part
            break

    # update model info from config
    model_cfg: HookedTransformerConfig = get_pretrained_model_config(model_name)
    model_info.update(
        {
            "name.from_cfg": model_cfg.model_name,
            "n_params.as_str": shorten_numerical_to_str(model_cfg.n_params),  # type: ignore[arg-type]
            "n_params.as_int": model_cfg.n_params,
            "n_params.from_name": param_count_from_name,
            **{f"cfg.{attr}": getattr(model_cfg, attr) for attr in CONFIG_ATTRS_COPY},
        }
    )

    # put the whole config as yaml (for readability)
    if include_cfg:
        # modify certain values to make them pretty-printable
        model_cfg_dict: dict = {
            key: (val if key not in CONFIG_VALUES_PROCESS else CONFIG_VALUES_PROCESS[key](val))
            for key, val in model_cfg.to_dict().items()
        }

        # raw config
        model_info["config.raw__"] = model_cfg_dict
        # dump to yaml
        model_info["config"] = yaml.dump(
            model_cfg_dict,
            default_flow_style=False,
            sort_keys=False,
            width=1000,
        )

    # get tensor shapes
    if include_tensor_dims or include_tokenizer_info:
        # set default device to meta, so that we don't actually allocate tensors
        # this can't be done at the root level because it would break other tests when we import this file
        # and it has to be done inside this function due to usage of multiprocessing
        with torch.device(DEVICE):
            got_model: bool = False
            try:
                # copy the config, so we can modify it
                model_cfg_copy: HookedTransformerConfig = deepcopy(model_cfg)
                # set device to "meta" -- don't actually initialize the model with real tensors
                model_cfg_copy.device = str(DEVICE)
                if not include_tokenizer_info:
                    # don't need to download the tokenizer
                    model_cfg_copy.tokenizer_name = None
                # init the fake model
                model: HookedTransformer = HookedTransformer(model_cfg_copy, move_to_device=True)
                # HACK: use https://huggingface.co/huggyllama to get tokenizers for original llama models
                if model.cfg.tokenizer_name in NON_HF_HOSTED_MODEL_NAMES:
                    model.set_tokenizer(
                        AutoTokenizer.from_pretrained(
                            f"huggyllama/{model.cfg.tokenizer_name.removesuffix('-hf')}",
                            add_bos_token=True,
                            token=HF_TOKEN,
                            legacy=False,
                        )
                    )
                got_model = True
            except Exception as e:
                msg: str = f"Failed to init model '{model_name}', can't get tensor shapes or tokenizer info"
                if allow_warn:
                    warnings.warn(f"{msg}:\n{e}")
                else:
                    raise ValueError(msg) from e

            if got_model:
                if include_tokenizer_info:
                    try:
                        tokenizer_info: dict = get_tokenizer_info(model)
                        model_info.update(tokenizer_info)
                    except Exception as e:
                        msg = f"Failed to get tokenizer info for model '{model_name}'"
                        if allow_warn:
                            warnings.warn(f"{msg}:\n{e}")
                        else:
                            raise ValueError(msg) from e

                if include_tensor_dims:
                    try:
                        tensor_shapes_info: dict = get_tensor_shapes(model, tensor_dims_fmt)
                        model_info.update(tensor_shapes_info)
                    except Exception as e:
                        msg = f"Failed to get tensor shapes for model '{model_name}'"
                        if allow_warn:
                            warnings.warn(f"{msg}:\n{e}")
                        else:
                            raise ValueError(msg) from e

    return model_name, model_info


def safe_try_get_model_info(
    model_name: str, kwargs: Optional[dict] = None
) -> tuple[str, Union[dict, Exception]]:
    """for parallel processing, to catch exceptions and return the exception instead of raising them"""
    if kwargs is None:
        kwargs = {}
    try:
        return get_model_info(model_name, **kwargs)
    except Exception as e:
        warnings.warn(f"Failed to get model info for '{model_name}': {e}")
        return model_name, e


def make_model_table(
    verbose: bool,
    allow_except: bool = False,
    parallelize: Union[bool, int] = True,
    model_names_pattern: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """make table of all models. kwargs passed to `get_model_info()`"""
    model_names: list[str] = list(supported_models.DEFAULT_MODEL_ALIASES)
    model_data: list[tuple[str, Union[dict, Exception]]] = list()

    # filter by regex pattern if provided
    if model_names_pattern:
        model_names = [
            model_name for model_name in model_names if model_names_pattern in model_name
        ]

    if parallelize:
        # parallel
        n_processes: int = parallelize if int(parallelize) > 1 else multiprocessing.cpu_count()
        if verbose:
            print(f"running in parallel with {n_processes=}")
        with multiprocessing.Pool(processes=n_processes) as pool:
            # Use imap for ordered results, wrapped with tqdm for progress bar
            imap_results: list[tuple[str, Union[dict, Exception]]] = list(
                tqdm.tqdm(
                    pool.imap(
                        partial(safe_try_get_model_info, **kwargs),
                        model_names,
                    ),
                    total=len(model_names),
                    desc="Loading model info",
                    disable=not verbose,
                )
            )

        model_data = imap_results

    else:
        # serial
        with tqdm.tqdm(
            supported_models.DEFAULT_MODEL_ALIASES,
            desc="Loading model info",
            disable=not verbose,
        ) as pbar:
            for model_name in pbar:
                pbar.set_postfix_str(f"model: '{model_name}'")
                try:
                    model_data.append(get_model_info(model_name, **kwargs))
                except Exception as e:
                    if allow_except:
                        # warn and continue if we allow exceptions
                        warnings.warn(f"Failed to get model info for '{model_name}': {e}")
                        model_data.append((model_name, e))
                    else:
                        # raise exception right away if we don't allow exceptions
                        # note that this differs from the parallel version, which will only except at the end
                        raise ValueError(f"Failed to get model info for '{model_name}'") from e

    # figure out what to do with failed models
    failed_models: dict[str, Exception] = {
        model_name: result for model_name, result in model_data if isinstance(result, Exception)
    }
    msg: str = (
        f"Failed to get model info for {len(failed_models)}/{len(model_names)} models: {failed_models}\n"
        + "\n".join(f"\t'{model_name}': {expt}" for model_name, expt in failed_models.items())
    )
    if not allow_except:
        if failed_models:
            # raise exception if we don't allow exceptions
            raise ValueError(msg + "\n\n" + "=" * 80 + "\n\n" + "NO DATA WRITTEN")
    else:
        if failed_models:
            warnings.warn(msg + "\n\n" + "-" * 80 + "\n\n" + "WRITING PARTIAL DATA")

    # filter out failed models if we allow exceptions
    model_data_filtered: list[dict] = [
        result for _, result in model_data if not isinstance(result, Exception)
    ]
    return pd.DataFrame(model_data_filtered)


OutputFormat = Literal["jsonl", "csv", "md"]


def huggingface_name_to_url(df: pd.DataFrame) -> pd.DataFrame:
    """convert the huggingface model name to a url"""
    df_new: pd.DataFrame = df.copy()
    df_new["name.huggingface"] = df_new["name.huggingface"].map(
        lambda x: f"[{x}](https://huggingface.co/{x})" if x else x
    )
    return df_new


MD_TABLE_HEARDER: str = """---
title: HookedTransformer
hide-toc: true
---
# HookedTransformer Model Properties

also see the [interactive model table](../_static/model_properties_table_interactive.html)
"""


def write_model_table(
    model_table: pd.DataFrame,
    path: Path,
    format: OutputFormat = "jsonl",
    include_TL_version: bool = True,
    md_hf_links: bool = True,
    md_header: str = MD_TABLE_HEARDER,
) -> None:
    """write the model table to disk in the specified format"""

    # make sure the directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    if include_TL_version:
        # get `transformer_lens` version
        tl_version: str = "unknown"
        try:
            from importlib.metadata import PackageNotFoundError, version

            tl_version = version("transformer_lens")
        except PackageNotFoundError as e:
            warnings.warn(f"Failed to get transformer_lens version: package not found\n{e}")
        except Exception as e:
            warnings.warn(f"Failed to get transformer_lens version: {e}")

        with open(path.with_suffix(".version"), "w") as f:
            json.dump({"version": tl_version}, f)

    if format == "jsonl":
        model_table.to_json(path.with_suffix(".jsonl"), orient="records", lines=True)
    elif format == "csv":
        model_table.to_csv(path.with_suffix(".csv"), index=False)
    elif format == "md":
        model_table_processed: pd.DataFrame = model_table
        # convert huggingface name to url
        if md_hf_links:
            model_table_processed = huggingface_name_to_url(model_table_processed)
        model_table_md_text: str = md_header + model_table_processed.to_markdown(index=False)
        with open(path.with_suffix(".md"), "w") as f:
            f.write(model_table_md_text)
    else:
        raise KeyError(f"Invalid format: {format}")


def abridge_model_table(
    model_table: pd.DataFrame,
    columns_keep: Sequence[str] = COLUMNS_ABRIDGED,
    null_to_empty: bool = True,
) -> pd.DataFrame:
    """keep only columns in COLUMNS_ABRIDGED

    primarily used to make the csv and md versions of the table readable

    also replaces `None` with empty string if `null_to_empty` is `True`
    """

    output: pd.DataFrame = model_table.copy()
    # filter columns
    output = output[list(columns_keep)]

    if null_to_empty:
        output = output.fillna("")

    return output


def get_model_table(
    model_table_path: Path,
    verbose: bool = True,
    force_reload: bool = True,
    do_write: bool = True,
    parallelize: Union[bool, int] = True,
    model_names_pattern: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """get the model table either by generating or reading from jsonl file

    # Parameters:
     - `model_table_path : Path`
        the path to the model table file, and the base name for the csv and md files
     - `verbose : bool`
        whether to show progress bar
       (defaults to `True`)
     - `force_reload : bool`
        force creating the table from scratch, even if file exists
       (defaults to `True`)
     - `do_write : bool`
        whether to write the table to disk, if generating
       (defaults to `True`)
     - `model_names_pattern : Optional[str]`
        filter the model names by making them include this string. passed to `make_model_table()`. no filtering if `None`
        (defaults to `None`)
     - `**kwargs`
        passed to `make_model_table()`

    # Returns:
     - `pd.DataFrame`
        the model table. rows are models, columns are model attributes
    """

    # modify the name if a pattern is provided
    if model_names_pattern is not None:
        model_table_path = model_table_path.with_name(
            model_table_path.stem + f"-{model_names_pattern}"
        )

    model_table: pd.DataFrame
    if not model_table_path.exists() or force_reload:
        # generate it from scratch
        model_table = make_model_table(
            verbose=verbose,
            parallelize=parallelize,
            model_names_pattern=model_names_pattern,
            **kwargs,
        )
        if do_write:
            # full data as jsonl
            write_model_table(model_table, model_table_path, format="jsonl")
            # abridged data as csv, md
            abridged_table: pd.DataFrame = abridge_model_table(model_table)
            write_model_table(abridged_table, model_table_path, format="csv")
            write_model_table(abridged_table, model_table_path, format="md")
    else:
        # read the table from jsonl
        model_table = pd.read_json(model_table_path, orient="records", lines=True)

    return model_table


def build_docs():
    """Build the docs."""
    get_model_table(
        model_table_path=GENERATED_DIR / "model_properties_table.jsonl",
        force_reload=True,
        allow_except=True,
    )
    copy_demos()
    generate_bridge_models_page()

    # Generating docs
    # Use sys.executable with -m sphinx to ensure we use the venv's sphinx
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            SOURCE_PATH,
            BUILD_PATH,
            # "-n",  # Nitpicky mode (warn about all missing references)
            # "-W",  # Turn warnings into errors - temporarily disabled due to duplicate object warnings
        ],
        check=True,
    )


def get_property(name: str, model_name: str) -> Any:
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


def generate_model_table(_app: Optional[Any] = None):
    """Generate a markdown table summarizing properties of pretrained models.

    This script extracts various properties of pretrained models from the `easy_transformer`
    library, such as the number of parameters, layers, and heads, among others, and generates a
    markdown table.
    """

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
        "n_key_value_heads",
    ]
    df = pd.DataFrame(
        {
            name: [
                get_property(name, model_name)
                for model_name in supported_models.DEFAULT_MODEL_ALIASES
            ]
            for name in column_names
        },
        index=supported_models.DEFAULT_MODEL_ALIASES,
    )

    # Convert to markdown (with a title)
    df["n_key_value_heads"] = df["n_key_value_heads"].fillna(-1).astype(int).replace(-1, "")
    markdown_string = df.to_markdown()
    markdown_string = "# Model Properties Table\n\n" + markdown_string

    # Save to the docs directory
    GENERATED_DIR.mkdir(exist_ok=True)
    file_path = GENERATED_DIR / "model_properties_table.md"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(markdown_string)


def copy_demos(_app: Optional[Any] = None):
    """Copy demo notebooks to the generated directory."""
    copy_to_dir = GENERATED_DIR / "demos"
    notebooks_to_copy = [
        "Exploratory_Analysis_Demo.ipynb",
        "Main_Demo.ipynb",
    ]

    if copy_to_dir.exists():
        shutil.rmtree(copy_to_dir)

    copy_to_dir.mkdir()
    for filename in notebooks_to_copy:
        shutil.copy(DEMOS_DIR / filename, copy_to_dir)


BRIDGE_MODELS_PAGE: str = """---
title: TransformerBridge Models
hide-toc: true
---
# TransformerBridge Models

The TransformerBridge provides automatic model compatibility for HuggingFace models
across supported architectures.

```{raw} html
<style>
/* Widen the Furo content area for this page */
.main > .content {
    width: 80vw !important;
    max-width: 80vw !important;
}

#bt-root { font-size: 14px; }
#bt-root .bt-controls {
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 16px;
}
#bt-root .bt-controls input[type="text"] {
    padding: 6px 10px; border: 1px solid var(--color-foreground-border, #ccc);
    border-radius: 4px; font-size: 14px; min-width: 260px;
    background: var(--color-background-primary, #fff); color: var(--color-foreground-primary, #333);
}
#bt-root .bt-controls select {
    padding: 6px 10px; border: 1px solid var(--color-foreground-border, #ccc);
    border-radius: 4px; font-size: 14px; min-width: 220px;
    background: var(--color-background-primary, #fff); color: var(--color-foreground-primary, #333);
}
#bt-root .bt-controls label {
    display: flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer; white-space: nowrap;
}
#bt-root .bt-count {
    margin-left: auto; font-size: 13px; color: var(--color-foreground-muted, #666); white-space: nowrap;
}
#bt-root .bt-wrap { overflow-x: auto; }
#bt-root table {
    width: 100%; border-collapse: collapse; font-size: 13px;
}
#bt-root thead th {
    background: var(--color-background-secondary, #f5f5f5);
    border-bottom: 2px solid var(--color-foreground-border, #ddd);
    padding: 8px 12px; text-align: left; font-weight: 600; white-space: nowrap;
    position: sticky; top: 0; z-index: 1;
}
#bt-root tbody td {
    padding: 6px 12px; border-bottom: 1px solid var(--color-background-border, #eee); white-space: nowrap;
}
#bt-root tbody tr:hover { background: var(--color-background-hover, #f8f8ff); }
#bt-root tbody td a { color: var(--color-link, #0366d6); text-decoration: none; }
#bt-root tbody td a:hover { text-decoration: underline; }
#bt-root .bt-rn { color: var(--color-foreground-muted, #999); font-size: 12px; width: 36px; text-align: right; }
#bt-root .bt-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
#bt-root .bt-s0 { background: #e8e8e8; color: #666; }
#bt-root .bt-s1 { background: #d4edda; color: #155724; }
#bt-root .bt-s2 { background: #fff3cd; color: #856404; }
#bt-root .bt-s3 { background: #f8d7da; color: #721c24; }
#bt-root .bt-muted { color: var(--color-foreground-muted, #999); }
#bt-root .bt-note { max-width: 280px; white-space: normal; word-wrap: break-word; font-size: 12px; line-height: 1.4; }
#bt-root .bt-note-toggle { color: var(--color-link, #2962ff); cursor: pointer; font-size: 11px; margin-left: 4px; white-space: nowrap; }
#bt-root .bt-detail-link { cursor: pointer; font-size: 12px; }
#bt-root .bt-detail-row td { padding: 0; border-bottom: 2px solid var(--color-foreground-border, #ddd); }
#bt-root .bt-detail-row:hover { background: transparent; }
#bt-root .bt-detail {
    padding: 12px 16px 12px 48px;
    background: var(--color-background-secondary, #f9f9f9);
    font-size: 13px;
}
#bt-root .bt-detail-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 6px 24px;
}
#bt-root .bt-detail-grid dt {
    font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em;
    color: var(--color-foreground-muted, #888); margin: 0;
}
#bt-root .bt-detail-grid dd {
    margin: 0 0 6px 0; font-family: monospace; font-size: 13px;
}
#bt-root .bt-detail-loading { color: var(--color-foreground-muted, #999); font-style: italic; }
#bt-root .bt-detail-error { color: #dc3545; font-style: italic; }
#bt-root .bt-pag {
    display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 16px; flex-wrap: wrap;
}
#bt-root .bt-pag button {
    padding: 6px 14px; border: 1px solid var(--color-foreground-border, #ccc); border-radius: 4px;
    background: var(--color-background-primary, #fff); color: var(--color-foreground-primary, #333);
    cursor: pointer; font-size: 13px;
}
#bt-root .bt-pag button:hover:not(:disabled) { background: var(--color-background-hover, #f0f0f0); }
#bt-root .bt-pag button:disabled { opacity: 0.4; cursor: default; }
#bt-root .bt-pag .bt-pinfo { font-size: 13px; color: var(--color-foreground-muted, #666); }
#bt-root .bt-pag .bt-pnums { display: flex; gap: 4px; }
#bt-root .bt-pag .bt-pbtn { padding: 4px 10px; min-width: 32px; text-align: center; }
#bt-root .bt-pag .bt-pbtn.active {
    background: var(--color-link, #0366d6); color: #fff; border-color: var(--color-link, #0366d6);
}
#bt-root .bt-pag .bt-ellip { padding: 4px 6px; color: var(--color-foreground-muted, #999); }
#bt-root .bt-empty { text-align: center; padding: 40px 20px; color: var(--color-foreground-muted, #999); }
</style>

<div id="bt-root">
  <div class="bt-controls">
    <input type="text" id="btSearch" placeholder="Search by model name...">
    <select id="btArch"><option value="">All Architectures</option></select>
    <select id="btStatus"><option value="">All Statuses</option><option value="1">Verified</option><option value="0">Unverified</option><option value="2">Skipped</option><option value="3">Failed</option></select>
    <span class="bt-count" id="btCount"></span>
  </div>
  <div class="bt-wrap">
    <table>
      <thead><tr>
        <th class="bt-rn">#</th>
        <th>Model</th>
        <th>Architecture</th>
        <th>Status</th>
        <th>Verified Date</th>
        <th>Note</th>
        <th></th>
      </tr></thead>
      <tbody id="btBody"><tr><td colspan="7" class="bt-empty">Loading models...</td></tr></tbody>
    </table>
  </div>
  <div class="bt-pag" id="btPag"></div>
</div>

<script>
(function() {
    const PS = 25, COLS = 7;
    const SM = {0:'Unverified',1:'Verified',2:'Skipped',3:'Failed'};
    const cfgCache = {};
    let all=[], filt=[], pg=1, dt=null;

    /* Fields to extract from HuggingFace config.json, with beautified labels.
       Each entry: [label, ...candidate keys to try in order].
       Keys starting with _ are computed/inferred fields handled in extractField(). */
    const CFG_FIELDS = [
        /* Size & Shape */
        ['Parameters',          '_n_params'],
        ['Layers',              'num_hidden_layers', 'n_layer', 'num_layers'],
        ['Heads',               'num_attention_heads', 'n_head', 'num_heads'],
        ['KV Heads',            'num_key_value_heads'],
        ['Model Dim (d_model)', 'hidden_size', 'n_embd', 'd_model'],
        ['Head Dim (d_head)',   'head_dim'],
        ['MLP Dim (d_mlp)',     'intermediate_size', 'd_ff'],
        ['Vocab Size',          'vocab_size'],
        ['Context Length',      'max_position_embeddings', 'n_positions', 'n_ctx'],
        /* Architecture details relevant for MI */
        ['Activation',          'hidden_act', 'activation_function', 'feed_forward_proj'],
        ['Normalization',       '_norm_type'],
        ['Positional Embedding','_pos_embed_type'],
        ['Parallel Attn & MLP', '_parallel_attn_mlp'],
        ['Gated MLP',           '_gated_mlp'],
        ['Tie Embeddings',      'tie_word_embeddings'],
        /* Tokenizer */
        ['Tokenizer',           '_tokenizer'],
    ];

    function fmtParam(n) {
        if (n === null || n === undefined) return null;
        if (n >= 1e12) return (n/1e12).toFixed(1) + 'T';
        if (n >= 1e9)  return (n/1e9).toFixed(1) + 'B';
        if (n >= 1e6)  return (n/1e6).toFixed(0) + 'M';
        if (n >= 1e3)  return (n/1e3).toFixed(0) + 'K';
        return String(n);
    }

    function estimateParams(cfg) {
        const d = cfg.hidden_size || cfg.n_embd || cfg.d_model;
        const L = cfg.num_hidden_layers || cfg.n_layer || cfg.num_layers;
        const V = cfg.vocab_size;
        const dff = cfg.intermediate_size || cfg.d_ff;
        if (!d || !L) return null;
        /* Rough: 2*V*d + L*(4*d^2 + 2*d*dff) */
        let p = 2 * (V||0) * d;
        p += L * (4 * d * d + (dff ? 2 * d * dff : 8 * d * d));
        return p;
    }

    function inferNormType(cfg) {
        if (cfg.rms_norm_eps !== undefined) return 'RMSNorm';
        if (cfg.layer_norm_epsilon !== undefined || cfg.layer_norm_eps !== undefined) return 'LayerNorm';
        if (cfg.norm_eps !== undefined) return 'LayerNorm';
        return null;
    }

    function inferPosEmbedType(cfg) {
        /* Check for rotary indicators */
        if (cfg.rope_theta || cfg.rope_parameters || cfg.rope_scaling ||
            cfg.rotary_pct || cfg.rotary_dim || cfg.rotary_emb_base) return 'Rotary (RoPE)';
        if (cfg.partial_rotary_factor !== undefined) return 'Partial Rotary (RoPE)';
        /* ALiBi */
        if (cfg.alibi || cfg.position_embedding_type === 'alibi') return 'ALiBi';
        /* Relative position bias (T5-style) */
        if (cfg.relative_attention_num_buckets !== undefined) return 'Relative Bias (T5)';
        /* Explicit type */
        if (cfg.position_embedding_type) return cfg.position_embedding_type;
        /* If max_position_embeddings is set and no rotary found, likely learned */
        if (cfg.max_position_embeddings || cfg.n_positions) return 'Learned';
        return null;
    }

    function inferParallelAttnMlp(cfg) {
        /* Explicit flags in some configs */
        if (cfg.parallel_attn !== undefined) return cfg.parallel_attn;
        /* Phi models use parallel attention+MLP */
        const arch = cfg.architectures?.[0] || cfg.model_type || '';
        if (/^phi$/i.test(cfg.model_type) && cfg.model_type !== 'phi3') return true;
        if (arch === 'GPTJForCausalLM' || cfg.model_type === 'gptj') return true;
        return null;
    }

    function inferGatedMlp(cfg) {
        /* SwiGLU / gated MLPs are used by LLaMA, Mistral, Qwen, Gemma, OLMo, etc.
           The tell-tale sign is hidden_act being 'silu' + intermediate_size present,
           or an explicit gate_proj in the architecture. */
        const act = cfg.hidden_act || '';
        const arch = cfg.architectures?.[0] || '';
        const gatedArchs = ['LlamaForCausalLM', 'MistralForCausalLM', 'MixtralForCausalLM',
            'Qwen2ForCausalLM', 'Qwen3ForCausalLM', 'GemmaForCausalLM', 'Gemma2ForCausalLM',
            'Gemma3ForCausalLM', 'Phi3ForCausalLM', 'OlmoForCausalLM', 'Olmo2ForCausalLM',
            'Olmo3ForCausalLM', 'OlmoeForCausalLM', 'StableLmForCausalLM', 'GptOssForCausalLM'];
        if (gatedArchs.includes(arch)) return true;
        /* T5 with gated-gelu or gated-silu */
        if (cfg.feed_forward_proj && cfg.feed_forward_proj.includes('gated')) return true;
        if (arch === 'GPT2LMHeadModel' || arch === 'GPTNeoForCausalLM' ||
            arch === 'GPTNeoXForCausalLM' || arch === 'GPTJForCausalLM' ||
            arch === 'OPTForCausalLM' || arch === 'BloomForCausalLM') return false;
        return null;
    }

    function inferTokenizer(cfg) {
        return cfg.tokenizer_class || null;
    }

    function extractField(cfg, keys) {
        for (const k of keys) {
            if (k === '_n_params') {
                const est = estimateParams(cfg);
                return est ? fmtParam(est) + ' (est.)' : null;
            }
            if (k === '_norm_type') return inferNormType(cfg);
            if (k === '_pos_embed_type') return inferPosEmbedType(cfg);
            if (k === '_parallel_attn_mlp') return inferParallelAttnMlp(cfg);
            if (k === '_gated_mlp') return inferGatedMlp(cfg);
            if (k === '_tokenizer') return inferTokenizer(cfg);
            if (cfg[k] !== undefined && cfg[k] !== null) return cfg[k];
        }
        return null;
    }

    async function fetchDetail(modelId) {
        if (cfgCache[modelId]) return cfgCache[modelId];
        const url = 'https://huggingface.co/' + modelId + '/resolve/main/config.json';
        const r = await fetch(url);
        if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
        const cfg = await r.json();
        cfgCache[modelId] = cfg;
        return cfg;
    }

    function renderDetail(cfg) {
        let html = '<dl class="bt-detail-grid">';
        for (const [label, ...keys] of CFG_FIELDS) {
            let val = extractField(cfg, keys);
            if (val === null || val === undefined) continue;
            if (typeof val === 'boolean') val = val ? 'Yes' : 'No';
            if (typeof val === 'number') val = val.toLocaleString();
            html += '<div><dt>' + esc(label) + '</dt><dd>' + esc(String(val)) + '</dd></div>';
        }
        html += '</dl>';
        return html;
    }

    async function toggleDetail(modelId, rowEl) {
        const nextRow = rowEl.nextElementSibling;
        if (nextRow && nextRow.classList.contains('bt-detail-row')) {
            nextRow.remove();
            rowEl.querySelector('.bt-detail-link').textContent = 'Details';
            return;
        }
        const detailRow = document.createElement('tr');
        detailRow.className = 'bt-detail-row';
        detailRow.innerHTML = '<td colspan="' + COLS + '"><div class="bt-detail"><span class="bt-detail-loading">Loading config from HuggingFace...</span></div></td>';
        rowEl.after(detailRow);
        rowEl.querySelector('.bt-detail-link').textContent = 'Hide';
        try {
            const cfg = await fetchDetail(modelId);
            detailRow.querySelector('.bt-detail').innerHTML = renderDetail(cfg);
        } catch(e) {
            detailRow.querySelector('.bt-detail').innerHTML = '<span class="bt-detail-error">Could not load config: ' + esc(String(e.message)) + '</span>';
        }
    }

    async function init() {
        try {
            const r = await fetch('../_static/supported_models.json');
            const d = await r.json();
            all = d.models;
            const ac = {};
            all.forEach(m => ac[m.architecture_id] = (ac[m.architecture_id]||0)+1);
            const sel = document.getElementById('btArch');
            Object.keys(ac).sort().forEach(a => {
                const o = document.createElement('option');
                o.value = a; o.textContent = a+' ('+ac[a]+')';
                sel.appendChild(o);
            });
            apply();
        } catch(e) {
            document.getElementById('btBody').innerHTML =
                '<tr><td colspan="'+COLS+'" class="bt-empty">Failed to load model data.</td></tr>';
        }
    }

    function apply() {
        const s = document.getElementById('btSearch').value.toLowerCase().trim();
        const a = document.getElementById('btArch').value;
        const sv = document.getElementById('btStatus').value;
        filt = all.filter(m => {
            if (s && !m.model_id.toLowerCase().includes(s)) return false;
            if (a && m.architecture_id !== a) return false;
            if (sv !== '' && m.status !== +sv) return false;
            return true;
        });
        pg = 1; render(); pag(); count();
    }

    function esc(str) { const d=document.createElement('div'); d.textContent=str; return d.innerHTML; }
    function cleanNote(note) {
        if (!note) return '';
        // "Benchmark passed with issues: P1=50.0% (failed: a, b); P3=88.9% (failed: c, d)"
        // → "Benchmark passed with issues: a, b, c, d"
        const m = note.match(/^(.*?:\s*)(.+)$/);
        if (!m) return note;
        const prefix = m[1];
        const failures = [...m[2].matchAll(/failed:\s*([^)]+)/g)].map(x => x[1].trim());
        if (!failures.length) return note;
        return prefix + failures.join(', ');
    }
    function renderNote(note) {
        if (!note) return '<span class="bt-muted">&mdash;</span>';
        const clean = cleanNote(note);
        if (clean.length <= 50) return esc(clean);
        return '<span class="bt-note-trunc">' + esc(clean.slice(0,50)) + '&hellip; <a class="bt-note-toggle" href="javascript:void(0)">more</a></span>' +
               '<span class="bt-note-full" style="display:none">' + esc(clean) + ' <a class="bt-note-toggle" href="javascript:void(0)">less</a></span>';
    }

    function render() {
        const tb = document.getElementById('btBody');
        const st = (pg-1)*PS, pm = filt.slice(st, st+PS);
        if (!pm.length) {
            tb.innerHTML='<tr><td colspan="'+COLS+'" class="bt-empty">No models match your filters.</td></tr>';
            return;
        }
        tb.innerHTML = pm.map((m,i) => {
            const id = esc(m.model_id);
            return '<tr data-model="'+id+'">' +
            '<td class="bt-rn">'+(st+i+1)+'</td>' +
            '<td><a href="https://huggingface.co/'+id+'" target="_blank" rel="noopener">'+id+'</a></td>' +
            '<td>'+esc(m.architecture_id)+'</td>' +
            '<td><span class="bt-badge bt-s'+m.status+'">'+SM[m.status]+'</span></td>' +
            '<td>'+(m.verified_date || '<span class="bt-muted">&mdash;</span>')+'</td>' +
            '<td class="bt-note">'+ renderNote(m.note) +'</td>' +
            '<td><a class="bt-detail-link" href="javascript:void(0)">Details</a></td>' +
            '</tr>';
        }).join('');

        tb.querySelectorAll('.bt-detail-link').forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const row = this.closest('tr');
                toggleDetail(row.dataset.model, row);
            });
        });
        tb.querySelectorAll('.bt-note-toggle').forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const cell = this.closest('.bt-note');
                const trunc = cell.querySelector('.bt-note-trunc');
                const full = cell.querySelector('.bt-note-full');
                const showing = trunc.style.display === 'none';
                trunc.style.display = showing ? '' : 'none';
                full.style.display = showing ? 'none' : '';
            });
        });
    }

    function pag() {
        const tp = Math.max(1, Math.ceil(filt.length/PS)), c = document.getElementById('btPag');
        if (tp<=1) { c.innerHTML=''; return; }
        let h = '<button id="btPrev" '+(pg===1?'disabled':'')+'>Previous</button><div class="bt-pnums">';
        const pages=[1]; if(tp>1) pages.push(tp);
        for(let i=Math.max(2,pg-1);i<=Math.min(tp-1,pg+1);i++) if(!pages.includes(i)) pages.push(i);
        pages.sort((a,b)=>a-b);
        let last=0;
        pages.forEach(p => {
            if(last&&p-last>1) h+='<span class="bt-ellip">...</span>';
            h+='<button class="bt-pbtn'+(p===pg?' active':'')+'" data-p="'+p+'">'+p+'</button>';
            last=p;
        });
        h += '</div><button id="btNext" '+(pg===tp?'disabled':'')+'>Next</button>';
        h += '<span class="bt-pinfo">Page '+pg+' of '+tp+'</span>';
        c.innerHTML = h;
        document.getElementById('btPrev').addEventListener('click', () => go(pg-1));
        document.getElementById('btNext').addEventListener('click', () => go(pg+1));
        c.querySelectorAll('.bt-pbtn').forEach(b => b.addEventListener('click', () => go(+b.dataset.p)));
    }

    function go(p) { const tp=Math.ceil(filt.length/PS); if(p<1||p>tp) return; pg=p; render(); pag(); count(); }
    function count() { document.getElementById('btCount').textContent='Showing '+filt.length+' of '+all.length+' models'; }

    document.getElementById('btSearch').addEventListener('input', () => { clearTimeout(dt); dt=setTimeout(apply,200); });
    document.getElementById('btArch').addEventListener('change', apply);
    document.getElementById('btStatus').addEventListener('change', apply);
    init();
})();
</script>
```
"""


def generate_bridge_models_page():
    """Generate the TransformerBridge models markdown page.

    The page fetches supported_models.json from _static/, which is a symlink
    to the canonical source at transformer_lens/tools/model_registry/data/.
    """
    GENERATED_DIR.mkdir(exist_ok=True)
    # Write the markdown wrapper page
    (GENERATED_DIR / "transformer_bridge_models.md").write_text(
        BRIDGE_MODELS_PAGE, encoding="utf-8"
    )


def docs_hot_reload():
    """Hot reload the docs."""
    get_model_table(
        model_table_path=GENERATED_DIR / "model_properties_table.jsonl", force_reload=False
    )
    copy_demos()
    generate_bridge_models_page()

    subprocess.run(
        [
            "sphinx-autobuild",
            "--watch",
            str(PACKAGE_DIR) + "," + str(DEMOS_DIR),
            "--open-browser",
            SOURCE_PATH,
            BUILD_PATH,
        ],
        check=True,
    )
