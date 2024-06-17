"""Build the API Documentation."""
import base64
import hashlib
import multiprocessing
import os
import shutil
import subprocess
import warnings
from copy import deepcopy
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import pandas as pd
import torch
import tqdm
import yaml
from muutils.dictmagic import condense_tensor_dict
from muutils.json_serialize import json_serialize
from muutils.misc import shorten_numerical_to_str
from transformers import PreTrainedTokenizer

import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig, loading

DEVICE: torch.device = torch.device("meta")
torch.set_default_device(DEVICE)


# make sure we have a HuggingFace token
try:
    _hf_token = os.environ.get("HF_TOKEN", None)
    if not _hf_token.startswith("hf_"):
        raise ValueError("Invalid HuggingFace token")
except Exception as e:
    warnings.warn(
        f"Failed to get Hugging Face token -- info about certain models will be limited\n{e}"
    )

# Docs Directories
CURRENT_DIR = Path(__file__).parent
SOURCE_PATH = CURRENT_DIR / "../docs/source"
BUILD_PATH = CURRENT_DIR / "../docs/build"
PACKAGE_DIR = CURRENT_DIR.parent
DEMOS_DIR = CURRENT_DIR.parent / "demos"
GENERATED_DIR = CURRENT_DIR.parent / "docs/source/generated"


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
KNOWN_MODEL_TYPES: list[str] = [
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
]

MODEL_ALIASES_MAP: dict[str, str] = transformer_lens.loading.make_model_alias_map()

# these will be copied as table columns
CONFIG_ATTRS_COPY: list[str] = [
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
]

# modify certain values when printing config as yaml
CONFIG_VALUES_PROCESS: dict[str, Callable] = {
    "initializer_range": float,
}


def get_tensor_shapes(model: HookedTransformer, tensor_dims_fmt: str = "yaml") -> dict:
    """get the tensor shapes from a model"""
    model_info: dict = dict()
    # state dict
    model_info["tensor_shapes.state_dict"] = condense_tensor_dict(
        model.state_dict(), fmt=tensor_dims_fmt
    )
    model_info["tensor_shapes.state_dict.raw__"] = condense_tensor_dict(
        model.state_dict(), fmt="dict"
    )
    # input shape for activations -- "847"~="bat", subtract 7 for the context window to make it unique
    input_shape: tuple[int, int, int] = (847, model.cfg.n_ctx - 7)
    # why? to replace the batch and seq_len dims with "batch" and "seq_len" in the yaml
    dims_names_map: dict[int, str] = {
        input_shape[0]: "batch",
        input_shape[1]: "seq_len",
    }
    # run with cache to activation cache
    _, cache = model.run_with_cache(
        torch.empty(input_shape, dtype=torch.long, device=DEVICE)
    )
    # condense using muutils and store
    model_info["tensor_shapes.activation_cache"] = condense_tensor_dict(
        cache,
        fmt=tensor_dims_fmt,
        dims_names_map=dims_names_map,
    )
    model_info["tensor_shapes.activation_cache.raw__"] = condense_tensor_dict(
        cache,
        fmt="dict",
        dims_names_map=dims_names_map,
    )

    return model_info


def tokenizer_vocab_hash(tokenizer: PreTrainedTokenizer) -> str:
    # sort
    vocab_hashable: list[tuple[str, int]] = list(
        sorted(
            tokenizer.vocab.items(),
            key=lambda x: x[1],
        )
    )
    # hash it
    hash_obj = hashlib.sha1(bytes(str(vocab_hashable), "UTF-8"))
    # convert to base64
    return (
        base64.b64encode(
            hash_obj.digest(),
            altchars=b"-_",  # - and _ as altchars
        )
        .decode("UTF-8")
        .rstrip("=")
    )


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
    tensor_dims_fmt: str = "yaml",
) -> dict:
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
     - `tensor_dims_fmt : str`
        the format of the tensor shapes. one of "yaml", "json", "dict"
       (defaults to `"yaml"`)
    """
    # assumes the input is a default alias
    if model_name not in transformer_lens.loading.DEFAULT_MODEL_ALIASES:
        raise ValueError(f"Model name {model_name} not found in default aliases")

    # get the names and model types
    official_name: str = MODEL_ALIASES_MAP.get(model_name, None)
    model_info: dict = {
        "name.default_alias": model_name,
        "name.huggingface": official_name,
        "name.aliases": ", ".join(
            list(transformer_lens.loading.MODEL_ALIASES.get(official_name, []))
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
    param_count_from_name: str | None = None
    for part in parts:
        if (
            part[-1].lower() in ["m", "b", "k"]
            and part[:-1].replace(".", "", 1).isdigit()
        ):
            param_count_from_name = part
            break

    # update model info from config
    model_cfg: HookedTransformerConfig = get_config(model_name)
    model_info.update(
        {
            "name.from_cfg": model_cfg.model_name,
            "n_params.as_str": shorten_numerical_to_str(model_cfg.n_params),
            "n_params.as_int": model_cfg.n_params,
            "n_params.from_name": param_count_from_name,
            **{f"cfg.{attr}": getattr(model_cfg, attr) for attr in CONFIG_ATTRS_COPY},
        }
    )

    # put the whole config as yaml (for readability)
    if include_cfg:
        model_cfg_dict: dict = model_cfg.to_dict()
        # modify certain values to make them pretty-printable
        for key, func_process in CONFIG_VALUES_PROCESS.items():
            if key in model_cfg_dict:
                model_cfg_dict[key] = func_process(model_cfg_dict[key])
        # dump to yaml
        model_cfg_dict = json_serialize(model_cfg_dict)
        model_info["config.raw__"] = model_cfg_dict
        model_info["config"] = yaml.dump(
            model_cfg_dict,
            default_flow_style=False,
            sort_keys=False,
            width=1000,
        )

    # get tensor shapes
    if include_tensor_dims or include_tokenizer_info:
        got_model: bool = False
        try:
            # copy the config, so we can modify it
            model_cfg_copy: HookedTransformerConfig = deepcopy(model_cfg)
            # set device to "meta" -- don't actually initialize the model with real tensors
            model_cfg_copy.device = DEVICE
            if not include_tokenizer_info:
                # don't need to download the tokenizer
                model_cfg_copy.tokenizer_name = None
            # init the fake model
            model: HookedTransformer = HookedTransformer(
                model_cfg_copy, move_to_device=True
            )
            got_model = True
        except Exception as e:
            warnings.warn(
                f"Failed to init model {model_name}, can't get tensor shapes or tokenizer info:\n{e}"
            )

        if got_model:
            if include_tokenizer_info:
                try:
                    tokenizer_info: dict = get_tokenizer_info(model)
                    model_info.update(tokenizer_info)
                except Exception as e:
                    warnings.warn(
                        f"Failed to get tokenizer info for model {model_name}:\n{e}"
                    )

            if include_tensor_dims:
                try:
                    tensor_shapes_info: dict = get_tensor_shapes(model, tensor_dims_fmt)
                    model_info.update(tensor_shapes_info)
                except Exception as e:
                    warnings.warn(
                        f"Failed to get tensor shapes for model {model_name}:\n{e}"
                    )

    return model_name, model_info


def safe_try_get_model_info(
    model_name: str, kwargs: dict | None = None
) -> dict | Exception:
    """for parallel processing, to catch exceptions and return the exception instead of raising them"""
    if kwargs is None:
        kwargs = {}
    try:
        return get_model_info(model_name, **kwargs)
    except Exception as e:
        warnings.warn(f"Failed to get model info for {model_name}: {e}")
        return model_name, e


def make_model_table(
    verbose: bool,
    allow_except: bool = False,
    parallelize: bool | int = True,
    model_names_pattern: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """make table of all models. kwargs passed to `get_model_info()`"""
    model_names: list[str] = list(transformer_lens.loading.DEFAULT_MODEL_ALIASES)
    model_data: list[tuple[str, dict | Exception]] = list()

    # filter by regex pattern if provided
    if model_names_pattern:
        model_names = [
            model_name
            for model_name in model_names
            if model_names_pattern in model_name
        ]

    if parallelize:
        # parallel
        n_processes: int = (
            parallelize if int(parallelize) > 1 else multiprocessing.cpu_count()
        )
        with multiprocessing.Pool(processes=n_processes) as pool:
            # Use imap for ordered results, wrapped with tqdm for progress bar
            imap_results: list[dict | Exception] = list(
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
            transformer_lens.loading.DEFAULT_MODEL_ALIASES,
            desc="Loading model info",
            disable=not verbose,
        ) as pbar:
            for model_name in pbar:
                pbar.set_postfix_str(f"model: {model_name}")
                try:
                    model_data.append(get_model_info(model_name, **kwargs))
                except Exception as e:
                    if allow_except:
                        # warn and continue if we allow exceptions
                        warnings.warn(f"Failed to get model info for {model_name}: {e}")
                        model_data.append(e)
                    else:
                        # raise exception right away if we don't allow exceptions
                        # note that this differs from the parallel version, which will only except at the end
                        raise ValueError(
                            f"Failed to get model info for {model_name}"
                        ) from e

    # figure out what to do with failed models
    failed_models: dict[str, Exception] = {
        model_name: result
        for model_name, result in model_data
        if isinstance(result, Exception)
    }
    msg: str = (
        f"Failed to get model info for {len(failed_models)}/{len(model_names)} models: {failed_models}\n"
        + "\n".join(
            f"\t{model_name}: {expt}" for model_name, expt in failed_models.items()
        )
    )
    if not allow_except:
        if failed_models:
            # raise exception if we don't allow exceptions
            raise ValueError(msg + "\n\n" + "=" * 80 + "\n\n" + "NO DATA WRITTEN")
    else:
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
        # not sure how to make this type error go away, but it will propagate a None if it's None and be a string otherwise
        lambda x: f"[{x}](https://huggingface.co/{x})" if x else x # type: ignore
    )
    return df_new


def write_model_table(
    model_table: pd.DataFrame,
    path: Path,
    format: OutputFormat = "jsonl",
    include_TL_version: bool = True,
    md_hf_links: bool = True,
    md_header: str = "# Model Properties Table\nalso see the [interactive model table](model_properties_table_interactive.html)\n",
) -> None:
    """write the model table to disk in the specified format"""
    if include_TL_version:
        # get `transformer_lens` version
        tl_version: str = "unknown"
        try:
            from importlib.metadata import PackageNotFoundError, version

            tl_version = version("transformer_lens")
        except PackageNotFoundError as e:
            warnings.warn(
                f"Failed to get transformer_lens version: package not found\n{e}"
            )
        except Exception as e:
            warnings.warn(f"Failed to get transformer_lens version: {e}")

        with open(path.with_suffix(".version"), "w") as f:
            f.write(tl_version)

    match format:
        case "jsonl":
            model_table.to_json(
                path.with_suffix(".jsonl"), orient="records", lines=True
            )
        case "csv":
            model_table.to_csv(path.with_suffix(".csv"), index=False)
        case "md":
            model_table_processed: pd.DataFrame = model_table
            # convert huggingface name to url
            if md_hf_links:
                model_table_processed = huggingface_name_to_url(model_table_processed)
            
            model_table_md: str = md_header + model_table_processed.to_markdown(index=False)
            with open(path.with_suffix(".md"), "w") as f:
                f.write(model_table_md)

        case _:
            raise KeyError(f"Invalid format: {format}")


def abridge_model_table(
    model_table: pd.DataFrame,
    max_mean_col_len: int = 100,
    null_to_empty: bool = True,
) -> pd.DataFrame:
    """remove columns which are too long from the model table, returning a new table

    primarily used to make the csv and md versions of the table readable

    also replaces `None` with empty string if `null_to_empty` is `True`
    """
    column_lengths: pd.Series = model_table.map(str).map(len).mean()
    columns_to_drop: list[str] = column_lengths[
        column_lengths > max_mean_col_len
    ].index.tolist()

    output: pd.DataFrame = model_table.drop(columns=columns_to_drop)

    if null_to_empty:
        output = output.fillna("")

    return output


def generate_model_table(
    base_path: Path|str = GENERATED_DIR,
    verbose: bool = True,
    parallelize: bool | int = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """get the model table either by generating or reading from jsonl file

    # Parameters:
     - `base_path : Path|str`
        base path of where to save the tables
     - `verbose : bool`
        whether to show progress bar
       (defaults to `True`)
     - `model_names_pattern : str|None`
        filter the model names by making them include this string. passed to `make_model_table()`. no filtering if `None`
        (defaults to `None`)
     - `**kwargs`
        eventually passed to `get_model_info()`

    # Returns:
     - `pd.DataFrame`
        the model table. rows are models, columns are model attributes
    """

    # convert to Path, and modify the name if a pattern is provided
    base_path = Path(base_path)

    # generate the table
    model_table: pd.DataFrame = make_model_table(
        verbose=verbose,
        parallelize=parallelize,
        **kwargs,
    )

    # full data as jsonl
    write_model_table(
        model_table=model_table,
        path=base_path / "model_properties_data" / "data.jsonl",
        format="jsonl",
    )
    # abridged data as csv, md
    abridged_table: pd.DataFrame = abridge_model_table(model_table)
    write_model_table(
        model_table=abridged_table,
        path=base_path / "model_properties_table.md",
        format="csv",
    )
    write_model_table(
        model_table=abridged_table,
        path=base_path / "model_properties_data" / "data.csv",
        format="md",
    )

    return model_table


# def get_property(name, model_name):
#     """Retrieve a specific property of a pretrained model.

#     Args:
#         name (str): Name of the property to retrieve.
#         model_name (str): Name of the pretrained model.

#     Returns:
#         str: Value of the specified property.
#     """
#     cfg = get_config(model_name)

#     if name == "act_fn":
#         if cfg.attn_only:
#             return "attn_only"
#         if cfg.act_fn == "gelu_new":
#             return "gelu"
#         if cfg.act_fn == "gelu_fast":
#             return "gelu"
#         if cfg.act_fn == "solu_ln":
#             return "solu"
#         return cfg.act_fn
#     if name == "n_params":
#         n_params = cfg.n_params
#         if n_params < 1e4:
#             return f"{n_params/1e3:.1f}K"
#         if n_params < 1e6:
#             return f"{round(n_params/1e3)}K"
#         if n_params < 1e7:
#             return f"{n_params/1e6:.1f}M"
#         if n_params < 1e9:
#             return f"{round(n_params/1e6)}M"
#         if n_params < 1e10:
#             return f"{n_params/1e9:.1f}B"
#         if n_params < 1e12:
#             return f"{round(n_params/1e9)}B"
#         raise ValueError(f"Passed in {n_params} above 1T?")
#     return cfg.to_dict()[name]


# def generate_model_table(_app: Optional[Any] = None):
#     """Generate a markdown table summarizing properties of pretrained models.

#     This script extracts various properties of pretrained models from the `easy_transformer`
#     library, such as the number of parameters, layers, and heads, among others, and generates a
#     markdown table.
#     """

#     # Create the table
#     column_names = [
#         "n_params",
#         "n_layers",
#         "d_model",
#         "n_heads",
#         "act_fn",
#         "n_ctx",
#         "d_vocab",
#         "d_head",
#         "d_mlp",
#         "n_key_value_heads",
#     ]
#     df = pd.DataFrame(
#         {
#             name: [get_property(name, model_name) for model_name in loading.DEFAULT_MODEL_ALIASES]
#             for name in column_names
#         },
#         index=loading.DEFAULT_MODEL_ALIASES,
#     )

#     # Convert to markdown (with a title)
#     df["n_key_value_heads"] = df["n_key_value_heads"].fillna(-1).astype(int).replace(-1, "")
#     markdown_string = df.to_markdown()
#     markdown_string = "# Model Properties Table\n\n" + markdown_string

#     # Save to the docs directory
#     GENERATED_DIR.mkdir(exist_ok=True)
#     file_path = GENERATED_DIR / "model_properties_table.md"
#     with open(file_path, "w", encoding="utf-8") as file:
#         file.write(markdown_string)


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


def build_docs():
    """Build the docs."""
    get_model_table()
    copy_demos()

    # Generating docs
    subprocess.run(
        [
            "sphinx-build",
            SOURCE_PATH,
            BUILD_PATH,
            # "-n",  # Nitpicky mode (warn about all missing references)
            "-W",  # Turn warnings into errors
        ],
        check=True,
    )


def docs_hot_reload():
    """Hot reload the docs."""
    get_model_table()
    copy_demos()

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
