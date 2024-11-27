"""utils.

This module is deprecated, but imports from the new utilities to maintain backwards compatibility
"""

from __future__ import annotations

import inspect
import json
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets.arrow_dataset import Dataset
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Int
from transformers import AutoTokenizer

from transformer_lens.FactoredMatrix import FactoredMatrix

CACHE_DIR = transformers.TRANSFORMERS_CACHE
USE_DEFAULT_VALUE = None


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


def print_gpu_mem(step_name=""):
    print(f"{step_name} ~ {np.round(torch.cuda.memory_allocated()/2e30, 2)} GiB allocated on GPU.")


def get_corner(tensor, n=3):
    # Prints the top left corner of the tensor
    if isinstance(tensor, torch.Tensor):
        return tensor[tuple(slice(n) for _ in range(tensor.ndim))]
    elif isinstance(tensor, FactoredMatrix):
        return tensor[tuple(slice(n) for _ in range(tensor.ndim))].AB


def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def lm_cross_entropy_loss(
    logits: Float[torch.Tensor, "batch pos d_vocab"],
    tokens: Int[torch.Tensor, "batch pos"],
    attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    per_token: bool = False,
) -> Union[Float[torch.Tensor, ""], Float[torch.Tensor, "batch pos"]]:
    """Cross entropy loss for the language model, gives the loss for predicting the NEXT token.

    Args:
        logits (torch.Tensor): Logits. Shape [batch, pos, d_vocab]
        tokens (torch.Tensor[int64]): Input tokens. Shape [batch, pos]
        attention_mask (torch.Tensor[int64], optional): Attention mask. Shape [batch, pos]. Used to
            mask out padding tokens. Defaults to None.
        per_token (bool, optional): Whether to return the log probs predicted for the correct token, or the loss (ie mean of the predicted log probs). Note that the returned array has shape [batch, seq-1] as we cannot predict the first token (alternately, we ignore the final logit). Defaults to False.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Use torch.gather to find the log probs of the correct tokens
    # Offsets needed because we're predicting the NEXT token (this means the final logit is meaningless)
    # None and [..., 0] needed because the tensor used in gather must have the same rank.
    predicted_log_probs = log_probs[..., :-1, :].gather(dim=-1, index=tokens[..., 1:, None])[..., 0]

    if attention_mask is not None:
        # Ignore token positions which are masked out or where the next token is masked out
        # (generally padding tokens)
        next_token_mask = torch.logical_and(attention_mask[:, :-1], attention_mask[:, 1:])
        predicted_log_probs *= next_token_mask
        n_tokens = next_token_mask.sum().item()
    else:
        n_tokens = predicted_log_probs.numel()

    if per_token:
        return -predicted_log_probs
    else:
        return -predicted_log_probs.sum() / n_tokens


def lm_accuracy(
    logits: Float[torch.Tensor, "batch pos d_vocab"],
    tokens: Int[torch.Tensor, "batch pos"],
    per_token: bool = False,
) -> Union[Float[torch.Tensor, ""], Float[torch.Tensor, "batch pos"]]:
    """Cross-Entropy Accuracy for Language Modelling. We measure the accuracy on the logits for predicting the NEXT token.

    If per_token is True, returns the boolean for top 1 accuracy for each token in the batch. Note that this has size [batch, seq_len-1], as we cannot predict the first token.
    """
    top_prediction = logits.argmax(dim=-1)
    correct_matches = top_prediction[:, :-1] == tokens[:, 1:]
    if per_token:
        return correct_matches
    else:
        return correct_matches.sum() / correct_matches.numel()


def gelu_new(
    input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    # Implementation of GeLU used by GPT2 - subtly different from PyTorch's
    return (
        0.5
        * input
        * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    )


def gelu_fast(
    input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


def solu(input: Float[torch.Tensor, "batch pos d_mlp"]) -> Float[torch.Tensor, "batch pos d_mlp"]:
    """
    SoLU activation function as described by
    https://transformer-circuits.pub/2022/solu/index.html.

    LayerNorm implemented by the MLP class.
    """
    return input * F.softmax(input, dim=-1)


ACTIVATION_FN_DICT = {
    "solu": solu,
    "solu_ln": solu,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda tensor: F.gelu(tensor, approximate="tanh"),
}


def calc_fan_in_and_fan_out(tensor):
    """
    Calculate the fan in and fan out of a tensor. We define it ourselves because Torch uses a
    different convention for weights (e.g. for an MLP they use d_out x d_in, and we use d_in x
    d_out, for attention they do (n_head d_head) x d_model, we do n_head x d_model x d_head).
    """
    shape = tensor.shape

    if len(shape) == 0:
        raise ValueError("Fan in and fan out can not be computed for scalars.")
    elif len(shape) == 1:
        fan_in = 1
        fan_out = shape[0]
    elif len(shape) == 2:  # Linear transform
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 3:  # Attention head weight, has shape n_head x d_model x d_head
        fan_in = shape[1]
        fan_out = shape[0] * shape[2]
    else:
        raise ValueError(f"Fan in and fan out can not be computed for shape {shape} tensors.")

    return fan_in, fan_out


def init_xavier_uniform_(param, gain=1.0):
    """
    Initializes the input tensor using the Xavier initialization method.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    max = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return nn.init.uniform_(param, -max, max)


def init_xavier_normal_(param, gain=1.0):
    """
    Initializes the input tensor using the Xavier initialization method.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return nn.init.normal_(param, mean=0.0, std=std)


def init_kaiming_uniform_(param, a=0, nonlinearity="relu", gain=1.0, mode="fan_in"):
    """
    Initializes the input tensor using the Kaiming initialization method.

    Starting from a std 1 uniform distribution, we scale the weights by c / sqrt(fan_in), where c =
    sqrt(2) if the params were immediately preceded by a relu and 1 for everything else.

    As with torch, `a` is a hyperparameter for `nonlinearity`, if it takes one.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    fan = fan_in if mode == "fan_in" else fan_out
    gain *= nn.init.calculate_gain(nonlinearity, a)
    max = gain * np.sqrt(3.0 / fan)
    return nn.init.uniform_(param, -max, max)


def init_kaiming_normal_(param, a=0, nonlinearity="relu", gain=1.0, mode="fan_in"):
    """
    Initializes the input tensor using the Kaiming initialization method.

    Starting from a std 1 normal distribution, we scale the weights by c / sqrt(fan_in), where c =
    sqrt(2) if the params were immediately preceded by a relu and 1 for everything else.

    As with torch, `a` is a hyperparameter for `nonlinearity`, if it takes one.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    fan = fan_in if mode == "fan_in" else fan_out
    gain *= nn.init.calculate_gain(nonlinearity, a)
    std = gain * np.sqrt(1.0 / fan)
    return nn.init.normal_(param, mean=0.0, std=std)


def keep_single_column(dataset: Dataset, col_name: str):
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name - useful when we want to tokenize and mix together different strings
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 10,
) -> Dataset:
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with padding, then remove padding at the end.

    This tokenization is useful for training language models, as it allows us to efficiently train on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding). Further, for models with absolute positional encodings, this avoids privileging early tokens (eg, news articles often begin with CNN, and models may learn to use early positional encodings to predict these)

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (AutoTokenizer): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.

    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"
    """
    dataset = keep_single_column(dataset, column_name)
    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(text)

        # Handle the case when full_text is empty
        if not full_text.strip():
            return {"tokens": np.array([], dtype=np.int64)}

        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)

        # Handle cases where num_tokens is less than seq_len
        if num_tokens < seq_len:
            num_batches = 1
            # Pad tokens if necessary
            tokens = tokens[:seq_len]
            if len(tokens) < seq_len:
                padding_length = seq_len - len(tokens)
                padding = np.full(padding_length, tokenizer.pad_token_id)
                tokens = np.concatenate([tokens, padding], axis=0)
        else:
            num_batches = num_tokens // seq_len
            # Drop the final tokens if not enough to make a full sequence
            tokens = tokens[: seq_len * num_batches]

        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=(num_proc if not streaming else None),
        remove_columns=[column_name],
    )
    tokenized_dataset.set_format(type="torch", columns=["tokens"])
    return tokenized_dataset


def sample_logits(
    final_logits: Float[torch.Tensor, "batch d_vocab"],
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
    freq_penalty: float = 0.0,
    tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
) -> Int[torch.Tensor, "batch"]:
    """
    Sample from the logits, in order to generate text

    final_logits has shape [batch, vocab_size]
    We divide the logits by temperature before softmaxing and sampling - high temperature = more uniform, low = more argmaxy. Temp = 0.0 is greedy sampling
    We apply top_k and top_p filtering to the logits, to encourage diversity. top_k = 10 means we only sample from the 10 most likely tokens. top_p = 0.9 means we only sample from the top 90% of tokens, and then renormalise the distribution. top_k and top_p are mutually exclusive. By default we apply neither and just sample from the full distribution.

    Frequency penalty is a penalty on the probability of a token, proportional to the number of times it has been generated so far. This encourages the model to generate new tokens, rather than repeating itself. It is a hyperparameter, and should be tuned. It is applied to the logits before sampling. If this is non-zero it is required to input the input_tokens

    #! TODO: Finish testing all the edge cases here. Useful testing code:
    logits = torch.randn(4)
    print(logits)
    np.unique(np.array([sample_logits(logits, top_k=2).item() for i in range(1000)]), return_counts=True)
    """
    if temperature == 0.0:
        # Greedy sampling
        return final_logits.argmax(dim=-1)
    else:
        # Sample from the distribution

        final_logits = final_logits / temperature
        if freq_penalty > 0:
            assert tokens is not None, "Must provide input_tokens if applying a frequency penalty"
            for batch_index in range(final_logits.shape[0]):
                # torch.bincount returns a tensor of length d_vocab, with the number of occurences of each token in the tokens.
                final_logits[batch_index] = final_logits[
                    batch_index
                ] - freq_penalty * torch.bincount(
                    tokens[batch_index], minlength=final_logits.shape[-1]
                )
        if top_k is not None:
            assert top_k > 0, "top_k has to be greater than 0"
            top_logits, top_idx = final_logits.topk(top_k, dim=-1)
            indices_to_remove = final_logits < top_logits[..., -1].unsqueeze(-1)
            final_logits = final_logits.masked_fill(indices_to_remove, -float("inf"))
        elif top_p is not None:
            assert 1.0 >= top_p > 0.0, "top_p has to be in (0, 1]"
            sorted_logits, sorted_indices = torch.sort(final_logits, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # We round up - we want prob >= top_p not <top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            final_logits = final_logits.masked_fill(indices_to_remove, -float("inf"))

        final_logits = final_logits.to(torch.float32)
        return torch.distributions.categorical.Categorical(logits=final_logits).sample()


# Type alias
SliceInput = Optional[
    Union[
        int,
        Tuple[int,],
        Tuple[int, int],
        Tuple[int, int, int],
        List[int],
        torch.Tensor,
        np.ndarray,
    ]
]
