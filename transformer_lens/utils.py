from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from datasets.arrow_dataset import Dataset
import einops
from transformers import AutoTokenizer
from typing import Optional, Union, Tuple, List, Dict, Type
from torchtyping import TensorType as TT
import transformers
from huggingface_hub import hf_hub_download
import re
from rich import print as rprint
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset

from transformer_lens import FactoredMatrix

CACHE_DIR = transformers.TRANSFORMERS_CACHE
import json


def download_file_from_hf(
    repo_name, file_name, subfolder=".", cache_dir=CACHE_DIR, force_is_torch=False
):
    """
    Helper function to download files from the HuggingFace Hub, from subfolder/file_name in repo_name, saving locally to cache_dir and returning the loaded file (if a json or Torch object) and the file path otherwise.

    If it's a Torch file without the ".pth" extension, set force_is_torch=True to load it as a Torch object.
    """
    file_path = hf_hub_download(
        repo_id=repo_name, filename=file_name, subfolder=subfolder, cache_dir=cache_dir
    )

    # Load to the CPU device if CUDA is not available
    map_location = None if torch.cuda.is_available() else torch.device('cpu')

    if file_path.endswith(".pth") or force_is_torch:
        return torch.load(file_path, map_location=map_location)
    elif file_path.endswith(".json"):
        return json.load(open(file_path, "r"))
    else:
        print("File type not supported:", file_path.split(".")[-1])
        return file_path

def print_gpu_mem(step_name=""):
    print(
        f"{step_name} ~ {np.round(torch.cuda.memory_allocated()/2e30, 2)} GiB allocated on GPU."
    )


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
    logits: TT["batch", "pos", "d_vocab"],
    tokens: TT["batch", "pos"],
    per_token: bool = False,
) -> Union[TT[()], TT["batch", "pos"]]:
    """Cross entropy loss for the language model, gives the loss for predicting the NEXT token.

    Args:
        logits (torch.Tensor): Logits. Shape [batch, pos, d_vocab]
        tokens (torch.Tensor[int64]): Input tokens. Shape [batch, pos]
        per_token (bool, optional): Whether to return the log probs predicted for the correct token, or the loss (ie mean of the predicted log probs). Note that the returned array has shape [batch, seq-1] as we cannot predict the first token (alternately, we ignore the final logit). Defaults to False.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Use torch.gather to find the log probs of the correct tokens
    # Offsets needed because we're predicting the NEXT token (this means the final logit is meaningless)
    # None and [..., 0] needed because the tensor used in gather must have the same rank.
    predicted_log_probs = log_probs[..., :-1, :].gather(
        dim=-1, index=tokens[..., 1:, None]
    )[..., 0]
    if per_token:
        return -predicted_log_probs
    else:
        return -predicted_log_probs.mean()


def lm_accuracy(
    logits: TT["batch", "pos", "d_vocab"],
    tokens: TT["batch", "pos"],
    per_token: bool = False,
) -> Union[TT[()], TT["batch", "pos"]]:
    """Cross-Entropy Accuracy for Language Modelling. We measure the accuracy on the logits for predicting the NEXT token.

    If per_token is True, returns the boolean for top 1 accuracy for each token in the batch. Note that this has size [batch, seq_len-1], as we cannot predict the first token.
    """
    top_prediction = logits.argmax(dim=-1)
    correct_matches = top_prediction[:, :-1] == tokens[:, 1:]
    if per_token:
        return correct_matches
    else:
        return correct_matches.sum() / correct_matches.numel()


def gelu_new(input: TT["batch", "pos", "d_mlp"]) -> TT["batch", "pos", "d_mlp"]:
    # Implementation of GeLU used by GPT2 - subtly different from PyTorch's
    return (
        0.5
        * input
        * (
            1.0
            + torch.tanh(
                np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))
            )
        )
    )


def gelu_fast(input: TT["batch", "pos", "d_mlp"]) -> TT["batch", "pos", "d_mlp"]:
    return (
        0.5
        * input
        * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))
    )


def solu(input: TT["batch", "pos", "d_mlp"]) -> TT["batch", "pos", "d_mlp"]:
    """
    SoLU activation function as described by
    https://transformer-circuits.pub/2022/solu/index.html.

    LayerNorm implemented by the MLP class.
    """
    return input * F.softmax(input, dim=-1)


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

    Note: There is a bug when inputting very small datasets (eg, <1 batch per process) where it just outputs nothing. I'm not super sure why
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
        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_length : (i + 1) * chunk_length]
            for i in range(num_chunks)
        ]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)[
            "input_ids"
        ].flatten()
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)
        num_batches = num_tokens // (seq_len)
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


""" 
Test ^

data = Dataset.from_dict({"text":[str(i) for i in range(1000)]})
tokenizer = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits")
print(data)
tokenize_and_concatenate(data, tokenizer, streaming=False, column_name="text")
"""

def sample_logits(
    final_logits: TT["batch", "d_vocab"],
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
    freq_penalty: float = 0.0,
    tokens: Optional[TT["batch", "pos"]] = None,
) -> TT["batch"]:
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
            assert (
                tokens is not None
            ), "Must provide input_tokens if applying a frequency penalty"
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
            assert 1.0 >= top_p > 0.0, "top_p has to be in [0, 1)"
            sorted_logits, sorted_indices = torch.sort(final_logits, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # We round up - we want prob >= top_p not <top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            final_logits = final_logits.masked_fill(indices_to_remove, -float("inf"))
        return torch.distributions.categorical.Categorical(logits=final_logits).sample()


# %%
# Type alias
SliceInput: Type = Optional[
    Union[int, Tuple[int, int], Tuple[int, int, int], List[int], torch.Tensor]
]


class Slice:
    """
    We use a custom slice syntax because Python/Torch's don't let us reduce the number of dimensions:

    Note that slicing with input_slice=None means do nothing, NOT add an extra dimension (use unsqueeze for that)

    There are several modes:
    int - just index with that integer (decreases number of dimensions)
    slice - Input is a tuple converted to a slice ((k,) means :k, (k, m) means m:k, (k, m, n) means m:k:n)
    array - Input is a list or tensor or numpy array, converted to a numpy array, and we take the stack of values at those indices
    identity - Input is None, leave it unchanged.

    Examples for dim=0:
    if input_slice=0, tensor -> tensor[0]
    elif input_slice = (1, 5), tensor -> tensor[1:5]
    elif input_slice = (1, 5, 2), tensor -> tensor[1:5:2] (ie indexing with [1, 3])
    elif input_slice = [1, 4, 5], tensor -> tensor[[1, 4, 5]] (ie changing the first axis to have length 3, and taking the indices 1, 4, 5 out).
    elif input_slice is a Tensor, same as list - Tensor is assumed to be a 1D list of indices.
    """

    def __init__(
        self,
        input_slice: SliceInput = None,
    ):
        if type(input_slice) == tuple:
            input_slice = slice(*input_slice)
            self.slice = input_slice
            self.mode = "slice"
        elif type(input_slice) == int:
            self.slice = input_slice
            self.mode = "int"
        elif type(input_slice) == slice:
            self.slice = input_slice
            self.mode = "slice"
        elif (
            type(input_slice) == list
            or type(input_slice) == torch.Tensor
            or type(input_slice) == np.ndarray
        ):
            self.slice = to_numpy(input_slice)
            self.mode = "array"
        elif input_slice is None:
            self.slice = slice(None)
            self.mode = "identity"
        else:
            raise ValueError(f"Invalid input_slice {input_slice}")

    def apply(self, tensor, dim=0):
        """
        Takes in a tensor and a slice, and applies the slice to the given dimension (supports positive and negative dimension syntax). Returns the sliced tensor.
        """
        ndim = tensor.ndim
        slices = [slice(None)] * ndim
        slices[dim] = self.slice
        return tensor[tuple(slices)]

    def indices(self, max_ctx=None):
        """
        Returns the indices when this slice is applied to an axis of size max_ctx. Returns them as a numpy array, for integer slicing it is eg array([4])
        """
        if self.mode == "int":
            return np.array([self.slice])
        else:
            return np.arange(max_ctx)[self.slice]

    def __repr__(self):
        return f"Slice: {self.slice} Mode: {self.mode} "


# %%


def get_act_name(
    name: str,
    layer: Optional[int] = None,
    layer_type: Optional[str] = None,
):
    """
    Helper function to convert shorthand to an activation name. Pretty hacky, intended to be useful for short feedback loop hacking stuff together, more so than writing good, readable code. But it is deterministic!

    eg:
    get_act_name('k', 6, 'a')=='blocks.6.attn.hook_k'
    get_act_name('pre', 2)=='blocks.2.mlp.hook_pre'
    get_act_name('embed')=='hook_embed'
    get_act_name('normalized', 27, 'ln2')=='blocks.27.ln2.hook_normalized'
    get_act_name('k6')=='blocks.6.attn.hook_k'
    get_act_name('scale4ln1')=='blocks.4.ln1.hook_scale'
    get_act_name('pre5')=='blocks.5.mlp.hook_pre'
    """
    if (
        ("." in name or name.startswith("hook_"))
        and layer is None
        and layer_type is None
    ):
        # If this was called on a full name, just return it
        return name
    match = re.match(r"([a-z]+)(\d+)([a-z]?.*)", name)
    if match is not None:
        name, layer, layer_type = match.groups(0)

    layer_type_alias = {
        "a": "attn",
        "m": "mlp",
        "b": "",
        "block": "",
        "blocks": "",
        "attention": "attn",
    }

    act_name_alias = {
        "attn":"pattern",
        "attn_logits":"attn_scores",
        "key":"k",
        "query":"q",
        "value":"v",
        "mlp_pre":"pre",
        "mlp_mid":"mid",
        "mlp_post":"post",
    }

    if name in act_name_alias:
        name = act_name_alias[name]

    full_act_name = ""
    if layer is not None:
        full_act_name += f"blocks.{layer}."
    if name in ["k", "v", "q", "z", "rot_k", "rot_q", "result", "pattern", "attn_scores"]:
        layer_type = "attn"
    elif name in ["pre", "post", "mid"]:
        layer_type = "mlp"
    elif layer_type in layer_type_alias:
        layer_type = layer_type_alias[layer_type]

    if layer_type:
        full_act_name += f"{layer_type}."
    full_act_name += f"hook_{name}"
    return full_act_name


def remove_batch_dim(tensor: TT[1, ...]) -> TT[...]:
    """
    Removes the first dimension of a tensor if it is size 1, otherwise returns the tensor unchanged
    """
    if tensor.shape[0] == 1:
        return tensor.squeeze(0)
    else:
        return tensor


def test_prompt(
    prompt: str,
    answer: str,
    model,
    prepend_space_to_answer: bool = True,
    print_details: bool = True,
    prepend_bos: bool = True,
    top_k: int = 10,
):
    """
    Function to test whether a model can give the correct answer to a prompt. Intended for exploratory analysis, so it prints things out rather than returning things.

    Works for multi-token answers and multi-token prompts.

    Will always print the ranks of the answer tokens, and if print_details will print the logit and prob for the answer tokens and the top k tokens returned for each answer position.
    """
    if prepend_space_to_answer and not answer.startswith(" "):
        answer = " " + answer
    # GPT-2 often treats the first token weirdly, so lets give it a resting position
    tokens = model.to_tokens(prompt + answer, prepend_bos=prepend_bos)
    prompt_str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)
    answer_str_tokens = model.to_str_tokens(answer, prepend_bos=False)
    prompt_length = len(prompt_str_tokens)
    answer_length = len(answer_str_tokens)
    if print_details:
        print("Tokenized prompt:", prompt_str_tokens)
        print("Tokenized answer:", answer_str_tokens)
    logits = remove_batch_dim(model(tokens))
    probs = logits.softmax(dim=-1)
    answer_ranks = []
    for index in range(prompt_length, prompt_length + answer_length):
        answer_token = tokens[0, index]
        answer_str_token = answer_str_tokens[index - prompt_length]
        # Offset by 1 because models predict the NEXT token
        token_probs = probs[index - 1]
        sorted_token_probs, sorted_token_values = token_probs.sort(descending=True)
        # Janky way to get the index of the token in the sorted list - I couldn't find a better way?
        correct_rank = torch.arange(len(sorted_token_values))[
            (sorted_token_values == answer_token).cpu()
        ].item()
        answer_ranks.append((answer_str_token, correct_rank))
        if print_details:
            # String formatting syntax - the first number gives the number of characters to pad to, the second number gives the number of decimal places.
            # rprint gives rich text printing
            rprint(
                f"Performance on answer token:\n[b]Rank: {correct_rank: <8} Logit: {logits[index-1, answer_token].item():5.2f} Prob: {token_probs[answer_token].item():6.2%} Token: |{answer_str_token}|[/b]"
            )
            for i in range(top_k):
                print(
                    f"Top {i}th token. Logit: {logits[index-1, sorted_token_values[i]].item():5.2f} Prob: {sorted_token_probs[i].item():6.2%} Token: |{model.to_string(sorted_token_values[i])}|"
                )
    rprint(f"[b]Ranks of the answer tokens:[/b] {answer_ranks}")


# %%
def transpose(tensor: TT[..., "a", "b"]) -> TT[..., "b", "a"]:
    """
    Utility to swap the last two dimensions of a tensor, regardless of the number of leading dimensions
    """
    return tensor.transpose(-1, -2)

def composition_scores(
    left: FactoredMatrix, right: FactoredMatrix, broadcast_dims=True
) -> Union[
    TT["leading_dims":...], TT["leading_dims_left":..., "leading_dims_right":...]
]:
    """
    See `HookedTransformer.all_composition_scores` for documentation.
    """
    if broadcast_dims:
        r_leading = right.ndim - 2
        l_leading = left.ndim - 2
        for i in range(l_leading):
            right = right.unsqueeze(i)
        for i in range(r_leading):
            left = left.unsqueeze(i + l_leading)
    assert (
        left.rdim == right.ldim
    ), f"Composition scores require left.rdim==right.ldim, shapes were left: {left.shape}, right:{right.shape}"

    right = right.collapse_r()
    left = left.collapse_l()
    r_norms = right.norm(dim=[-2, -1])
    l_norms = left.norm(dim=[-2, -1])
    comp_norms = (left @ right).norm(dim=[-2, -1])
    return comp_norms / r_norms / l_norms


# %%
def get_dataset(dataset_name: str) -> Dataset:
    """
    Returns a small HuggingFace dataset, for easy testing and exploration. Accesses several convenience datasets with 10,000 elements (dealing with the enormous 100GB - 2TB datasets is a lot of effort!). Note that it returns a dataset (ie a dictionary containing all the data), *not* a DataLoader (iterator over the data + some fancy features). But you can easily convert it to a DataLoader. 
    
    Each dataset has a 'text' field, which contains the relevant info, some also have several meta data fields

    Possible inputs:
    * openwebtext (approx the GPT-2 training data https://huggingface.co/datasets/openwebtext)
    * pile (The Pile, a big mess of tons of diverse data https://pile.eleuther.ai/)
    * c4 (Colossal, Cleaned, Common Crawl - basically openwebtext but bigger https://huggingface.co/datasets/c4)
    * code (Codeparrot Clean, a Python code dataset https://huggingface.co/datasets/codeparrot/codeparrot-clean )
    * c4_code (c4 + code - the 20K data points from c4-10k and code-10k. This is the mix of datasets used to train my interpretability-friendly models, though note that they are *not* in the correct ratio! There's 10K texts for each, but about 22M tokens of code and 5M tokens of C4)
    * wiki (Wikipedia, generated from the 20220301.en split of https://huggingface.co/datasets/wikipedia )
    """
    if dataset_name in ["openwebtext", "owt"]:
        dataset = load_dataset("stas/openwebtext-10k", split='train')
    elif dataset_name == "pile":
        dataset = load_dataset("NeelNanda/pile-10k", split='train')
    elif dataset_name == "c4":
        dataset = load_dataset("NeelNanda/c4-10k", split='train')
    elif dataset_name in ["code", "python"]:
        dataset = load_dataset("NeelNanda/code-10k", split='train')
    elif dataset_name in ["c4_code", "c4-code"]:
        # Note that this one has 20K 
        dataset = load_dataset("NeelNanda/c4-code-20k", split='train')
    elif dataset_name == "wiki":
        dataset = load_dataset("NeelNanda/wiki-10k", split="train")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return dataset