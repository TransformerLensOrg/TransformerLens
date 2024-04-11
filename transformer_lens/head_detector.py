"""Head Detector.

Utilities for detecting specific types of heads (e.g. previous token heads).
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from typing_extensions import Literal, get_args

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.utils import is_lower_triangular, is_square

HeadName = Literal["previous_token_head", "duplicate_token_head", "induction_head"]
HEAD_NAMES = cast(List[HeadName], get_args(HeadName))
ErrorMeasure = Literal["abs", "mul"]

LayerHeadTuple = Tuple[int, int]
LayerToHead = Dict[int, List[int]]

INVALID_HEAD_NAME_ERR = (
    f"detection_pattern must be a Tensor or one of head names: {HEAD_NAMES}; got %s"
)

SEQ_LEN_ERR = "The sequence must be non-empty and must fit within the model's context window."

DET_PAT_NOT_SQUARE_ERR = "The detection pattern must be a lower triangular matrix of shape (sequence_length, sequence_length); sequence_length=%d; got detection patern of shape %s"


def detect_head(
    model: HookedTransformer,
    seq: Union[str, List[str]],
    detection_pattern: Union[torch.Tensor, HeadName],
    heads: Optional[Union[List[LayerHeadTuple], LayerToHead]] = None,
    cache: Optional[ActivationCache] = None,
    *,
    exclude_bos: bool = False,
    exclude_current_token: bool = False,
    error_measure: ErrorMeasure = "mul",
) -> torch.Tensor:
    """Search for a Particular Type of Attention Head.

    Searches the model (or a set of specific heads, for circuit analysis) for a particular type of
    attention head. This head is specified by a detection pattern, a (sequence_length,
    sequence_length) tensor representing the attention pattern we expect that type of attention head
    to show. The detection pattern can be also passed not as a tensor, but as a name of one of
    pre-specified types of attention head (see `HeadName` for available patterns), in which case the
    tensor is computed within the function itself.

    There are two error measures available for quantifying the match between the detection pattern
    and the actual attention pattern.

    1. `"mul"` (default) multiplies both tensors element-wise and divides the sum of the result by
        the sum of the attention pattern. Typically, the detection pattern should in this case
        contain only ones and zeros, which allows a straightforward interpretation of the score: how
        big fraction of this head's attention is allocated to these specific query-key pairs? Using
        values other than 0 or 1 is not prohibited but will raise a warning (which can be disabled,
        of course).

    2. `"abs"` calculates the mean element-wise absolute difference between the detection pattern
        and the actual attention pattern. The "raw result" ranges from 0 to 2 where lower score
        corresponds to greater accuracy. Subtracting it from 1 maps that range to (-1, 1) interval,
        with 1 being perfect match and -1 perfect mismatch.

    Which one should you use?

    `"mul"` is likely better for quick or exploratory investigations. For precise examinations where
    you're trying to reproduce as much functionality as possible or really test your understanding
    of the attention head, you probably want to switch to `"abs"`.

    The advantage of `"abs"` is that you can make more precise predictions, and have that measured
    in the score. You can predict, for instance, 0.2 attention to X, and 0.8 attention to Y, and
    your score will be better if your prediction is closer. The "mul" metric does not allow this,
    you'll get the same score if attention is 0.2, 0.8 or 0.5, 0.5 or 0.8, 0.2.

    Args:
        model: Model being used.
        seq: String or list of strings being fed to the model.
        head_name: Name of an existing head in HEAD_NAMES we want to check. Must pass either a
            head_name or a detection_pattern, but not both!
        detection_pattern: (sequence_length, sequence_length)nTensor representing what attention
            pattern corresponds to the head we're looking for or the name of a pre-specified head.
            Currently available heads are: `["previous_token_head", "duplicate_token_head",
            "induction_head"]`.
        heads: If specific attention heads is given here, all other heads' score is set to -1.
            Useful for IOI-style circuit analysis. Heads can be spacified as a list tuples (layer,
            head) or a dictionary mapping a layer to heads within that layer that we want to
            analyze. cache: Include the cache to save time if you want.
        exclude_bos: Exclude attention paid to the beginning of sequence token.
        exclude_current_token: Exclude attention paid to the current token.
        error_measure: `"mul"` for using element-wise multiplication. `"abs"` for using absolute
            values of element-wise differences as the error measure.

    Returns:
        Tensor representing the score for each attention head.
    """

    cfg = model.cfg
    tokens = model.to_tokens(seq).to(cfg.device)
    seq_len = tokens.shape[-1]

    # Validate error_measure

    assert error_measure in get_args(
        ErrorMeasure
    ), f"Invalid error_measure={error_measure}; valid values are {get_args(ErrorMeasure)}"

    # Validate detection pattern if it's a string
    if isinstance(detection_pattern, str):
        assert detection_pattern in HEAD_NAMES, INVALID_HEAD_NAME_ERR % detection_pattern
        if isinstance(seq, list):
            batch_scores = [detect_head(model, seq, detection_pattern) for seq in seq]
            return torch.stack(batch_scores).mean(0)
        detection_pattern = cast(
            torch.Tensor,
            eval(f"get_{detection_pattern}_detection_pattern(tokens.cpu())"),
        ).to(cfg.device)

    # if we're using "mul", detection_pattern should consist of zeros and ones
    if error_measure == "mul" and not set(detection_pattern.unique().tolist()).issubset({0, 1}):
        logging.warning(
            "Using detection pattern with values other than 0 or 1 with error_measure 'mul'"
        )

    # Validate inputs and detection pattern shape
    assert 1 < tokens.shape[-1] < cfg.n_ctx, SEQ_LEN_ERR
    assert (
        is_lower_triangular(detection_pattern) and seq_len == detection_pattern.shape[0]
    ), DET_PAT_NOT_SQUARE_ERR % (seq_len, detection_pattern.shape)

    if cache is None:
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    if heads is None:
        layer2heads = {layer_i: list(range(cfg.n_heads)) for layer_i in range(cfg.n_layers)}
    elif isinstance(heads, list):
        layer2heads = defaultdict(list)
        for layer, head in heads:
            layer2heads[layer].append(head)
    else:
        layer2heads = heads

    matches = -torch.ones(cfg.n_layers, cfg.n_heads, dtype=cfg.dtype)

    for layer, layer_heads in layer2heads.items():
        # [n_heads q_pos k_pos]
        layer_attention_patterns = cache["pattern", layer, "attn"]
        for head in layer_heads:
            head_attention_pattern = layer_attention_patterns[head, :, :]
            head_score = compute_head_attention_similarity_score(
                head_attention_pattern,
                detection_pattern=detection_pattern,
                exclude_bos=exclude_bos,
                exclude_current_token=exclude_current_token,
                error_measure=error_measure,
            )
            matches[layer, head] = head_score
    return matches


# Previous token head
def get_previous_token_head_detection_pattern(
    tokens: torch.Tensor,  # [batch (1) x pos]
) -> torch.Tensor:
    """Outputs a detection score for [previous token heads](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=0O5VOHe9xeZn8Ertywkh7ioc).

    Args:
      tokens: Tokens being fed to the model.
    """
    detection_pattern = torch.zeros(tokens.shape[-1], tokens.shape[-1])
    # Adds a diagonal of 1's below the main diagonal.
    detection_pattern[1:, :-1] = torch.eye(tokens.shape[-1] - 1)
    return torch.tril(detection_pattern)


# Duplicate token head
def get_duplicate_token_head_detection_pattern(
    tokens: torch.Tensor,  # [batch (1) x pos]
) -> torch.Tensor:
    """Outputs a detection score for [duplicate token heads](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=2UkvedzOnghL5UHUgVhROxeo).

    Args:
      sequence: String being fed to the model.
    """
    # [pos x pos]
    token_pattern = tokens.repeat(tokens.shape[-1], 1).numpy()

    # If token_pattern[i][j] matches its transpose, then token j and token i are duplicates.
    eq_mask = np.equal(token_pattern, token_pattern.T).astype(int)

    np.fill_diagonal(eq_mask, 0)  # Current token is always a duplicate of itself. Ignore that.
    detection_pattern = eq_mask.astype(int)
    return torch.tril(torch.as_tensor(detection_pattern).float())


# Induction head
def get_induction_head_detection_pattern(
    tokens: torch.Tensor,  # [batch (1) x pos]
) -> torch.Tensor:
    """Outputs a detection score for [induction heads](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=_tFVuP5csv5ORIthmqwj0gSY).

    Args:
      sequence: String being fed to the model.
    """
    duplicate_pattern = get_duplicate_token_head_detection_pattern(tokens)

    # Shift all items one to the right
    shifted_tensor = torch.roll(duplicate_pattern, shifts=1, dims=1)

    # Replace first column with 0's
    # we don't care about bos but shifting to the right moves the last column to the first,
    # and the last column might contain non-zero values.
    zeros_column = torch.zeros(duplicate_pattern.shape[0], 1)
    result_tensor = torch.cat((zeros_column, shifted_tensor[:, 1:]), dim=1)
    return torch.tril(result_tensor)


def get_supported_heads() -> None:
    """Returns a list of supported heads."""
    print(f"Supported heads: {HEAD_NAMES}")


def compute_head_attention_similarity_score(
    attention_pattern: torch.Tensor,  # [q_pos k_pos]
    detection_pattern: torch.Tensor,  # [seq_len seq_len] (seq_len == q_pos == k_pos)
    *,
    exclude_bos: bool,
    exclude_current_token: bool,
    error_measure: ErrorMeasure,
) -> float:
    """Compute the similarity between `attention_pattern` and `detection_pattern`.

    Args:
      attention_pattern: Lower triangular matrix (Tensor) representing the attention pattern of a particular attention head.
      detection_pattern: Lower triangular matrix (Tensor) representing the attention pattern we are looking for.
      exclude_bos: `True` if the beginning-of-sentence (BOS) token should be omitted from comparison. `False` otherwise.
      exclude_bcurrent_token: `True` if the current token at each position should be omitted from comparison. `False` otherwise.
      error_measure: "abs" for using absolute values of element-wise differences as the error measure. "mul" for using element-wise multiplication (legacy code).
    """
    assert is_square(
        attention_pattern
    ), f"Attention pattern is not square; got shape {attention_pattern.shape}"

    # mul

    if error_measure == "mul":
        if exclude_bos:
            attention_pattern[:, 0] = 0
        if exclude_current_token:
            attention_pattern.fill_diagonal_(0)
        score = attention_pattern * detection_pattern
        return (score.sum() / attention_pattern.sum()).item()

    # abs

    abs_diff = (attention_pattern - detection_pattern).abs()
    assert (abs_diff - torch.tril(abs_diff).to(abs_diff.device)).sum() == 0

    size = len(abs_diff)
    if exclude_bos:
        abs_diff[:, 0] = 0
    if exclude_current_token:
        abs_diff.fill_diagonal_(0)

    return 1 - round((abs_diff.mean() * size).item(), 3)
    return 1 - round((abs_diff.mean() * size).item(), 3)
