from collections import defaultdict
from typing import cast, get_args, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import tril_abs_error, is_lower_triangular, is_square

HeadName = Literal["previous_token_head", "duplicate_token_head", "induction_head"]
HEAD_NAMES = cast(List[HeadName], get_args(HeadName))
ErrorMeasure = Literal["abs", "mul"]

LayerHeadTuple = Tuple[int, int]
LayerToHead = Dict[int, List[int]]

INVALID_HEAD_NAME_ERR = (
    f"detection_pattern must be a Tensor or one of head names: {HEAD_NAMES}; got %s"
)
SEQ_LEN_ERR = (
    "The sequence must be non-empty and must fit within the model's context window."
)
DET_PAT_NOT_SQUARE_ERR = (
    "The detection pattern must be a lower triangular matrix of shape "
    "(sequence_length, sequence_length); sequence_length=%d; got detection patern of shape %s"
)


def detect_head(
    model: HookedTransformer,
    seq: Union[str, List[str]],
    detection_pattern: Union[torch.Tensor, HeadName],
    heads: Optional[Union[List[LayerHeadTuple], LayerToHead]] = None,
    cache: Optional[ActivationCache] = None,
    *,
    exclude_bos: bool = False,
    exclude_current_token: bool = False,
    error_measure: ErrorMeasure = "abs",
) -> torch.Tensor:
    """Searches the model (or a set of specific heads, for circuit analysis) for a particular type of attention head. This head is specified
    by a detection pattern, a (sequence_length, sequence_length) tensor representing how much attention to keep at each position. That pattern
    can be also given a name of one specified types of attention head (see `HeadName` for available patterns), in which case the tensor is
    computed by the function itself. #TODO: wording

    #TODO: to abs
    We take the element-wise absolute difference of the attention pattern and the detection pattern. The resulting score expresses how well our
    expectation (detection pattern) match reality (attention pattern).

    We element-wise
    multiply the attention pattern by the detection pattern, and our score is how much attention is left, divided by the total attention.
    (1 per token other than the first).

    For instance, a perfect previous token head would put 1 attention to the previous token and 0 to everything else. To write a detection pattern
    for this head, we would write a (sequence_length, sequence_length) tensor with `1`s below the diagonal and `0`s everywhere else. We would then
    element-wise subtract that from the attention pattern and take the absolute value of each element-wise difference. The resulting scores would
    express how far the detection pattern was off from the attention pattern for each key-value pair

    , zeroing out all attention that did not attend to the previous token. The attention
    remaining divided by the total attention would then be our score for that attention head. (This particular head is already implemented in HEAD_NAMES)

    Args:
      model: Model being used.
      seq: String or list of strings being fed to the model.
      head_name: Name of an existing head in HEAD_NAMES we want to check. Must pass either a head_name or a detection_pattern, but not both!
      detection_pattern: (sequence_length, sequence_length) Tensor representing what attention pattern corresponds to the head we're looking for **or** the name of a pre-specified head. Currently available heads are: `["previous_token_head", "duplicate_token_head", "induction_head"]`.
      heads: If specific attention heads is given here, all other heads' score is set to -1. Useful for IOI-style circuit analysis. Heads can be spacified as a list of tuples (layer, head) or a dictionary mapping a layer to heads within that layer that we want to analyze.
      cache: Include the cache to save time if you want.
      exclude_bos: Exclude attention paid to the beginning of sequence token.
      exclude_current_token: Exclude attention paid to the current token.

    Returns a (n_layers, n_heads) Tensor representing the score for each attention head.

    Example:
    --------
    .. code-block:: python

        >>> from transformer_lens.head_detector import detect_head
        >>> from transformer_lens.HookedTransformer import HookedTransformer
        >>> import plotly.express as px
        >>> import transformer_lens.utils as utils

        >>> def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
        >>>     px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> sequence = "This is a test sequence. This is a test sequence."

        >>> attention_score = detect_head(model, sequence, "previous_token_head")
        >>> imshow(attention_score, zmin=-1, zmax=1, xaxis="Head", yaxis="Layer", title="Previous Head Matches")
    """

    cfg = model.cfg
    tokens = model.to_tokens(seq).to(cfg.device)
    seq_len = tokens.shape[-1]

    # Validate detection pattern if it's a string
    if isinstance(detection_pattern, str):
        assert detection_pattern in HEAD_NAMES, (
            INVALID_HEAD_NAME_ERR % detection_pattern
        )
        if isinstance(seq, list):
            batch_scores = [detect_head(model, seq, detection_pattern) for seq in seq]
            return torch.stack(batch_scores).mean(0)
        detection_pattern = cast(
            torch.Tensor,
            eval(f"get_{detection_pattern}_detection_pattern(tokens.cpu())"),
        ).to(cfg.device)

    # Validate inputs and detection pattern shape
    assert 1 < tokens.shape[-1] < cfg.n_ctx, SEQ_LEN_ERR
    assert (
        is_lower_triangular(detection_pattern) and seq_len == detection_pattern.shape[0]
    ), DET_PAT_NOT_SQUARE_ERR % (seq_len, detection_pattern.shape)

    if cache is None:
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    if heads is None:
        layer2heads = {
            layer_i: list(range(cfg.n_heads)) for layer_i in range(cfg.n_layers)
        }
    elif isinstance(heads, list):
        layer2heads = defaultdict(list)
        for layer, head in heads:
            layer2heads[layer].append(head)
    else:
        layer2heads = heads

    matches = -torch.ones(cfg.n_layers, cfg.n_heads)

    for layer, layer_heads in layer2heads.items():
        # [n_heads q_pos k_pos]
        layer_attention_patterns = cache["pattern", layer, "attn"]
        for head in layer_heads:
            head_attention_pattern = layer_attention_patterns[head, :, :]
            head_score = process_head_attention_pattern_score(
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

    np.fill_diagonal(
        eq_mask, 0
    )  # Current token is always a duplicate of itself. Ignore that.
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

    # Replace first column with 0's - we don't care about bos but shifting to the right moves the last column to the first, and the last column might contain non-zero values.
    zeros_column = torch.zeros(duplicate_pattern.shape[0], 1)
    result_tensor = torch.cat((zeros_column, shifted_tensor[:, 1:]), dim=1)
    return torch.tril(result_tensor)


def get_supported_heads() -> None:
    """Returns a list of supported heads."""
    print(f"Supported heads: {HEAD_NAMES}")


def process_head_attention_pattern_score(
    attention_pattern: torch.Tensor,  # [q_pos k_pos]
    detection_pattern: torch.Tensor,  # [seq_len seq_len]
    *,
    exclude_bos: bool,
    exclude_current_token: bool,
    error_measure: ErrorMeasure,
) -> float:
    """
    seq_len == q_pos == k_pos
    """
    assert is_square(attention_pattern)

    if exclude_bos:
        attention_pattern[:, 0] = 0
    if exclude_current_token:
        attention_pattern.fill_diagonal_(0)

    if error_measure == "abs":
        matched_attention_val = 1 - tril_abs_error(attention_pattern, detection_pattern)
        return matched_attention_val
    else:
        matched_attention = attention_pattern * detection_pattern
        return (matched_attention.sum() / attention_pattern.sum()).item()
