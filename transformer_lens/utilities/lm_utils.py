"""lm_utils.

This module contains utility functions related to langauge models
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


def lm_cross_entropy_loss(
    logits: Float[torch.Tensor, "batch pos d_vocab"],
    tokens: Int[torch.Tensor, "batch pos"],
    per_token: bool = False,
) -> Union[Float[torch.Tensor, ""], Float[torch.Tensor, "batch pos"]]:
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
    predicted_log_probs = log_probs[..., :-1, :].gather(dim=-1, index=tokens[..., 1:, None])[..., 0]
    if per_token:
        return -predicted_log_probs
    else:
        return -predicted_log_probs.mean()


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
