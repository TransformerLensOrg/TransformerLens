"""logits_utils.

This module contains utility functions related to logits
"""

from __future__ import annotations

from typing import Optional

import torch
from jaxtyping import Float, Int


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
            assert (
                len(tokens.shape) == 2
            ), "Frequency penalty do not support input in the form of embeddings"
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
