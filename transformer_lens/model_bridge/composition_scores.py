"""Tensor-like container for composition score results with layer-index metadata."""
from typing import List

import torch


class CompositionScores:
    """Composition scores that behave like a tensor but carry layer-index metadata.

    Delegates indexing, .shape, arithmetic, and torch.* functions to the
    underlying ``scores`` tensor via ``__torch_function__``. On hybrid models
    where n_attn_layers < n_layers, ``layer_indices`` maps tensor position i
    to the original layer number.

    Attributes:
        scores: Upper-triangular composition score tensor.
        layer_indices: Original layer numbers, e.g. [0, 2, 5].
        head_labels: Labels matching scores dims, e.g. ["L0H0", "L0H1", ...].
    """

    def __init__(self, scores: torch.Tensor, layer_indices: List[int], head_labels: List[str]):
        self.scores = scores
        self.layer_indices = layer_indices
        self.head_labels = head_labels

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Unwrap CompositionScores args so torch.isnan, torch.where, etc. work."""
        if kwargs is None:
            kwargs = {}
        unwrapped_args = tuple(a.scores if isinstance(a, CompositionScores) else a for a in args)
        unwrapped_kwargs = {
            k: v.scores if isinstance(v, CompositionScores) else v for k, v in kwargs.items()
        }
        return func(*unwrapped_args, **unwrapped_kwargs)

    @property
    def shape(self) -> torch.Size:
        return self.scores.shape

    @property
    def device(self) -> torch.device:
        return self.scores.device

    @property
    def dtype(self) -> torch.dtype:
        return self.scores.dtype

    def __getitem__(self, key):
        return self.scores[key]

    def __getattr__(self, name):
        # Guard against recursion during pickle/deepcopy when self.scores isn't set yet
        try:
            scores = object.__getattribute__(self, "scores")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(scores, name)

    def __gt__(self, other):
        return self.scores > other

    def __lt__(self, other):
        return self.scores < other

    def __ge__(self, other):
        return self.scores >= other

    def __le__(self, other):
        return self.scores <= other

    def __eq__(self, other):
        if isinstance(other, CompositionScores):
            return self.scores == other.scores
        return self.scores == other

    def __ne__(self, other):
        if isinstance(other, CompositionScores):
            return self.scores != other.scores
        return self.scores != other

    def __repr__(self) -> str:
        return (
            f"CompositionScores(shape={self.shape}, "
            f"layer_indices={self.layer_indices}, "
            f"n_head_labels={len(self.head_labels)})"
        )
