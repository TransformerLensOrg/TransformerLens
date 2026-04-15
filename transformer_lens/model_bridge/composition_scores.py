"""CompositionScores — tensor-like container for composition score results."""
from typing import List

import torch


class CompositionScores:
    """Composition scores bundled with layer-index metadata.

    Behaves like a tensor for backward compatibility — indexing, .shape,
    arithmetic, and ``torch.*`` namespace functions all delegate to the
    underlying scores tensor via ``__torch_function__``. The additional
    ``layer_indices`` and ``head_labels`` attributes provide metadata that
    prevents silent misinterpretation of indices on hybrid models.

    For hybrid models, the scores tensor has shape
    (n_attn_layers, n_heads, n_attn_layers, n_heads) where n_attn_layers
    may be less than n_layers. ``layer_indices`` maps tensor position i
    to the original layer number.

    Attributes:
        scores: Upper-triangular composition score tensor.
        layer_indices: Original layer numbers for each position in scores.
            E.g., [0, 2, 5] means position 0 = layer 0, position 1 = layer 2, etc.
        head_labels: Labels like ["L0H0", "L0H1", "L2H0", ...] matching scores dims.
    """

    def __init__(self, scores: torch.Tensor, layer_indices: List[int], head_labels: List[str]):
        self.scores = scores
        self.layer_indices = layer_indices
        self.head_labels = head_labels

    # --- Tensor protocol ---

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Delegate torch.* calls (torch.isnan, torch.where, etc.) to .scores."""
        if kwargs is None:
            kwargs = {}
        # Unwrap any CompositionScores args to their underlying tensor
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

    # Python 3 automatically sets __hash__ = None when __eq__ is defined,
    # making instances unhashable. No explicit __hash__ needed.

    def __getitem__(self, key):
        return self.scores[key]

    def __getattr__(self, name):
        # Delegate tensor methods (.abs(), .sum(), .any(), etc.) to .scores.
        # Guard against infinite recursion during pickling/unpickling where
        # self.scores may not exist yet.
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
