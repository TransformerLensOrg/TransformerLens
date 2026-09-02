"""Deprecated: train.py moved to tools.training.

Use transformer_lens.tools.training instead.
"""

import warnings

import torch.nn as nn
from torch.utils.data import Dataset

from transformer_lens.model_protocol import TrainableTransformerLensModel
from transformer_lens.tools.training import TrainConfig as _TrainConfig
from transformer_lens.tools.training import train as _train


class HookedTransformerTrainConfig(_TrainConfig):
    """Deprecated alias for TrainConfig. Use transformer_lens.tools.training.TrainConfig instead."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        warnings.warn(
            "HookedTransformerTrainConfig is deprecated; use "
            "transformer_lens.tools.training.TrainConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]


def train(
    model: "TrainableTransformerLensModel", config: _TrainConfig, dataset: Dataset
) -> "TrainableTransformerLensModel":
    """Deprecated. Use transformer_lens.tools.training.train instead."""
    warnings.warn(
        "transformer_lens.train is deprecated; use "
        "transformer_lens.tools.training.train instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _train(model, config, dataset)
