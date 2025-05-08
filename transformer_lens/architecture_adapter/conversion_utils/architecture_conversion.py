"""Architecture conversion base class.

This module contains the base class for architecture conversions.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils.helpers.merge_quantiziation_fields import (
    merge_quantiziation_fields,
)

from .conversion_steps.types import FIELD_SET
from .conversion_steps.weight_conversion_set import WeightConversionSet


class ArchitectureConversion(ABC):
    """Base class for architecture conversions."""

    def __init__(self, cfg: HookedTransformerConfig):
        """Initialize the architecture conversion.

        Args:
            cfg: The config to use for the conversion.
        """
        self.cfg = cfg
        self.field_set = WeightConversionSet(cfg.fields)

    def enable_quantiziation(
        self, cfg: HookedTransformerConfig, quantiziation_fields: FIELD_SET
    ) -> None:
        if cfg.load_in_4bit:
            self.field_set = merge_quantiziation_fields(self.field_set, quantiziation_fields)

    def convert(self, remote_module: torch.nn.Module):
        return self.field_set.convert(input_value=remote_module)

    @abstractmethod
    def convert_weights(self, hf_model: Any) -> dict[str, torch.Tensor]:
        """Convert the weights from the HuggingFace format to the HookedTransformer format.

        Args:
            hf_model: The HuggingFace model to convert.

        Returns:
            dict[str, torch.Tensor]: The converted weights.
        """
        pass
