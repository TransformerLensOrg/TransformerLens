from torch import nn

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils.helpers.merge_quantiziation_fields import (
    merge_quantiziation_fields,
)

from .conversion_steps.types import FIELD_SET
from .conversion_steps.weight_conversion_set import WeightConversionSet


class ArchitectureConversion:
    def __init__(self, fields: FIELD_SET) -> None:
        self.field_set = WeightConversionSet(fields)

    def enable_quantiziation(
        self, cfg: HookedTransformerConfig, quantiziation_fields: FIELD_SET
    ) -> None:
        if cfg.load_in_4bit:
            self.field_set = merge_quantiziation_fields(self.field_set, quantiziation_fields)

    def convert(self, remote_module: nn.Module):
        state_dict = self.field_set.convert(input_value=remote_module)

        # Flatten state dictionary such that PyTorch can load it properly
        flattened_state_dict = self.flatten_nested_dict(state_dict)
        return flattened_state_dict

    def flatten_nested_dict(self, input, parent_key="", sep="."):
        """
        Flattens a nested dictionary/list structure into a flat dictionary with dot notation.

        Args:
            input: The input structure (can be dict, list, or a value)
            parent_key: The parent key for the current item (used in recursion)
            sep: Separator to use between nested keys (default '.')

        Returns:
            dict: Flattened dictionary with dot notation keys
        """
        items = {}

        if isinstance(input, dict):
            for k, v in input.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.update(self.flatten_nested_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v

        elif isinstance(input, list):
            for i, v in enumerate(input):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                if isinstance(v, (dict, list)):
                    items.update(self.flatten_nested_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
        else:
            items[parent_key] = input

        return items
