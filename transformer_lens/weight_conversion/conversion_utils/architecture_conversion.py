from torch import nn

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils.helpers.merge_quantiziation_fields import (
    merge_quantiziation_fields,
)

from .conversion_steps.types import FIELD_SET
from .conversion_steps.weight_conversion_set import WeightConversionSet


class ArchitectureConversion:
    def __init__(self, fields: FIELD_SET, modules: FIELD_SET) -> None:
        self.field_set = WeightConversionSet(fields)
        self.modules = modules

    def enable_quantiziation(
        self, cfg: HookedTransformerConfig, quantiziation_fields: FIELD_SET
    ) -> None:
        if cfg.load_in_4bit:
            self.field_set = merge_quantiziation_fields(self.field_set, quantiziation_fields)

    def convert(self, remote_module: nn.Module):
        return self.field_set.convert(input_value=remote_module)
