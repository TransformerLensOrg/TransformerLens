from torch import nn

from .conversion_steps.types import FIELD_SET
from .conversion_steps.weight_conversion_set import WeightConversionSet


class ArchitectureConversion:
    def __init__(self, fields: FIELD_SET) -> None:
        self.field_set = WeightConversionSet(fields)

    def convert(self, remote_module: nn.Module):
        return self.field_set.convert(input_value=remote_module)
