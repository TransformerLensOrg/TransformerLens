from torch import nn

from .conversion_steps.base_weight_conversion import FIELD_SET
from .conversion_steps.weight_conversion_set import WeightConversionSet
from .weight_conversion_utils import WeightConversionUtils


class ArchitectureConversion:
    def __init__(self, fields: FIELD_SET) -> None:
        self.field_set = WeightConversionSet(fields)

    def convert(self, remote_module: nn.Module):
        return self.field_set.convert(input_value=remote_module)

    def __repr__(self) -> str:
        return WeightConversionUtils.create_conversion_string(self.field_set.weights)
