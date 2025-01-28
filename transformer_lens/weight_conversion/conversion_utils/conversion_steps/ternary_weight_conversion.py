from typing import Callable

import torch

from .base_weight_conversion import CONVERSION_ACTION, BaseWeightConversion

PRIMARY_CONVERSION = torch.Tensor | BaseWeightConversion | None


class TernaryWeightConversion(BaseWeightConversion):
    # TODO add none as fallback
    def __init__(
        self,
        fallback_conversion: CONVERSION_ACTION,
        primary_conversion: PRIMARY_CONVERSION = None,
        input_filter: Callable | None = None,
        output_filter: Callable | None = None,
    ):
        super().__init__(input_filter=input_filter, output_filter=output_filter)
        self.primary_conversion = primary_conversion
        self.fallback_conversion = fallback_conversion

    def handle_primary_conversion(self, input_value: torch.Tensor) -> torch.Tensor:
        if self.primary_conversion is None:
            return input_value
        if isinstance(self.primary_conversion, torch.Tensor):
            return self.primary_conversion
        else:
            return self.primary_conversion.convert(input_value=input_value)

    def handle_conversion(self, input_value: torch.Tensor or None) -> torch.Tensor:
        if input_value is not None:
            return self.handle_primary_conversion(input_value=input_value)
        else:
            return super().process_weight_conversion(
                input_value=input_value, conversion_details=self.fallback_conversion
            )

    def __repr__(self):
        return f"Is a ternary operation with the following primary conversion: {self.primary_conversion} and fallback conversion: {self.fallback_conversion}"
