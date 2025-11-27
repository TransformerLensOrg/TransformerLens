"""Weight conversion that performs ternary operations on weights."""

from collections.abc import Callable
from typing import Any, Optional, Union

import torch

from transformer_lens.conversion_utils.helpers.find_property import find_property

from .base_tensor_conversion import BaseTensorConversion

PRIMARY_CONVERSION = torch.Tensor | BaseTensorConversion | None


class TernaryTensorConversion(BaseTensorConversion):
    def __init__(
        self,
        fallback_conversion: Any,
        primary_conversion: PRIMARY_CONVERSION = None,
        input_filter: Optional[Callable] = None,
        output_filter: Optional[Callable] = None,
    ):
        super().__init__(input_filter=input_filter, output_filter=output_filter)
        self.primary_conversion = primary_conversion
        self.fallback_conversion = fallback_conversion

    def handle_conversion(
        self, input_value: Union[torch.Tensor | None], *full_context
    ) -> torch.Tensor | None:
        if input_value is not None:
            return self.handle_primary_conversion(input_value, *full_context)
        else:
            return self.handle_fallback_conversion(*full_context)

    def handle_primary_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        if self.primary_conversion is None:
            return input_value
        elif isinstance(self.primary_conversion, torch.Tensor):
            return self.primary_conversion
        else:
            return self.primary_conversion.convert(input_value, *full_context)

    def handle_fallback_conversion(self, *full_context) -> torch.Tensor | None:
        if isinstance(self.fallback_conversion, torch.Tensor):
            return self.fallback_conversion
        elif isinstance(self.fallback_conversion, str):
            return self.find_context_field(self.fallback_conversion, *full_context)
        else:
            (backup_field, conversion) = self.fallback_conversion
            backup_input = self.find_context_field(backup_field, *full_context)
            return conversion.convert(backup_input, *full_context)

    def find_context_field(self, field_key: str, *full_context):
        for context in full_context:
            maybe_field = find_property(field_key, context)
            if maybe_field is not None:
                return maybe_field

        return None

    def __repr__(self):
        return f"Is a ternary operation with the following primary conversion: {self.primary_conversion} and fallback conversion: {self.fallback_conversion}"
