"""Tensor conversion set."""

from typing import Any

import torch

from transformer_lens.conversion_utils.helpers.find_property import find_property
from transformer_lens.conversion_utils.hook_conversion_utils import (
    get_weight_conversion_field_set,
)

from .base_tensor_conversion import BaseTensorConversion
from .rearrange_tensor_conversion import RearrangeTensorConversion


class TensorConversionSet(BaseTensorConversion):
    def __init__(
        self,
        fields: dict[str, Any],
    ):
        super().__init__()
        self.fields = fields

    def get_component(self, model: Any, name: str) -> Any:
        """Get a component from the model using the field mapping.

        Args:
            model: The model to get the component from.
            name: The name of the component to get.

        Returns:
            The requested component.
        """
        if name not in self.fields:
            raise ValueError(f"Unknown component name: {name}")

        field_info = self.fields[name]
        if isinstance(field_info, str):
            field_name = field_info
            conversion_step = None
        else:
            field_name, conversion_step = field_info

        # Get the component from the model
        component = find_property(field_name, model)

        # Apply conversion step if specified
        if conversion_step is not None:
            component = conversion_step(component)

        return component

    def handle_conversion(self, input_value: Any, *full_context: Any) -> dict[str, Any]:
        result = {}
        for fields_name in self.fields:
            conversion_action = self.fields[fields_name]
            result[fields_name] = self.process_conversion_action(
                input_value,
                conversion_details=conversion_action,
            )

        return result

    def process_conversion_action(
        self, input_value: Any, conversion_details: Any, *full_context: Any
    ) -> Any:
        if isinstance(conversion_details, torch.Tensor):
            return conversion_details
        elif isinstance(conversion_details, str):
            return find_property(conversion_details, input_value)
        else:
            (remote_field, conversion) = conversion_details
            return self.process_conversion(input_value, remote_field, conversion, *full_context)

    def process_conversion(
        self,
        input_value: Any,
        remote_field: str,
        conversion: BaseTensorConversion,
        *full_context: Any,
    ) -> Any:
        field = find_property(remote_field, input_value)
        if isinstance(conversion, TensorConversionSet):
            result = []
            for layer in field:
                result.append(conversion.convert(layer, input_value, *full_context))
            return result

        else:
            return conversion.convert(field, *[input_value, *full_context])

    def get_conversion_action(self, field: str) -> BaseTensorConversion:
        conversion_details = self.fields[field]
        if isinstance(conversion_details, tuple):
            return conversion_details[1]
        else:
            # Return no op if not a specific conversion
            return RearrangeTensorConversion("... -> ...")

    def __repr__(self) -> str:
        conversion_string = (
            "Is composed of a set of nested conversions with the following details {\n\t"
        )
        # This is a bit of a hack to get the string representation of nested conversions
        conversion_string += get_weight_conversion_field_set(self.fields)[:-1].replace("\n", "\n\t")
        conversion_string += "\n}"
        return conversion_string
