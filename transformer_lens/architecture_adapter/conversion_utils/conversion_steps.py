"""Conversion steps for architecture adapters."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from transformers import PreTrainedModel


class RearrangeWeightConversion:
    """Rearrange weight conversion step."""

    def __init__(self, pattern: str) -> None:
        """Initialize the rearrange weight conversion.

        Args:
            pattern: The pattern to use for rearrangement.
        """
        self.pattern = pattern

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Rearrange the tensor according to the pattern.

        Args:
            tensor: The tensor to rearrange.

        Returns:
            The rearranged tensor.
        """
        # TODO: Implement tensor rearrangement based on pattern
        return tensor


class WeightConversionSet:
    """Set of weight conversions."""

    def __init__(self, field_mapping: Dict[str, Union[str, Tuple[str, Callable[[torch.Tensor], torch.Tensor]]]]) -> None:
        """Initialize the weight conversion set.

        Args:
            field_mapping: The mapping from HookedTransformer field names to model field names and optional conversion steps.
        """
        self.field_mapping = field_mapping

    def get_component(self, model: PreTrainedModel, name: str) -> Any:
        """Get a component from the model using the field mapping.

        Args:
            model: The model to get the component from.
            name: The name of the component to get.

        Returns:
            The requested component.
        """
        if name not in self.field_mapping:
            raise ValueError(f"Unknown component name: {name}")

        field_info = self.field_mapping[name]
        if isinstance(field_info, str):
            field_name = field_info
            conversion_step = None
        else:
            field_name, conversion_step = field_info

        # Get the component from the model
        component = self._get_nested_attr(model, field_name)

        # Apply conversion step if specified
        if conversion_step is not None:
            component = conversion_step(component)

        return component

    def convert(self, input_value: PreTrainedModel) -> Dict[str, torch.Tensor]:
        """Convert the weights from the model format to the HookedTransformer format.

        Args:
            input_value: The model to convert.

        Returns:
            dict[str, torch.Tensor]: The converted weights.
        """
        state_dict = {}
        for name, field_info in self.field_mapping.items():
            if isinstance(field_info, str):
                field_name = field_info
                conversion_step = None
            else:
                field_name, conversion_step = field_info

            # Get the component from the model
            component = self._get_nested_attr(input_value, field_name)

            # Apply conversion step if specified
            if conversion_step is not None:
                component = conversion_step(component)

            state_dict[name] = component

        return state_dict

    def _get_nested_attr(self, obj: Any, attr_path: str) -> Any:
        """Get a nested attribute from an object.

        Args:
            obj: The object to get the attribute from.
            attr_path: The path to the attribute, using dots for nesting.

        Returns:
            The requested attribute.
        """
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj 