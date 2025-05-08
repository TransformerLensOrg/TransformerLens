"""Merge quantization fields helper.

This module contains helper functions for merging quantization fields.
"""

from typing import Any

from transformer_lens.architecture_adapter.conversion_utils.conversion_steps import (
    WeightConversionSet,
)


def merge_quantiziation_fields(
    field_set: WeightConversionSet, quantiziation_fields: dict[str, Any]
) -> WeightConversionSet:
    """Merge quantization fields into a field set.

    Args:
        field_set: The field set to merge into.
        quantiziation_fields: The quantization fields to merge.

    Returns:
        The merged field set.
    """
    # Create a new field set with the same fields as the original
    new_field_set = WeightConversionSet(field_set.fields)

    # Merge the quantization fields
    for field_name, field_value in quantiziation_fields.items():
        if field_name in new_field_set.fields:
            # If the field exists in both sets, merge them
            if isinstance(field_value, WeightConversionSet):
                # If both fields are field sets, recursively merge them
                new_field_set.fields[field_name] = merge_quantiziation_fields(
                    new_field_set.fields[field_name], field_value.fields
                )
            else:
                # If the field in the quantization set is not a field set, replace the field
                new_field_set.fields[field_name] = field_value
        else:
            # If the field only exists in the quantization set, add it
            new_field_set.fields[field_name] = field_value

    return new_field_set
