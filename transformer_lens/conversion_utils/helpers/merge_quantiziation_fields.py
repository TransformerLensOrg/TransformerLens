"""Merge quantization fields helper.

This module contains helper functions for merging quantization fields.
"""

from typing import Any


def merge_quantization_fields(field_set: Any, quantization_fields: dict[str, Any]) -> Any:
    """Merge quantization fields into a field set.

    Args:
        field_set: The field set to merge into.
        quantization_fields: The quantization fields to merge.

    Returns:
        The merged field set (same object, modified in-place).
    """
    # Merge the quantization fields into the existing field_set
    for field_name, new_field_value in quantization_fields.items():
        existing_field = field_set.fields.get(field_name)

        # Check if existing field is None and raise error as expected by tests
        if existing_field is None:
            raise RuntimeError(
                "Attempted to merge quantization field into existing conversion without original field configured"
            )

        # Handle different cases based on the types of existing and new fields
        if isinstance(new_field_value, tuple) and len(new_field_value) == 2:
            # new_field_value is (str, TensorConversionSet)
            new_remote, new_sub_wcs = new_field_value

            if isinstance(existing_field, tuple) and len(existing_field) == 2:
                # existing_field is also (str, TensorConversionSet)
                existing_remote, existing_sub_wcs = existing_field

                # Check if the second element is a TensorConversionSet-like object
                if hasattr(existing_sub_wcs, "fields") and hasattr(new_sub_wcs, "fields"):
                    # Recursively merge the sub-TensorConversionSets
                    merge_quantization_fields(existing_sub_wcs, new_sub_wcs.fields)
                    # Update the remote field name
                    field_set.fields[field_name] = (new_remote, existing_sub_wcs)
                else:
                    raise RuntimeError(
                        "Attempted to merge TensorConversionSet into a field that is not configured as a TensorConversionSet"
                    )
            else:
                # existing_field is not a tuple, but new_field_value is
                raise RuntimeError(
                    "Attempted to merge TensorConversionSet into a field that is not configured as a TensorConversionSet"
                )
        else:
            # new_field_value is a simple value (like torch.Tensor)
            # Simply overwrite the existing field
            field_set.fields[field_name] = new_field_value

    return field_set
