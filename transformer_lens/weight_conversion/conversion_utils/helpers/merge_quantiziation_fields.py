from transformer_lens.weight_conversion.conversion_utils.conversion_steps.types import (
    FIELD_SET,
)
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    WeightConversionSet,
)

def merge_quantiziation_fields(conversion_set: WeightConversionSet, quantiziation_fields: FIELD_SET) -> WeightConversionSet:

    for transformer_lens_field in quantiziation_fields:
        existing_field = conversion_set.fields[transformer_lens_field]
        new_field = quantiziation_fields[transformer_lens_field]
        if existing_field is None:
            raise RuntimeError("Attempted to merge quantization field into existing conversion without original field configured!")

        if isinstance(new_field, tuple) and isinstance(new_field[1], WeightConversionSet):
            if not isinstance(existing_field, tuple) or not isinstance(existing_field[1], WeightConversionSet):
                raise RuntimeError("Attempted to merge WeightConversionSet into a field that is not configured as a WeightConversionSet")
            existing_conversion_set = existing_field[1]
            replacement_conversion_set = new_field[1]
            conversion_set.fields[transformer_lens_field] = (
                new_field[0],
                merge_quantiziation_fields(existing_conversion_set, replacement_conversion_set.fields)
            )

        else:
            conversion_set.fields[transformer_lens_field] = new_field
            
    return conversion_set
            
