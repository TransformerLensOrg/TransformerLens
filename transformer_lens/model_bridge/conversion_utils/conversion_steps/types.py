import torch


class WeightConversionInterface:
    def convert(self, input_value, *full_context):
        raise NotImplementedError(f"WeightConversionInterface called directly!")


# This type is used to indicate the position of a field in the remote model
REMOTE_FIELD = str
# This is the typing for a weight conversion when operations are needed on the REMOTE_FIELD.
# The WeightConversionInterface will be the instructions on the operations needed to bring the
# field into TransformerLens
CONVERSION = tuple[REMOTE_FIELD, WeightConversionInterface]
# This is the full range of actions that can be taken to bring a field into TransformerLens
# These can be configured as a predefined tensor, or a direction copy of the REMOTE_FIELD into
# TransformerLens, or a more in depth CONVERSION
CONVERSION_ACTION = torch.Tensor | REMOTE_FIELD | CONVERSION
# This type is for a full set of conversions from a remote model into TransformerLens. Each key in
# this dictionary will correspond to a field within a TransformerLens module, and each
# CONVERSION_ACTION will instruction TransformerLens on how to bring the field into a
# TransformerLens model. This type is repeated in both the root level of a model, as well as any
# layers within the model
FIELD_SET = dict[str, CONVERSION_ACTION]
