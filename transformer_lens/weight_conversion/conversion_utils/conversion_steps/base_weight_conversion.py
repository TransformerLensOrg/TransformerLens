import torch


class BaseWeightConversion:
    def convert(self, input_value):
        raise Exception(
            f"The conversion function for {type(self).__name__} needs to be implemented."
        )


CONVERSION = tuple[str, BaseWeightConversion]
FIELD_SET = dict[str, torch.Tensor | str | CONVERSION]
