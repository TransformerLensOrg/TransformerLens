from torch import nn
from .conversion_steps.base_weight_conversion import FIELD_SET
from .conversion_steps.weight_conversion_set import WeightConversionSet


class ArchitectureConversion:
    def __init__(self, fields: FIELD_SET) -> None:
        self.field_set = WeightConversionSet(fields)
        
    def get_model(self, remote_module: nn.Module) -> dict:
        """This function allows a child conversion to return the model containing all of the
        weights. By default this is going to be the model parameter on the module, but there are
        some cases where it could have a different name

        Args:
            remote_module nn.Module: The module from hugging face

        Returns:
            dict: The model
        """
        return remote_module.model

    def convert(self, remote_module: nn.Module):
        return self.field_set.convert(input_value=self.get_model(remote_module))
