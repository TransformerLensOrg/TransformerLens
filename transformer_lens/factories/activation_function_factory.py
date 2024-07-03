from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.activation_functions import ActivationFunction, SUPPORTED_ACTIVATIONS

class ActivationFunctionFactory:
    
    @staticmethod
    def pick_activation_function(config: HookedTransformerConfig) -> ActivationFunction:
        activation_function = SUPPORTED_ACTIVATIONS.get(config.act_fn)
            
        if (activation_function is None):
            raise ValueError(f"Invalid activation function name: {config.act_fn}")
        
        return activation_function