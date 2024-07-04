from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.activation_functions import (
    SUPPORTED_ACTIVATIONS,
    ActivationFunction,
)


class ActivationFunctionFactory:
    @staticmethod
    def pick_activation_function(cfg: HookedTransformerConfig) -> ActivationFunction:
        activation_function = SUPPORTED_ACTIVATIONS.get(cfg.act_fn)

        if activation_function is None:
            raise ValueError(f"Invalid activation function name: {cfg.act_fn}")

        return activation_function
