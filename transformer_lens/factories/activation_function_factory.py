"""Activation Function Factory

Centralized location for selection supported activation functions throughout TransformerLens
"""

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.activation_functions import (
    SUPPORTED_ACTIVATIONS,
    ActivationFunction,
)


class ActivationFunctionFactory:
    @staticmethod
    def pick_activation_function(cfg: HookedTransformerConfig) -> ActivationFunction:
        act_fn = cfg.act_fn

        if act_fn is None:
            raise ValueError("act_fn not set when trying to select Activation Function")

        activation_function = SUPPORTED_ACTIVATIONS.get(act_fn)

        if activation_function is None:
            raise ValueError(f"Invalid activation function name: {act_fn}")

        return activation_function
