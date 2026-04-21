"""Activation Function Factory

Centralized location for selection supported activation functions throughout TransformerLens
"""

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.activation_functions import (
    SUPPORTED_ACTIVATIONS,
    XIELU,
    ActivationFunction,
)


class ActivationFunctionFactory:
    @staticmethod
    def pick_activation_function(cfg: HookedTransformerConfig) -> ActivationFunction:
        """Use this to select what activation function is needed based on configuration.

        Args:
            cfg (HookedTransformerConfig): The already created hooked transformer config

        Raises:
            ValueError: If there is a problem with the requested activation function.

        Returns:
            ActivationFunction: The activation function based on the dictionary of supported activations.
        """
        act_fn = cfg.act_fn

        if act_fn is None:
            raise ValueError("act_fn not set when trying to select Activation Function")

        # XIeLU has trainable parameters (alpha_p, alpha_n, beta) that are loaded
        # from pretrained weights via load_state_dict. Return a class instance so
        # the parameters are registered as nn.Parameters on the MLP module.
        if act_fn == "xielu":
            return XIELU()

        activation_function = SUPPORTED_ACTIVATIONS.get(act_fn)

        if activation_function is None:
            raise ValueError(f"Invalid activation function name: {act_fn}")

        return activation_function
