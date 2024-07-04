import pytest
import torch

from transformer_lens.factories.activation_function_factory import (
    ActivationFunctionFactory,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.activation_functions import SUPPORTED_ACTIVATIONS


@pytest.mark.parametrize("act_function", SUPPORTED_ACTIVATIONS.keys())
def test_pick_activation_function_runs(act_function):
    config = HookedTransformerConfig.unwrap(
        {"n_layers": 12, "n_ctx": 1024, "d_head": 64, "d_model": 128, "act_fn": act_function}
    )
    function = ActivationFunctionFactory.pick_activation_function(config)
    assert function is not None
    dummy_data = torch.zeros((1, 4, 32))
    result = function(dummy_data)
    assert isinstance(result, torch.Tensor)
