"""Shared eligibility gate for mask-derived position IDs."""

import inspect

from torch import nn


def accepts_mask_derived_position_ids(model: nn.Module) -> bool:
    """Accept explicit/kwargs positions unless the model owns their derivation."""
    parameters = inspect.signature(model.forward).parameters
    if "position_ids" not in parameters and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        return False
    for module in (model, getattr(model, "model", None), getattr(model, "language_model", None)):
        if module is not None and hasattr(module, "get_rope_index"):
            return False
    config = getattr(model, "config", None)
    for candidate in (config, getattr(config, "text_config", None)):
        scaling = getattr(candidate, "rope_scaling", None)
        if isinstance(scaling, dict) and "mrope_section" in scaling:
            return False
    for module in model.modules():
        if isinstance(module, nn.Embedding) and type(module).forward is not nn.Embedding.forward:
            if "attention_mask" in inspect.signature(module.forward).parameters:
                return False
    return True
