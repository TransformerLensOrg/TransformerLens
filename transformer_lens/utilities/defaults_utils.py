"""attribute_utils.

This module contains utility functions related to defaults
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from .attribute_utils import get_nested_attr, set_nested_attr

USE_DEFAULT_VALUE = None


def override_or_use_default_value(
    default_flag: Any,
    override: Optional[Any] = None,
) -> Any:
    """
    Determines which flag to return based on whether an overriding flag is provided.
    If a not-None overriding flag is provided, it is returned.
    Otherwise, the global flag is returned.
    """
    return override if override is not None else default_flag


class LocallyOverridenDefaults:
    """
    Context manager that allows temporary overriding of default values within a model.
    Once the context is exited, the default values are restored.

    WARNING: This context manager must be used for any function/method that directly accesses
    default values which may be overridden by the user using the function/method's arguments,
    e.g., `model.cfg.default_prepend_bos` and `model.tokenizer.padding_side` which can be
    overriden by `prepend_bos` and `padding_side` arguments, respectively, in the `to_tokens`.
    """

    def __init__(self, model, **overrides):
        """
        Initializes the context manager.

        Args:
            model (HookedTransformer): The model whose default values will be overridden.
            overrides (dict): Key-value pairs of properties to override and their new values.
        """
        self.model = model
        self.overrides = overrides

        # Dictionary defining valid defaults, valid values, and locations to find and store them
        self.values_with_defaults = {
            "prepend_bos": {
                "default_location": "model.cfg.default_prepend_bos",
                "valid_values": [USE_DEFAULT_VALUE, True, False],
                "skip_overriding": False,
                "default_value_to_restore": None,  # Will be set later
            },
            "padding_side": {
                "default_location": "model.tokenizer.padding_side",
                "valid_values": [USE_DEFAULT_VALUE, "left", "right"],
                "skip_overriding": model.tokenizer is None,  # Do not override if tokenizer is None
                "default_value_to_restore": None,  # Will be set later
            },
        }

        # Ensure provided overrides are defined in the dictionary above
        for override in overrides:
            assert override in self.values_with_defaults, (
                f"{override} is not a valid parameter to override. "
                f"Valid parameters are {self.values_with_defaults.keys()}."
            )

    def __enter__(self):
        """
        Override default values upon entering the context.
        """
        for property, override in self.overrides.items():
            info = self.values_with_defaults[property]
            if info["skip_overriding"]:
                continue  # Skip if overriding for this property is disabled

            # Ensure the override is a valid value
            valid_values = info["valid_values"]
            assert (
                override in valid_values  # type: ignore
            ), f"{property} must be one of {valid_values}, but got {override}."

            # Fetch current default and store it to restore later
            default_location = info["default_location"]
            default_value = get_nested_attr(self, default_location)
            info["default_value_to_restore"] = deepcopy(default_value)

            # Override the default value
            locally_overriden_value = override_or_use_default_value(default_value, override)
            set_nested_attr(self, default_location, locally_overriden_value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restore default values upon exiting the context.
        """
        for property in self.overrides:
            info = self.values_with_defaults[property]
            if info["skip_overriding"]:
                continue

            # Restore the default value from before the context was entered
            default_location = info["default_location"]
            default_value = info["default_value_to_restore"]
            set_nested_attr(self, default_location, default_value)
