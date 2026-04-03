"""Library availability utilities.

Utilities for checking if optional libraries are available without importing them.
"""

import importlib.util
import sys


def is_library_available(name: str) -> bool:
    """
    Checks if a library is installed in the current environment without importing it.
    Prevents crash or segmentation fault.

    Args:
        name: The name of the library to check (e.g., "wandb", "transformers")

    Returns:
        True if the library is available, False otherwise
    """
    return name in sys.modules or importlib.util.find_spec(name) is not None
