"""attribute_utils.

This module contains utility functions related to attributes
"""

from __future__ import annotations


def get_nested_attr(obj, attr_str):
    """
    Retrieves a nested attribute from an object based on a dot-separated string.

    For example, if `attr_str` is "a.b.c", this function will return `obj.a.b.c`.

    Args:
        obj (Any): The object from which to retrieve the attribute.
        attr_str (str): A dot-separated string representing the attribute hierarchy.

    Returns:
        Any: The value of the nested attribute.
    """
    attrs = attr_str.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_str, value):
    """
    Sets a nested attribute of an object based on a dot-separated string.

    For example, if `attr_str` is "a.b.c", this function will set the value of `obj.a.b.c` to `value`.

    Args:
        obj (Any): The object on which to set the attribute.
        attr_str (str): A dot-separated string representing the attribute hierarchy.
        value (Any): The value to set for the nested attribute.
    """
    attrs = attr_str.split(".")

    # Navigate to the deepest object containing the attribute to be set
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)

    # Set the nested attribute's value
    setattr(obj, attrs[-1], value)
