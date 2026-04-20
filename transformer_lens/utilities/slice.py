"""Slice.

This module contains the functionailty for the Slice object
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .tensors import to_numpy

# Type alias
SliceInput = Optional[
    Union[
        int,
        Tuple[int,],
        Tuple[int, int],
        Tuple[int, int, int],
        List[int],
        torch.Tensor,
        np.ndarray,
    ]
]
"""An object that represents a slice input. It can be a tuple of integers or a slice object.

An optional type alias for a slice input used in the `ActivationCache` module.

A `SliceInput` can be one of the following types:
    - `int`: an integer representing a single position
    - `Tuple[int, int]`: a tuple of two integers representing a range of positions
    - `Tuple[int, int, int]`: a tuple of three integers representing a range of positions with a step size
    - `List[int]`: a list of integers representing multiple positions
    - `torch.Tensor`: a tensor containing a boolean mask or a list of indices to be selected from the input tensor.

`SliceInput` is used in the `apply_ln_to_stack` method in the `ActivationCache` module.
"""


class Slice:
    """An object that represents a slice input. It can be a tuple of integers or a slice object.

    We use a custom slice syntax because Python/Torch's don't let us reduce the number of dimensions:

    Note that slicing with input_slice=None means do nothing, NOT add an extra dimension (use unsqueeze for that)

    There are several modes:
    int - just index with that integer (decreases number of dimensions)
    slice - Input is a tuple converted to a slice ((k,) means :k, (k, m) means m:k, (k, m, n) means m:k:n)
    array - Input is a list or tensor or numpy array, converted to a numpy array, and we take the stack of values at those indices
    identity - Input is None, leave it unchanged.

    Examples for dim=0:
    if input_slice=0, tensor -> tensor[0]
    elif input_slice = (1, 5), tensor -> tensor[1:5]
    elif input_slice = (1, 5, 2), tensor -> tensor[1:5:2] (ie indexing with [1, 3])
    elif input_slice = [1, 4, 5], tensor -> tensor[[1, 4, 5]] (ie changing the first axis to have length 3, and taking the indices 1, 4, 5 out).
    elif input_slice is a Tensor, same as list - Tensor is assumed to be a 1D list of indices.
    """

    slice: Union[int, slice, np.ndarray]

    def __init__(
        self,
        input_slice: SliceInput = None,
    ):
        """
        Modular component for slicing tensors. Can be used to slice a tensor along a given dimension, or to index into a tensor along a given dimension.

        Args:
            input_slice (SliceInput): The slice to apply. Can be an int, a tuple, a list, a torch.Tensor, or None. If None, do nothing.

        Raises:
            ValueError: If the input_slice is not one of the above types.
        """
        if isinstance(input_slice, tuple):
            self.slice = slice(*input_slice)
            self.mode = "slice"
        elif isinstance(input_slice, int):
            self.slice = input_slice
            self.mode = "int"
        elif isinstance(input_slice, slice):
            self.slice = input_slice
            self.mode = "slice"
        elif type(input_slice) in [list, torch.Tensor, np.ndarray]:
            self.slice = to_numpy(input_slice)
            self.mode = "array"
        elif input_slice is None:
            self.slice = slice(None)
            self.mode = "identity"
        else:
            raise ValueError(f"Invalid input_slice {input_slice}")

    def apply(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """
        Takes in a tensor and a slice, and applies the slice to the given dimension (supports positive and negative dimension syntax). Returns the sliced tensor.

        Args:
            tensor (torch.Tensor): The tensor to slice.
            dim (int, optional): The dimension to slice along. Supports positive and negative dimension syntax.

        Returns:
            torch.Tensor: The sliced tensor.
        """
        ndim = tensor.ndim
        slices = [slice(None)] * ndim
        slices[dim] = self.slice  # type: ignore
        return tensor[tuple(slices)]

    def indices(
        self,
        max_ctx: Optional[int] = None,
    ) -> Union[np.ndarray, np.int32, np.int64]:
        """
        Returns the indices when this slice is applied to an axis of size max_ctx. Returns them as a numpy array, for integer slicing it is eg array([4])

        Args:
            max_ctx (int, optional): The size of the axis to slice. Only used if the slice is not an integer.

        Returns:
            np.ndarray: The indices that this slice will select.

        Raises:
            ValueError: If the slice is not an integer and max_ctx is not specified.
        """
        if self.mode == "int":
            return np.array([self.slice], dtype=np.int64)
        if max_ctx is None:
            raise ValueError("max_ctx must be specified if slice is not an integer")
        return np.arange(max_ctx, dtype=np.int64)[self.slice]

    def __repr__(
        self,
    ) -> str:
        return f"Slice: {self.slice} Mode: {self.mode} "

    @classmethod
    def unwrap(
        cls,
        slice_input: Union["Slice", SliceInput],
    ) -> "Slice":
        """
        Takes a Slice-like input and converts it into a Slice, if it is not already.

        Args:
            slice_input (Union[Slice, SliceInput]): The input to turn into a Slice.

        Returns:
            Slice: A Slice object.
        """
        if not isinstance(slice_input, Slice):
            if isinstance(
                slice_input, int
            ):  # slicing with an int collapses the dimension so this stops the pos dimension from collapsing
                slice_input = [slice_input]
            slice_input = Slice(slice_input)
        return slice_input
