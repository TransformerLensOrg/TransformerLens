"""Exceptions for TransformerBridge.

This module contains custom exceptions used by the TransformerBridge.
"""
from __future__ import annotations

import torch


class StopAtLayerException(Exception):
    """Exception raised when execution should stop at a specific layer.

    This exception is used to implement stop_at_layer functionality by having
    blocks check if they should stop and raise this exception, which is then
    caught in the main forward pass.
    """

    def __init__(self, layer_output: torch.Tensor):
        """Initialize with the output tensor from the layer where we stopped.

        Args:
            layer_output: The output tensor from the last layer that executed
        """
        self.layer_output = layer_output
        super().__init__()
