"""Exceptions for the TransformerBridge module."""
import torch


class StopAtLayerException(Exception):
    """Exception raised to stop forward pass at a specific layer."""

    def __init__(self, layer_output: torch.Tensor, layer_idx: int):
        self.layer_output = layer_output
        self.layer_idx = layer_idx
        super().__init__(f"Stopped at layer {layer_idx}")
