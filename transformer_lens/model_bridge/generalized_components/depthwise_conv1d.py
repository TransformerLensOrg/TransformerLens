"""Depthwise 1D convolution bridge component.

Wraps an `nn.Conv1d` used as a depthwise causal convolution in Mamba (and
similar SSM) models. Distinct from `Conv1DBridge`, which wraps the GPT-2
style Conv1D that is actually a linear layer stored in transposed form.

Tensor format note: HF's MambaMixer transposes hidden states to channel-first
(`[batch, channels, seq_len]`) before invoking `conv1d`, and transposes back
after. The conv1d wrapper only sees this channel-first format; `hook_out`
captures the raw conv output prior to the causal trim that MambaMixer applies
outside the conv1d call.
"""
from typing import Any

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class DepthwiseConv1DBridge(GeneralizedComponent):
    """Bridge component for depthwise 1D convolutions used in Mamba models.

    Hooks:
        hook_in:  shape [batch, channels, seq_len] (channel-first)
        hook_out: shape [batch, channels, seq_len + conv_kernel - 1]
                  (channel-first, before the causal trim applied by the mixer)
    """

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the wrapped nn.Conv1d with input/output hooks."""
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )
        input = self.hook_in(input)
        output = self.original_component(input, *args, **kwargs)
        output = self.hook_out(output)
        return output

    def __repr__(self) -> str:
        if self.original_component is not None:
            try:
                in_channels = self.original_component.in_channels
                out_channels = self.original_component.out_channels
                kernel_size = self.original_component.kernel_size
                groups = self.original_component.groups
                return (
                    f"DepthwiseConv1DBridge({in_channels} -> {out_channels}, "
                    f"kernel_size={kernel_size}, groups={groups})"
                )
            except AttributeError:
                return (
                    f"DepthwiseConv1DBridge(name={self.name}, "
                    f"original_component={type(self.original_component).__name__})"
                )
        return f"DepthwiseConv1DBridge(name={self.name}, original_component=None)"
