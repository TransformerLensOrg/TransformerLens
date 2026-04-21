"""Bridge for Mamba-style depthwise causal Conv1d (distinct from GPT-2's Conv1D linear)."""
from typing import Any

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class DepthwiseConv1DBridge(GeneralizedComponent):
    """Wraps an ``nn.Conv1d`` depthwise causal convolution with input/output hooks.

    Hook shapes (channel-first, as HF's MambaMixer transposes before the call):
        hook_in:  [batch, channels, seq_len]
        hook_out: [batch, channels, seq_len + conv_kernel - 1]  (pre causal trim)

    Decode-step limitation: on stateful generation, HF's Mamba/Mamba-2 mixers
    bypass ``self.conv1d(...)`` and read ``self.conv1d.weight`` directly, so the
    forward hook never fires on decode steps — only on prefill. For per-step
    conv output during decode, compute it manually from the cached conv_states
    and ``conv1d.original_component.weight``, or run token-by-token via
    ``forward()`` instead of ``generate()``.
    """

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
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
