"""Wrap-don't-reimplement bridge for HF's MambaMixer (Mamba-1)."""
from typing import Any

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class SSMMixerBridge(GeneralizedComponent):
    """Opaque wrapper around Mamba-1's MambaMixer.

    Submodules (in_proj, conv1d, x_proj, dt_proj, out_proj) are swapped into
    the HF mixer by ``replace_remote_component``, so their hooks fire when
    slow_forward accesses them. ``A_log`` and ``D`` reach the user via
    ``GeneralizedComponent.__getattr__`` delegation.

    Decode-step caveat: ``conv1d.hook_out`` fires only on prefill during
    stateful generation; see ``DepthwiseConv1DBridge`` for the reason.
    """

    hook_aliases = {
        "hook_in_proj": "in_proj.hook_out",
        "hook_conv": "conv1d.hook_out",
        "hook_x_proj": "x_proj.hook_out",
        "hook_dt_proj": "dt_proj.hook_out",
        "hook_ssm_out": "hook_out",
    }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Hook the input, delegate to HF slow_forward, hook the output."""
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        # Hook the hidden_states input (positional or keyword)
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked = self.hook_in(args[0])
            args = (hooked,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])

        output = self.original_component(*args, **kwargs)

        # Hook the primary output tensor, preserving tuple structure
        if isinstance(output, tuple) and len(output) > 0:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return (self.hook_out(first),) + output[1:]
            return output
        if isinstance(output, torch.Tensor):
            return self.hook_out(output)
        return output
