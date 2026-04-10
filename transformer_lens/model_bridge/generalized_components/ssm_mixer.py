"""Mamba-1 mixer bridge component.

Wraps HF's MambaMixer as an opaque component with hook_in/hook_out. Submodule
bridges (in_proj, conv1d, x_proj, dt_proj, out_proj) are swapped into the HF
mixer's module tree by `replace_remote_component`, so they fire automatically
when the HF forward accesses those attributes.

This is a wrap-don't-reimplement bridge: the HF mixer's slow_forward runs as-is.
Submodule hooks give researchers access to all projection inputs and outputs.
Mamba-1 `nn.Parameter`s (`A_log`, `D`) are accessible via GeneralizedComponent's
`__getattr__` fallback, which delegates to `self.original_component`.
"""
from typing import Any

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class SSMMixerBridge(GeneralizedComponent):
    """Bridge component for Mamba-1 MambaMixer.

    Has `x_proj` and `dt_proj` as submodules (these do not exist in Mamba-2).
    Does not have head structure or inner norm.
    """

    hook_aliases = {
        "hook_in_proj": "in_proj.hook_out",
        "hook_conv": "conv1d.hook_out",
        "hook_x_proj": "x_proj.hook_out",
        "hook_dt_proj": "dt_proj.hook_out",
        "hook_ssm_out": "hook_out",
    }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass: hook_in → HF mixer → hook_out.

        The HF mixer's slow_forward calls `self.in_proj`, `self.conv1d`, etc.
        — each of which has been swapped for a bridge submodule — so submodule
        hooks fire automatically inside the opaque forward.
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked = self.hook_in(args[0])
            args = (hooked,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])

        output = self.original_component(*args, **kwargs)

        if isinstance(output, tuple) and len(output) > 0:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return (self.hook_out(first),) + output[1:]
            return output
        if isinstance(output, torch.Tensor):
            return self.hook_out(output)
        return output
