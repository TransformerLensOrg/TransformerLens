"""SSM block bridge component.

Block container for State Space Model (Mamba) layers. Unlike BlockBridge, this
component inherits directly from GeneralizedComponent to avoid transformer-
specific hook aliases (hook_attn_*, hook_mlp_*, hook_resid_mid → ln2.hook_in)
which don't apply to SSM blocks. SSM blocks have the structure norm → mixer → residual.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.exceptions import StopAtLayerException
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class SSMBlockBridge(GeneralizedComponent):
    """Block bridge for SSM (Mamba) layers.

    Direct subclass of GeneralizedComponent (NOT BlockBridge) to avoid inheriting
    transformer-specific hook aliases. SSM blocks contain a norm and a mixer; there
    is no attn/mlp/ln2 structure.
    """

    is_list_item: bool = True
    hook_aliases = {
        "hook_resid_pre": "hook_in",
        "hook_resid_post": "hook_out",
        "hook_mixer_in": "mixer.hook_in",
        "hook_mixer_out": "mixer.hook_out",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        hook_alias_overrides: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            name,
            config,
            submodules=submodules if submodules is not None else {},
            hook_alias_overrides=hook_alias_overrides,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward through the wrapped HF Mamba block with hook_in/hook_out.

        Delegates to the original HF component so submodule bridges (norm, mixer,
        and the mixer's own submodules like in_proj/conv1d) fire via the swapped
        attribute chain.
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        self._check_stop_at_layer(*args, **kwargs)
        args, kwargs = self._hook_input_hidden_states(args, kwargs)
        output = self.original_component(*args, **kwargs)
        return self._apply_output_hook(output)

    def _apply_output_hook(self, output: Any) -> Any:
        """Apply hook_out to the primary tensor in the output."""
        if isinstance(output, tuple) and len(output) > 0:
            first = output[0]
            if isinstance(first, torch.Tensor):
                first = self.hook_out(first)
                return (first,) + output[1:]
            return output
        if isinstance(output, torch.Tensor):
            return self.hook_out(output)
        return output

    def _hook_input_hidden_states(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Apply hook_in to the hidden_states input, whether in args or kwargs."""
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked = self.hook_in(args[0])
            args = (hooked,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])
        return args, kwargs

    def _check_stop_at_layer(self, *args: Any, **kwargs: Any) -> None:
        """Raise StopAtLayerException if this block matches the configured stop index."""
        if not (hasattr(self, "_stop_at_layer_idx") and self._stop_at_layer_idx is not None):
            return
        if self.name is None:
            return
        match = re.search(r"\.layers\.(\d+)", self.name) or re.search(r"blocks\.(\d+)", self.name)
        if not match:
            return
        layer_idx = int(match.group(1))
        if layer_idx != self._stop_at_layer_idx:
            return
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            input_tensor = args[0]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            input_tensor = kwargs["hidden_states"]
        else:
            raise ValueError(f"Cannot find input tensor to stop at layer {layer_idx}")
        input_tensor = self.hook_in(input_tensor)
        raise StopAtLayerException(input_tensor)
