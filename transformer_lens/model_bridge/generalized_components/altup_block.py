"""Block bridge for AltUp (Alternating Updates) decoder layers."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.exceptions import StopAtLayerException
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class AltUpBlockBridge(GeneralizedComponent):
    """Block bridge for a decoder layer that operates on a stacked AltUp residual.

    Direct GeneralizedComponent subclass (not BlockBridge) because the layer's residual is a
    stacked ``[num_altup_inputs, batch, seq, d_model]`` tensor, not a single stream.
    ``hook_in``/``hook_out`` carry the full stack; ``hook_resid_pre``/``hook_resid_post`` expose
    the active stream (``altup_active_idx``) as a conventional ``[batch, seq, d_model]`` residual
    and are patchable (written back into the stack).
    """

    is_list_item: bool = True
    hook_aliases = {
        "hook_attn_out": "self_attn.hook_out",
        "hook_mlp_out": "mlp.hook_out",
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
        self.altup_active_idx = int(getattr(config, "altup_active_idx", 0) or 0)
        # Active AltUp stream as a conventional residual (patchable).
        self.hook_resid_pre = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the HF layer, hooking the AltUp stack and the active residual stream."""
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        self._check_stop_at_layer(*args, **kwargs)
        args, kwargs = self._hook_input(args, kwargs)
        output = self.original_component(*args, **kwargs)
        return self._hook_output(output)

    def _patch_active_stream(self, stack: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        """Fire ``hook`` on the active AltUp stream and write the (possibly patched) result back."""
        if (
            not isinstance(stack, torch.Tensor)
            or stack.dim() < 1
            or stack.shape[0] <= self.altup_active_idx
        ):
            return stack
        active = hook(stack[self.altup_active_idx])
        stack = stack.clone()
        stack[self.altup_active_idx] = active
        return stack

    def _hook_input(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Hook the stacked hidden_states then the active residual, positional or by name."""
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden = self._patch_active_stream(self.hook_in(args[0]), self.hook_resid_pre)
            args = (hidden,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self._patch_active_stream(
                self.hook_in(kwargs["hidden_states"]), self.hook_resid_pre
            )
        return args, kwargs

    def _hook_output(self, output: Any) -> Any:
        """Hook the active residual then the stacked output, preserving any tuple structure."""
        primary = output[0] if isinstance(output, tuple) and len(output) > 0 else output
        if isinstance(primary, torch.Tensor):
            primary = self.hook_out(self._patch_active_stream(primary, self.hook_resid_post))
        if isinstance(output, tuple) and len(output) > 0:
            return (primary,) + output[1:]
        return primary

    def _check_stop_at_layer(self, *args: Any, **kwargs: Any) -> None:
        """Raise StopAtLayerException (carrying the AltUp stack) at the configured stop index."""
        if getattr(self, "_stop_at_layer_idx", None) is None or self.name is None:
            return
        match = re.search(r"\.layers\.(\d+)", self.name) or re.search(r"blocks\.(\d+)", self.name)
        if not match or int(match.group(1)) != self._stop_at_layer_idx:
            return
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            tensor = args[0]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            tensor = kwargs["hidden_states"]
        else:
            raise ValueError(f"Cannot find input tensor to stop at layer {self.name}")
        raise StopAtLayerException(self.hook_in(tensor))
