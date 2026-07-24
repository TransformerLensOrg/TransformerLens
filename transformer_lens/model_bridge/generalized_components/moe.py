"""Mixture of Experts bridge component.

This module contains the bridge component for Mixture of Experts layers.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class MoEBridge(GeneralizedComponent):
    """Bridge component for Mixture of Experts layers.

    This component wraps a Mixture of Experts layer from a remote model and provides a consistent interface
    for accessing its weights and performing MoE operations.

    hook_router_scores fires only when the wrapped block returns a tuple
    (gpt_oss, LLaDA2 remote); 5.13-native SparseMoeBlocks return a plain
    tensor, so router observability comes from the ``gate`` submodule's
    hook_out instead.
    """

    hook_aliases = {"hook_pre": "hook_in", "hook_post": "hook_out"}

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
        optional: bool = False,
    ):
        """Initialize the MoE bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration (unused for MoEBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
            optional: If True, setup skips this subtree when absent (dense layers)
        """
        super().__init__(name, config, submodules=submodules, optional=optional)
        self.hook_router_scores = HookPoint()

    def get_random_inputs(
        self,
        batch_size: int = 2,
        seq_len: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Generate random inputs for component testing.

        Args:
            batch_size: Batch size for generated inputs
            seq_len: Sequence length for generated inputs
            device: Device to place tensors on
            dtype: Dtype for generated tensors (defaults to float32)

        Returns:
            Dictionary of input tensors matching the component's expected input signature
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        d_model = self.config.d_model if self.config and hasattr(self.config, "d_model") else 768
        # Use positional args to avoid parameter name mismatches across MoE implementations
        # (e.g., Mixtral uses "hidden_states", GraniteMoe uses "layer_input")
        return {"args": (torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype),)}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the MoE bridge.

        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments

        Returns:
            Same return type as original component (tuple or tensor).
            For MoE models that return (hidden_states, router_scores), preserves the tuple.
            Router scores are also captured via hook for inspection.
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        target_dtype = None
        try:
            target_dtype = next(self.original_component.parameters()).dtype
        except StopIteration:
            pass
        if len(args) > 0:
            hooked = self.hook_in(args[0])
            if (
                target_dtype is not None
                and isinstance(hooked, torch.Tensor)
                and hooked.is_floating_point()
            ):
                hooked = hooked.to(dtype=target_dtype)
            args = (hooked,) + args[1:]
        elif "hidden_states" in kwargs:
            hooked = self.hook_in(kwargs["hidden_states"])
            if (
                target_dtype is not None
                and isinstance(hooked, torch.Tensor)
                and hooked.is_floating_point()
            ):
                hooked = hooked.to(dtype=target_dtype)
            kwargs = {**kwargs, "hidden_states": hooked}
        output = self.original_component(*args, **kwargs)
        if isinstance(output, tuple):
            hidden_states = output[0]
            if len(output) > 1:
                router_scores = output[1]
                # Some MoEs pack extras with the logits (LLaDA2 returns
                # (router_logits, topk_idx)); hook the first tensor.
                if isinstance(router_scores, tuple):
                    router_scores = next(
                        (t for t in router_scores if isinstance(t, torch.Tensor)), None
                    )
                if isinstance(router_scores, torch.Tensor):
                    self.hook_router_scores(router_scores)
            hidden_states = self.hook_out(hidden_states)
            return (hidden_states,) + output[1:]
        else:
            hidden_states = self.hook_out(output)
            return hidden_states


class MoERouterBridge(LinearBridge):
    """Bridge MoE router logits while preserving HF's tuple return.

    5.13 TopKRouters return ``(router_logits, topk_weights, topk_indices)``;
    hook_out fires on the logits (element ``logits_index`` — JetMoe puts them
    last) and the tuple is re-packed so HF's unpacking is undisturbed.
    """

    def __init__(self, *args: Any, logits_index: int = 0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.logits_index = logits_index

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        input = self.hook_in(input)
        output = self.original_component(input, *args, **kwargs)
        if not isinstance(output, tuple) or len(output) == 0:
            return self.hook_out(output)
        idx = self.logits_index % len(output)
        router_logits = self.hook_out(output[idx])
        return output[:idx] + (router_logits,) + output[idx + 1 :]

    def set_processed_weights(
        self, weights: Mapping[str, Optional[torch.Tensor]], verbose: bool = False
    ) -> None:
        """Accept routers whose Linear is nested inside a gating module.

        JetMoe's TopKGating holds its projection at ``router.layer.weight``, so
        this bridge receives ``{"layer.weight": ...}`` and the wrapped module has
        no ``.weight`` for the base writer. Weight processing never transforms
        router weights, so copy nested tensors back onto the wrapped module's
        own parameters by dotted path.
        """
        if "weight" in weights:
            super().set_processed_weights(weights, verbose=verbose)
            return
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        for key, tensor in weights.items():
            if tensor is None:
                continue
            target: Any = self.original_component
            *path, leaf = key.split(".")
            for part in path:
                target = getattr(target, part)
            param = getattr(target, leaf)
            if param.shape != tensor.shape:
                raise ValueError(
                    f"Router weight {key} shape {tuple(tensor.shape)} does not match "
                    f"parameter shape {tuple(param.shape)} on {self.name}"
                )
            with torch.no_grad():
                param.copy_(tensor.to(dtype=param.dtype, device=param.device))
