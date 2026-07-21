"""Normalization bridge component implementation."""
import contextlib
import warnings
from typing import Any, ContextManager, Dict, Optional, cast

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

# The native-autograd path returns HF's own output, so hook edits and backward hooks
# can only be honored by switching to the python-norm computation, whose numerics
# differ from HF's at float-rounding scale.
NATIVE_PATH_BWD_FALLBACK_WARNING = (
    "Backward hooks on hook_scale/hook_normalized require grad-connected hook tensors; "
    "falling back from the native-autograd path to the python-norm path. Output numerics "
    "may differ from the unhooked forward at float-rounding scale."
)
NATIVE_PATH_EDIT_FALLBACK_WARNING = (
    "A forward hook edited hook_scale/hook_normalized on the native-autograd path; the "
    "output is reconstructed from the hooked values instead of HF's native forward. "
    "Output numerics may differ from the unhooked forward at float-rounding scale."
)


class NormalizationBridge(GeneralizedComponent):
    """Normalization bridge that wraps transformer normalization layers but implements the calculation from scratch.

    This component provides standardized input/output hooks.
    """

    property_aliases = {"w": "weight", "b": "bias"}

    def __init__(
        self,
        name: str,
        config: Any,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
        use_native_layernorm_autograd: bool = False,
        uses_rms_norm: Optional[bool] = None,
    ):
        """Initialize the normalization bridge.

        Args:
            name: The name of this component
            config: Optional configuration
            submodules: Dictionary of GeneralizedComponent submodules to register
            use_native_layernorm_autograd: If True, use HuggingFace's native LayerNorm
                                          autograd for exact gradient matching. If False,
                                          use custom implementation. Defaults to False.
            uses_rms_norm: Force RMSNorm vs LayerNorm; None defers to introspection
                then ``config.uses_rms_norm``.
        """
        super().__init__(name, config, submodules=submodules)
        self.hook_normalized = HookPoint()
        self.hook_scale = HookPoint()
        self.use_native_layernorm_autograd = use_native_layernorm_autograd
        self._uses_rms_norm_override = uses_rms_norm

    @property
    def uses_rms_norm(self) -> bool:
        """Whether this bridge treats the wrapped module as RMSNorm.

        Override > module introspection > config. Introspection guards against
        a shared config (RMSNorm LM + LayerNorm vision tower) misclassifying
        a real ``nn.LayerNorm``.
        """
        if self._uses_rms_norm_override is not None:
            return self._uses_rms_norm_override
        component = self.original_component
        if component is not None:
            if isinstance(component, torch.nn.LayerNorm):
                return False
            if "RMSNorm" in type(component).__name__:
                return True
        return bool(getattr(self.config, "uses_rms_norm", False))

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the normalization bridge.

        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Normalized output
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        assert self.config is not None
        hidden_states = self.hook_in(hidden_states)
        self._last_input_before_norm = hidden_states
        if self.use_native_layernorm_autograd:
            result = self._hf_autograd_forward_with_hooks(hidden_states)
        elif hasattr(self.config, "layer_norm_folding") and self.config.layer_norm_folding:
            result = self._hf_autograd_forward_with_hooks(hidden_states)
        else:
            result = self._python_norm_forward(hidden_states)
        output = self.hook_out(result)
        return output

    def _python_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """From-scratch normalization with live hooks: edits propagate, gradients flow."""
        # Upcast to float32 for normalization precision (matches HT's RMSNorm behavior)
        input_dtype = hidden_states.dtype
        if input_dtype not in (torch.float32, torch.float64):
            hidden_states = hidden_states.float()
        if not self.uses_rms_norm:
            hidden_states = hidden_states - hidden_states.mean(-1, keepdim=True)
        scale = self.hook_scale(
            (
                hidden_states.pow(2).mean(-1, keepdim=True) + getattr(self.config, "eps", 1e-05)
            ).sqrt()
        )
        hidden_states = self.hook_normalized(hidden_states / scale)
        return self._apply_weight_and_bias(hidden_states, input_dtype)

    def _apply_weight_and_bias(
        self, hidden_states: torch.Tensor, input_dtype: torch.dtype
    ) -> torch.Tensor:
        """Apply weight/bias in float32 before casting back (matches HF precision)."""
        hidden_states = hidden_states * self.weight
        component = self.original_component
        if (
            not self.uses_rms_norm
            and component is not None
            and hasattr(component, "bias")
            and component.bias is not None
        ):
            hidden_states = hidden_states + cast(torch.Tensor, component.bias)
        return hidden_states.to(input_dtype)

    def _hf_autograd_forward_with_hooks(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that preserves HF's autograd while firing intermediate hooks.

        When hooks only observe (return ``None``, e.g. ``run_with_cache``), the result is
        HF's own forward — bit-identical numerics and exact autograd. When a forward hook
        edits ``hook_scale`` / ``hook_normalized``, the output is reconstructed from the
        hooked values so the edit propagates; when backward hooks are attached, the whole
        computation takes the python-norm path so hook tensors stay in the autograd graph.
        Both fallbacks warn, since their numerics differ from HF's at rounding scale.

        Args:
            x: Input tensor

        Returns:
            Normalized output tensor
        """
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        if self.hook_scale.bwd_hooks or self.hook_normalized.bwd_hooks:
            warnings.warn(NATIVE_PATH_BWD_FALLBACK_WARNING)
            return self._python_norm_forward(x)
        has_fwd_hooks = bool(self.hook_scale.fwd_hooks or self.hook_normalized.fwd_hooks)
        # No hooks: skip building a graph for observation-only intermediates. With hooks,
        # keep grad so an edited value stays connected to the input.
        grad_ctx: ContextManager[Any] = (
            contextlib.nullcontext() if has_fwd_hooks else torch.no_grad()
        )
        with grad_ctx:
            # Upcast to float32 for hook precision (matches HT's RMSNorm/LayerNorm behavior)
            x_float = x.float() if x.dtype not in (torch.float32, torch.float64) else x
            if not self.uses_rms_norm:
                x_centered = x_float - x_float.mean(-1, keepdim=True)
            else:
                x_centered = x_float
            eps_tensor = getattr(self.original_component, "eps", None)
            if eps_tensor is None:
                eps_tensor = getattr(self.original_component, "variance_epsilon", None)
            if eps_tensor is None:
                eps_value: float | torch.Tensor = getattr(self.config, "eps", 1e-05)
            else:
                eps_value = eps_tensor
            variance = x_centered.pow(2).mean(-1, keepdim=True)
            if isinstance(eps_value, torch.Tensor):
                inv_rms = torch.rsqrt(variance + eps_value)
                scale = (variance + eps_value).sqrt()
            else:
                inv_rms = torch.rsqrt(variance + float(eps_value))
                scale = (variance + float(eps_value)).sqrt()
            # Use rsqrt for x_normalized to match HF's actual computation path
            # (LlamaRMSNorm uses x * rsqrt(variance + eps)). Keep scale as sqrt
            # for hook_scale (denominator convention used by HookedTransformer).
            x_normalized = x_centered * inv_rms
            hooked_scale = self.hook_scale(scale)
            if hooked_scale is not scale:
                # Edited scale: recompute with the denominator convention so the edit
                # feeds hook_normalized, mirroring the python-norm path's ordering.
                x_normalized = x_centered / hooked_scale
            hooked_normalized = self.hook_normalized(x_normalized)
        input_dtype = x.dtype
        # A hook returning None keeps the original tensor object (see HookPoint), so
        # identity is the edit signal. Note in-place mutation of the hook value without
        # returning it is NOT detected — return the tensor from the hook to edit.
        if hooked_scale is scale and hooked_normalized is x_normalized:
            result = self.original_component(x)
            if result.dtype != input_dtype:
                result = result.to(input_dtype)
            return result
        warnings.warn(NATIVE_PATH_EDIT_FALLBACK_WARNING)
        return self._apply_weight_and_bias(hooked_normalized, input_dtype)
