"""Gated MLP bridge component.

This module contains the bridge component for gated MLP layers (e.g., LLaMA, Gemma).
"""
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, cast

import torch
import torch.nn as nn

from transformer_lens.model_bridge._relevance_rules import (
    half_rule,
    identity_rule,
    scale_gradient,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.mlp import MLPBridge


def _resolve_activation_fn_name(config: Any) -> Optional[str]:
    """The raw activation-name attribute a config exposes, in adapter priority order."""
    if config is None:
        return None
    for attr in ("activation_function", "hidden_activation", "hidden_act", "act_fn"):
        name = getattr(config, attr, None)
        if name is not None:
            return str(name)
    return None


_IDENTITY_RULE_UNSUPPORTED_ACTIVATIONS = {"relu", "relu2", "relu_2", "relu_squared"}


def identity_rule_supports_activation(config: Any) -> bool:
    """Whether the config's activation suits the Identity-rule's ``f(x) / x`` multiplier; the
    relu family is excluded (relu-squared's ratio is ``relu(x)``, not ``2 * relu(x)``)."""
    return _resolve_activation_fn_name(config) not in _IDENTITY_RULE_UNSUPPORTED_ACTIVATIONS


def resolve_activation_fn(config: Any) -> Callable:
    """Resolve activation function from a model config.

    Checks config attributes in order: activation_function, hidden_activation,
    hidden_act, act_fn. Maps common aliases to torch.nn.functional callables.
    """
    act_fn_name = _resolve_activation_fn_name(config)

    if act_fn_name is None or act_fn_name in ("silu", "swish"):
        return torch.nn.functional.silu
    if act_fn_name == "gelu":
        return torch.nn.functional.gelu
    if act_fn_name in ("gelu_new", "gelu_pytorch_tanh"):

        def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(x, approximate="tanh")

        return gelu_tanh
    if act_fn_name == "relu":
        return torch.nn.functional.relu
    if act_fn_name in ("relu2", "relu_2", "relu_squared"):

        def relu_squared(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.relu(x).square()

        return relu_squared
    return torch.nn.functional.silu


class _IdentityRuleActivation(nn.Module):
    """Identity-rule wrapper swapped in for an opaque HF MLP's activation callable; a wrapped
    ``nn.Module`` (the ``ACT2FN`` case) registers as a child so its parameters stay live."""

    def __init__(self, wrapped: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self._wrapped_activation = wrapped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return identity_rule(x, self._wrapped_activation)


class GatedMLPBridge(MLPBridge):
    """Bridge component for gated MLP layers.

    This component wraps a gated MLP layer from a remote model (e.g., LLaMA, Gemma)
    and provides a consistent interface for accessing its weights and performing MLP operations.

    Gated MLPs have the structure:
    output = down_proj(act_fn(gate_proj(x)) * up_proj(x))

    Where:
    - gate_proj: The gating projection (produces the activation to be gated)
    - up_proj (in): The input projection (produces the linear component)
    - down_proj (out): The output projection
    """

    hook_aliases = {
        "hook_pre": "gate.hook_out",
        "hook_pre_linear": "in.hook_out",
        "hook_post": "out.hook_in",
    }
    # property_aliases inherited from MLPBridge (W_gate, b_gate, W_in, b_in, W_out, b_out)

    def __init__(
        self,
        name: Optional[str],
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        optional: bool = False,
    ):
        """Initialize the gated MLP bridge.

        Args:
            name: The name of the component in the model (None if no container exists)
            config: Optional configuration (unused for GatedMLPBridge)
            submodules: Dictionary of submodules to register (e.g., gate_proj, up_proj, down_proj)
            optional: If True, setup skips this bridge when absent (hybrid architectures).
        """
        super().__init__(name, config, submodules=submodules or {}, optional=optional)
        self._relevance_rule_activation_active = False
        self._relevance_rule_gate_active = False
        # Opaque-path teardown state: (attr_name, original_value, was_child_module) restores the
        # swapped activation exactly; the handle is the down projection's gradient-scale pre-hook.
        self._relevance_activation_wrap: Optional[Tuple[str, Any, bool]] = None
        self._relevance_gate_hook_handle: Optional[Any] = None

    def _is_gated_mlp_shaped(self) -> bool:
        """Whether gate/in/out are wired; if not, this mount is benignly not a gated MLP."""
        if self.original_component is None:
            return False
        gate_module = getattr(self, "gate", None)
        in_module = getattr(self, "in", None)
        out_module = getattr(self, "out", None)
        return gate_module is not None and in_module is not None and out_module is not None

    def _find_activation_attr(self) -> Optional[str]:
        """Name of the HF module's callable activation attribute the opaque path swaps, or None."""
        component = self.original_component
        if component is None:
            return None
        for attr in ("act_fn", "activation_fn", "act", "activation"):
            if callable(getattr(component, attr, None)):
                return attr
        return None

    def _activation_rule_installable(self) -> bool:
        """Needs a supported activation form and, on the opaque path, a swappable callable."""
        return (
            identity_rule_supports_activation(self.config)
            and self._find_activation_attr() is not None
        )

    @property
    def _relevance_rule_kinds(self) -> Tuple[str, ...]:
        """Gate rule on any gated-MLP node, since it needs no weight or activation access;
        activation rule only when installable."""
        if not self._is_gated_mlp_shaped():
            return ()
        kinds: Tuple[str, ...] = ("multiplicative_gate",)
        if self._activation_rule_installable():
            kinds = ("activation",) + kinds
        return kinds

    @property
    def _relevance_rule_unsupported_kinds(self) -> Tuple[str, ...]:
        """``("activation",)`` on a gated-MLP node that cannot install it, so a request raises."""
        if not self._is_gated_mlp_shaped():
            return ()
        if "activation" not in self._relevance_rule_kinds:
            return ("activation",)
        return ()

    def _install_activation_rule(self) -> None:
        """Swap the HF activation callable for the Identity-rule wrapper; a missing callable is
        already refused through ``_relevance_rule_unsupported_kinds``."""
        component = self.original_component
        attr = self._find_activation_attr()
        if component is None or attr is None:
            return
        was_child_module = attr in component._modules
        original = component._modules[attr] if was_child_module else getattr(component, attr, None)
        # _find_activation_attr only returns an attribute whose value is callable.
        wrapper = _IdentityRuleActivation(cast(Callable[[torch.Tensor], torch.Tensor], original))
        # Avoid recursively changing the original activation's training state.
        wrapper.training = component.training
        if not was_child_module:
            component.__dict__.pop(attr, None)
        component._modules[attr] = wrapper
        self._relevance_activation_wrap = (attr, original, was_child_module)

    def _teardown_activation_rule(self) -> None:
        """Restore the activation callable swapped in by ``_install_activation_rule``."""
        if self._relevance_activation_wrap is None:
            return
        attr, original, was_child_module = self._relevance_activation_wrap
        component = self.original_component
        if component is not None:
            component._modules.pop(attr, None)
            if was_child_module:
                component._modules[attr] = original
            else:
                component.__dict__[attr] = original
        self._relevance_activation_wrap = None

    def _install_gate_rule(self) -> None:
        """Half-rule on the opaque path: halve the gradient into the down projection's input (the
        gate*up product); down's own weight gradient stays ordinary."""
        out_module = getattr(self, "out", None)
        down_component = getattr(out_module, "original_component", None)
        if down_component is None:
            return

        def _scale_product_gradient(
            module: nn.Module, args: Tuple[Any, ...]
        ) -> Optional[Tuple[Any, ...]]:
            if not args:
                return None
            return (scale_gradient(args[0], 0.5),) + tuple(args[1:])

        self._relevance_gate_hook_handle = down_component.register_forward_pre_hook(
            _scale_product_gradient
        )

    def _teardown_gate_rule(self) -> None:
        """Remove the down-projection gradient-scale hook."""
        if self._relevance_gate_hook_handle is not None:
            self._relevance_gate_hook_handle.remove()
            self._relevance_gate_hook_handle = None

    def _enable_relevance_rule(self, kind: str) -> None:
        """Set the flag (read by reconstructed forwards) and install the opaque-path hook (which
        the native forward needs, since a flag alone cannot alter it)."""
        if kind == "activation":
            self._relevance_rule_activation_active = True
            self._install_activation_rule()
        elif kind == "multiplicative_gate":
            self._relevance_rule_gate_active = True
            self._install_gate_rule()

    def _disable_relevance_rule(self, kind: str) -> None:
        """Deactivate the named rule and tear down its opaque-path hook."""
        if kind == "activation":
            self._teardown_activation_rule()
            self._relevance_rule_activation_active = False
        elif kind == "multiplicative_gate":
            self._teardown_gate_rule()
            self._relevance_rule_gate_active = False

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the gated MLP bridge.

        Intermediate hooks (gate.hook_out, in.hook_out, out.hook_in) only fire in
        compatibility mode with processed weights enabled. In non-compatibility mode,
        the HF component is called as an opaque forward and only hook_in/hook_out fire.

        Args:
            *args: Positional arguments for the original component
            **kwargs: Keyword arguments for the original component

        Returns:
            Output hidden states
        """
        if hasattr(self, "_use_processed_weights") and self._use_processed_weights:
            assert hasattr(self, "_processed_W_gate") and hasattr(self, "_processed_W_in"), (
                "Processed weights flag is set but weights are missing. "
                "This indicates a bug in set_processed_weights()."
            )
            assert self._processed_W_in is not None
            assert self._processed_W_out is not None
            hidden_states = args[0]
            hidden_states = self.hook_in(hidden_states)
            gate_output = torch.nn.functional.linear(
                hidden_states, self._processed_W_gate, self._processed_b_gate
            )
            if hasattr(self, "gate") and hasattr(self.gate, "hook_out"):
                gate_output = self.gate.hook_out(gate_output)
            linear_output = torch.nn.functional.linear(
                hidden_states, self._processed_W_in, self._processed_b_in
            )
            in_module = getattr(self, "in", None)
            if in_module is not None and hasattr(in_module, "hook_out"):
                linear_output = in_module.hook_out(linear_output)  # type: ignore[misc]
            act_fn = resolve_activation_fn(self.config)
            activated = (
                identity_rule(gate_output, act_fn)
                if self._relevance_rule_activation_active
                else act_fn(gate_output)
            )
            hidden = (
                half_rule(activated, linear_output)
                if self._relevance_rule_gate_active
                else activated * linear_output
            )
            if hasattr(self, "out") and hasattr(self.out, "hook_in"):
                hidden = self.out.hook_in(hidden)
            output = torch.nn.functional.linear(
                hidden, self._processed_W_out, self._processed_b_out
            )
            # The functional path bypasses the wrapped `out` projection, so fire
            # its hook_out here — it is the down-projection output.
            if hasattr(self, "out") and hasattr(self.out, "hook_out"):
                output = self.out.hook_out(output)
            output = self.hook_out(output)
            return output
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        hidden_states = args[0]
        hidden_states = self.hook_in(hidden_states)
        new_args = (hidden_states,) + args[1:]
        # Rules live on the HF submodules (see _enable_relevance_rule), so the native forward's
        # own hooks, gate multipliers and activation sparsity stay in the backward graph.
        output = self.original_component(*new_args, **kwargs)
        output = self.hook_out(output)
        return output

    def set_processed_weights(
        self, weights: Mapping[str, torch.Tensor | None], verbose: bool = False
    ) -> None:
        """Set the processed weights to use when layer norm is folded.

        Args:
            W_gate: The processed MLP gate weight tensor
            W_in: The processed MLP input weight tensor
            W_out: The processed MLP output weight tensor
            b_gate: The processed MLP gate bias tensor (optional)
            b_in: The processed MLP input bias tensor (optional)
            b_out: The processed MLP output bias tensor (optional)
            verbose: If True, print detailed information about weight setting
        """
        if verbose:
            print(
                f"\n  set_processed_weights: GatedMLPBridge (name={getattr(self, 'name', 'unknown')})"
            )
            print(f"    Received {len(weights)} weight keys")

        super().set_processed_weights(weights, verbose=verbose)  # type: ignore[arg-type]
        W_gate = weights.get("gate.weight")
        if W_gate is None:
            return
        b_gate = weights.get("gate.bias")

        W_in = weights.get("in.weight")
        b_in = weights.get("in.bias")
        W_out = weights.get("out.weight")
        b_out = weights.get("out.bias")

        if verbose:
            print(f"    Setting W_gate with shape: {W_gate.shape}")
            if b_gate is not None:
                print(f"    Setting b_gate with shape: {b_gate.shape}")
            if W_in is not None:
                print(f"    Setting W_in with shape: {W_in.shape}")
            if W_out is not None:
                print(f"    Setting W_out with shape: {W_out.shape}")

        self._use_processed_weights = True
        self._processed_W_gate = W_gate
        self._processed_b_gate = b_gate
        self._processed_W_in = W_in
        self._processed_b_in = b_in
        self._processed_W_out = W_out
        self._processed_b_out = b_out

        # Distribute to submodules if they support it
        gate_module = getattr(self, "gate", None)
        if gate_module and hasattr(gate_module, "set_processed_weights"):
            gate_weights: Dict[str, torch.Tensor] = {"weight": W_gate}
            if b_gate is not None:
                gate_weights["bias"] = b_gate
            gate_module.set_processed_weights(gate_weights, verbose=verbose)

        in_module = getattr(self, "in", None)
        if in_module and hasattr(in_module, "set_processed_weights") and W_in is not None:
            in_weights: Dict[str, torch.Tensor] = {"weight": W_in}
            if b_in is not None:
                in_weights["bias"] = b_in
            in_module.set_processed_weights(in_weights, verbose=verbose)

        out_module = getattr(self, "out", None)
        if out_module and hasattr(out_module, "set_processed_weights") and W_out is not None:
            out_weights: Dict[str, torch.Tensor] = {"weight": W_out}
            if b_out is not None:
                out_weights["bias"] = b_out
            out_module.set_processed_weights(out_weights, verbose=verbose)
