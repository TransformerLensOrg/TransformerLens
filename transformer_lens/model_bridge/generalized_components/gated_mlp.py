"""Gated MLP bridge component.

This module contains the bridge component for gated MLP layers (e.g., LLaMA, Gemma).
"""
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch

from transformer_lens.model_bridge._relevance_rules import half_rule, identity_rule
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.mlp import (
    MLPBridge,
    weight_layout_in_out,
)


def resolve_activation_fn(config: Any) -> Callable:
    """Resolve activation function from a model config.

    Checks config attributes in order: activation_function, hidden_activation,
    hidden_act, act_fn. Maps common aliases to torch.nn.functional callables.
    """
    act_fn_name = None
    if config is not None:
        for attr in ("activation_function", "hidden_activation", "hidden_act", "act_fn"):
            act_fn_name = getattr(config, attr, None)
            if act_fn_name is not None:
                break

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


class _GatedMLPRecomputeRule(torch.autograd.Function):
    """Wrap a raw gated-MLP module's own forward call in the Identity-/Half-rule VJP.

    Forward returns ``component(x, ...)`` unchanged, so the result is bit-identical
    to the native forward by construction. The opaque call gives no access to its own
    internal gate/up/down intermediates, so backward recomputes them from the
    TL-oriented ``W_gate``/``W_in``/``W_out`` (checkpointing-style: the values are
    rederived here rather than saved from forward) and reapplies the Identity-rule to
    the activation and/or the Half-rule to the gate*up product, per whichever of the
    two is active. Weight and bias gradients come out of the same recomputed graph,
    via ``torch.autograd.grad``, so they keep their ordinary form -- the rules only
    redefine how relevance reaches the input, not parameter training gradients.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        w_gate: torch.Tensor,
        b_gate: Optional[torch.Tensor],
        w_in: torch.Tensor,
        b_in: Optional[torch.Tensor],
        w_out: torch.Tensor,
        b_out: Optional[torch.Tensor],
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        activation_rule_active: bool,
        gate_rule_active: bool,
        component: torch.nn.Module,
        extra_args: Tuple[Any, ...],
        extra_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        ctx.save_for_backward(x, w_gate, w_in, w_out)
        ctx.b_gate = b_gate
        ctx.b_in = b_in
        ctx.b_out = b_out
        ctx.act_fn = act_fn
        ctx.activation_rule_active = activation_rule_active
        ctx.gate_rule_active = gate_rule_active
        result: torch.Tensor = component(x, *extra_args, **extra_kwargs)
        return result

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        x, w_gate, w_in, w_out = ctx.saved_tensors
        b_gate, b_in, b_out = ctx.b_gate, ctx.b_in, ctx.b_out
        act_fn = ctx.act_fn

        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            w_gate_ = w_gate.detach().requires_grad_(True)
            w_in_ = w_in.detach().requires_grad_(True)
            w_out_ = w_out.detach().requires_grad_(True)
            b_gate_ = b_gate.detach().requires_grad_(True) if b_gate is not None else None
            b_in_ = b_in.detach().requires_grad_(True) if b_in is not None else None
            b_out_ = b_out.detach().requires_grad_(True) if b_out is not None else None

            gate_output = x_ @ w_gate_
            if b_gate_ is not None:
                gate_output = gate_output + b_gate_
            up_output = x_ @ w_in_
            if b_in_ is not None:
                up_output = up_output + b_in_

            activated = (
                identity_rule(gate_output, act_fn)
                if ctx.activation_rule_active
                else act_fn(gate_output)
            )
            gated = (
                half_rule(activated, up_output) if ctx.gate_rule_active else activated * up_output
            )

            down = gated @ w_out_
            if b_out_ is not None:
                down = down + b_out_

        leaves = [x_, w_gate_, b_gate_, w_in_, b_in_, w_out_, b_out_]
        needed = [leaf for leaf in leaves if leaf is not None]
        grads = torch.autograd.grad(down, needed, grad_outputs=grad_output, allow_unused=True)
        grad_iter = iter(grads)
        results = [next(grad_iter) if leaf is not None else None for leaf in leaves]
        grad_x, grad_w_gate, grad_b_gate, grad_w_in, grad_b_in, grad_w_out, grad_b_out = results

        return (
            grad_x,
            grad_w_gate,
            grad_b_gate,
            grad_w_in,
            grad_b_in,
            grad_w_out,
            grad_b_out,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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

    @property
    def _relevance_rule_kinds(self) -> Tuple[str, ...]:
        """``("activation", "multiplicative_gate")`` only when the gate/up projections
        are backed by an allowlisted HF module class (``nn.Linear`` or ``Conv1D``) in a
        recognized orientation; empty otherwise, so an opaque or unrecognized backing
        module is reported skipped rather than silently installing a rule the
        weights-recompute cannot honor correctly.
        """
        if self.original_component is None:
            return ()
        gate_module = getattr(self, "gate", None)
        in_module = getattr(self, "in", None)
        out_module = getattr(self, "out", None)
        if gate_module is None or in_module is None or out_module is None:
            return ()
        if weight_layout_in_out(gate_module) is None or weight_layout_in_out(in_module) is None:
            return ()
        if weight_layout_in_out(out_module) is None:
            return ()
        return ("activation", "multiplicative_gate")

    def _enable_relevance_rule(self, kind: str) -> None:
        """Activate the named rule for this instance's recompute path only."""
        if kind == "activation":
            self._relevance_rule_activation_active = True
        elif kind == "multiplicative_gate":
            self._relevance_rule_gate_active = True

    def _disable_relevance_rule(self, kind: str) -> None:
        """Deactivate the named rule, restoring today's native-forward behavior."""
        if kind == "activation":
            self._relevance_rule_activation_active = False
        elif kind == "multiplicative_gate":
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
            activated = act_fn(gate_output)
            hidden = activated * linear_output
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
        if self._relevance_rule_activation_active or self._relevance_rule_gate_active:
            act_fn = resolve_activation_fn(self.config)
            output = _GatedMLPRecomputeRule.apply(
                hidden_states,
                self.W_gate,
                getattr(self.gate, "bias", None),
                self.W_in,
                getattr(getattr(self, "in"), "bias", None),
                self.W_out,
                getattr(self.out, "bias", None),
                act_fn,
                self._relevance_rule_activation_active,
                self._relevance_rule_gate_active,
                self.original_component,
                new_args[1:],
                kwargs,
            )
        else:
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
