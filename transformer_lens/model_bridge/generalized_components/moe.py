"""Mixture of Experts bridge component.

This module contains the bridge component for Mixture of Experts layers.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.model_bridge.generalized_components.mlp import (
    normalize_mlp_weight,
    weight_layout_in_out,
)


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

    # Deliberately NOT gate/in/out: ``gate`` is the ROUTER on sparse layers of
    # the same model, so reusing it would flip the meaning of
    # blocks.N.mlp.gate.hook_out per layer (#1645's own confusion). dense_gate
    # is absent on ungated feed-forwards (Switch's wi/wo); the alias set
    # follows the bound shape.
    DENSE_SUBMODULE_KEYS = ("dense_in", "dense_out")
    DENSE_GATE_KEY = "dense_gate"
    _DENSE_HOOK_ALIASES = {
        "hook_pre": "dense_in.hook_out",
        "hook_post": "dense_out.hook_in",
    }
    _DENSE_GATED_HOOK_ALIASES = {
        "hook_pre": "dense_gate.hook_out",
        "hook_pre_linear": "dense_in.hook_out",
        "hook_post": "dense_out.hook_in",
    }
    _DENSE_PROPERTY_ALIASES = {
        "b_gate": "dense_gate.bias",
        "b_in": "dense_in.bias",
        "b_out": "dense_out.bias",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
        optional: bool = False,
        sparse_required: Tuple[str, ...] = (),
    ):
        """Initialize the MoE bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration (unused for MoEBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
            optional: If True, setup skips this subtree when absent (dense layers)
            sparse_required: Submodule keys that must be declared ``optional``
                (dense layers of an interleaved stack do not have them) but whose
                absence on a SPARSE layer is an error rather than a silent skip.
                Routers belong here: HF creates them unconditionally on sparse
                blocks, so a skip means the attribute was renamed or moved, and
                plain ``optional`` would drop their hooks without a word.
        """
        super().__init__(name, config, submodules=submodules, optional=optional)
        self.hook_router_scores = HookPoint()
        self._bound_dense = False
        self._bound_dense_gate = False
        # A misspelled key would silently disable the very guard that exists to
        # stop silent degradation, so the opt-in is validated on construction.
        unknown = set(sparse_required) - set(submodules or {})
        if unknown:
            raise ValueError(
                f"{name}: sparse_required {sorted(unknown)} are not declared "
                f"submodules (declared: {sorted(submodules or {})})"
            )
        self._sparse_required = sparse_required

    def _binds_dense_projections(self, component: torch.nn.Module) -> bool:
        """Whether this layer is the dense variant of an interleaved MoE stack.

        Positive detection only: the adapter must have declared the dense
        projections AND the wrapped module must actually expose them. Guessing
        from the absence of ``experts`` would risk silently stripping hooks off
        a sparse block whose experts are named differently.
        """
        declared = [self.submodules.get(key) for key in self.DENSE_SUBMODULE_KEYS]
        if not all(declared):
            return False
        return all(
            sub is not None and sub.name is not None and hasattr(component, sub.name)
            for sub in declared
        )

    def _binds_dense_gate(self, component: torch.nn.Module) -> bool:
        """Whether the bound dense MLP is gated (SiLU-gated) rather than plain.

        A declared-but-unresolvable gate means the HF attribute was renamed,
        not that the MLP is ungated — binding it as ungated would silently
        alias hook_pre to the UP projection, so raise instead.
        """
        gate = self.submodules.get(self.DENSE_GATE_KEY)
        if gate is None or gate.name is None:
            return False  # never declared: a genuinely ungated dense MLP
        if hasattr(component, gate.name):
            return True
        raise ValueError(
            f"{self.name}: dense layer wrapped {type(component).__name__} which has "
            f"no {gate.name!r}, but this adapter declares {self.DENSE_GATE_KEY!r} — "
            "so its dense MLPs are gated and the attribute was renamed or moved. "
            "Binding it as ungated would alias hook_pre to the up projection. "
            "Update the adapter's submodule name, or drop the declaration if this "
            "architecture's dense layers really are ungated."
        )

    def set_original_component(self, component: torch.nn.Module) -> None:
        """Bind the layer, adopting gated-MLP semantics on dense layers.

        Dense layers of interleaved MoE stacks get neuron-basis hooks and
        weight accessors instead of MoE boundary tensors under those names.
        """
        super().set_original_component(component)
        is_dense = self._binds_dense_projections(component)
        if is_dense == self._bound_dense:
            # First sparse bind, or an idempotent rebind: nothing to morph.
            if not is_dense:
                return
        self._bound_dense = is_dense
        if is_dense:
            self._bound_dense_gate = self._binds_dense_gate(component)
            self.hook_aliases = dict(
                self._DENSE_GATED_HOOK_ALIASES
                if self._bound_dense_gate
                else self._DENSE_HOOK_ALIASES
            )
            dense_props = dict(self._DENSE_PROPERTY_ALIASES)
            if not self._bound_dense_gate:
                dense_props.pop("b_gate", None)
            self.property_aliases = {**self.property_aliases, **dense_props}
            # A dense layer has no router; leaving the hook in hook_dict would
            # advertise an intervention point that can never fire.
            if hasattr(self, "hook_router_scores"):
                self._hook_registry.pop("hook_router_scores", None)
                del self.hook_router_scores
        else:
            # Symmetric restore so a rebinding harness cannot leave a chimera.
            self.hook_aliases = dict(type(self).hook_aliases)
            self.property_aliases = {
                key: value
                for key, value in self.property_aliases.items()
                if key not in self._DENSE_PROPERTY_ALIASES
            }
            self._bound_dense_gate = False
            if not hasattr(self, "hook_router_scores"):
                self.hook_router_scores = HookPoint()

    def validate_after_setup(self, skipped_optional: list[str]) -> None:
        """Fail loudly when a sparse layer is missing a submodule only dense
        layers may lack (see ``sparse_required``).

        Called by setup_submodules once every submodule has resolved — the
        skipped set is not knowable at bind time.
        """
        if self._bound_dense or not self._sparse_required:
            return
        missing = [key for key in self._sparse_required if key in skipped_optional]
        if missing:
            component = type(self.original_component).__name__
            raise ValueError(
                f"{self.name}: sparse MoE layer wrapped {component} but required "
                f"submodule(s) {missing} were absent. These are optional only so "
                "dense layers of an interleaved stack can skip them; on a sparse "
                "layer their absence means the HF attribute was renamed or moved. "
                "Update the adapter's submodule name(s) rather than losing the hooks."
            )

    @property
    def bound_dense(self) -> bool:
        """Whether this layer bound the dense variant of an interleaved MoE stack.

        Public so weight-collection helpers can find the projections under
        ``DENSE_SUBMODULE_KEYS`` instead of the sparse ``gate`` (the router).
        """
        return self._bound_dense

    def _dense_projection(self, key: str) -> Any:
        """Return a bound dense projection.

        Raises AttributeError on sparse layers (per-expert weights, no single
        W_*) so ``hasattr`` stays False and weight collection skips them.
        """
        if not self._bound_dense:
            raise AttributeError(f"{self.name}: {key} exists only on dense layers")
        module = getattr(self, key, None)
        if module is None:
            raise AttributeError(f"{self.name}: dense projection {key!r} is not bound")
        return module

    @property
    def W_gate(self) -> torch.Tensor:
        """Gated dense layer's gate weight in TL orientation [d_model, d_mlp]."""
        if not self._bound_dense_gate:
            raise AttributeError(f"{self.name}: this dense layer is ungated (no W_gate)")
        module = self._dense_projection("dense_gate")
        return normalize_mlp_weight(
            module.weight, weight_layout_in_out(module), module, pattern="in"
        )

    @property
    def W_in(self) -> torch.Tensor:
        """Dense-layer input weight in TL orientation [d_model, d_mlp]."""
        module = self._dense_projection("dense_in")
        return normalize_mlp_weight(
            module.weight, weight_layout_in_out(module), module, pattern="in"
        )

    @property
    def W_out(self) -> torch.Tensor:
        """Dense-layer output weight in TL orientation [d_mlp, d_model]."""
        module = self._dense_projection("dense_out")
        return normalize_mlp_weight(
            module.weight, weight_layout_in_out(module), module, pattern="out"
        )

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
            if not output:
                raise TypeError(
                    f"{self.name}: expected a non-empty tuple whose first "
                    "element is a torch.Tensor from the wrapped MoE component, "
                    "got an empty tuple."
                )

            hidden_states = output[0]
            if not isinstance(hidden_states, torch.Tensor):
                raise TypeError(
                    f"{self.name}: expected the first tuple element from the "
                    f"wrapped MoE component to be a torch.Tensor, got "
                    f"{type(hidden_states).__name__}."
                )
            if len(output) > 1:
                router_scores = output[1]
                # Some MoEs pack extras with the logits (LLaDA2 returns
                # (router_logits, topk_idx)); hook the first tensor.
                if isinstance(router_scores, tuple):
                    router_scores = next(
                        (t for t in router_scores if isinstance(t, torch.Tensor)), None
                    )
                # The hook is removed on dense binds (no router exists there), so
                # a dense layer whose wrapped module still returns a tuple must
                # pass the extras through untouched rather than raise.
                if isinstance(router_scores, torch.Tensor) and hasattr(self, "hook_router_scores"):
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
        """Copy router weights onto nested params by dotted path (JetMoe nests its
        Linear at ``router.layer.weight``); router weights are never processed."""
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
