"""Block bridge component.

This module contains the bridge component for transformer blocks.
"""
from __future__ import annotations

import inspect
import re
import weakref
from typing import Any, Callable, Dict, Optional, cast

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.exceptions import StopAtLayerException
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

# Layer-type variant submodule names. Tuple for deterministic iteration order.
# Extend here when adding new hybrid variant types.
VARIANT_SUBMODULE_NAMES: tuple[str, ...] = ("attn", "linear_attn", "mamba", "mixer", "ssm")
_VARIANT_SUBMODULE_SET: frozenset[str] = frozenset(VARIANT_SUBMODULE_NAMES)

# Infrastructure modules excluded from submodule introspection.
_BLOCK_INTERNAL_MODULES: frozenset[str] = frozenset({"hook_in", "hook_out", "_original_component"})

# Norm-module prefixes excluded from layer_types() labels.
_NORM_PREFIXES: tuple[str, ...] = ("ln", "layer_norm", "norm", "rms")


class BlockBridge(GeneralizedComponent):
    """Bridge component for transformer blocks.

    This component provides standardized input/output hooks and monkey-patches
    HuggingFace blocks to insert hooks at positions matching HookedTransformer.
    """

    is_list_item: bool = True
    hook_out_is_single_residual_stream: bool = True
    # hook_mlp_in is a direct HookPoint on this class (not aliased) so it can
    # fire on the MLP-branch entry (pre-ln2, or the MLP input on post-norm
    # blocks); see __init__. The normalized mlp input stays at block.mlp.hook_in.
    hook_aliases = {
        "hook_resid_pre": "hook_in",
        "hook_resid_mid": "ln2.hook_in",
        "hook_resid_post": "hook_out",
        "hook_attn_in": "attn.hook_attn_in",
        "hook_attn_out": "attn.hook_out",
        "hook_q_input": "attn.hook_q_input",
        "hook_k_input": "attn.hook_k_input",
        "hook_v_input": "attn.hook_v_input",
        "hook_mlp_out": "mlp.hook_out",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        hook_alias_overrides: Optional[Dict[str, str]] = None,
        mlp_reads_resid_directly: bool = False,
    ):
        """Initialize the block bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration (unused for BlockBridge)
            submodules: Dictionary of submodules to register
            hook_alias_overrides: Optional dictionary to override default hook aliases.
                For example, {"hook_attn_out": "ln1_post.hook_out"} will make hook_attn_out
                point to ln1_post.hook_out instead of the default attn.hook_out.
            mlp_reads_resid_directly: True for post-norm blocks where the MLP consumes
                the mid-residual with no pre-MLP norm (OLMo 2 layout). Moves the
                hook_mlp_in capture from ln2 (whose input there is the raw MLP output)
                to the MLP itself.
        """
        # ln1_post/ln2_post redirect attn_out/mlp_out to match HookedTransformer's
        # placement (hook fires after the post-norm, not before).
        auto_overrides = {}
        if submodules is not None:
            if "ln1_post" in submodules:
                auto_overrides["hook_attn_out"] = "ln1_post.hook_out"
            if "ln2_post" in submodules:
                auto_overrides["hook_mlp_out"] = "ln2_post.hook_out"
        merged_overrides = {**auto_overrides, **(hook_alias_overrides or {})}

        # Guard against the bug where a sequential block (attn + mlp) with no ln2
        # silently points hook_resid_mid at the wrong tensor. Use
        # ParallelBlockBridge for parallel-residual architectures.
        # Skip the check on generic-container / attn-only uses (no mlp).
        has_attn_like = submodules is not None and any(
            k in submodules for k in _VARIANT_SUBMODULE_SET
        )
        has_mlp = submodules is not None and "mlp" in submodules
        has_ln2 = submodules is not None and "ln2" in submodules
        if has_attn_like and has_mlp and not has_ln2 and type(self) is BlockBridge:
            raise ValueError(
                f"BlockBridge at '{name}': 'ln2' submodule not declared. "
                f"Either declare ln2, or use ParallelBlockBridge for a "
                f"parallel-residual architecture."
            )

        super().__init__(
            name,
            config,
            submodules=submodules if submodules is not None else {},
            hook_alias_overrides=merged_overrides if merged_overrides else None,
        )

        self._original_block_forward: Optional[Callable[..., Any]] = None
        self._capture_hooks_wired: bool = False
        self._capture_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        # Fallback for _read_use_hook_mlp_in when block.config is None.
        self._use_hook_mlp_in: bool = False
        self.mlp_reads_resid_directly = mlp_reads_resid_directly
        # Fires on the MLP-branch entry (pre-ln2, or the MLP input on post-norm
        # blocks) when use_hook_mlp_in is set. See #1317.
        self.hook_mlp_in = HookPoint()

    def _wire_ln1_module(self) -> None:
        """Keep the raw ln1 execution reference outside the ownership tree."""
        from transformer_lens.model_bridge.generalized_components.attention import (
            AttentionBridge,
        )

        ln1 = self.submodules.get("ln1") if self.submodules else None
        attn = self.submodules.get("attn") if self.submodules else None
        if not isinstance(attn, AttentionBridge):
            return

        ln1_module = None
        if (
            ln1 is not None
            and getattr(attn, "supports_split_qkv_fork", False)
            and getattr(ln1, "original_component", None) is not None
        ):
            ln1_module = ln1.original_component

        attn._modules.pop("_ln1_module", None)
        object.__setattr__(attn, "_ln1_module", ln1_module)

    def _maybe_wire_capture_hooks(self) -> None:
        """Install the block's capture hooks (split-qkv fork, hook_mlp_in).

        Registered on the bridge submodule, not ``original_component`` — the
        manual bridge forward never calls the raw module. Idempotent.
        """
        self._wire_ln1_module()
        if self._capture_hooks_wired:
            return
        from transformer_lens.model_bridge.generalized_components.attention import (
            AttentionBridge,
        )

        ln1 = self.submodules.get("ln1") if self.submodules else None
        attn = self.submodules.get("attn") if self.submodules else None
        if (
            ln1 is not None
            and isinstance(attn, AttentionBridge)
            and getattr(attn, "supports_split_qkv_fork", False)
            and getattr(ln1, "original_component", None) is not None
        ):
            attn_ref = cast(AttentionBridge, weakref.proxy(attn))

            def _capture_pre_ln1(_module: torch.nn.Module, args: tuple) -> None:
                if args and isinstance(args[0], torch.Tensor):
                    attn_ref._captured_pre_ln_residual = args[0]

            handle = ln1.register_forward_pre_hook(_capture_pre_ln1)
            self._capture_hook_handles.append(handle)

        # hook_mlp_in must capture the MLP-branch entry point: ln2's input on
        # pre-norm blocks, the MLP's own input on post-norm blocks (where ln2
        # follows the MLP and its input is the raw MLP output).
        if self.mlp_reads_resid_directly:
            capture_target = self.submodules.get("mlp") if self.submodules else None
        else:
            capture_target = self.submodules.get("ln2") if self.submodules else None
        if (
            capture_target is not None
            and getattr(capture_target, "original_component", None) is not None
        ):
            hook_mlp_in = self.hook_mlp_in
            block_ref = weakref.proxy(self)

            def _capture_mlp_in(_module: torch.nn.Module, args: tuple) -> Any:
                if not block_ref._read_use_hook_mlp_in():
                    return None
                if args and isinstance(args[0], torch.Tensor):
                    hooked = hook_mlp_in(args[0])
                    return (hooked,) + args[1:]
                return None

            handle = capture_target.register_forward_pre_hook(_capture_mlp_in)
            self._capture_hook_handles.append(handle)

        self._capture_hooks_wired = True

    def _teardown_capture_hooks(self) -> None:
        """Remove the capture hooks installed by _maybe_wire_capture_hooks (and subclass extensions)."""
        for handle in self._capture_hook_handles:
            handle.remove()
        self._capture_hook_handles.clear()
        self._capture_hooks_wired = False

    def _read_use_hook_mlp_in(self) -> bool:
        """Prefer ``block.config.use_hook_mlp_in``; fall back to the block-local flag."""
        cfg = self.config
        if cfg is not None and hasattr(cfg, "use_hook_mlp_in"):
            return bool(cfg.use_hook_mlp_in)
        return self._use_hook_mlp_in

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the block bridge.

        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments

        Returns:
            The output from the original component

        Raises:
            StopAtLayerException: If stop_at_layer is set and this block should stop execution
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        self._maybe_wire_capture_hooks()
        args, kwargs = self._maybe_inject_start_residual(args, kwargs)
        self._check_stop_at_layer(*args, **kwargs)
        args, kwargs = self._hook_input_hidden_states(args, kwargs)

        # Filter kwargs to only include parameters accepted by the original component
        # This prevents errors when passing encoder-specific params to decoder-only models
        filtered_kwargs = self._filter_kwargs_for_forward(kwargs, len(args))

        output = self.original_component(*args, **filtered_kwargs)
        force_tuple_for_bare_tensor = self._is_standalone_hidden_state_call(args, filtered_kwargs)
        return self._apply_output_hook(
            output, force_tuple_for_bare_tensor=force_tuple_for_bare_tensor
        )

    def _apply_output_hook(
        self,
        output: Any,
        wrap_single_element: bool = True,
        force_tuple_for_bare_tensor: bool = False,
    ) -> Any:
        """Hook the primary tensor in the output and return the result.

        Args:
            output: Raw output from the original component (tensor or tuple).
            wrap_single_element: If True, single-element tuples stay as tuples after
                hooking (default, required by most HF models). If False, single-element
                tuples are unwrapped to a bare tensor (Bloom convention).
            force_tuple_for_bare_tensor: If True, bare tensor outputs are wrapped into
                a one-element tuple after hooking. This keeps standalone BlockBridge
                calls compatible with HF block APIs that expose tuple-like block outputs,
                while preserving tensor outputs during newer HF parent-model execution.
        """
        if isinstance(output, tuple) and len(output) > 0:
            first = output[0]
            if isinstance(first, torch.Tensor):
                first = self.hook_out(first)
                if len(output) == 1:
                    return (first,) if wrap_single_element else first
                output = (first,) + output[1:]
            return output
        if isinstance(output, torch.Tensor):
            output = self.hook_out(output)
            if force_tuple_for_bare_tensor and wrap_single_element:
                return (output,)
            return output
        return output

    @staticmethod
    def _is_standalone_hidden_state_call(args: tuple, kwargs: dict) -> bool:
        """Return True for direct block(hidden_states) style calls.

        Transformers versions differ on whether parent model loops expect block
        outputs as tuples or tensors. We preserve the original tensor return during
        full-model execution, but expose tuple-like output for standalone component
        calls so `output[0]` does not accidentally drop the batch dimension.
        """
        if len(args) == 1 and isinstance(args[0], torch.Tensor) and not kwargs:
            return True
        return (
            len(args) == 0
            and set(kwargs.keys()) == {"hidden_states"}
            and isinstance(kwargs["hidden_states"], torch.Tensor)
        )

    def _extract_layer_idx(self) -> Optional[int]:
        """Parse this block's layer index from its name (TL/GPT-2/LLaMA patterns)."""
        if self.name is None:
            return None
        match = re.search(r"(?:^|\.)(?:blocks|h|layers)\.(\d+)", self.name)
        return int(match.group(1)) if match else None

    def _check_stop_at_layer(self, *args: Any, **kwargs: Any) -> None:
        """Check if execution should stop before this block. Raises StopAtLayerException.

        The _stop_at_layer_idx attribute is set by the bridge's forward method.
        Supports TL/GPT-2/LLaMA naming patterns for layer index extraction.
        """
        if getattr(self, "_stop_at_layer_idx", None) is None:
            return
        layer_idx = self._extract_layer_idx()
        if layer_idx is not None and layer_idx == self._stop_at_layer_idx:
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                input_tensor = args[0]
            elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                input_tensor = kwargs["hidden_states"]
            else:
                raise ValueError(f"Cannot find input tensor to stop at layer {layer_idx}")
            input_tensor = self.hook_in(input_tensor)
            raise StopAtLayerException(input_tensor)

    def _maybe_inject_start_residual(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """If this is the start_at_layer block, swap in the caller's residual.

        Mirror of ``_check_stop_at_layer``: the bridge's forward stashes the
        residual-stream input on the block via ``_start_residual`` and sets
        ``_start_at_layer_idx``. This block replaces its incoming hidden states
        with that residual; ``_hook_input_hidden_states`` then fires ``hook_in``
        on it, so ``hook_resid_pre`` reflects the injected value.
        """
        if getattr(self, "_start_at_layer_idx", None) is None:
            return args, kwargs
        if self._extract_layer_idx() != self._start_at_layer_idx:
            return args, kwargs
        residual = getattr(self, "_start_residual", None)
        if residual is None:
            return args, kwargs
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            args = (residual,) + args[1:]
        elif "hidden_states" in kwargs:
            kwargs = {**kwargs, "hidden_states": residual}
        return args, kwargs

    def _hook_input_hidden_states(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Apply hook_in to the hidden_states input, whether in args or kwargs."""
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            args = (hooked_input,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])
        return args, kwargs

    def _filter_kwargs_for_forward(
        self, kwargs: Dict[str, Any], num_positional_args: int = 0
    ) -> Dict[str, Any]:
        """Filter kwargs to only include parameters accepted by original_component.forward().

        This prevents TypeErrors when the bridge passes parameters (like encoder_attention_mask)
        that aren't accepted by decoder-only models. It also removes any kwargs that would
        conflict with positional arguments already being passed.

        Args:
            kwargs: The full set of keyword arguments
            num_positional_args: Number of positional arguments being passed (to avoid conflicts)

        Returns:
            Filtered kwargs containing only accepted parameters
        """
        if self.original_component is None:
            return kwargs

        try:
            # Get the signature of the original component's forward method
            sig = inspect.signature(self.original_component.forward)
            param_list = list(sig.parameters.keys())
            valid_params = set(param_list)

            # Check if the signature accepts **kwargs (VAR_KEYWORD)
            accepts_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )

            # If it accepts **kwargs, pass everything through
            if accepts_var_keyword:
                return kwargs

            # Skip params already provided positionally
            positional_param_names = set(param_list[:num_positional_args])

            # Filter kwargs: include only if in signature AND not already provided positionally
            filtered = {
                k: v
                for k, v in kwargs.items()
                if k in valid_params and k not in positional_param_names
            }
            return filtered

        except (ValueError, TypeError):
            # If we can't inspect the signature, pass through all kwargs
            # (better to potentially fail than to silently drop important params)
            return kwargs


class MLABlockBridge(BlockBridge):
    """Block wrapping Multi-Head Latent Attention (DeepSeek V2/V3/R1).

    MLA has no standalone q/k/v projections — Q flows through compressed
    q_a_proj→q_a_layernorm→q_b_proj, and K/V share a joint kv_a_proj_with_mqa
    entry point. There is no single HookPoint that represents "input that
    becomes Q/K/V", so the block-level ``hook_q_input``/``hook_k_input``/
    ``hook_v_input``/``hook_attn_in`` aliases do not apply. Type-level
    distinction means a reader of the adapter sees ``MLABlockBridge`` and
    knows those hooks are absent.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        hook_alias_overrides: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            name,
            config=config,
            submodules=submodules,
            hook_alias_overrides=hook_alias_overrides,
        )
        if self.hook_aliases is BlockBridge.hook_aliases:
            self.hook_aliases = dict(self.hook_aliases)
        for alias in ("hook_q_input", "hook_k_input", "hook_v_input", "hook_attn_in"):
            self.hook_aliases.pop(alias, None)


class ParallelBlockBridge(BlockBridge):
    """Block where attn and MLP both read the pre-attention residual.

    For GPT-J, NeoX, Pythia, Phi, Cohere, CodeGen, and some Falcon variants,
    output = resid_pre + attn_out + mlp_out — no distinct post-attention
    residual exists. Matches legacy HookedTransformer which omits hook_resid_mid
    when ``cfg.parallel_attn_mlp=True``. Type-level distinction means a reader
    of the adapter sees ``ParallelBlockBridge`` and knows the hook is absent.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        hook_alias_overrides: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            name,
            config=config,
            submodules=submodules,
            hook_alias_overrides=hook_alias_overrides,
        )
        # Ensure instance-level copy before mutating; base may have left the
        # class-level dict shared when no overrides were passed.
        if self.hook_aliases is BlockBridge.hook_aliases:
            self.hook_aliases = dict(self.hook_aliases)
        self.hook_aliases.pop("hook_resid_mid", None)


class ScaledResidualBlockBridge(BlockBridge):
    """Block whose sublayer outputs are scaled before the residual add.

    Granite-family HF blocks compute ``residual + sublayer_out * residual_multiplier``
    inline, so no submodule output equals the tensor added to the residual stream
    and the legacy aliases cannot be fixed by re-pointing. ``hook_attn_out`` /
    ``hook_mlp_out`` become real HookPoints on the block firing on the scaled
    contribution:

    - no hooks attached: the forward is untouched (bit-exact);
    - read-only hooks: they observe ``module_out * scale``, forward stays bit-exact;
    - a hook that changes the tensor (returned new or mutated in place): the module
      output is rewritten to ``hooked / scale`` so HF's multiply reconstructs the
      written value as the contribution (~1-ulp rounding; exact for zero-ablation);
    - any backward hooks: the rewrite always happens so the autograd graph routes
      through the HookPoint.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        hook_alias_overrides: Optional[Dict[str, str]] = None,
        mlp_reads_resid_directly: bool = False,
        residual_contribution_scale: float = 1.0,
        scaled_attn_submodule: str = "attn",
        scaled_mlp_submodule: Optional[str] = "mlp",
    ):
        """scaled_mlp_submodule=None when no single submodule feeds the MLP-side
        add (GraniteMoeHybrid sums moe + shared_mlp inline) — hook_mlp_out then
        stays absent rather than firing with a partial tensor."""
        super().__init__(
            name,
            config=config,
            submodules=submodules,
            hook_alias_overrides=hook_alias_overrides,
            mlp_reads_resid_directly=mlp_reads_resid_directly,
        )
        scale = float(residual_contribution_scale)
        if scale == 0.0:
            raise ValueError(
                f"ScaledResidualBlockBridge at '{name}': residual_contribution_scale "
                f"must be nonzero (the write path divides by it)."
            )
        self.residual_contribution_scale = scale
        self.scaled_attn_submodule = scaled_attn_submodule
        self.scaled_mlp_submodule = scaled_mlp_submodule
        if self.hook_aliases is BlockBridge.hook_aliases:
            self.hook_aliases = dict(self.hook_aliases)
        for alias in ("hook_attn_out", "hook_mlp_out"):
            self.hook_aliases.pop(alias, None)
        self.hook_attn_out = HookPoint()
        if scaled_mlp_submodule is not None:
            self.hook_mlp_out = HookPoint()

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Prune contribution HookPoints the bound layer cannot fire.

        Heterogeneous blocks (GraniteMoeHybrid mamba layers) lack the attn
        submodule; a HookPoint that exists but never fires is a silent-no-op
        intervention trap, so the name must be absent instead.
        """
        super().set_original_component(original_component)
        targets = [(self.scaled_attn_submodule, "hook_attn_out")]
        if self.scaled_mlp_submodule is not None:
            targets.append((self.scaled_mlp_submodule, "hook_mlp_out"))
        for sub_name, hook_name in targets:
            sub = self.submodules.get(sub_name) if self.submodules else None
            remote = getattr(sub, "name", None)
            first = remote.split(".", 1)[0] if isinstance(remote, str) else None
            missing = sub is None or (
                first is not None and getattr(original_component, first, None) is None
            )
            if missing:
                if hasattr(self, hook_name):
                    delattr(self, hook_name)
                # __setattr__ auto-registered the HookPoint; get_hooks() serves
                # from this registry, so the module deletion alone is not enough.
                self._hook_registry.pop(hook_name, None)

    def _maybe_wire_capture_hooks(self) -> None:
        """Extend the base wiring with the scaled-contribution forward hooks.

        Shares the base flag and handle list so _teardown_capture_hooks also
        removes these hooks and re-wiring stays idempotent.
        """
        if self._capture_hooks_wired:
            return
        super()._maybe_wire_capture_hooks()
        targets = [(self.scaled_attn_submodule, "hook_attn_out")]
        if self.scaled_mlp_submodule is not None:
            targets.append((self.scaled_mlp_submodule, "hook_mlp_out"))
        for sub_name, hook_name in targets:
            hook_point = getattr(self, hook_name, None)
            sub = self.submodules.get(sub_name) if self.submodules else None
            if (
                hook_point is None
                or sub is None
                or getattr(sub, "original_component", None) is None
            ):
                continue
            handle = sub.register_forward_hook(self._make_scaled_contribution_hook(hook_point))
            self._capture_hook_handles.append(handle)

    def _make_scaled_contribution_hook(
        self, hook_point: HookPoint
    ) -> Callable[[torch.nn.Module, tuple, Any], Any]:
        """Build a forward hook exposing ``output * scale`` through hook_point."""
        scale = self.residual_contribution_scale

        def _hook(_module: torch.nn.Module, _args: tuple, output: Any) -> Any:
            if not hook_point.has_hooks(dir="both"):
                return None
            is_tuple = isinstance(output, tuple)
            out = output[0] if is_tuple else output
            if not isinstance(out, torch.Tensor):
                return None
            scaled = out * scale
            hooked = hook_point(scaled)
            # Compare against a freshly computed reference: an in-place mutation
            # alters `scaled` itself, so an identity check would miss it.
            if not hook_point.has_hooks(dir="bwd") and torch.equal(hooked, out * scale):
                return None
            new = hooked / scale
            return ((new,) + output[1:]) if is_tuple else new

        return _hook


class DelegatedAttentionBlockBridge(BlockBridge):
    """Block whose attention is delegated wholesale to HF (no split-qkv fork).

    For architectures with heterogeneous per-layer attention structure — e.g.
    Gemma 4, where KV-shared layers have no ``k_proj``/``v_proj`` at all and
    K==V layers have no ``v_proj`` — there is no uniform HookPoint that
    represents "input that becomes Q/K/V", so the block-level ``hook_q_input``/
    ``hook_k_input``/``hook_v_input``/``hook_attn_in`` aliases do not apply.
    Type-level distinction means a reader of the adapter sees
    ``DelegatedAttentionBlockBridge`` and knows those hooks are absent.
    """

    # Tell the component benchmark this block's attention is delegated wholesale
    # to HF and cannot be tested in isolation (requires model-specific kwargs).
    maintain_native_attention: bool = True

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        hook_alias_overrides: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            name,
            config=config,
            submodules=submodules,
            hook_alias_overrides=hook_alias_overrides,
        )
        if self.hook_aliases is BlockBridge.hook_aliases:
            self.hook_aliases = dict(self.hook_aliases)
        for alias in ("hook_q_input", "hook_k_input", "hook_v_input", "hook_attn_in"):
            self.hook_aliases.pop(alias, None)
