"""Framework-agnostic bridge surface shared by TransformerBridge and RemoteBridge."""
from __future__ import annotations

import re
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.driver_protocol import to_torch
from transformer_lens.model_bridge.exceptions import StopAtLayerException
from transformer_lens.utilities.aliases import resolve_alias
from transformer_lens.utilities.lm_utils import lm_cross_entropy_loss
from transformer_lens.utilities.slice import Slice, SliceInput

_BLOCK_PATTERN = re.compile("blocks\\.(\\d+)")


def build_alias_to_canonical_map(hook_dict: Any, prefix: str = "") -> dict:
    """Map alias hook names to their canonical names (where ``.name`` differs from the key)."""
    aliases: dict = {}
    for key, value in hook_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            aliases.update(build_alias_to_canonical_map(value, full_key))
        elif hasattr(value, "name"):
            if key != value.name:
                aliases[full_key] = value.name
    return aliases


class BridgeCore:
    """Framework-agnostic bridge surface: hooks, cache, run_with_*, driver wiring.

    Holds state shared by every bridge (adapter, cfg, tokenizer, driver, hook
    registries). Subclasses add framework-specific state — ``TransformerBridge``
    walks the wrapped ``nn.Module``; ``RemoteBridge`` builds components from
    adapter metadata.
    """

    hook_aliases: Dict[str, Union[str, List[str]]] = {
        # Prefer embed_ln.hook_out for post-LN models (Bloom, BERT)
        "hook_embed": ["embed_ln.hook_out", "embed.hook_out"],
        "hook_pos_embed": ["pos_embed.hook_out", "rotary_emb.hook_out"],
        "hook_unembed": "unembed.hook_out",
    }

    def __init__(
        self,
        adapter: ArchitectureAdapter,
        tokenizer: Any,
        driver: Any,
    ) -> None:
        """Subclasses call this AFTER ``nn.Module.__init__`` (if applicable),
        then do framework-specific setup."""
        self.adapter = adapter
        self.cfg = adapter.cfg
        self.tokenizer = tokenizer
        if self.cfg.d_vocab == -1 and self.tokenizer is not None:
            if hasattr(self.tokenizer, "get_vocab"):
                vocab = self.tokenizer.get_vocab()
                self.cfg.d_vocab = max(vocab.values()) + 1
            elif hasattr(self.tokenizer, "vocab"):
                self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
            else:
                self.cfg.d_vocab = getattr(self.tokenizer, "vocab_size", 50257)
        if self.cfg.d_vocab_out == -1:
            self.cfg.d_vocab_out = self.cfg.d_vocab
        self.compatibility_mode = False
        self._weights_processed = False
        self._hook_cache = None
        self._hook_registry: Dict[str, HookPoint] = {}
        self._hook_registry_initialized = False
        self._hook_alias_registry: Dict[str, Union[str, List[str]]] = {}
        self._property_alias_registry: Dict[str, str] = {}
        self._driver = driver
        if not hasattr(adapter, "component_mapping") or adapter.component_mapping is None:
            raise ValueError("Adapter must have a component_mapping attribute")

    # ---- hook registry ----

    def _initialize_hook_registry(self) -> None:
        """Initialize the hook registry by scanning existing components."""
        if self._hook_registry_initialized:
            return
        self._scan_existing_hooks(self, "")
        self._hook_registry_initialized = True

    def _scan_existing_hooks(self, module: Any, prefix: str = "") -> None:
        """Walk components for HookPoint instances. Framework-specific."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _scan_existing_hooks "
            "(walks the component tree to find HookPoint instances)."
        )

    def clear_hook_registry(self) -> None:
        """Clear the hook registry and force re-initialization."""
        self._hook_registry.clear()
        self._hook_registry_initialized = False

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        """All HookPoint objects, including aliases — TransformerLens-compatible."""
        hooks = self._hook_registry.copy()
        self._add_aliases_to_hooks(hooks)
        return hooks

    @property
    def mod_dict(self) -> Dict[str, Any]:
        """Module/hook name -> object, HookedRootModule-compatible.

        Union of the named-module tree and the aliased hook view, so both
        canonical (``blocks.0.mlp.hook_out``) and HT-style
        (``blocks.0.hook_mlp_out``) names resolve to the same HookPoint.
        """
        # BridgeCore is a mixin; concrete bridges (TransformerBridge/RemoteBridge)
        # are nn.Modules, so named_modules() is always present at runtime.
        mods: Dict[str, Any] = {
            name: module
            for name, module in cast(torch.nn.Module, self).named_modules()
            if name != ""
        }
        mods.update(self.hook_dict)
        return mods

    # ---- alias registry ----

    def _register_aliases(self) -> None:
        """Register bridge-level aliases (hook_embed, hook_pos_embed, etc.) by
        resolving each alias target path and installing the target HookPoint
        as a direct attribute under the alias name."""
        if self.hook_aliases:
            self._hook_alias_registry.update(self.hook_aliases)
            for alias_name, target_path in self.hook_aliases.items():
                try:
                    if isinstance(target_path, list):
                        for single_target in target_path:
                            try:
                                target_obj = self
                                for part in single_target.split("."):
                                    target_obj = getattr(target_obj, part)
                                object.__setattr__(self, alias_name, target_obj)
                                break
                            except AttributeError:
                                continue
                    else:
                        target_obj = self
                        for part in target_path.split("."):
                            target_obj = getattr(target_obj, part)
                        object.__setattr__(self, alias_name, target_obj)
                except AttributeError:
                    pass

    def _collect_component_aliases(self, component_mapping: Any, prefix: str = "") -> dict:
        """Recursively collect aliases from components."""
        aliases: dict = {}
        if isinstance(component_mapping, dict):
            for name, component in component_mapping.items():
                sub_prefix = f"{prefix}.{name}" if prefix else name
                aliases.update(self._collect_component_aliases(component, sub_prefix))
        else:
            if hasattr(component_mapping, "hook_aliases") and component_mapping.hook_aliases:
                for alias_name, target in component_mapping.hook_aliases.items():
                    full_alias = f"{prefix}.{alias_name}" if prefix else alias_name
                    full_target = f"{prefix}.{target}" if prefix else target
                    aliases[full_alias] = full_target
            if hasattr(component_mapping, "submodules") and component_mapping.submodules:
                for sub_name, sub_component in component_mapping.submodules.items():
                    sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
                    aliases.update(self._collect_component_aliases(sub_component, sub_prefix))
        return aliases

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_hook_aliases_cached(
        hook_names_tuple: Tuple[str, ...],
        component_aliases_tuple: Tuple[Tuple[str, str], ...],
    ) -> Tuple[Tuple[str, str], ...]:
        """Cached computation of hook aliases."""
        aliases: dict = {}
        component_aliases = dict(component_aliases_tuple)
        for hook_name in hook_names_tuple:
            for alias_pattern, target_pattern in component_aliases.items():
                if "blocks." in target_pattern and "blocks." in hook_name:
                    block_match = _BLOCK_PATTERN.search(hook_name)
                    if block_match:
                        block_num = block_match.group(1)
                        dynamic_alias_pattern = alias_pattern.replace(
                            "blocks.", f"blocks.{block_num}."
                        )
                        dynamic_target_pattern = target_pattern.replace(
                            "blocks.", f"blocks.{block_num}."
                        )
                        if hook_name.endswith(dynamic_target_pattern):
                            target_len = len(dynamic_target_pattern)
                            alias_name = hook_name[:-target_len] + dynamic_alias_pattern
                            aliases[alias_name] = hook_name
                elif hook_name.endswith(target_pattern):
                    target_len = len(target_pattern)
                    alias_name = hook_name[:-target_len] + alias_pattern
                    aliases[alias_name] = hook_name
        return tuple(aliases.items())

    def _collect_hook_aliases_from_registry(self) -> dict:
        """Collect aliases based on existing hooks in the registry."""
        if hasattr(self.adapter, "component_mapping"):
            component_aliases = self._collect_component_aliases(self.adapter.component_mapping)
            hook_names_tuple = tuple(sorted(self._hook_registry.keys()))
            component_aliases_tuple = tuple(sorted(component_aliases.items()))
            aliases_tuple = self._compute_hook_aliases_cached(
                hook_names_tuple, component_aliases_tuple
            )
            return dict(aliases_tuple)
        return {}

    def _add_aliases_to_hooks(self, hooks: Dict[str, HookPoint]) -> None:
        """Add aliases to hooks in place. Registry-first so RemoteBridge works."""
        component_aliases = self._collect_hook_aliases_from_registry()
        all_aliases = {**self.hook_aliases, **component_aliases}
        if not all_aliases:
            return
        for alias_name, target in all_aliases.items():
            targets = target if isinstance(target, list) else [target]
            for t in targets:
                hp = self._hook_registry.get(t)
                if hp is not None:
                    hooks[alias_name] = hp
                    break
                # Fall back to attribute walk for TransformerBridge nested paths
                # not directly keyed in the registry.
                try:
                    target_hook = resolve_alias(self, alias_name, {alias_name: t})
                    if target_hook is not None:
                        hooks[alias_name] = target_hook
                        break
                except AttributeError:
                    continue

    # ---- captures from driver ----

    def _replay_captures(self, captured: Mapping[str, Any]) -> None:
        """Fire driver-delivered captures through the registry. Unknown names dropped silently."""
        for hook_name, activation in captured.items():
            hp = self._hook_registry.get(hook_name)
            if hp is None:
                continue
            hp(to_torch(activation))

    # ---- driver dispatch (subclasses concrete-override) ----

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Subclasses implement how the driver gets called."""
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def to_tokens(self, *args: Any, **kwargs: Any) -> Any:
        """Subclasses implement against their tokenizer surface."""
        raise NotImplementedError(f"{type(self).__name__} must implement to_tokens()")

    def close(self) -> None:
        """Release driver-managed resources. Idempotent — safe to call multiple times."""
        self._driver.close()

    def _input_device(self) -> Any:
        """Driver's expected device for inputs; None for remote / meta / dispatched drivers."""
        if not self._driver.supports("parameters"):
            return None
        params = getattr(self._driver, "parameters", None)
        if not callable(params):
            return None
        try:
            device = next(params()).device
        except (StopIteration, NotImplementedError, RuntimeError):
            return None
        # Meta device → inputs would silently become meta tensors with no data.
        if device.type == "meta":
            return None
        return device

    def loss_fn(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        per_token: bool = False,
    ) -> torch.Tensor:
        """Cross-entropy loss matching HookedTransformer's formula (log_softmax + gather)."""
        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)
        return lm_cross_entropy_loss(logits, tokens, attention_mask, per_token)

    def _finalize_return(
        self,
        return_type: Optional[str],
        logits: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        *,
        is_audio_model: bool = False,
        inputs_embeds_was_used: bool = False,
        loss_per_token: bool = False,
    ) -> Any:
        """Post-process driver output into the user's requested return_type."""
        if return_type == "logits":
            return logits
        if return_type is None:
            return None
        self._check_loss_supported(return_type)
        if return_type == "loss":
            if is_audio_model:
                raise ValueError(
                    "Audio models do not support return_type='loss'. "
                    "CTC loss requires aligned frame-level labels."
                )
            if inputs_embeds_was_used:
                raise ValueError(
                    "Cannot compute loss with inputs_embeds — token IDs required for labels."
                )
            assert isinstance(logits, torch.Tensor), f"Expected logits tensor, got {type(logits)}"
            assert input_ids is not None, "input_ids required for return_type='loss'"
            return self.loss_fn(logits, input_ids, per_token=loss_per_token)
        if return_type == "both":
            if is_audio_model:
                raise ValueError(
                    "Audio models do not support return_type='both'. "
                    "CTC loss requires aligned frame-level labels."
                )
            if inputs_embeds_was_used:
                raise ValueError(
                    "Cannot compute loss with inputs_embeds — token IDs required for labels."
                )
            assert isinstance(logits, torch.Tensor), f"Expected logits tensor, got {type(logits)}"
            assert input_ids is not None, "input_ids required for return_type='both'"
            loss = self.loss_fn(logits, input_ids, per_token=loss_per_token)
            return (logits, loss)
        if return_type == "predictions":
            assert self.tokenizer is not None, "Tokenizer required for return_type='predictions'"
            assert isinstance(logits, torch.Tensor), f"Expected logits tensor, got {type(logits)}"
            if logits.shape[-1] == 2:
                # Next Sentence Prediction — 2-class output
                logprobs = logits.log_softmax(dim=-1)
                predictions = [
                    "The sentences are sequential",
                    "The sentences are NOT sequential",
                ]
                return predictions[int(logprobs.argmax(dim=-1).item())]
            else:
                # Masked Language Modeling — decode [MASK] tokens
                assert input_ids is not None, "input_ids required for MLM predictions"
                logprobs = logits[input_ids == self.tokenizer.mask_token_id].log_softmax(dim=-1)
                preds = self.tokenizer.decode(logprobs.argmax(dim=-1))
                if " " in preds:
                    parts = preds.split(" ")
                    return [f"Prediction {i}: {p}" for i, p in enumerate(parts)]
                return preds
        raise ValueError(f"Invalid return_type: {return_type}")

    def _check_loss_supported(self, return_type: Optional[str]) -> None:
        """Loss needs full-sequence logits; final-position-only drivers would NaN."""
        if return_type in ("loss", "both") and not getattr(
            self._driver, "provides_sequence_logits", True
        ):
            raise NotImplementedError(
                f"return_type={return_type!r} is unsupported on this driver: it "
                "provides next-token logits for the final position only, so loss "
                "over earlier positions is undefined. Use return_type='logits' "
                "and read logits[..., -1, :]."
            )

    # ---- hook lookup / mutation ----

    @staticmethod
    def _is_embedding_stage_hook(name: str) -> bool:
        """Hooks belonging to the pre-block token/positional embedding stage.

        Excluded from ``start_at_layer`` output: the caller's residual already
        carries the embedding, so the embedding stage is logically skipped even
        though HF still computes (and discards) it. ``unembed``/``hook_unembed``
        are the output stage and deliberately not matched.
        """
        return name in ("hook_embed", "hook_pos_embed", "hook_tokens") or name.startswith(
            ("embed.", "pos_embed.")
        )

    def _check_hook_fireable(self, *names: str) -> None:
        """Fail loud when the driver declares it can't fire a requested hook —
        attaching anyway would yield a silently-unhooked forward / empty cache."""
        non_fireable: frozenset = getattr(self._driver, "non_fireable_hook_points", frozenset())
        for name in names:
            if name in non_fireable:
                raise NotImplementedError(
                    f"this backend cannot fire {name!r}; use boot_transformers() "
                    "for full hook coverage."
                )

    def _resolve_hook_point(
        self, name: str, aliases: Dict[str, str], hook_dict: Dict[str, HookPoint]
    ) -> Tuple[str, HookPoint]:
        """Resolve an (aliased) string hook name to its HookPoint, enforcing
        fireability. A name that resolves to nothing raises — a typo'd name must
        not run unhooked."""
        canonical = aliases.get(name, name)
        self._check_hook_fireable(name, canonical)
        hook_point = hook_dict.get(canonical)
        if hook_point is None:
            raise KeyError(f"Hook name {name!r} does not exist on this model.")
        return canonical, hook_point

    def get_hook_point(self, hook_name: str) -> Optional[HookPoint]:
        """Get a hook point by name from the bridge's hook system."""
        if hook_name in self._hook_registry:
            return self._hook_registry[hook_name]
        try:
            parts = hook_name.split(".")
            current: Any = self
            for part in parts:
                current = getattr(current, part)
            if isinstance(current, HookPoint):
                return current
        except AttributeError:
            pass
        return None

    def check_hooks_to_add(
        self,
        hook_point: HookPoint,
        hook_point_name: str,
        hook: Callable,
        dir: Literal["fwd", "bwd"] = "fwd",
        is_permanent: bool = False,
        prepend: bool = False,
    ) -> None:
        """Validate a hook before it is added; override to add checks.

        No-op by default (mirrors ``HookedRootModule.check_hooks_to_add``).
        Driver-fireability is enforced separately in ``_check_hook_fireable``.
        """

    def _add_fn_to_hook_point(
        self,
        hook_point: HookPoint,
        name: str,
        hook_fn: Callable,
        dir: Literal["fwd", "bwd"],
        is_permanent: bool,
    ) -> None:
        """Run the extension-point check, then attach the hook function."""
        self.check_hooks_to_add(hook_point, name, hook_fn, dir=dir, is_permanent=is_permanent)
        hook_point.add_hook(hook_fn, dir=dir, is_permanent=is_permanent)

    def add_hook(
        self,
        name: Union[str, Callable[[str], bool]],
        hook_fn: Any,
        dir: Literal["fwd", "bwd"] = "fwd",
        is_permanent: bool = False,
    ) -> None:
        """Add a hook to a specific component or to all components matching a filter.

        Args:
            name: Either a string hook point name (e.g. "blocks.0.attn.hook_q")
                or a callable filter ``(str) -> bool`` that is applied to every
                hook point name; the hook is added to each point where the filter
                returns True.
            hook_fn: The hook function ``(activation, hook) -> activation | None``.
            dir: Hook direction, ``"fwd"`` or ``"bwd"``.
            is_permanent: If True the hook survives ``reset_hooks()`` calls.
        """
        if callable(name) and not isinstance(name, str):
            hook_dict = self.hook_dict
            seen_hooks: set = set()
            for hook_name, hook_point in hook_dict.items():
                if name(hook_name):
                    hook_id = id(hook_point)
                    if hook_id in seen_hooks:
                        continue
                    seen_hooks.add(hook_id)
                    self._add_fn_to_hook_point(hook_point, hook_name, hook_fn, dir, is_permanent)
            return

        # Fast path: canonical registry names skip the alias-map build (hook_dict +
        # map construction cost ~ms on large models; add_hook is often called per layer).
        registry_hp = self._hook_registry.get(name)
        if registry_hp is not None:
            self._check_hook_fireable(name)
            self._add_fn_to_hook_point(registry_hp, name, hook_fn, dir, is_permanent)
            return
        # Same alias resolution run_with_hooks uses, so HT-style names work here too.
        canonical = build_alias_to_canonical_map(self.hook_dict).get(name, name)
        if canonical != name:
            self._check_hook_fireable(name, canonical)
        else:
            self._check_hook_fireable(name)
        registry_hp = self._hook_registry.get(canonical)
        if registry_hp is not None:
            self._add_fn_to_hook_point(registry_hp, canonical, hook_fn, dir, is_permanent)
            return

        component: Any = self
        parts = name.split(".")
        for part in parts[:-1]:
            if hasattr(component, part):
                component = getattr(component, part)
            else:
                raise AttributeError(f"Component path '{'.'.join(parts[:-1])}' not found")
        hook_name = parts[-1]
        if hasattr(component, hook_name):
            hook_point = getattr(component, hook_name)
            if isinstance(hook_point, HookPoint):
                self._add_fn_to_hook_point(hook_point, name, hook_fn, dir, is_permanent)
            else:
                raise AttributeError(
                    f"'{hook_name}' is not a hook point. Found object of type: {type(hook_point)} with value: {hook_point}"
                )
        else:
            raise AttributeError(f"Hook point '{hook_name}' not found on component")

    def add_perma_hook(
        self,
        name: Union[str, Callable[[str], bool]],
        hook_fn: Callable,
        dir: Literal["fwd", "bwd"] = "fwd",
    ) -> None:
        """Add a permanent hook that survives ``reset_hooks()`` calls.

        Convenience wrapper for ``add_hook(..., is_permanent=True)``. To remove,
        call ``reset_hooks(including_permanent=True)`` or remove from the
        underlying ``HookPoint`` directly.
        """
        self.add_hook(name, hook_fn, dir=dir, is_permanent=True)

    def hook_points(self) -> Iterable[HookPoint]:
        """All :class:`HookPoint` instances (registry is canonical and complete)."""
        return self._hook_registry.values()

    def clear_contexts(self) -> None:
        """Clear the stored ``ctx`` on every hook point."""
        for hp in self._hook_registry.values():
            hp.clear_context()

    def remove_all_hook_fns(
        self,
        direction: Literal["fwd", "bwd", "both"] = "both",
        including_permanent: bool = False,
        level: Optional[int] = None,
    ) -> None:
        """Remove hook functions from every hook point."""
        for hp in self._hook_registry.values():
            hp.remove_hooks(direction, including_permanent=including_permanent, level=level)

    def reset_hooks(
        self,
        clear_contexts: bool = True,
        direction: Literal["fwd", "bwd", "both"] = "both",
        including_permanent: bool = False,
        level: Optional[int] = None,
    ) -> None:
        """Remove hooks from every hook point; mirrors ``HookedRootModule.reset_hooks``.

        The hook registry is canonical and complete (every component's HookPoint
        is registered), so a single pass covers the whole model.

        Args:
            clear_contexts: Also clear each hook point's stored ``ctx``.
            direction: Which direction(s) to remove — ``"fwd"``, ``"bwd"``, or ``"both"``.
            including_permanent: If True, also remove hooks added via ``add_perma_hook``.
            level: If set, only remove hooks registered at this context level.
        """
        if clear_contexts:
            self.clear_contexts()
        self.remove_all_hook_fns(direction, including_permanent=including_permanent, level=level)

    def hooks(
        self,
        fwd_hooks: List = [],
        bwd_hooks: List = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ) -> Any:
        """Context manager for temporarily adding hooks.

        Example:
            with model.hooks(fwd_hooks=[("hook_embed", my_hook)]):
                output = model("Hello world")
        """

        @contextmanager
        def _hooks_context() -> Iterator["BridgeCore"]:
            added_hooks: List[Tuple[HookPoint, Literal["fwd", "bwd"]]] = []

            def add_hook_to_point(
                hook_point: HookPoint,
                hook_fn: Callable,
                name: str,
                dir: Literal["fwd", "bwd"] = "fwd",
            ) -> None:
                if self.compatibility_mode and name != hook_point.name:
                    alias_names_list: list = []
                    if hook_point.name is not None:
                        alias_names_list.append(hook_point.name)
                    alias_names_list.append(name)
                    hook_point.add_hook(hook_fn, dir=dir, alias_names=alias_names_list)
                else:
                    hook_point.add_hook(hook_fn, dir=dir)
                added_hooks.append((hook_point, dir))

            def apply_hooks(hook_list: List[Tuple[Any, Callable]], is_fwd: bool) -> None:
                direction: Literal["fwd", "bwd"] = "fwd" if is_fwd else "bwd"
                hook_dict = self.hook_dict
                aliases = build_alias_to_canonical_map(hook_dict)
                for hook_name_or_filter, hook_fn in hook_list:
                    if isinstance(hook_name_or_filter, str):
                        actual_hook_name, hook_point = self._resolve_hook_point(
                            hook_name_or_filter, aliases, hook_dict
                        )
                        add_hook_to_point(hook_point, hook_fn, actual_hook_name, direction)
                    else:
                        seen_hooks = set()
                        for n, hook_point in hook_dict.items():
                            if hook_name_or_filter(n):
                                hook_id = id(hook_point)
                                if hook_id in seen_hooks:
                                    continue
                                seen_hooks.add(hook_id)
                                hook_name_to_use = hook_point.name if hook_point.name else n
                                add_hook_to_point(hook_point, hook_fn, hook_name_to_use, direction)

            try:
                apply_hooks(fwd_hooks, True)
                apply_hooks(bwd_hooks, False)
                yield self
            finally:
                if reset_hooks_end:
                    for hook_point, direction in added_hooks:
                        hook_point.remove_hooks(dir=direction)

        return _hooks_context()

    # ---- high-level execution: run_with_hooks ----

    def run_with_hooks(
        self,
        input: Any,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        return_type: Optional[str] = "logits",
        stop_at_layer: Optional[int] = None,
        start_at_layer: Optional[int] = None,
        remove_batch_dim: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run the model with specified forward and backward hooks.

        ``stop_at_layer`` raises :class:`StopAtLayerException` to stop early
        (KV cache cleaned up on stop). ``start_at_layer`` treats ``input`` as the
        residual entering block ``k`` (see :meth:`forward`); hooks on blocks below
        ``k`` are skipped to match HookedTransformer. ``remove_batch_dim``
        squeezes/unsqueezes the batch dim around hook callbacks (batch_size==1 only).
        """
        if "names_filter" in kwargs:
            # **kwargs would silently absorb it; fail loud.
            raise TypeError(
                "run_with_hooks() got an unexpected keyword argument 'names_filter'; "
                "use run_with_cache(names_filter=...) to scope caching."
            )
        added_hooks: List[Tuple[HookPoint, Literal["fwd", "bwd"]]] = []
        effective_stop_layer = None
        if stop_at_layer is not None and hasattr(self, "blocks"):
            if stop_at_layer < 0:
                effective_stop_layer = len(self.blocks) + stop_at_layer
            else:
                effective_stop_layer = stop_at_layer
        effective_start_layer = None
        if start_at_layer is not None and hasattr(self, "blocks"):
            effective_start_layer = (
                len(self.blocks) + start_at_layer if start_at_layer < 0 else start_at_layer
            )

        def add_hook_to_point(
            hook_point: HookPoint,
            hook_fn: Callable,
            name: str,
            dir: Literal["fwd", "bwd"] = "fwd",
        ) -> None:
            if effective_start_layer is not None and self._is_embedding_stage_hook(name):
                return
            if name.startswith("blocks."):
                try:
                    layer_num: Optional[int] = int(name.split(".")[1])
                except (IndexError, ValueError):
                    layer_num = None
                if layer_num is not None:
                    if effective_stop_layer is not None and layer_num >= effective_stop_layer:
                        return
                    if effective_start_layer is not None and layer_num < effective_start_layer:
                        return
            if self.compatibility_mode and name != hook_point.name:
                alias_names_list: list = []
                if hook_point.name is not None:
                    alias_names_list.append(hook_point.name)
                alias_names_list.append(name)
                hook_point.add_hook(hook_fn, dir=dir, alias_names=alias_names_list)
            else:
                hook_point.add_hook(hook_fn, dir=dir)
            added_hooks.append((hook_point, dir))

        if stop_at_layer is not None and hasattr(self, "blocks"):
            if stop_at_layer < 0:
                stop_at_layer = len(self.blocks) + stop_at_layer
            if stop_at_layer >= 0 and stop_at_layer < len(self.blocks):

                def stop_hook(tensor: Any, *, hook: Any) -> Any:
                    raise StopAtLayerException(tensor)

                # Stop at the beginning of the specified block, not at the end of the previous block
                block_hook_name = f"blocks.{stop_at_layer}.hook_in"
                hook_dict = self.hook_dict
                if block_hook_name in hook_dict:
                    add_hook_to_point(hook_dict[block_hook_name], stop_hook, block_hook_name, "fwd")

        def apply_hooks(
            hook_list: List[Tuple[Union[str, Callable], Callable]], is_fwd: bool
        ) -> None:
            direction: Literal["fwd", "bwd"] = "fwd" if is_fwd else "bwd"
            hook_dict = self.hook_dict
            aliases = build_alias_to_canonical_map(hook_dict)
            for hook_name_or_filter, hook_fn in hook_list:
                if remove_batch_dim:
                    original_hook_fn = hook_fn

                    # Default arg captures hook_fn by value (avoids closure issue)
                    def wrapped_hook_fn(tensor, hook, _orig_fn=original_hook_fn):
                        if tensor.shape[0] == 1:
                            tensor_no_batch = tensor.squeeze(0)
                            result = _orig_fn(tensor_no_batch, hook)
                            if result.dim() == tensor_no_batch.dim():
                                result = result.unsqueeze(0)
                            return result
                        else:
                            return _orig_fn(tensor, hook)

                    hook_fn = wrapped_hook_fn
                if isinstance(hook_name_or_filter, str):
                    actual_hook_name, hook_point = self._resolve_hook_point(
                        hook_name_or_filter, aliases, hook_dict
                    )
                    add_hook_to_point(hook_point, hook_fn, actual_hook_name, direction)
                else:
                    seen_hooks: set = set()
                    for n, hook_point in hook_dict.items():
                        if hook_name_or_filter(n):
                            hook_id = id(hook_point)
                            if hook_id in seen_hooks:
                                continue
                            seen_hooks.add(hook_id)
                            hook_name_to_use = hook_point.name if hook_point.name else n
                            add_hook_to_point(hook_point, hook_fn, hook_name_to_use, direction)

        try:
            apply_hooks(fwd_hooks, True)
            apply_hooks(bwd_hooks, False)
            if start_at_layer is not None:
                kwargs["start_at_layer"] = start_at_layer
            try:
                output = self.forward(
                    input, return_type=return_type, stop_at_layer=stop_at_layer, **kwargs
                )
            except StopAtLayerException as e:
                output = e.layer_output
            return output
        finally:
            if reset_hooks_end:
                for hook_point, direction in added_hooks:
                    hook_point.remove_hooks(dir=direction)

    # ---- high-level execution: run_with_cache ----

    @overload
    def run_with_cache(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_cache_object: Literal[True] = True,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[Any, ActivationCache]:
        """Run with cache - placeholder implementation."""
        pass

    @overload
    def run_with_cache(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_cache_object: Literal[False],
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Run with cache - placeholder implementation."""
        pass

    def run_with_cache(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        names_filter: Optional[Union[str, List[str], Callable[[str], bool]]] = None,
        stop_at_layer: Optional[int] = None,
        start_at_layer: Optional[int] = None,
        pos_slice: Optional[Union[Slice, SliceInput]] = None,
        **kwargs,
    ) -> Tuple[Any, Union[ActivationCache, Dict[str, torch.Tensor]]]:
        """Run the model and cache activations. Returns ``(output, cache)``.

        ``stop_at_layer`` raises :class:`StopAtLayerException` to stop early.
        ``start_at_layer`` treats ``input`` as the residual entering block ``k``
        (see :meth:`forward`); blocks below ``k`` are excluded from the cache to
        match HookedTransformer. ``pos_slice`` slices each cached activation along
        its position dimension (``-3`` for per-head q/k/v/z/result, ``-2`` otherwise).
        ``device`` offloads cached activations (matches ``ActivationCache.to``); the
        model and inputs stay where the caller put them.
        """
        pos_slice_obj = Slice.unwrap(pos_slice)
        aliases = build_alias_to_canonical_map(self.hook_dict)

        def create_names_filter_fn(filter_input):
            if filter_input is None:
                return lambda name: True
            elif isinstance(filter_input, str):
                mapped_name = aliases.get(filter_input, None)
                if mapped_name:
                    return lambda name: name == mapped_name or name == filter_input
                else:
                    return lambda name: name == filter_input
            elif isinstance(filter_input, list):
                mapped_list = []
                for item in filter_input:
                    mapped_list.append(item)
                    mapped_name = aliases.get(item, None)
                    if mapped_name:
                        mapped_list.append(mapped_name)
                return lambda name: name in mapped_list
            elif callable(filter_input):
                return filter_input
            else:
                raise ValueError("names_filter must be a string, list of strings, or callable")

        names_filter_fn = create_names_filter_fn(names_filter)
        if isinstance(names_filter, (str, list)):
            requested = [names_filter] if isinstance(names_filter, str) else names_filter
            for name in requested:
                self._check_hook_fireable(name, aliases.get(name, name))
        cache: Dict[str, torch.Tensor] = {}
        hooks: List[Tuple[HookPoint, str]] = []
        visited: set[int] = set()

        # None → no-op .to(None), tensors stay on their current device.
        cache_device = kwargs.pop("device", None)

        def _store(name: str, value: torch.Tensor) -> None:
            stored = value.detach().to(cache_device)
            if pos_slice_obj is not None and stored.dim() >= 2:
                # Every bridge activation lays out position at dim 1 (batch is dim 0):
                # resid [b,p,d], per-head [b,p,h,d], token ids [b,p]. The exception is
                # attention patterns/scores [b,head,q_pos,k_pos] — slice the query
                # (destination) position at -2, matching HookedTransformer.
                pos_dim = -2 if name.endswith(("hook_pattern", "hook_attn_scores")) else 1
                stored = pos_slice_obj.apply(stored, dim=pos_dim)
            cache[name] = stored

        def make_cache_hook(name: str):
            def cache_hook(tensor: torch.Tensor, *, hook: Any) -> torch.Tensor:
                if tensor is None:
                    cache[name] = None
                elif isinstance(tensor, torch.Tensor):
                    _store(name, tensor)
                elif isinstance(tensor, tuple):
                    if len(tensor) > 0 and isinstance(tensor[0], torch.Tensor):
                        _store(name, tensor[0])
                    else:
                        pass
                else:
                    try:
                        if hasattr(tensor, "detach"):
                            _store(name, tensor)
                    except:
                        pass
                return tensor

            return cache_hook

        hook_dict = self.hook_dict
        effective_stop_layer = None
        if stop_at_layer is not None and hasattr(self, "blocks"):
            if stop_at_layer < 0:
                effective_stop_layer = len(self.blocks) + stop_at_layer
            else:
                effective_stop_layer = stop_at_layer
        effective_start_layer = None
        if start_at_layer is not None and hasattr(self, "blocks"):
            effective_start_layer = (
                len(self.blocks) + start_at_layer if start_at_layer < 0 else start_at_layer
            )
        matched_any = False
        for hook_name, hook in hook_dict.items():
            if names_filter_fn(hook_name):
                matched_any = True
                if effective_start_layer is not None and self._is_embedding_stage_hook(hook_name):
                    continue
                if hook_name.startswith("blocks."):
                    try:
                        layer_num = int(hook_name.split(".")[1])
                    except (IndexError, ValueError):
                        layer_num = None
                    if layer_num is not None:
                        # stop/start bound the executed range; blocks outside it
                        # either don't run (stop) or run on discarded input (start),
                        # so their activations must not enter the cache.
                        if effective_stop_layer is not None and layer_num >= effective_stop_layer:
                            continue
                        if effective_start_layer is not None and layer_num < effective_start_layer:
                            continue
                hooks.append((hook, hook_name))
        # Explicit string/list filters matching nothing must not return (logits, {}) silently.
        if not matched_any and names_filter and isinstance(names_filter, (str, list)):
            raise KeyError(
                f"names_filter {names_filter!r} matched no hook points on this model; "
                "check the name against model.hook_dict (this backend may not serve it)."
            )
        for hp, name in hooks:
            hp.add_hook(make_cache_hook(name))
        processed_args = [input]
        # Driver-aware input placement: torch drivers move input_ids to the model's
        # device; remote drivers (no local parameters) leave them as-is.
        target_device = self._input_device()
        if processed_args and isinstance(processed_args[0], str):
            assert self.tokenizer is not None, "Tokenizer must be set to pass string input."
            input_ids = self.to_tokens(processed_args[0])
            if target_device is not None:
                input_ids = input_ids.to(target_device)
            kwargs["input_ids"] = input_ids
            processed_args = processed_args[1:]
        elif "input" in kwargs and isinstance(kwargs["input"], str):
            assert self.tokenizer is not None, "Tokenizer must be set to pass string input."
            input_ids = self.to_tokens(kwargs["input"])
            if target_device is not None:
                input_ids = input_ids.to(target_device)
            kwargs["input_ids"] = input_ids
            del kwargs["input"]
        if stop_at_layer is not None and hasattr(self, "blocks"):
            if stop_at_layer < 0:
                stop_at_layer = len(self.blocks) + stop_at_layer
            last_layer_to_process = stop_at_layer - 1

            def stop_hook(tensor: torch.Tensor, *, hook: Any) -> torch.Tensor:
                raise StopAtLayerException(tensor)

            if stop_at_layer >= 0 and stop_at_layer < len(self.blocks):
                # Stop at the beginning of the specified block, not at the end of the previous block
                block_hook_name = f"blocks.{stop_at_layer}.hook_in"
                hook_dict = self.hook_dict
                if block_hook_name in hook_dict:
                    hook_dict[block_hook_name].add_hook(stop_hook)
                    hooks.append((hook_dict[block_hook_name], block_hook_name))
        filtered_kwargs = kwargs.copy()
        # ``cache_device`` is honored by ``make_cache_hook`` above (``tensor.detach().to(cache_device)``);
        # the model and inputs stay where the caller put them, matching ``ActivationCache.to``.
        if cache_device is not None and getattr(self.cfg, "n_devices", 1) > 1:
            # Moving a dispatched model to a single device collapses accelerate's
            # split and breaks its routing hooks. The cache will stay spread across
            # the per-layer devices; callers can .to(cache_device) on cache entries
            # after the fact if they need a single-device cache.
            warnings.warn(
                f"run_with_cache(device={cache_device!r}) ignored: model is dispatched "
                f"across {self.cfg.n_devices} devices via device_map. Cached activations "
                "will remain on their per-layer devices.",
                stacklevel=2,
            )
        if start_at_layer is not None:
            filtered_kwargs["start_at_layer"] = start_at_layer
        try:
            if (
                "output_attentions" not in filtered_kwargs
                and self.adapter.supports_hf_output_attentions
            ):
                filtered_kwargs["output_attentions"] = True
            if processed_args:
                output = self.forward(processed_args[0], **filtered_kwargs)
            elif "input_ids" in filtered_kwargs:
                output = self.forward(
                    filtered_kwargs["input_ids"],
                    **{k: v for k, v in filtered_kwargs.items() if k != "input_ids"},
                )
            else:
                output = self.forward(**filtered_kwargs)
            if hasattr(output, "logits"):
                output = output.logits
        except StopAtLayerException as e:
            output = e.layer_output
        except Exception as e:
            raise e
        finally:
            for hp, _ in hooks:
                hp.remove_hooks(dir="fwd")
        if self.compatibility_mode == True:
            reverse_aliases = {}
            for old_name, new_name in aliases.items():
                if isinstance(new_name, list):
                    for single_new_name in new_name:
                        reverse_aliases[single_new_name] = old_name
                else:
                    reverse_aliases[new_name] = old_name
            cache_items_to_add = {}
            for cache_name, cached_value in cache.items():
                for new_name, old_name in reverse_aliases.items():
                    if cache_name == new_name:
                        cache_items_to_add[old_name] = cached_value
                        break
            cache.update(cache_items_to_add)
            for alias_name, target_name in aliases.items():
                if isinstance(target_name, list):
                    for single_target in target_name:
                        if single_target in cache and alias_name not in cache:
                            cache[alias_name] = cache[single_target]
                            break
                elif target_name in cache and alias_name not in cache:
                    cache[alias_name] = cache[target_name]
        if return_cache_object:
            activation_cache = ActivationCache(cache, self, has_batch_dim=True)
            if remove_batch_dim:
                activation_cache.remove_batch_dim()
            return (output, activation_cache)
        else:
            if remove_batch_dim:
                for key in cache:
                    if cache[key] is not None and isinstance(cache[key], torch.Tensor):
                        if cache[key].size(0) == 1:
                            cache[key] = cache[key][0]
            return (output, cache)
