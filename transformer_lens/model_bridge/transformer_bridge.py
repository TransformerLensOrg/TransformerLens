"""Bridge module for connecting different model architectures.

This module provides the bridge components that wrap remote model components and provide
a consistent interface for accessing their weights and performing operations.
"""
import logging
import re
import warnings
from collections.abc import Generator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

import einops
import numpy as np
import torch
import tqdm
from torch import nn

from transformer_lens import utilities as utils
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookIntrospectionMixin, HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.bridge_core import BridgeCore
from transformer_lens.model_bridge.component_setup import set_original_components
from transformer_lens.model_bridge.composition_scores import CompositionScores
from transformer_lens.model_bridge.driver_protocol import (
    TensorLike,
    to_torch,
    validate_driver,
)
from transformer_lens.model_bridge.exceptions import StopAtLayerException
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.block import (
    _BLOCK_INTERNAL_MODULES,
    _NORM_PREFIXES,
    _VARIANT_SUBMODULE_SET,
    VARIANT_SUBMODULE_NAMES,
)
from transformer_lens.model_bridge.get_params_util import get_bridge_params
from transformer_lens.utilities.devices import move_to_and_update_config

if TYPE_CHECKING:
    pass

_BLOCK_PATTERN = re.compile("blocks\\.(\\d+)")


def _resolve_attr_path(obj: nn.Module, attr_path: str) -> torch.Tensor:
    """Walk a dot-separated attribute path and return the final tensor."""
    result = obj
    for attr in attr_path.split("."):
        result = getattr(result, attr)
    return cast(torch.Tensor, result)


# build_alias_to_canonical_map lives in bridge_core.py; re-import for the module's
# internal use (run_with_cache, hooks() context manager in this file).
from transformer_lens.model_bridge.bridge_core import (  # noqa: E402
    build_alias_to_canonical_map,
)


class TransformerBridge(BridgeCore, HookIntrospectionMixin, nn.Module):
    """Torch-backed bridge: HF, vLLM-via-torch, anything that wraps an ``nn.Module``.

    Provides a standardized interface to access components of a transformer
    model, regardless of the underlying architecture. It uses an architecture adapter
    to map between the TransformerLens and HuggingFace model structures.

    Tokenization notes
    ------------------

    :meth:`to_tokens`, :meth:`to_str_tokens`, :meth:`get_token_position`,
    :meth:`forward` (string input), and :meth:`generate` accept ``prepend_bos``
    to control BOS prepending. Resolution: explicit arg →
    ``cfg.default_prepend_bos`` (defaults ``True``, even for non-BOS-trained
    models — attention heads tend to use position 0 as a resting state).
    **Pass ``prepend_bos=False`` when tokenizing a fragment of a larger
    prompt** — off-by-one position errors usually trace back here.

    Reconciliation with ``cfg.tokenizer_prepends_bos`` (tokenizers that add
    BOS automatically) is handled internally — pass the value you want;
    the bridge adds or strips manually as needed. When
    ``cfg.tokenizer_appends_eos=True`` (OLMo, Apertus, etc.),
    :meth:`to_tokens` also strips trailing EOS tokens so the model receives
    a continuation rather than a terminated sequence; this path is
    bridge-specific.

    BPE/SentencePiece tokenizers treat ``"hello"``, ``" hello"``, and
    ``"Hello"`` as distinct tokens. Concatenated prompts may not tokenize
    as the sum of parts — inspect with :meth:`to_str_tokens` when in doubt.
    """

    # hook_aliases inherited from BridgeCore

    def __init__(
        self,
        model: nn.Module,
        adapter: ArchitectureAdapter,
        tokenizer: Any,
        *,
        driver: Any = None,
    ):
        """Initialize the bridge.

        Args:
            model: The model to bridge (must be a PyTorch nn.Module or PreTrainedModel)
            adapter: The architecture adapter to use
            tokenizer: The tokenizer to use (required)
            driver: Optional pre-built :class:`Driver`. Sources that construct
                exotic drivers (vLLM, Inspect) pass them here. When ``None``,
                a :class:`TransformersDriver` is built from the supplied
                ``model``/``adapter``/``tokenizer`` — kept for backward
                compatibility with direct ``TransformerBridge(...)`` callers.
        """
        nn.Module.__init__(self)
        self.__dict__["original_model"] = model
        # Production sources construct their Driver and pass it via ``driver=``.
        # The fallback covers tests / direct callers with a hand-rolled triple.
        if driver is None:
            from transformer_lens.model_bridge.sources.transformers_driver import (
                TransformersDriver,
            )

            driver = TransformersDriver(model, adapter, tokenizer)
        BridgeCore.__init__(self, adapter, tokenizer, driver)
        # real_components maps TL keys to (remote_path, actual_instance) tuples;
        # for list components, actual_instance is a list of instances.
        self.real_components: Dict[str, tuple] = {}
        if not hasattr(self.cfg, "device") or self.cfg.device is None:
            try:
                self.cfg.device = str(next(self.original_model.parameters()).device)
            except StopIteration:
                self.cfg.device = "cpu"
        set_original_components(self, self.adapter, self.__dict__["original_model"])
        self._initialize_hook_registry()
        self._register_aliases()
        self._register_all_aliases_recursive()
        # Re-scan after alias registration so alias names (hook_resid_pre, …)
        # join the registry alongside their canonical targets. Shared HookPoint
        # instances, so no double-firing.
        self._scan_existing_hooks(self, "")
        self._setup_hook_compatibility()
        # Backfill supported_hook_points = registry − non_fireable. Whitelist-
        # semantic drivers (Inspect) declared their own non-empty set and skip.
        if not self._driver.supported_hook_points:
            self._driver.supported_hook_points = (
                frozenset(self._hook_registry) - self._driver.non_fireable_hook_points
            )
        # Fail fast on a misshapen driver here, not at first capture.
        validate_driver(self._driver, after_bridge_construction=True)
        self.processor = None

    @classmethod
    def boot_transformers(
        cls,
        model_name: str,
        hf_config_overrides: Optional[dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        tokenizer: Optional[Any] = None,
        load_weights: bool = True,
        trust_remote_code: bool = False,
        model_class: Optional[type] = None,
        hf_model: Optional[Any] = None,
        device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
        n_devices: Optional[int] = None,
        max_memory: Optional[Dict[Union[str, int], str]] = None,
        n_ctx: Optional[int] = None,
        revision: Optional[str] = None,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
    ) -> "TransformerBridge":
        """Boot a model from HuggingFace (alias for sources.transformers.boot).

        Returns raw HF weights by default — logits/activations match HF, *not*
        legacy ``HookedTransformer`` (which folds LayerNorm + centers weights).
        Call ``enable_compatibility_mode()`` on the result for HookedTransformer-
        equivalent numerics. Generation, argmax, and CE loss are unaffected.

        Attention implementation is forced to ``"eager"`` so hooks can capture scores
        and patterns. For an apples-to-apples HF comparison, load the HF model with
        ``attn_implementation="eager"`` too; comparing against the default ``"sdpa"``
        shows ~1e-3 fp32 drift from kernel-level op reordering, not a bridge bug.

        Args:
            model_name: The name of the model to load.
            hf_config_overrides: Optional overrides applied to the HuggingFace config before model load.
            device: The device to use. If None, will be determined automatically. Mutually exclusive
                with ``device_map``.
            dtype: The dtype to use for the model.
            tokenizer: Optional pre-initialized tokenizer to use; if not provided one will be created.
            load_weights: If False, load model without weights (on meta device) for config inspection only.
            trust_remote_code: Whether to trust remote code for custom model architectures.
            model_class: Optional HuggingFace model class to use instead of the default
                auto-detected class (e.g., BertForNextSentencePrediction).
            hf_model: Optional pre-loaded HuggingFace model to use instead of loading one. Useful
                for models loaded with custom configurations (e.g., quantization via
                BitsAndBytesConfig). When provided, load_weights is ignored. If the pre-loaded
                model was built with a ``device_map``, ``cfg.device`` and ``cfg.n_devices`` are
                derived from its ``hf_device_map`` automatically.
            device_map: HuggingFace-style device map for multi-GPU inference. Pass ``"auto"``,
                ``"balanced"``, ``"sequential"``, or an explicit ``{submodule_path: device}`` dict.
                Mutually exclusive with ``device``.
            n_devices: Convenience shortcut: split the model across this many CUDA devices.
                Translated to a ``max_memory`` dict over devices 0..n_devices-1 and passed as
                ``device_map`` to HF. Requires CUDA with at least this many visible devices.
            max_memory: Optional per-device memory budget, passed through to HF's dispatcher.
                Only used when ``device_map`` or ``n_devices`` is in effect.
            n_ctx: Optional context length override. Writes to the appropriate HF config field
                for this model automatically (callers don't need to know the field name).
                Warns if larger than the model's default context length.
            revision: Optional HF revision (branch, tag, or commit). Forwarded to the underlying
                ``AutoConfig.from_pretrained`` and ``AutoModelForCausalLM.from_pretrained`` calls.
                Mutually exclusive with ``checkpoint_index`` / ``checkpoint_value``.
            checkpoint_index: Index into the available training checkpoints for the model family
                (currently ``EleutherAI/pythia*`` and ``stanford-crfm/*``). Resolved to a revision
                string via known per-family naming conventions.
            checkpoint_value: Training step or token count of the desired checkpoint. Alternative
                to ``checkpoint_index``; must match an entry in the family's checkpoint label list.

        Returns:
            The bridge to the loaded model.
        """
        from transformer_lens.model_bridge.sources.transformers import boot

        return boot(
            model_name=model_name,
            hf_config_overrides=hf_config_overrides,
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            load_weights=load_weights,
            trust_remote_code=trust_remote_code,
            model_class=model_class,
            hf_model=hf_model,
            device_map=device_map,
            n_devices=n_devices,
            max_memory=max_memory,
            n_ctx=n_ctx,
            revision=revision,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
        )

    @property
    def original_model(self) -> nn.Module:
        """The wrapped ``nn.Module``. Raises :class:`AttributeError` for
        non-torch drivers (vLLM, Inspect) that don't expose a local module."""
        underlying = getattr(self._driver, "underlying_model", None)
        if underlying is None:
            raise AttributeError(
                f"{type(self._driver).__name__} does not expose an nn.Module — "
                "non-torch drivers (vLLM, Inspect) operate without a local module."
            )
        return underlying

    @original_model.setter
    def original_model(self, value: nn.Module) -> None:
        """Used by weight-processing paths that move the model across devices."""
        self.__dict__["original_model"] = value
        # Sync via the driver's public API; non-torch drivers don't implement it.
        setter = getattr(self._driver, "set_underlying_model", None)
        if callable(setter):
            setter(value)

    def _set_processed_weight_attributes(self) -> None:
        """Create 3D processed weight attributes for attention components.

        For each attention component, if it has 2D weights (q.weight, k.weight, v.weight),
        reshape them to 3D format [n_heads, d_model, d_head] and set as:
        - _processed_W_Q
        - _processed_W_K
        - _processed_W_V
        - _processed_b_Q
        - _processed_b_K
        - _processed_b_V

        This allows property aliases (W_Q, W_K, W_V) to return 3D format for
        HookedTransformer compatibility while keeping 2D format for calculations.
        """

        n_heads = self.cfg.n_heads
        d_head = self.cfg.d_head
        d_model = self.cfg.d_model
        if not hasattr(self, "blocks"):
            return
        for block in self.blocks:
            if "attn" not in block._modules:
                continue
            attn = block.attn
            if not (hasattr(attn, "q") and hasattr(attn.q, "weight")):
                continue
            try:
                w_q_2d = attn.q.weight.data
                w_k_2d = attn.k.weight.data
                w_v_2d = attn.v.weight.data
                attn._processed_W_Q = einops.rearrange(
                    w_q_2d, "m (i h) -> i m h", i=n_heads, h=d_head
                )
                attn._processed_W_K = einops.rearrange(
                    w_k_2d, "m (i h) -> i m h", i=n_heads, h=d_head
                )
                attn._processed_W_V = einops.rearrange(
                    w_v_2d, "m (i h) -> i m h", i=n_heads, h=d_head
                )
                if hasattr(attn.q, "bias") and attn.q.bias is not None:
                    b_q_2d = attn.q.bias.data
                    b_k_2d = attn.k.bias.data
                    b_v_2d = attn.v.bias.data
                    attn._processed_b_Q = einops.rearrange(
                        b_q_2d, "(i h) -> i h", i=n_heads, h=d_head
                    )
                    attn._processed_b_K = einops.rearrange(
                        b_k_2d, "(i h) -> i h", i=n_heads, h=d_head
                    )
                    attn._processed_b_V = einops.rearrange(
                        b_v_2d, "(i h) -> i h", i=n_heads, h=d_head
                    )
                if hasattr(attn, "o") and hasattr(attn.o, "weight"):
                    w_o_2d = attn.o.weight.data
                    w_o_transposed = w_o_2d.T
                    attn._processed_W_O = einops.rearrange(
                        w_o_transposed, "m (i h) -> i h m", i=n_heads, h=d_head
                    )
                    if hasattr(attn.o, "bias") and attn.o.bias is not None:
                        attn._processed_b_O = attn.o.bias.data
            except Exception:
                pass

    def _register_all_aliases_recursive(self) -> None:
        """Recursively register aliases on all bridge components.

        This walks through all components and calls _register_aliases() on each one.
        Used after weight processing to ensure aliases point to processed weights.
        """
        if hasattr(self, "_register_aliases"):
            self._register_aliases()
        for module in self.modules():
            if module is not self and hasattr(module, "_register_aliases"):
                getattr(module, "_register_aliases")()

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to track HookPoint objects dynamically."""
        super().__setattr__(name, value)
        if isinstance(value, HookPoint):
            value.name = name
            self._hook_registry[name] = value
        elif hasattr(value, "get_hooks") and callable(getattr(value, "get_hooks")):
            component_hooks = value.get_hooks()
            for hook_name, hook in component_hooks.items():
                full_name = f"{name}.{hook_name}"
                hook.name = full_name
                self._hook_registry[full_name] = hook

    def _scan_existing_hooks(self, module: nn.Module, prefix: str = "") -> None:
        """Scan existing modules for hooks and add them to registry."""
        visited = set()
        # Protect canonical HookPoint names from alias overwrites
        named_hook_ids: set = set()

        def scan_module(mod: nn.Module, path: str = "") -> None:
            obj_id = id(mod)
            if obj_id in visited:
                return
            visited.add(obj_id)
            if hasattr(mod, "get_hooks") and callable(getattr(mod, "get_hooks")):
                component_hooks = mod.get_hooks()  # type: ignore[operator]
                if isinstance(component_hooks, dict):
                    hooks_dict = cast(Dict[str, HookPoint], component_hooks)
                    for hook_name, hook in hooks_dict.items():
                        full_name = f"{path}.{hook_name}" if path else hook_name
                        hook_id = id(hook)
                        if hook_id not in named_hook_ids:
                            hook.name = full_name
                            named_hook_ids.add(hook_id)
                        self._hook_registry[full_name] = hook
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                if attr_name == "original_component" or attr_name == "original_model":
                    continue
                if attr_name in [
                    "OV",
                    "QK",
                    "W_V",
                    "W_O",
                    "W_Q",
                    "W_K",
                    "W_in",
                    "W_gate",
                    "W_out",
                    "b_V",
                    "b_O",
                    "b_Q",
                    "b_K",
                    "b_in",
                    "b_out",
                ]:
                    continue
                try:
                    attr = getattr(mod, attr_name)
                except (AttributeError, NameError, RuntimeError, TypeError):
                    continue
                name = f"{path}.{attr_name}" if path else attr_name
                if isinstance(attr, HookPoint):
                    hook_id = id(attr)
                    if hook_id not in named_hook_ids:
                        attr.name = name
                        named_hook_ids.add(hook_id)
                    self._hook_registry[name] = attr
            for child_name, child_module in mod.named_children():
                if (
                    child_name == "original_component"
                    or child_name == "_original_component"
                    or child_name == "original_model"
                ):
                    continue
                child_path = f"{path}.{child_name}" if path else child_name
                scan_module(child_module, child_path)

        scan_module(module, prefix)

    @property
    def n_params_total(self) -> int:
        """Total number of parameters in the model, including embeddings, biases,
        and layer norm weights.

        Mirrors :attr:`HookedTransformer.n_params_total`. Use this when you want
        the actual parameter count for memory budgeting, comparison with
        HuggingFace's ``model.num_parameters()``, or alignment with reported
        model sizes in papers (e.g. the Pythia suite).

        Returns:
            int: ``sum(p.numel() for p in self.parameters())``
        """
        return sum(p.numel() for p in self.parameters())

    def __getattr__(self, name: str) -> Any:
        """Provide a clear error message for missing attributes."""
        # Re-invoke original_model's property so its descriptive AttributeError
        # for non-torch drivers isn't shadowed by the __dict__ fallback below.
        if name == "original_model":
            prop = type(self).__dict__.get("original_model")
            if isinstance(prop, property) and prop.fget is not None:
                return prop.fget(self)
        if name in self.__dict__:  # type: ignore[arg-type]
            return self.__dict__[name]
        # Use __dict__ directly to avoid recursion
        if "_modules" in self.__dict__ and name in self.__dict__["_modules"]:  # type: ignore[arg-type]
            return self.__dict__["_modules"][name]
        if "original_model" in self.__dict__ and self.__dict__["original_model"] is not None:
            try:
                name_split = name.split(".")
                if len(name_split) > 1:
                    current = getattr(self.__dict__["original_model"], name_split[0])
                    for part in name_split[1:]:  # type: ignore[operator]
                        current = getattr(current, part)
                    return current
                else:
                    return getattr(self.__dict__["original_model"], name)
            except AttributeError:
                pass  # type: ignore[operator,assignment]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __str__(self) -> str:
        """Get a string representation of the bridge.
        # type: ignore[operator]
               Returns:
                   A string describing the bridge's components # type: ignore[operator]
        """
        lines = ["TransformerBridge:"]
        mapping = self.adapter.get_component_mapping()
        lines.extend(self._format_component_mapping(mapping, indent=1))
        return "\n".join(lines)

    def enable_compatibility_mode(
        self,
        disable_warnings: bool = False,
        no_processing: bool = False,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
    ) -> None:
        """Apply HookedTransformer-equivalent weight processing and legacy hook compatibility.

        Defaults match HookedTransformer's load-time processing (fold_ln + weight
        centering) — required for analyses that reason in HookedTransformer's
        post-processed coordinate system: logit lens, direct logit attribution,
        residual-stream norms. Also enables legacy hook/component name aliases.

        Args:
            disable_warnings: Whether to disable warnings about legacy components/hooks
            no_processing: Whether to disable ALL pre-processing steps of the model.
                If True, overrides fold_ln, center_writing_weights, and center_unembed to False.
            fold_ln: Whether to fold layer norm weights into the subsequent linear layers.
                Default: True. Ignored if no_processing=True.
            center_writing_weights: Whether to center the writing weights (W_out in attention and MLPs).
                Default: True. Ignored if no_processing=True.
            center_unembed: Whether to center the unembedding matrix.
                Default: True. Ignored if no_processing=True.
            fold_value_biases: Whether to fold value biases into output bias.
                Default: True. Ignored if no_processing=True.
            refactor_factored_attn_matrices: Whether to refactor factored attention matrices.
                Default: False. Ignored if no_processing=True.
        """
        from transformer_lens.utilities.bridge_components import (
            apply_fn_to_all_components,
        )

        self.compatibility_mode = True

        def set_compatibility_mode(component: Any) -> None:
            """Set compatibility mode on a component."""
            component.compatibility_mode = True
            component.disable_warnings = disable_warnings

        apply_fn_to_all_components(self, set_compatibility_mode)
        self.clear_hook_registry()
        try:
            if not no_processing:
                self.process_weights(
                    fold_ln=fold_ln,
                    center_writing_weights=center_writing_weights,
                    center_unembed=center_unembed,
                    fold_value_biases=fold_value_biases,
                    refactor_factored_attn_matrices=refactor_factored_attn_matrices,
                )
        finally:
            # Re-initialize hooks even on failure so bridge stays usable
            self._initialize_hook_registry()
            self._setup_hook_compatibility()
            self._register_all_aliases_recursive()

    def _setup_hook_compatibility(self) -> None:
        """Setup hook compatibility transformations to match HookedTransformer behavior.

        This method sets up hook conversions and wrappers that ensure Bridge hooks
        have the same shapes and behavior as HookedTransformer hooks. This includes:
        1. hook_z reshaping from [batch, seq, d_model] to [batch, seq, n_heads, d_head]
        2. Wrapping HF attention forward to inject position embeddings/attention masks
        3. Architecture-specific setup (e.g., rotary embedding references)

        This is called during __init__ and should always be run, regardless of whether
        compatibility mode or weight processing is enabled.

        Note: This method is idempotent - can be called multiple times safely.
        """
        if hasattr(self.adapter, "setup_hook_compatibility"):
            self.adapter.setup_hook_compatibility(self)
        elif hasattr(self.adapter, "setup_no_processing_hooks"):
            self.adapter.setup_no_processing_hooks(self)
        blocks_to_process = []
        if hasattr(self, "blocks"):
            blocks_to_process.extend(self.blocks)
        if hasattr(self, "encoder_blocks"):
            blocks_to_process.extend(self.encoder_blocks)
        if hasattr(self, "decoder_blocks"):
            blocks_to_process.extend(self.decoder_blocks)
        for block in blocks_to_process:
            for attn_name in ["attn", "self_attn", "cross_attn"]:
                if hasattr(block, attn_name):
                    attn = getattr(block, attn_name)
                    if hasattr(attn, "setup_hook_compatibility"):
                        attn.setup_hook_compatibility()
                    elif hasattr(attn, "setup_no_processing_hooks"):
                        attn.setup_no_processing_hooks()

    def process_weights(
        self,
        verbose: bool = False,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
    ) -> None:
        """Process weights directly using ProcessWeights and architecture adapter.

        This method applies weight processing transformations to improve model interpretability
        without requiring a reference HookedTransformer model. Works with all architectures
        supported by TransformerBridge, including GPT-OSS and other new models.

        Args:
            verbose: If True, print detailed progress messages. Default: False
            fold_ln: Fold LayerNorm weights/biases into subsequent layers. Default: True
            center_writing_weights: Center weights that write to residual stream. Default: True
            center_unembed: Center unembedding weights (translation invariant). Default: True
            fold_value_biases: Fold value biases into output bias. Default: True
            refactor_factored_attn_matrices: Experimental QK/OV factorization. Default: False
        """
        from transformer_lens.weight_processing import ProcessWeights

        if verbose:
            print(f"Processing weights for {self.cfg.model_name}...")

        # Soft capping (tanh) is not translation-invariant; centering would change output.
        if center_unembed and getattr(self.cfg, "output_logits_soft_cap", -1.0) > 0.0:
            import logging

            logging.warning(
                "center_unembed=True is incompatible with logit softcapping "
                "(output_logits_soft_cap=%.1f). Disabling center_unembed.",
                self.cfg.output_logits_soft_cap,
            )
            center_unembed = False

        if verbose:
            print("  Extracting state dict from existing model...")
        state_dict = self.state_dict()
        adapter = self.adapter

        # Untie embed/unembed weights (GPT-2) so centering affects only unembed
        embed_key = "embed.weight"
        unembed_key = "unembed.weight"

        if embed_key in state_dict and unembed_key in state_dict:
            # Check if they point to the same tensor (weight tying)
            if state_dict[embed_key].data_ptr() == state_dict[unembed_key].data_ptr():
                if verbose:
                    print("  Breaking weight tying between embed and unembed in state dict...")
                # Clone the unembed weight to break the tie
                state_dict[unembed_key] = state_dict[unembed_key].clone()

        if adapter and hasattr(adapter, "preprocess_weights"):
            adapter._fold_ln_requested = fold_ln  # type: ignore[union-attr]
            state_dict = adapter.preprocess_weights(state_dict)

        # Use unified ProcessWeights.process_weights() like HookedTransformer does.
        # Float32 upcasting for precision is handled centrally in process_weights().
        if verbose:
            print("  Processing weights (fold_ln, center_writing_weights, etc.)...")
        state_dict = ProcessWeights.process_weights(
            state_dict,
            self.cfg,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            adapter=adapter,
        )

        # Normalize HF-prefix keys to TL format for weight routing
        import re

        hf_to_tl_prefix = {}
        for tl_name, (remote_path, _component) in self.real_components.items():
            if remote_path and remote_path != tl_name:
                hf_to_tl_prefix[remote_path] = tl_name

        normalized_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for hf_prefix, tl_prefix in hf_to_tl_prefix.items():
                if key.startswith(hf_prefix + "."):
                    suffix = key[len(hf_prefix) + 1 :]
                    new_key = f"{tl_prefix}.{suffix}"
                    break
            normalized_state_dict[new_key] = value
        state_dict = normalized_state_dict

        if verbose:
            print("  Distributing weights to generalized components...")
        ProcessWeights.distribute_weights_to_components(
            state_dict=state_dict,
            component_mapping=self.real_components,
        )

    def _calculate_loss(self, logits, tokens, loss_per_token=False):
        """Calculate cross-entropy loss."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none" if loss_per_token else "mean")
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        loss = loss_fct(flat_logits, flat_labels)
        if loss_per_token:
            return loss.view(shift_labels.shape)
        else:
            return loss

    def _extract_hf_weights(self):
        """Extract weights from the original HuggingFace model."""
        hf_state_dict = self.state_dict()
        for layer_idx in range(self.cfg.n_layers):
            combined_qkv_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"
            combined_qkv_bias_key = f"transformer.h.{layer_idx}.attn.c_attn.bias"
            if combined_qkv_key in hf_state_dict:
                separate_keys_to_remove = [
                    f"transformer.h.{layer_idx}.attn.q.weight",
                    f"transformer.h.{layer_idx}.attn.q.bias",
                    f"transformer.h.{layer_idx}.attn.k.weight",
                    f"transformer.h.{layer_idx}.attn.k.bias",
                    f"transformer.h.{layer_idx}.attn.v.weight",
                    f"transformer.h.{layer_idx}.attn.v.bias",
                ]
                for key_to_remove in separate_keys_to_remove:
                    if key_to_remove in hf_state_dict:
                        del hf_state_dict[key_to_remove]
        return hf_state_dict

    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> torch.Tensor:
        """Converts a string to a tensor of tokens.

        See the class-level "Tokenization notes" for full ``prepend_bos``
        semantics, the ``default_prepend_bos`` /
        ``tokenizer_prepends_bos`` interaction, and the whitespace-
        sensitivity gotcha. **Pass ``prepend_bos=False`` whenever you're
        tokenizing only part of a prompt.**

        Args:
            input: The input to tokenize.
            prepend_bos: Overrides ``self.cfg.default_prepend_bos``. Defaults
                to ``None`` (use the cfg setting). Pass ``True`` or ``False``
                to override locally.
            padding_side: Which side to pad on when tokenizing multiple
                strings of different lengths. Defaults to the tokenizer's
                ``padding_side``.
            move_to_device: Whether to move the result to ``cfg.device``.
            truncate: Whether to truncate inputs longer than ``cfg.n_ctx``.

        Returns:
            Token tensor of shape ``[batch, pos]``.
        """
        assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"
        if prepend_bos is None:
            prepend_bos = getattr(self.cfg, "default_prepend_bos", True)
        if padding_side is None:
            padding_side = getattr(self.tokenizer, "padding_side", "right")
        tokenizer_prepends_bos = getattr(self.cfg, "tokenizer_prepends_bos", True)
        if prepend_bos and (not tokenizer_prepends_bos):
            input = utils.get_input_with_manually_prepended_bos(self.tokenizer.bos_token, input)
        if isinstance(input, str):
            input = [input]
        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.cfg.n_ctx if truncate else None,
        )["input_ids"]
        # Strip auto-appended EOS tokens (e.g., OLMo)
        if (
            getattr(self.cfg, "tokenizer_appends_eos", False)
            and self.tokenizer.eos_token_id is not None
        ):
            # Remove trailing EOS, keep at least 1 token
            while tokens.shape[-1] > 1 and (tokens[:, -1] == self.tokenizer.eos_token_id).all():
                tokens = tokens[:, :-1]
        if not prepend_bos and tokenizer_prepends_bos:
            tokens = utils.get_tokens_with_bos_removed(self.tokenizer, tokens)
        if move_to_device:
            tokens = tokens.to(self.cfg.device)
        return tokens

    def to_string(
        self, tokens: Union[List[int], torch.Tensor, np.ndarray]
    ) -> Union[str, List[str]]:
        """Convert tokens to string(s).

        Args:
            tokens: Tokens to convert

        Returns:
            Decoded string(s)
        """
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)
        if len(tokens.shape) == 2:
            return self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        elif len(tokens.shape) <= 1:
            return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        else:
            raise ValueError(f"Invalid shape passed in: {tokens.shape}")

    def to_str_tokens(
        self,
        input: Union[str, torch.Tensor, np.ndarray, List],
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
    ) -> Union[List[str], List[List[str]]]:
        """Map text or tokens to a list of tokens as strings.

        See the class-level "Tokenization notes" for full ``prepend_bos``
        semantics. **Pass ``prepend_bos=False`` whenever you're tokenizing
        only part of a prompt.** When ``input`` is already a tensor or
        array, ``prepend_bos`` and ``padding_side`` are ignored.

        Args:
            input: A string, list of strings, or tensor/array of token IDs.
            prepend_bos: Overrides ``self.cfg.default_prepend_bos``. Only
                applies when ``input`` is a string. Defaults to ``None``
                (use the cfg setting).
            padding_side: Which side to pad on. Only applies when ``input``
                is a string.

        Returns:
            List of token strings.
        """
        if isinstance(input, list):
            return cast(
                List[List[str]],
                [self.to_str_tokens(item, prepend_bos, padding_side) for item in input],
            )
        elif isinstance(input, str):
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)[0]
        elif isinstance(input, torch.Tensor):
            tokens = input.squeeze()
            if tokens.dim() == 0:
                tokens = tokens.unsqueeze(0)
            assert (
                tokens.dim() == 1
            ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
        elif isinstance(input, np.ndarray):
            tokens_np = input.squeeze()
            if tokens_np.ndim == 0:
                tokens_np = np.expand_dims(tokens_np, axis=0)
            assert (
                tokens_np.ndim == 1
            ), f"Invalid tokens input to to_str_tokens, has shape: {tokens_np.shape}"
            tokens = torch.tensor(tokens_np)
        else:
            raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")
        # v5 compat: wrap each token so batch_decode decodes them individually
        tokens_list = [[int(t)] for t in tokens.tolist()]
        str_tokens = self.tokenizer.batch_decode(tokens_list, clean_up_tokenization_spaces=False)
        return str_tokens

    def to_single_token(self, string: str) -> int:
        """Map a string that makes up a single token to the id for that token.

        Args:
            string: The string to convert

        Returns:
            Token ID

        Raises:
            AssertionError: If string is not a single token
        """
        token = self.to_tokens(string, prepend_bos=False).squeeze()
        if token.numel() != 1:
            raise AssertionError(f"Input string: {string} is not a single token!")
        return int(token.item())

    def get_token_position(
        self,
        single_token: Union[str, int],
        input: Union[str, torch.Tensor],
        mode="first",
        prepend_bos: Optional[Union[bool, None]] = None,
        padding_side: Optional[Union[Literal["left", "right"], None]] = None,
    ):
        """Get the position of a single_token in a string or sequence of tokens.

        Raises an error if the token is not present.

        When ``input`` is a string it's tokenized internally — see the
        class-level "Tokenization notes" for ``prepend_bos`` semantics.
        Off-by-one position errors usually mean ``prepend_bos`` is on
        when it shouldn't be (or vice versa); pass ``prepend_bos=False``
        when ``input`` is a fragment of a larger prompt.

        Args:
            single_token (Union[str, int]): The token to search for. Can
                be a token index, or a string (but the string must correspond to a single token).
            input (Union[str, torch.Tensor]): The sequence to
                search in. Can be a string or a rank 1 tensor of tokens or a rank 2 tensor of tokens
                with a dummy batch dimension.
            mode (str, optional): If there are multiple matches, which match to return. Supports
                "first" or "last". Defaults to "first".
            prepend_bos (bool, optional): Overrides ``self.cfg.default_prepend_bos``. Only
                applies when ``input`` is a string. Defaults to ``None`` (use the cfg setting).
            padding_side (Union[Literal["left", "right"], None], optional): Specifies which
                side to pad when tokenizing multiple strings of different lengths.
        """
        if isinstance(input, str):
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            tokens = input
        if len(tokens.shape) == 2:
            assert (
                tokens.shape[0] == 1
            ), f"If tokens are rank two, they must have shape [1, seq_len], not {tokens.shape}"
            tokens = tokens[0]
        if isinstance(single_token, str):
            single_token = self.to_single_token(single_token)
        elif isinstance(single_token, torch.Tensor):
            single_token = single_token.item()
        indices = torch.arange(len(tokens), device=tokens.device)[tokens == single_token]
        assert len(indices) > 0, "The token does not occur in the prompt"
        if mode == "first":
            return indices[0].item()
        elif mode == "last":
            return indices[-1].item()
        else:
            raise ValueError(f"mode must be 'first' or 'last', not {mode}")

    def to_single_str_token(self, int_token: int) -> str:
        """Get the single token corresponding to an int in string form.

        Args:
            int_token: The token ID

        Returns:
            The token string
        """
        assert isinstance(int_token, int)
        token = self.to_str_tokens(torch.tensor([int_token]))
        if isinstance(token, list) and len(token) == 1:
            return str(token[0])
        raise AssertionError("Expected a single string token.")

    def blocks_with(self, submodule: str) -> List[Tuple[int, "GeneralizedComponent"]]:
        """Return (index, block) pairs for blocks with the named bridged submodule.

        Checks _modules (not hasattr) so HF-internal attrs don't match.
        Use instead of assuming blocks[0] is representative on hybrid models.
        """
        if not hasattr(self, "blocks"):
            return []
        return [(i, block) for i, block in enumerate(self.blocks) if submodule in block._modules]

    def stack_params_for(
        self, submodule: str, attr_path: str, reshape_fn: Optional[Callable] = None
    ) -> Tuple[List[int], torch.Tensor]:
        """Stack a parameter across matching blocks only. Returns (layer_indices, tensor).

        Use for hybrid models where not all blocks have the submodule.
        """
        matching = self.blocks_with(submodule)
        if not matching:
            raise ValueError(
                f"No blocks have submodule '{submodule}'. "
                f"Available submodules can be checked with blocks_with()."
            )
        indices: List[int] = []
        weights: List[torch.Tensor] = []
        for idx, block in matching:
            w = _resolve_attr_path(block, attr_path)
            if reshape_fn is not None:
                w = reshape_fn(w)
            weights.append(w)
            indices.append(idx)
        return indices, torch.stack(weights, dim=0)

    def _stack_block_params(
        self, attr_path: str, reshape_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """Stack a parameter across all blocks; falls back to matching-only on hybrids.

        On hybrid models, logs a warning about index mapping and returns only
        blocks that have the submodule. First path segment is checked against
        _modules; deeper segments resolve via getattr (intentional — W_Q etc.
        are exposed via __getattr__ delegation).
        """
        first_attr = attr_path.split(".")[0]
        matching_blocks = [
            (i, block) for i, block in enumerate(self.blocks) if first_attr in block._modules
        ]

        if len(matching_blocks) == 0:
            raise AttributeError(
                f"No blocks have submodule '{first_attr}'. "
                f"Use bridge.blocks_with('{first_attr}') to check availability."
            )

        if len(matching_blocks) < len(self.blocks):
            indices = [i for i, _ in matching_blocks]
            logging.warning(
                "Hybrid model: only %d/%d blocks have '%s'. Returning stacked tensor "
                "for layers %s only. Tensor index i corresponds to original layer "
                "indices[i], not layer i. For explicit index mapping, use "
                "bridge.stack_params_for('%s', '%s').",
                len(matching_blocks),
                len(self.blocks),
                first_attr,
                indices,
                first_attr,
                attr_path,
            )

        weights: List[torch.Tensor] = []
        for _, block in matching_blocks:
            w = _resolve_attr_path(block, attr_path)
            if reshape_fn is not None:
                w = reshape_fn(w)
            weights.append(w)
        # Under a device_map split, per-block tensors live on different devices.
        # torch.stack requires a common device; gather onto cfg.device (the embedding /
        # input device — a natural "home" for cross-layer reductions).
        if getattr(self.cfg, "n_devices", 1) > 1 and weights and self.cfg.device is not None:
            target_device = torch.device(self.cfg.device)
            weights = [w.to(target_device) for w in weights]
        return torch.stack(weights, dim=0)

    def _reshape_qkv(self, w: torch.Tensor) -> torch.Tensor:
        """Reshape 2D [d_model, d_model] QKV weight to 3D [n_heads, d_model, d_head]."""
        if w.shape == (self.cfg.d_model, self.cfg.d_model):
            d_head = self.cfg.d_model // self.cfg.n_heads
            return w.reshape(self.cfg.n_heads, self.cfg.d_model, d_head)
        return w

    def _reshape_o(self, w: torch.Tensor) -> torch.Tensor:
        """Reshape 2D [d_model, d_model] O weight to 3D [n_heads, d_head, d_model]."""
        if w.shape == (self.cfg.d_model, self.cfg.d_model):
            d_head = self.cfg.d_model // self.cfg.n_heads
            return w.reshape(self.cfg.n_heads, d_head, self.cfg.d_model)
        return w

    @property
    def W_K(self) -> torch.Tensor:
        """Stack the key weights across all layers."""
        return self._stack_block_params("attn.W_K", self._reshape_qkv)

    @property
    def W_Q(self) -> torch.Tensor:
        """Stack the query weights across all layers."""
        return self._stack_block_params("attn.W_Q", self._reshape_qkv)

    @property
    def W_V(self) -> torch.Tensor:
        """Stack the value weights across all layers."""
        return self._stack_block_params("attn.W_V", self._reshape_qkv)

    @property
    def W_O(self) -> torch.Tensor:
        """Stack the attn output weights across all layers."""
        return self._stack_block_params("attn.W_O", self._reshape_o)

    @property
    def W_in(self) -> torch.Tensor:
        """Stack the MLP input weights across all layers."""
        return self._stack_block_params("mlp.W_in")

    @property
    def W_gate(self) -> Union[torch.Tensor, None]:
        """Stack the MLP gate weights across all layers (gated MLPs only)."""
        if getattr(self.cfg, "gated_mlp", False):
            return self._stack_block_params("mlp.W_gate")
        return None

    @property
    def W_out(self) -> torch.Tensor:
        """Stack the MLP output weights across all layers."""
        return self._stack_block_params("mlp.W_out")

    @property
    def b_K(self) -> torch.Tensor:
        """Stack the key biases across all layers."""
        return self._stack_block_params("attn.b_K")

    @property
    def b_Q(self) -> torch.Tensor:
        """Stack the query biases across all layers."""
        return self._stack_block_params("attn.b_Q")

    @property
    def b_V(self) -> torch.Tensor:
        """Stack the value biases across all layers."""
        return self._stack_block_params("attn.b_V")

    @property
    def b_O(self) -> torch.Tensor:
        """Stack the attn output biases across all layers."""
        return self._stack_block_params("attn.b_O")

    @property
    def b_in(self) -> torch.Tensor:
        """Stack the MLP input biases across all layers."""
        return self._stack_block_params("mlp.b_in")

    @property
    def b_out(self) -> torch.Tensor:
        """Stack the MLP output biases across all layers."""
        return self._stack_block_params("mlp.b_out")

    @property
    def W_U(self) -> torch.Tensor:
        """Unembedding matrix (d_model, d_vocab). Maps residual stream to logits."""
        return self.unembed.W_U

    @property
    def b_U(self) -> torch.Tensor:
        """Unembedding bias (d_vocab)."""
        return self.unembed.b_U

    @property
    def W_E(self) -> torch.Tensor:
        """Token embedding matrix (d_vocab, d_model)."""
        return self.embed.W_E

    @property
    def QK(self):
        """QK circuit. On hybrids, returns attn layers only (with warning). See QK_for_attn_layers()."""
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self):
        """OV circuit. On hybrids, returns attn layers only (with warning). See OV_for_attn_layers()."""
        return FactoredMatrix(self.W_V, self.W_O)

    def QK_for_attn_layers(self) -> Tuple[List[int], FactoredMatrix]:
        """QK circuit for attention layers only. Returns (layer_indices, FactoredMatrix)."""
        q_indices, W_Q = self.stack_params_for("attn", "attn.W_Q", self._reshape_qkv)
        _, W_K = self.stack_params_for("attn", "attn.W_K", self._reshape_qkv)
        return q_indices, FactoredMatrix(W_Q, W_K.transpose(-2, -1))

    def OV_for_attn_layers(self) -> Tuple[List[int], FactoredMatrix]:
        """OV circuit for attention layers only. Returns (layer_indices, FactoredMatrix)."""
        v_indices, W_V = self.stack_params_for("attn", "attn.W_V", self._reshape_qkv)
        _, W_O = self.stack_params_for("attn", "attn.W_O", self._reshape_o)
        return v_indices, FactoredMatrix(W_V, W_O)

    # ------------------------------------------------------------------
    # Mechanistic interpretability analysis methods
    # ------------------------------------------------------------------

    def tokens_to_residual_directions(
        self,
        tokens: Union[str, int, torch.Tensor],
    ) -> torch.Tensor:
        """Map tokens to their unembedding vectors (residual stream directions).

        Returns the columns of W_U corresponding to the given tokens — i.e. the
        directions in the residual stream that the model dots with to produce the
        logit for each token.

        WARNING: If you use this without folding in LayerNorm (compatibility mode),
        the results will be misleading because LN weights change the unembed map.

        Args:
            tokens: A single token (str, int, or scalar tensor), a 1-D tensor of
                token IDs, or a 2-D batch of token IDs.

        Returns:
            Tensor of unembedding vectors with shape matching the input token shape
            plus a trailing d_model dimension.
        """
        if isinstance(tokens, torch.Tensor) and tokens.numel() > 1:
            residual_directions = self.W_U[:, tokens]
            residual_directions = einops.rearrange(
                residual_directions, "d_model ... -> ... d_model"
            )
            return residual_directions
        else:
            if isinstance(tokens, str):
                token = self.to_single_token(tokens)
            elif isinstance(tokens, int):
                token = tokens
            elif isinstance(tokens, torch.Tensor) and tokens.numel() == 1:
                token = int(tokens.item())
            else:
                raise ValueError(f"Invalid token type: {type(tokens)}")
            residual_direction = self.W_U[:, token]
            return residual_direction

    # Variant → attr paths for the output bias that feeds the residual stream.
    _VARIANT_OUTPUT_BIAS_ATTRS: Dict[str, tuple] = {
        "attn": ("b_O",),
        "linear_attn": ("out_proj.bias",),
        "mamba": ("out_proj.bias",),
        "mixer": ("out_proj.bias",),
        "ssm": ("out_proj.bias",),
    }

    def _get_block_variant_bias(self, block: "GeneralizedComponent") -> Optional[torch.Tensor]:
        """Return the output bias from this block's variant submodule, or None."""
        for name in VARIANT_SUBMODULE_NAMES:
            if name not in block._modules:
                continue
            variant = block._modules[name]
            for attr_path in self._VARIANT_OUTPUT_BIAS_ATTRS.get(name, ()):
                obj = variant
                try:
                    for attr in attr_path.split("."):
                        obj = getattr(obj, attr)
                except AttributeError:
                    continue
                if obj is not None and isinstance(obj, torch.Tensor):
                    return obj
        return None

    def accumulated_bias(
        self,
        layer: int,
        mlp_input: bool = False,
        include_mlp_biases: bool = True,
    ) -> torch.Tensor:
        """Sum of variant + MLP output biases through the residual stream up to `layer`.

        Includes all layer types (attn, SSM, linear-attn). Set mlp_input=True
        to include the variant bias of the target layer itself.
        """
        accumulated = torch.zeros(self.cfg.d_model, device=self.cfg.device)
        for i in range(layer):
            block = self.blocks[i]
            b_O = self._get_block_variant_bias(block)
            if b_O is not None:
                accumulated = accumulated + b_O.to(accumulated.device)
            if include_mlp_biases and "mlp" in block._modules:
                b_out = getattr(block.mlp, "b_out", None)
                if b_out is not None:
                    accumulated = accumulated + b_out.to(accumulated.device)
        if mlp_input:
            assert layer < self.cfg.n_layers, "Cannot include attn_bias from beyond the final layer"
            block = self.blocks[layer]
            b_O = self._get_block_variant_bias(block)
            if b_O is not None:
                accumulated = accumulated + b_O.to(accumulated.device)
        return accumulated

    def all_composition_scores(self, mode: str) -> CompositionScores:
        """Composition scores for all attention head pairs. Returns CompositionScores.

        See https://transformer-circuits.pub/2021/framework/index.html
        On hybrid models, only attention layers are included; layer_indices
        maps tensor position i to original layer number.
        """
        attn_blocks = self.blocks_with("attn")
        if not attn_blocks:
            raise ValueError("No attention layers found — cannot compute composition scores.")

        indices = [idx for idx, _ in attn_blocks]
        blocks_list = [block for _, block in attn_blocks]

        def _stack(attr_path: str, reshape_fn: Optional[Callable] = None) -> torch.Tensor:
            weights: List[torch.Tensor] = []
            for block in blocks_list:
                w = _resolve_attr_path(block, attr_path)
                if reshape_fn is not None:
                    w = reshape_fn(w)
                weights.append(w)
            # See _stack_block_params: gather per-block tensors onto cfg.device when split.
            if getattr(self.cfg, "n_devices", 1) > 1 and weights and self.cfg.device is not None:
                target_device = torch.device(self.cfg.device)
                weights = [w.to(target_device) for w in weights]
            return torch.stack(weights, dim=0)

        W_V = _stack("attn.W_V", self._reshape_qkv)
        W_O = _stack("attn.W_O", self._reshape_o)
        left = FactoredMatrix(W_V, W_O)

        if mode == "Q":
            W_Q = _stack("attn.W_Q", self._reshape_qkv)
            W_K = _stack("attn.W_K", self._reshape_qkv)
            right = FactoredMatrix(W_Q, W_K.transpose(-2, -1))
        elif mode == "K":
            W_Q = _stack("attn.W_Q", self._reshape_qkv)
            W_K = _stack("attn.W_K", self._reshape_qkv)
            right = FactoredMatrix(W_Q, W_K.transpose(-2, -1)).T
        elif mode == "V":
            right = left
        else:
            raise ValueError(f"mode must be one of ['Q', 'K', 'V'] not {mode}")

        scores = utils.composition_scores(left, right, broadcast_dims=True)
        n_attn = len(indices)
        idx_tensor = torch.arange(n_attn, device=self.cfg.device)
        mask = idx_tensor[:, None, None, None] < idx_tensor[None, None, :, None]
        scores = torch.where(mask, scores, torch.zeros_like(scores))

        labels = [f"L{l}H{h}" for l in indices for h in range(self.cfg.n_heads)]
        return CompositionScores(scores=scores, layer_indices=indices, head_labels=labels)

    def composition_layer_indices(self) -> List[int]:
        """Original layer indices for attention layers (maps composition score positions)."""
        return [idx for idx, _ in self.blocks_with("attn")]

    def block_hooks(self, layer_idx: int) -> List[str]:
        """Sorted hook names available on block `layer_idx` (block-relative paths)."""
        prefix = f"blocks.{layer_idx}."
        return sorted(name[len(prefix) :] for name in self.hook_dict if name.startswith(prefix))

    def block_submodules(self, layer_idx: int) -> List[str]:
        """Return bridged submodule names on block `layer_idx`."""
        block = self.blocks[layer_idx]
        return [name for name in block._modules if name not in _BLOCK_INTERNAL_MODULES]

    def layer_types(self) -> List[str]:
        """Per-block type labels, e.g. ["attn+mlp", "ssm+mlp", ...]. Deterministic order."""
        types = []
        for block in self.blocks:
            variants = [n for n in VARIANT_SUBMODULE_NAMES if n in block._modules]
            universals = sorted(
                n
                for n in block._modules
                if n not in _VARIANT_SUBMODULE_SET
                and n not in _BLOCK_INTERNAL_MODULES
                and not n.startswith(_NORM_PREFIXES)
            )
            parts = variants + universals
            types.append("+".join(parts) if parts else "unknown")
        return types

    @property
    def all_head_labels(self) -> list[str]:
        """Human-readable labels for all attention heads, e.g. ['L0H0', 'L0H1', ...]."""
        return [f"L{l}H{h}" for l in range(self.cfg.n_layers) for h in range(self.cfg.n_heads)]

    @property
    def attn_head_labels(self) -> list[str]:
        """Head labels for attention layers only — matches all_composition_scores() dims."""
        return [
            f"L{l}H{h}" for l in self.composition_layer_indices() for h in range(self.cfg.n_heads)
        ]

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Returns parameters following standard PyTorch semantics.

        This method delegates to the underlying HuggingFace model's parameters().
        For TransformerLens-style parameter generator, use tl_parameters() instead.

        Args:
            recurse: If True, yields parameters of this module and all submodules

        Returns:
            Iterator of nn.Parameter objects
        """
        return self.original_model.parameters(recurse=recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Returns named parameters following standard PyTorch semantics.

        This method delegates to the underlying HuggingFace model's named_parameters().
        For TransformerLens-style generator, use tl_named_parameters() instead.

        Args:
            prefix: Prefix to prepend to all parameter names
            recurse: If True, yields parameters of this module and all submodules
            remove_duplicate: If True, removes duplicate parameters

        Returns:
            Iterator of (name, parameter) tuples
        """
        return self.original_model.named_parameters(prefix, recurse, remove_duplicate)

    def tl_parameters(self) -> dict[str, torch.Tensor]:
        """Returns TransformerLens-style parameter dictionary.

        Parameter names follow TransformerLens conventions (e.g., 'blocks.0.attn.W_Q') and may
        include processed weights (non-leaf tensors). This format is expected by SVDInterpreter
        among other analysis tools.

        Returns:
            Dictionary mapping TransformerLens parameter names to tensors

        Example:
            >>> bridge = TransformerBridge.boot_transformers("gpt2")
            >>> tl_params = bridge.tl_parameters()
            >>> W_Q = tl_params["blocks.0.attn.W_Q"]  # Shape: [n_heads, d_model, d_head]
        """
        return self.get_params()

    def tl_named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Returns iterator of TransformerLens-style named parameters.

        This provides the same parameters as tl_parameters() but as an iterator
        for consistency with PyTorch's named_parameters() API pattern.

        Returns:
            Iterator of (name, tensor) tuples with TransformerLens naming conventions

        Example:
            >>> bridge = TransformerBridge.boot_transformers("gpt2")
            >>> for name, param in bridge.tl_named_parameters():
            ...     if "attn.W_Q" in name:
            ...         print(f"{name}: {param.shape}")  # doctest: +ELLIPSIS
            blocks.0.attn.W_Q: torch.Size([12, 768, 64])
            ...
        """
        return iter(self.get_params().items())

    def forward(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_type: Optional[str] = "logits",
        loss_per_token: bool = False,
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_at_layer: Optional[int] = None,
        stop_at_layer: Optional[int] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        """Forward pass through the model.

        Args:
            input: Input to the model
            return_type: Type of output to return ('logits', 'loss', 'both', 'predictions', None)
            loss_per_token: Whether to return loss per token
            prepend_bos: Whether to prepend BOS token
            padding_side: Which side to pad on
            start_at_layer: Not implemented in TransformerBridge. The bridge delegates
                to HuggingFace's model.forward() which owns the layer iteration loop,
                making start_at_layer infeasible without monkey-patching HF internals
                (fragile across HF versions) or exception-based layer skipping (corrupts
                model state). Raises NotImplementedError if a non-None value is passed.
            stop_at_layer: Layer to stop forward pass at
            pixel_values: Optional image tensor for multimodal models (e.g., LLaVA, Gemma3).
                The tensor is passed directly to the underlying HuggingFace model.
                Only valid when cfg.is_multimodal is True.
            input_values: Optional audio waveform tensor for audio models (e.g., HuBERT).
                The tensor is passed directly to the underlying HuggingFace model.
                Only valid when cfg.is_audio_model is True.
            **kwargs: Additional arguments passed to model

        Returns:
            Model output based on return_type
        """

        if start_at_layer is not None:
            raise NotImplementedError(
                "start_at_layer is not supported in TransformerBridge. "
                "The bridge delegates to HuggingFace's model.forward() which controls "
                "the layer iteration loop. See the TransformerBridge review plan for a "
                "detailed analysis of implementation approaches and their tradeoffs."
            )

        # Set stop_at_layer flag on all blocks if requested
        if stop_at_layer is not None and hasattr(self, "blocks"):
            for block in self.blocks:
                block._stop_at_layer_idx = stop_at_layer

        # Map HookedEncoderDecoder-style kwargs to HF-compatible names
        if "decoder_input" in kwargs:
            kwargs["decoder_input_ids"] = kwargs.pop("decoder_input")
        if "one_zero_attention_mask" in kwargs:
            if attention_mask is None:
                attention_mask = kwargs.pop("one_zero_attention_mask")
            else:
                kwargs.pop("one_zero_attention_mask")

        # Detect batched list input that will need padding. For this case we force
        # left-padding internally and auto-compute attention_mask + position_ids
        # (unless the caller passed them explicitly) so pad tokens don't contaminate
        # attention or position embeddings.
        _is_batched_list = (
            isinstance(input, list)
            and len(input) > 1
            and not getattr(self.cfg, "is_audio_model", False)
        )

        try:
            if isinstance(input, (str, list)):
                if getattr(self.cfg, "is_audio_model", False):
                    raise ValueError(
                        "Audio models require tensor input (raw waveform), not text. "
                        "Pass a torch.Tensor or use the input_values parameter."
                    )
                if _is_batched_list and padding_side is None:
                    # Force left-padding so real tokens are flush-right.
                    _orig_padding_side = self.tokenizer.padding_side
                    self.tokenizer.padding_side = "left"
                    try:
                        input_ids = self.to_tokens(
                            input, prepend_bos=prepend_bos, padding_side=padding_side
                        )
                    finally:
                        self.tokenizer.padding_side = _orig_padding_side
                else:
                    input_ids = self.to_tokens(
                        input, prepend_bos=prepend_bos, padding_side=padding_side
                    )
            else:
                input_ids = input
                # Promote 1D integer token tensors to 2D [batch=1, seq] to match
                # HookedTransformer's contract. Float tensors (inputs_embeds,
                # audio waveforms) are passed through unchanged.
                if (
                    isinstance(input_ids, torch.Tensor)
                    and input_ids.ndim == 1
                    and not input_ids.is_floating_point()
                ):
                    input_ids = input_ids.unsqueeze(0)

            # Detect inputs_embeds: if the tensor is floating point, it's pre-computed
            # embeddings (e.g., from multimodal models) rather than token IDs.
            _is_inputs_embeds = (
                isinstance(input_ids, torch.Tensor) and input_ids.is_floating_point()
            )

            # Auto-compute attention_mask + position_ids for batched list input
            # when the caller didn't supply them. Matches HF generation convention.
            if (
                _is_batched_list
                and attention_mask is None
                and self.tokenizer is not None
                and self.tokenizer.pad_token_id is not None
                and not _is_inputs_embeds
            ):
                _prev_side = self.tokenizer.padding_side
                self.tokenizer.padding_side = "left"
                try:
                    attention_mask = utils.get_attention_mask(
                        self.tokenizer,
                        input_ids,
                        prepend_bos=getattr(self.cfg, "default_prepend_bos", True),
                    ).to(self.cfg.device)
                finally:
                    self.tokenizer.padding_side = _prev_side
                if "position_ids" not in kwargs:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    kwargs["position_ids"] = position_ids

            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            if kwargs.pop("use_past_kv_cache", False) or kwargs.get("use_cache", False):
                kwargs["use_cache"] = True
            # Auto-generate decoder_input_ids for encoder-decoder models
            if (
                "decoder_input_ids" not in kwargs
                and hasattr(self.original_model, "config")
                and getattr(self.original_model.config, "is_encoder_decoder", False)
            ):
                decoder_start_token_id = getattr(
                    self.original_model.config, "decoder_start_token_id", None
                )
                if decoder_start_token_id is not None:
                    shifted = input_ids[:, :-1]
                    start_tokens = torch.full(
                        (input_ids.shape[0], 1),
                        decoder_start_token_id,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                    kwargs["decoder_input_ids"] = torch.cat([start_tokens, shifted], dim=1)
                else:
                    kwargs["decoder_input_ids"] = input_ids

            # Tell PosEmbedBridge to expand batch=1 position_ids to full batch.
            if hasattr(self, "pos_embed"):
                self.pos_embed._current_batch_size = input_ids.shape[0]

            # Handle pixel_values for multimodal models
            if pixel_values is not None:
                if not getattr(self.cfg, "is_multimodal", False):
                    raise ValueError(
                        "pixel_values can only be passed to multimodal models "
                        "(cfg.is_multimodal must be True)"
                    )
                kwargs["pixel_values"] = pixel_values

            # Handle input_values for audio models
            if input_values is not None:
                if not getattr(self.cfg, "is_audio_model", False):
                    raise ValueError(
                        "input_values can only be passed to audio models "
                        "(cfg.is_audio_model must be True)"
                    )
                kwargs["input_values"] = input_values

            # Audio models use input_values (waveform), not input_ids
            if getattr(self.cfg, "is_audio_model", False):
                if input_values is not None:
                    result = self._driver.forward(**kwargs)
                elif isinstance(input, torch.Tensor):
                    kwargs["input_values"] = input
                    result = self._driver.forward(**kwargs)
                else:
                    raise ValueError(
                        "Audio models require tensor input (raw waveform). "
                        "Pass a torch.Tensor or use input_values parameter."
                    )
            elif _is_inputs_embeds:
                result = self._driver.forward(inputs_embeds=input_ids, **kwargs)
            else:
                # By keyword so kw-only ``input_ids`` drivers don't TypeError.
                result = self._driver.forward(input_ids=input_ids, **kwargs)
            output = result.raw_output
            # No-op for HF (its hooks already fired); load-bearing for vLLM/Inspect.
            if result.captured:
                self._replay_captures(result.captured)
            # Convert TensorLike to torch at the boundary; let weird shapes
            # (audio/CTC dataclasses, tuple-of-tuples) pass through unchanged —
            # downstream return_type branches catch them with specific errors.
            logits = result.logits
            if isinstance(logits, torch.Tensor):
                pass
            elif logits is not None and isinstance(logits, TensorLike):
                logits = to_torch(logits)
            # Stash only the cache object (not the full output) for generate().
            if getattr(self, "_capture_hf_cache", False):
                self._last_hf_cache = getattr(output, "past_key_values", None)
            return self._finalize_return(
                return_type,
                logits,
                input_ids,
                is_audio_model=getattr(self.cfg, "is_audio_model", False),
                inputs_embeds_was_used=_is_inputs_embeds,
                loss_per_token=loss_per_token,
            )
        except StopAtLayerException as e:
            # Execution stopped at the requested layer
            return e.layer_output
        finally:
            # Clean up state that may be inconsistent after StopAtLayerException
            if stop_at_layer is not None and hasattr(self, "blocks"):
                # Reset the stop flag on all blocks
                for block in self.blocks:
                    block._stop_at_layer_idx = None

                # Clear any stale KV cache — layers after the stop point didn't
                # execute, so the cache is incomplete and would corrupt subsequent
                # generate() calls that expect a full cache.
                if hasattr(self, "_last_hf_cache"):
                    del self._last_hf_cache

    # loss_fn inherited from BridgeCore

    def _generate_tokens(
        self,
        current_tokens: torch.Tensor,
        input_tokens: torch.Tensor,
        batch_size: int,
        *,
        max_new_tokens: int,
        do_sample: bool,
        top_k: Optional[int],
        top_p: Optional[float],
        temperature: float,
        freq_penalty: float,
        repetition_penalty: float,
        stop_at_eos: bool,
        stop_tokens: List[int],
        eos_token_for_padding: int,
        finished_sequences: torch.Tensor,
        use_past_kv_cache: bool,
        use_stateful_cache: bool,
        mamba_cache: Any,
        mamba_conv_kernel: int,
        is_encoder_decoder: bool,
        _is_batched_list: bool,
        _generate_from_embeds: bool,
        encoder_input: Optional[torch.Tensor],
        decoder_tokens: Optional[torch.Tensor],
        generated_token_ids: Optional[List[torch.Tensor]],
        pixel_values: Optional[torch.Tensor],
        multimodal_kwargs: Dict[str, Any],
        verbose: bool,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, bool], None, None]:
        """Core generation loop. Yields (sampled_tokens, final_logits, all_finished) per step.

        Owns the forward pass, sampling, EOS handling, token accumulation, and
        KV cache management. Callers are responsible for try/finally cleanup of
        ``_capture_hf_cache``.
        """
        _hf_kv_cache = None

        for gen_step_idx in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
            with torch.no_grad():
                if is_encoder_decoder:
                    logits = self(
                        encoder_input,
                        return_type="logits",
                        decoder_input=decoder_tokens,
                    )
                else:
                    forward_kwargs: Dict[str, Any] = {}
                    # Compute attention mask and position_ids for batched
                    # inputs with padding.
                    if (
                        _is_batched_list
                        and self.tokenizer is not None
                        and self.tokenizer.pad_token_id is not None
                    ):
                        _prev_side = self.tokenizer.padding_side
                        self.tokenizer.padding_side = "left"
                        attn_mask = utils.get_attention_mask(
                            self.tokenizer,
                            current_tokens,
                            prepend_bos=getattr(self.cfg, "default_prepend_bos", True),
                        ).to(self.cfg.device)
                        self.tokenizer.padding_side = _prev_side
                        forward_kwargs["attention_mask"] = attn_mask
                        position_ids = attn_mask.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attn_mask == 0, 1)
                        forward_kwargs["position_ids"] = position_ids
                    if gen_step_idx == 0:
                        if pixel_values is not None:
                            forward_kwargs["pixel_values"] = pixel_values
                        if multimodal_kwargs:
                            forward_kwargs.update(multimodal_kwargs)
                    if use_stateful_cache:
                        forward_kwargs["cache_params"] = mamba_cache
                        forward_kwargs["use_cache"] = True
                        if gen_step_idx == 0:
                            cache_position = torch.arange(
                                0, mamba_conv_kernel, device=self.cfg.device
                            )
                            forward_kwargs["cache_position"] = cache_position
                            logits = self(
                                current_tokens,
                                return_type="logits",
                                **forward_kwargs,
                            )
                        else:
                            input_seq_pos = input_tokens.shape[1] + gen_step_idx - 1
                            cache_position = torch.tensor([input_seq_pos], device=self.cfg.device)
                            forward_kwargs["cache_position"] = cache_position
                            if "position_ids" in forward_kwargs:
                                forward_kwargs["position_ids"] = forward_kwargs["position_ids"][
                                    :, -1:
                                ]
                            logits = self(
                                current_tokens[:, -1:],
                                return_type="logits",
                                **forward_kwargs,
                            )
                    elif use_past_kv_cache:
                        forward_kwargs["use_cache"] = True
                        if _hf_kv_cache is not None:
                            forward_kwargs["past_key_values"] = _hf_kv_cache
                            if "position_ids" in forward_kwargs:
                                forward_kwargs["position_ids"] = forward_kwargs["position_ids"][
                                    :, -1:
                                ]
                            logits = self(
                                current_tokens[:, -1:],
                                return_type="logits",
                                **forward_kwargs,
                            )
                        else:
                            logits = self(
                                current_tokens,
                                return_type="logits",
                                **forward_kwargs,
                            )
                    else:
                        logits = self(current_tokens, return_type="logits", **forward_kwargs)
                    if use_past_kv_cache and hasattr(self, "_last_hf_cache"):
                        _hf_kv_cache = self._last_hf_cache or _hf_kv_cache
                        del self._last_hf_cache
                final_logits = logits[:, -1, :]

                # Sample next token
                penalty_tokens = (
                    torch.stack(generated_token_ids, dim=1)
                    if _generate_from_embeds and generated_token_ids
                    else None
                )
                if do_sample:
                    sampled_tokens = utils.sample_logits(
                        final_logits,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        freq_penalty=freq_penalty,
                        repetition_penalty=repetition_penalty,
                        tokens=penalty_tokens
                        if _generate_from_embeds
                        else (decoder_tokens if is_encoder_decoder else current_tokens),
                    ).to(self.cfg.device)
                else:
                    sampled_tokens = utils.sample_logits(
                        final_logits,
                        temperature=0.0,
                        repetition_penalty=repetition_penalty,
                        tokens=penalty_tokens
                        if _generate_from_embeds
                        else (decoder_tokens if is_encoder_decoder else current_tokens),
                    ).to(self.cfg.device)

                # Handle EOS
                if stop_at_eos:
                    sampled_tokens[finished_sequences] = eos_token_for_padding
                    finished_sequences.logical_or_(
                        torch.isin(
                            sampled_tokens.to(self.cfg.device),
                            torch.tensor(stop_tokens).to(self.cfg.device),
                        )
                    )

                # Update token sequences
                if is_encoder_decoder:
                    assert decoder_tokens is not None
                    decoder_tokens = torch.cat([decoder_tokens, sampled_tokens.unsqueeze(1)], dim=1)
                elif _generate_from_embeds:
                    assert generated_token_ids is not None
                    generated_token_ids.append(sampled_tokens)
                    embed_fn = self.original_model.get_input_embeddings()  # type: ignore[operator]
                    assert embed_fn is not None
                    new_embed = embed_fn(sampled_tokens.unsqueeze(1)).to(current_tokens.dtype)
                    current_tokens = torch.cat([current_tokens, new_embed], dim=1)
                else:
                    current_tokens = torch.cat([current_tokens, sampled_tokens.unsqueeze(1)], dim=1)

                all_finished = bool(stop_at_eos and finished_sequences.all().item())

            yield sampled_tokens, final_logits, all_finished

            if all_finished:
                return

    def generate(
        self,
        input: Union[str, List[str], torch.Tensor] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
        return_type: Optional[str] = "input",
        verbose: bool = True,
        output_logits: bool = False,
        pixel_values: Optional[torch.Tensor] = None,
        **multimodal_kwargs,
    ) -> str | list[str] | torch.Tensor | Any:  # Any for transformers.utils.ModelOutput
        # Any: beartype forward ref limitation (beartype#546)
        """Sample tokens from the model.

        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        This implementation is based on HookedTransformer.generate() to ensure consistent behavior.

        Args:
            input: Text string, list of strings, or tensor of tokens
            max_new_tokens: Maximum number of tokens to generate
            stop_at_eos: If True, stop generating tokens when the model outputs eos_token
            eos_token_id: The token ID to use for end of sentence
            do_sample: If True, sample from the model's output distribution. Otherwise, use greedy search
            top_k: Number of tokens to sample from. If None, sample from all tokens
            top_p: Probability mass to sample from. If 1.0, sample from all tokens
            temperature: Temperature for sampling. Higher values will make the model more random
            freq_penalty: Frequency penalty for sampling - how much to penalise previous tokens
            repetition_penalty: HuggingFace-style repetition penalty. Values > 1.0 discourage
                repetition by dividing positive logits and multiplying negative logits for
                previously seen tokens. Default 1.0 (no penalty).
            use_past_kv_cache: If True, use KV caching for faster generation
            prepend_bos: Accepted for API compatibility but not applied during generation.
                The HF model expects tokens in its native format (tokenizer defaults).
                Overriding BOS can silently degrade generation quality.
            padding_side: Which side to pad when tokenizing multiple strings of different
                lengths. For batched list inputs, left-padding is forced internally for
                correct generation behavior. Defaults to None (tokenizer default).
            return_type: The type of output to return - 'input', 'str', or 'tokens'
            verbose: Not used in Bridge (kept for API compatibility)
            output_logits: If True, return a ModelOutput with sequences and logits tuple
            pixel_values: Optional image tensor for multimodal models. Only passed on the
                first generation step (the vision encoder processes the image once, then
                embeddings are part of the token sequence for subsequent steps).

        Returns:
            Generated sequence as string, list of strings, or tensor depending on input type and return_type.
            If output_logits=True, returns a ModelOutput-like object with 'sequences' and 'logits' attributes.
        """
        # prepend_bos is intentionally not applied during generation.
        # The HF model expects tokens in its native format. Overriding BOS can silently
        # degrade quality.
        if prepend_bos is not None:
            import warnings

            warnings.warn(
                "prepend_bos is ignored during TransformerBridge.generate(). "
                "The HF model expects tokens with the tokenizer's default BOS handling. "
                "To control BOS, tokenize with to_tokens(prepend_bos=...) and pass the "
                "resulting tensor to generate().",
                stacklevel=2,
            )
        # padding_side is handled internally: for batched list inputs, left-padding
        # is forced to ensure correct generation. See _is_batched_list logic below.

        # Stateful dispatch is decided after input parsing so we can fall back
        # to hf_generate() for input types the stateful loop doesn't handle.
        is_stateful_model = getattr(self.cfg, "is_stateful", False)

        _is_batched_list = isinstance(input, list) and len(input) > 1

        _generate_from_embeds = False
        if isinstance(input, str):
            input_tokens = self.to_tokens(input, move_to_device=True, truncate=False)
            input_type = "str"
        elif isinstance(input, list):
            # Force left-padding for batched generation so real tokens are
            # flush-right and logits[:, -1, :] is always the last real token.
            if _is_batched_list:
                _orig_padding_side = self.tokenizer.padding_side
                self.tokenizer.padding_side = "left"
            input_tokens = self.to_tokens(input, move_to_device=True, truncate=False)
            if _is_batched_list:
                self.tokenizer.padding_side = _orig_padding_side
            input_type = "list"
        elif isinstance(input, torch.Tensor) and input.is_floating_point():
            # inputs_embeds: pre-computed embeddings (e.g., from multimodal models)
            input_tokens = input.to(self.cfg.device)
            input_type = "embeds"
            _generate_from_embeds = True
        else:
            input_tokens = input.to(self.cfg.device)
            input_type = "tokens"

        # Determine return type
        if return_type == "input":
            if input_type in ["str", "list"]:
                return_type = "str"
            elif input_type == "embeds":
                return_type = "tokens"
            else:
                return_type = "tokens"

        batch_size = input_tokens.shape[0]

        # Setup EOS token handling
        stop_tokens = []
        eos_token_for_padding = 0
        if stop_at_eos:
            tokenizer_has_eos_token = (
                self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            )
            if eos_token_id is None:
                assert (
                    tokenizer_has_eos_token
                ), "Must pass eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
                assert self.tokenizer is not None
                eos_token_id = self.tokenizer.eos_token_id

            if isinstance(eos_token_id, int):
                stop_tokens = [eos_token_id]
                eos_token_for_padding = eos_token_id
            else:
                stop_tokens = list(eos_token_id)
                if tokenizer_has_eos_token:
                    assert self.tokenizer is not None
                    eos_token_for_padding = self.tokenizer.eos_token_id
                else:
                    eos_token_for_padding = eos_token_id[0]

        # Track which sequences have finished
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)

        # Optionally collect logits at each generation step for downstream tooling/tests
        logits_seq_list: list[torch.Tensor] | None = [] if output_logits else None

        # Detect encoder-decoder models (T5, BART, etc.)
        is_encoder_decoder = hasattr(self.original_model, "config") and getattr(
            self.original_model.config, "is_encoder_decoder", False
        )

        # HF cache flows opaquely through the component chain via
        # _reconstruct_attention() → _update_kv_cache() on each layer.
        _hf_kv_cache = None
        if use_past_kv_cache and is_encoder_decoder:
            # Encoder-decoder models (T5, BART) don't support the opaque
            # cache path — silently disable rather than crash, since
            # use_past_kv_cache=True is the default.
            use_past_kv_cache = False

        # SSMs (Mamba/Mamba-2) run through a dedicated cache path so hooks
        # fire on every step. Unsupported input types fall back to hf_generate().
        use_stateful_cache = (
            is_stateful_model
            and use_past_kv_cache
            and not is_encoder_decoder
            and not _generate_from_embeds
            and pixel_values is None
            and not multimodal_kwargs
        )
        if is_stateful_model and not use_stateful_cache:
            hf_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
            }
            if top_k is not None:
                hf_kwargs["top_k"] = top_k
            if top_p is not None:
                hf_kwargs["top_p"] = top_p
            if eos_token_id is not None:
                hf_kwargs["eos_token_id"] = eos_token_id
            return self.hf_generate(input, **hf_kwargs)

        # SSM cache is built once and mutated in place across forward calls.
        # Adapter owns the cache-type choice; new SSMs just override
        # create_stateful_cache().
        mamba_cache: Any = None
        mamba_conv_kernel: int = 0
        if use_stateful_cache:
            hf_model: Any = self.original_model
            mamba_conv_kernel = int(getattr(hf_model.config, "conv_kernel", 4))
            cache_dtype = self.cfg.dtype or torch.float32
            mamba_cache = self.adapter.create_stateful_cache(
                hf_model=hf_model,
                batch_size=batch_size,
                device=self.cfg.device,
                dtype=cache_dtype,
            )

        if use_past_kv_cache and not use_stateful_cache:
            self._capture_hf_cache = True  # Signal forward() to stash cache

        # Generate tokens
        current_tokens = input_tokens.clone()
        # For inputs_embeds generation, also track generated token IDs for decoding
        if _generate_from_embeds:
            generated_token_ids: list[torch.Tensor] = []
        sampled_tokens_list = []

        # For encoder-decoder models, keep encoder input fixed and grow decoder input
        if is_encoder_decoder:
            encoder_input = input_tokens.clone()
            decoder_start_token_id = getattr(
                self.original_model.config, "decoder_start_token_id", 0
            )
            decoder_tokens = torch.full(
                (batch_size, 1),
                decoder_start_token_id,
                dtype=input_tokens.dtype,
                device=self.cfg.device,
            )

        try:
            for sampled_tokens, final_logits, all_finished in self._generate_tokens(
                current_tokens,
                input_tokens,
                batch_size,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                freq_penalty=freq_penalty,
                repetition_penalty=repetition_penalty,
                stop_at_eos=stop_at_eos,
                stop_tokens=stop_tokens,
                eos_token_for_padding=eos_token_for_padding,
                finished_sequences=finished_sequences,
                use_past_kv_cache=use_past_kv_cache,
                use_stateful_cache=use_stateful_cache,
                mamba_cache=mamba_cache,
                mamba_conv_kernel=mamba_conv_kernel,
                is_encoder_decoder=is_encoder_decoder,
                _is_batched_list=_is_batched_list,
                _generate_from_embeds=_generate_from_embeds,
                encoder_input=encoder_input if is_encoder_decoder else None,
                decoder_tokens=decoder_tokens if is_encoder_decoder else None,
                generated_token_ids=generated_token_ids if _generate_from_embeds else None,
                pixel_values=pixel_values,
                multimodal_kwargs=multimodal_kwargs if multimodal_kwargs else {},
                verbose=verbose,
            ):
                sampled_tokens_list.append(sampled_tokens.unsqueeze(1))
                if logits_seq_list is not None:
                    logits_seq_list.append(final_logits.clone())
                if all_finished:
                    break
        finally:
            self._capture_hf_cache = False
            if hasattr(self, "_last_hf_cache"):
                del self._last_hf_cache

        # Concatenate all sampled tokens
        sampled_tokens = torch.cat(sampled_tokens_list, dim=1)
        if is_encoder_decoder:
            # Reconstruct full decoder sequence: start token + generated tokens
            output_tokens = torch.cat([decoder_tokens[:, :1], sampled_tokens], dim=1)
        elif _generate_from_embeds:
            # For inputs_embeds, we only have the generated token IDs (no input token IDs)
            output_tokens = sampled_tokens
        else:
            output_tokens = torch.cat([input_tokens, sampled_tokens], dim=1)

        # Return ModelOutput if output_logits was requested
        if output_logits and logits_seq_list is not None:
            from transformers.utils import ModelOutput  # type: ignore

            def _logits_to_tuple(logits_list: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
                assert logits_list is not None
                # Convert list of [batch, vocab] tensors to tuple
                return tuple(logits_list)

            try:
                from transformers.generation.utils import GenerateDecoderOnlyOutput

                # Return a HF-compatible ModelOutput structure
                # GenerateDecoderOnlyOutput expects: sequences, scores (optional), logits (optional)
                return GenerateDecoderOnlyOutput(
                    sequences=cast(torch.LongTensor, output_tokens),
                    # HF's type hint says tuple[FloatTensor] but should be tuple[FloatTensor, ...]
                    # (variable-length tuple with one element per generated token)
                    logits=_logits_to_tuple(logits_seq_list),  # type: ignore[arg-type]
                )
            except (ImportError, AttributeError):
                # Fallback if GenerateDecoderOnlyOutput not available in this transformers version
                return ModelOutput(
                    sequences=output_tokens,
                    logits=_logits_to_tuple(logits_seq_list),
                )

        # Format output
        if return_type == "str":
            assert self.tokenizer is not None
            if input_type == "str":
                return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            else:
                decoded_texts = [
                    self.tokenizer.decode(tokens, skip_special_tokens=True)
                    for tokens in output_tokens
                ]
                return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts
        else:  # return_type == "tokens"
            return output_tokens

    @torch.no_grad()
    def generate_stream(
        self,
        input: Union[str, List[str], torch.Tensor] = "",
        max_new_tokens: int = 10,
        max_tokens_per_yield: int = 25,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
        return_type: Optional[str] = "input",
        verbose: bool = True,
    ) -> Generator[Union[torch.Tensor, str], None, None]:
        """Stream tokens from the model as they are generated.

        Yields batches of tokens progressively during generation rather than
        waiting for the entire sequence. Uses the same core loop as generate().

        Args:
            input: Text string, list of strings, or tensor of tokens.
            max_new_tokens: Maximum number of tokens to generate.
            max_tokens_per_yield: Yield accumulated tokens every this many steps.
            stop_at_eos: If True, stop when eos_token is produced.
            eos_token_id: Token ID(s) for end of sentence. Defaults to tokenizer's.
            do_sample: If True, sample; otherwise greedy.
            top_k: Top-k sampling. None means no filtering.
            top_p: Nucleus sampling threshold.
            temperature: Sampling temperature.
            freq_penalty: Frequency penalty for previous tokens.
            repetition_penalty: HF-style repetition penalty (>1.0 discourages repeats).
            use_past_kv_cache: Use KV caching for faster generation.
            prepend_bos: Not applied (API compatibility). See generate() docstring.
            padding_side: Which side to pad for batched list inputs. Left-padding
                is forced internally for batched generation.
            return_type: 'input' (match input type), 'str', or 'tokens'.
            verbose: Show progress bar.

        Yields:
            Token tensors [batch, seq_len] or strings, accumulated up to
            max_tokens_per_yield tokens between yields. First yield includes
            the input tokens; subsequent yields contain only new tokens.
        """
        if prepend_bos is not None:
            warnings.warn(
                "prepend_bos is ignored during TransformerBridge.generate_stream(). "
                "The HF model expects tokens with the tokenizer's default BOS handling.",
                stacklevel=2,
            )

        # --- Input parsing (mirrors generate()) ---
        _is_batched_list = isinstance(input, list) and len(input) > 1

        if isinstance(input, str):
            input_tokens = self.to_tokens(input, move_to_device=True, truncate=False)
            input_type = "str"
        elif isinstance(input, list):
            if _is_batched_list:
                _orig_ps = self.tokenizer.padding_side
                self.tokenizer.padding_side = "left"
            try:
                input_tokens = self.to_tokens(input, move_to_device=True, truncate=False)
            finally:
                if _is_batched_list:
                    self.tokenizer.padding_side = _orig_ps
            input_type = "list"
        else:
            input_tokens = input.to(self.cfg.device)
            input_type = "tokens"

        if return_type == "input":
            return_type = "str" if input_type in ["str", "list"] else "tokens"

        batch_size = input_tokens.shape[0]

        # --- EOS setup ---
        stop_tokens: List[int] = []
        eos_token_for_padding = 0
        if stop_at_eos:
            tokenizer_has_eos_token = (
                self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            )
            if eos_token_id is None:
                assert (
                    tokenizer_has_eos_token
                ), "Must pass eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
                assert self.tokenizer is not None
                eos_token_id = self.tokenizer.eos_token_id
            if isinstance(eos_token_id, int):
                stop_tokens = [eos_token_id]
                eos_token_for_padding = eos_token_id
            else:
                stop_tokens = list(eos_token_id)
                if tokenizer_has_eos_token:
                    assert self.tokenizer is not None
                    eos_token_for_padding = self.tokenizer.eos_token_id
                else:
                    eos_token_for_padding = eos_token_id[0]

        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)

        # --- Cache setup ---
        if use_past_kv_cache:
            self._capture_hf_cache = True

        current_tokens = input_tokens.clone()

        # --- Streaming loop ---
        # All yields are token tensors [batch, seq_len]. Each yield contains
        # only the newly generated tokens since the previous yield (the first
        # yield additionally prepends the input tokens for context).
        accumulated_tokens: Optional[torch.Tensor] = None
        tokens_since_last_yield = 0

        def _maybe_decode(
            tokens: torch.Tensor,
        ) -> Union[torch.Tensor, str]:
            if return_type == "str":
                assert self.tokenizer is not None
                return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
            return tokens

        try:
            for step_idx, (sampled_tokens, _, all_finished) in enumerate(
                self._generate_tokens(
                    current_tokens,
                    input_tokens,
                    batch_size,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    freq_penalty=freq_penalty,
                    repetition_penalty=repetition_penalty,
                    stop_at_eos=stop_at_eos,
                    stop_tokens=stop_tokens,
                    eos_token_for_padding=eos_token_for_padding,
                    finished_sequences=finished_sequences,
                    use_past_kv_cache=use_past_kv_cache,
                    use_stateful_cache=False,
                    mamba_cache=None,
                    mamba_conv_kernel=0,
                    is_encoder_decoder=False,
                    _is_batched_list=_is_batched_list,
                    _generate_from_embeds=False,
                    encoder_input=None,
                    decoder_tokens=None,
                    generated_token_ids=None,
                    pixel_values=None,
                    multimodal_kwargs={},
                    verbose=verbose,
                )
            ):
                new_tokens = sampled_tokens.unsqueeze(-1)

                if step_idx == 0:
                    accumulated_tokens = torch.cat([input_tokens, new_tokens], dim=-1)
                    tokens_since_last_yield = accumulated_tokens.shape[1]
                else:
                    if accumulated_tokens is None:
                        accumulated_tokens = new_tokens
                    else:
                        accumulated_tokens = torch.cat([accumulated_tokens, new_tokens], dim=-1)
                    tokens_since_last_yield += 1

                if tokens_since_last_yield >= max_tokens_per_yield:
                    yield _maybe_decode(accumulated_tokens)
                    tokens_since_last_yield = 0
                    accumulated_tokens = None

                if all_finished:
                    if accumulated_tokens is not None:
                        yield _maybe_decode(accumulated_tokens)
                    break

            # Yield remainder after loop completes without break
            if accumulated_tokens is not None:
                yield _maybe_decode(accumulated_tokens)
        finally:
            self._capture_hf_cache = False
            if hasattr(self, "_last_hf_cache"):
                del self._last_hf_cache

    def hf_generate(
        self,
        input: str | list[str] | torch.Tensor = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: int | None = None,
        do_sample: bool = True,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
        use_past_kv_cache: bool = True,
        return_type: str | None = "input",
        pixel_values: torch.Tensor | None = None,
        **generation_kwargs,
    ) -> str | list[str] | torch.Tensor | Any:  # Any for HF ModelOutput types
        # Any: beartype forward ref limitation (beartype#546)
        """Generate text using the underlying HuggingFace model with full HF API support.

        This method provides direct access to HuggingFace's generation API, forwarding all
        generation parameters (including output_scores, output_logits, output_attentions,
        output_hidden_states) directly to the underlying HF model. Use this when you need
        full HuggingFace generation features not supported by the standard generate() method.

        For standard generation compatible with HookedTransformer, use generate() instead.

        Args:
            input: Text string, list of strings, or tensor of tokens
            max_new_tokens: Maximum number of tokens to generate
            stop_at_eos: If True, stop generating tokens when the model outputs eos_token
            eos_token_id: The token ID to use for end of sentence
            do_sample: If True, sample from the model's output distribution
            top_k: Number of tokens to sample from
            top_p: Probability mass to sample from
            temperature: Temperature for sampling
            use_past_kv_cache: If True, use KV caching for faster generation
            return_type: The type of output to return - 'input', 'str', or 'tokens'
            **generation_kwargs: Additional HuggingFace generation parameters including:
                - output_scores: Return generation scores
                - output_logits: Return generation logits
                - output_attentions: Return attention weights
                - output_hidden_states: Return hidden states
                - return_dict_in_generate: Return ModelOutput object
                - And any other HF generation parameters

        Returns:
            Generated sequence as string, list of strings, tensor, or HF ModelOutput
            depending on input type, return_type, and generation_kwargs.

        Example::

            # Get full HF ModelOutput with logits and attentions
            from transformer_lens import HookedTransformer
            model = HookedTransformer.from_pretrained("tiny-stories-1M")
            result = model.hf_generate(
                "Hello world",
                max_new_tokens=5,
                output_logits=True,
                output_attentions=True,
                return_dict_in_generate=True
            )
            print(result.sequences)  # Generated tokens
            print(result.logits)  # Logits for each generation step
            print(result.attentions)  # Attention weights
        """
        # Handle string input by tokenizing it
        if isinstance(input, str):
            inputs = self.tokenizer(input, return_tensors="pt", padding=False, truncation=False).to(
                self.cfg.device
            )
            input_ids = inputs["input_ids"]
            input_type = "str"
        elif isinstance(input, list):
            inputs = self.tokenizer(input, return_tensors="pt", padding=True, truncation=False).to(
                self.cfg.device
            )
            input_ids = inputs["input_ids"]
            input_type = "list"
        else:
            input_ids = input
            if input_ids.device != self.cfg.device:
                input_ids = input_ids.to(self.cfg.device)
            input_type = "tokens"

        # Build generation_kwargs from explicit args and kwargs
        generation_kwargs = dict(generation_kwargs) if generation_kwargs is not None else {}
        generation_kwargs.update(
            {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
        )

        if top_k is not None:
            generation_kwargs["top_k"] = top_k
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        elif stop_at_eos and self.tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        if pixel_values is not None:
            generation_kwargs["pixel_values"] = pixel_values

        if use_past_kv_cache:
            generation_kwargs["use_cache"] = True

        # HF dict flags that trigger ModelOutput returns
        hf_dict_flags = (
            "output_scores",
            "output_logits",
            "output_attentions",
            "output_hidden_states",
        )

        # If any HF-style output flags are provided, ensure return_dict_in_generate is set
        any_flag_set = False
        for f in hf_dict_flags:
            if generation_kwargs.get(f) is not None:
                generation_kwargs[f] = bool(generation_kwargs[f])
                any_flag_set = True

        if any_flag_set:
            generation_kwargs.setdefault("return_dict_in_generate", True)

        # Generate using the original HuggingFace model
        with torch.no_grad():
            outputs = self.original_model.generate(input_ids, **generation_kwargs)  # type: ignore[operator]

        # Check if output is a ModelOutput
        try:
            from transformers.utils import ModelOutput  # type: ignore

            is_model_output = isinstance(outputs, ModelOutput)
        except Exception:
            is_model_output = False

        # Return based on return_type and input format
        if return_type == "input" or return_type is None:
            if input_type == "str":
                # Decode the full output back to string
                if is_model_output and hasattr(outputs, "sequences"):
                    return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif input_type == "list":
                # Decode each sequence in the batch
                if is_model_output and hasattr(outputs, "sequences"):
                    return [
                        self.tokenizer.decode(seq, skip_special_tokens=True)
                        for seq in outputs.sequences
                    ]
                return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
            else:
                # Return the full token sequence including input
                return outputs
        elif return_type == "tokens":
            return outputs
        else:
            # For other return types, default to the decoded text
            if input_type == "str":
                if is_model_output and hasattr(outputs, "sequences"):
                    return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif input_type == "list":
                if is_model_output and hasattr(outputs, "sequences"):
                    return [
                        self.tokenizer.decode(seq, skip_special_tokens=True)
                        for seq in outputs.sequences
                    ]
                return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
            else:
                return outputs

    def prepare_multimodal_inputs(
        self,
        text: Union[str, List[str]],
        images: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare multimodal inputs using the model's processor.

        Converts text and images into model-ready tensors (input_ids, pixel_values,
        attention_mask, etc.) using the HuggingFace processor loaded during boot().

        Args:
            text: Text prompt(s), typically containing image placeholder tokens
                (e.g., "<image>" for LLaVA).
            images: PIL Image or list of PIL Images to process. Pass None for
                text-only inputs on a multimodal model.

        Returns:
            Dictionary with 'input_ids', 'pixel_values', 'attention_mask', etc.
            All tensors are moved to the model's device.

        Raises:
            ValueError: If model is not multimodal or processor is not available.
        """
        if not getattr(self.cfg, "is_multimodal", False):
            raise ValueError(
                "prepare_multimodal_inputs() requires a multimodal model "
                "(cfg.is_multimodal must be True)"
            )
        if self.processor is None:
            raise ValueError(
                "No processor available. Load model with boot_transformers() or "
                "set bridge.processor = AutoProcessor.from_pretrained(...) manually."
            )
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        return {k: v.to(self.cfg.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def to(self, *args, **kwargs) -> "TransformerBridge":
        """Move model to device and/or change dtype.

        Args:
            args: Positional arguments for nn.Module.to
            kwargs: Keyword arguments for nn.Module.to
            print_details: Whether to print details about device/dtype changes (default: True)

        Returns:
            Self for chaining
        """
        # Extract print_details if provided
        print_details = kwargs.pop("print_details", True)

        # Handle both device and dtype changes
        # torch.nn.Module.to() supports: to(device), to(dtype), to(device, dtype),
        # to(device=...), to(dtype=...), to(device=..., dtype=...)
        target_device, target_dtype = None, None

        if len(args) >= 1:
            first_arg = args[0]
            if isinstance(first_arg, (torch.device, str)):
                target_device = first_arg
            elif isinstance(first_arg, torch.dtype):
                target_dtype = first_arg
        if len(args) >= 2:
            second_arg = args[1]
            if isinstance(second_arg, torch.dtype):
                target_dtype = second_arg

        # these override positional args
        if "device" in kwargs:
            target_device = kwargs["device"]
        if "dtype" in kwargs:
            target_dtype = kwargs["dtype"]

        # Moving a multi-device (device_map-dispatched) model to a single device would
        # collapse the split and break accelerate's hook routing. Warn and drop the
        # device move; still honor dtype changes.
        if target_device is not None and getattr(self.cfg, "n_devices", 1) > 1:
            warnings.warn(
                f"TransformerBridge.to({target_device!r}) ignored: model is dispatched "
                f"across {self.cfg.n_devices} devices via device_map. Reload with "
                "device=... (and no device_map/n_devices) to move to a single device.",
                stacklevel=2,
            )
            target_device = None

        if target_device is not None:
            move_to_and_update_config(self, target_device, print_details)
        if target_dtype is not None:
            move_to_and_update_config(self, target_dtype, print_details)

        # Move the original model with all original args/kwargs (with print_details removed).
        # When we've nulled target_device for multi-GPU safety, strip device args so the
        # underlying module isn't moved either.
        if target_device is None and (len(args) > 0 or "device" in kwargs):
            kwargs.pop("device", None)
            # Filter positional args: drop devices/strings, keep dtypes.
            args = tuple(a for a in args if not isinstance(a, (torch.device, str)))
        self.original_model = self.original_model.to(*args, **kwargs)
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> "TransformerBridge":
        """Move model to CUDA.

        Args:
            device: CUDA device

        Returns:
            Self for chaining
        """
        if isinstance(device, int):
            return self.to(f"cuda:{device}")
        elif device is None:
            return self.to("cuda")
        else:
            return self.to(device)

    def cpu(self) -> "TransformerBridge":
        """Move model to CPU.

        Returns:
            Self for chaining
        """
        return self.to(torch.device("cpu"))

    def mps(self) -> "TransformerBridge":
        """Move model to MPS.

        Returns:
            Self for chaining
        """
        return self.to(torch.device("mps"))

    def set_use_attn_result(self, use_attn_result: bool):
        """Toggle whether to explicitly calculate and expose the result for each attention head.

        Useful for interpretability but can easily burn through GPU memory.
        """
        if use_attn_result:
            self._validate_attention_fork_supported("use_attn_result")
        self.cfg.use_attn_result = use_attn_result
        self._propagate_attention_flag("use_attn_result", use_attn_result)

    def set_use_split_qkv_input(self, use_split_qkv_input: bool):
        """Toggle independent residual copies for Q/K/V so each path can be patched alone.

        Mutually exclusive with `use_attn_in` — set that flag off first if it's on.
        """
        if use_split_qkv_input:
            if bool(getattr(self.cfg, "use_attn_in", False)):
                raise ValueError(
                    "use_split_qkv_input and use_attn_in are mutually exclusive. "
                    "Call set_use_attn_in(False) before enabling use_split_qkv_input."
                )
            self._validate_attention_fork_supported("use_split_qkv_input")
        self.cfg.use_split_qkv_input = use_split_qkv_input
        self._propagate_attention_flag("use_split_qkv_input", use_split_qkv_input)

    def set_use_attn_in(self, use_attn_in: bool):
        """Toggle a single 4D residual copy feeding all three Q/K/V projections.

        Mutually exclusive with `use_split_qkv_input` — set that flag off first
        if it's on. When on, `hook_attn_in` fires at
        `[batch, pos, n_heads, d_model]`, enabling coarse-grained interventions
        on the residual-stream copy shared across Q/K/V.
        """
        if use_attn_in:
            if bool(getattr(self.cfg, "use_split_qkv_input", False)):
                raise ValueError(
                    "use_attn_in and use_split_qkv_input are mutually exclusive. "
                    "Call set_use_split_qkv_input(False) before enabling use_attn_in."
                )
            self._validate_attention_fork_supported("use_attn_in")
        self.cfg.use_attn_in = use_attn_in
        self._propagate_attention_flag("use_attn_in", use_attn_in)

    def _propagate_attention_flag(self, flag_name: str, value: bool) -> None:
        """Mirror `bridge.cfg.<flag>` onto every block's attention config.

        Some adapters (Llama family) deep-copy the block template during
        `setup_blocks_bridge`, cloning the attention bridge's config along
        with it. Others (Pythia, GPT-2) override `__deepcopy__` to share the
        config. Setting the flag only on `self.cfg` silently misses the
        cloned-config case. Propagating explicitly keeps both patterns
        honest — a no-op when configs are shared, a correctness fix when
        they aren't.
        """
        if not hasattr(self, "blocks"):
            return
        for block in self.blocks:
            attn = block._modules.get("attn") if hasattr(block, "_modules") else None
            if attn is None:
                continue
            attn_cfg = getattr(attn, "config", None)
            if attn_cfg is not None and attn_cfg is not self.cfg:
                try:
                    setattr(attn_cfg, flag_name, value)
                except Exception:
                    # Some cfg objects may be frozen/immutable. Skip silently —
                    # the block simply won't honor the flag, which is the
                    # same outcome as before this fix.
                    pass

    def _validate_attention_fork_supported(self, flag_name: str) -> None:
        """Raise / warn if the model can't honor a fine-grained attention flag.

        The post-ln1 fork path lives on JointQKVAttentionBridge and
        PositionEmbeddingsAttentionBridge. Plain AttentionBridge delegates to
        HF and exposes no fork point; we raise rather than setting the flag
        silently. For hybrid models (some attention layers, some not), we warn
        and list which layers will honor the flag.
        """
        # Deferred imports: tight circular dependency with bridge setup.
        from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
            JointQKVAttentionBridge,
        )
        from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
            PositionEmbeddingsAttentionBridge,
        )

        if not hasattr(self, "blocks"):
            raise NotImplementedError(
                f"{flag_name}: this bridge has no `blocks` attribute, so no "
                "attention bridges to apply the flag to."
            )
        supported_classes = (JointQKVAttentionBridge, PositionEmbeddingsAttentionBridge)
        supporting_layers: list[int] = []
        attn_classes: set[str] = set()
        total_with_attn = 0
        for idx, block in enumerate(self.blocks):
            attn = block._modules.get("attn") if hasattr(block, "_modules") else None
            if attn is None:
                continue
            total_with_attn += 1
            attn_classes.add(type(attn).__name__)
            if isinstance(attn, supported_classes):
                supporting_layers.append(idx)
        if total_with_attn == 0:
            raise NotImplementedError(f"{flag_name}: no attention bridges found on self.blocks.")
        if not supporting_layers:
            raise NotImplementedError(
                f"{flag_name}: none of this model's attention bridges support "
                "the fine-grained Q/K/V hook fork. Found attention classes: "
                f"{sorted(attn_classes)}. Supported classes: "
                f"{[c.__name__ for c in supported_classes]}. Plain "
                "AttentionBridge delegates to HuggingFace and exposes no hook "
                "point before the Q/K/V projection."
            )
        if len(supporting_layers) < total_with_attn:
            skipped = total_with_attn - len(supporting_layers)
            warnings.warn(
                f"{flag_name}: {skipped} of {total_with_attn} attention layers "
                "use an attention-bridge class that cannot honor this flag "
                f"(attention classes present: {sorted(attn_classes)}). "
                f"The flag will affect layers: {supporting_layers}.",
                stacklevel=3,
            )

    def _is_valid_bridge_path(self, hf_path: str) -> bool:
        """Check if a HuggingFace path corresponds to a valid bridge component.

        This validates that the path follows the bridge component structure and doesn't
        contain nested HuggingFace components that should have been wrapped.

        Args:
            hf_path: HuggingFace path after removing _original_component

        Returns:
            True if the path is valid, False if it contains nested HF components
        """
        # Split the path into parts
        parts = hf_path.split(".")

        # Get the component mapping for validation
        component_mapping = self.adapter.component_mapping
        if not component_mapping:
            return True  # If no mapping, accept all keys

        # Walk through the path and check if each level is a registered bridge component
        # For example, transformer.h.0.mlp.in.weight should be valid
        # but transformer.h.0.mlp.c_fc.weight should be invalid (c_fc is nested HF component)

        # Start from the root
        current_component = None
        idx = 0

        # Find which top-level component this belongs to
        for tl_name, component in component_mapping.items():
            if component.name and hf_path.startswith(component.name + "."):
                current_component = component
                # Skip past the HF prefix
                remaining_path = hf_path[len(component.name) + 1 :]
                parts = remaining_path.split(".")
                idx = 0
                break

        if current_component is None:
            return True  # Path doesn't match any component, let it through

        # Special handling for blocks
        if hasattr(current_component, "is_list_item") and current_component.is_list_item:
            # Skip the layer index
            if idx < len(parts) and parts[idx].isdigit():
                idx += 1

        # Now validate the rest of the path against submodules
        while idx < len(parts):
            part = parts[idx]

            # If we hit 'weight' or 'bias', we're at a parameter - this is valid
            if part in ("weight", "bias"):
                return True

            # Check if this part is a registered submodule
            if hasattr(current_component, "submodules") and current_component.submodules:
                if part in current_component.submodules:
                    current_component = current_component.submodules[part]
                    idx += 1
                    continue
                else:
                    # This part is not a registered bridge component
                    # It's likely a nested HF component (like c_fc, c_proj, c_attn)
                    return False
            else:
                # No submodules to check, but not at a parameter yet
                # Check if next is weight/bias
                if idx + 1 < len(parts) and parts[idx + 1] in ("weight", "bias"):
                    return True
                # Otherwise this is likely a nested HF component
                return False

            idx += 1

        return True

    def _normalize_bridge_key_to_hf(self, key: str) -> str:
        """Normalize a key that uses bridge attribute names to use HF module names.

        PyTorch's state_dict uses the Python attribute names (e.g., 'ln1')
        but the conversion logic expects HF module names (e.g., 'ln_1'). This
        function only replaces non-nested component names, leaving bridge
        subcomponents (like 'in', 'out', 'q', 'k', 'v') unchanged since they're
        handled by the component structure.

        Args:
            key: Key that may use bridge attribute names

        Returns:
            Key with attribute names replaced by module names where needed
        """
        component_mapping = self.adapter.component_mapping
        if not component_mapping:
            return key

        # Build a mapping of only the direct module attribute names to HF names
        # We only care about top-level and block-level component names, NOT subcomponents
        attr_to_hf = {}

        # Map top-level components
        for tl_name, component in component_mapping.items():
            if component.name and tl_name != "blocks":
                # Skip if TL name is already a suffix of the HF path (avoids doubling).
                if tl_name != component.name and not component.name.endswith("." + tl_name):
                    attr_to_hf[tl_name] = component.name

        # Map block-level components (ln1, ln2, attn, mlp)
        blocks_component = component_mapping.get("blocks")
        if blocks_component and hasattr(blocks_component, "submodules"):
            for tl_subname, subcomponent in blocks_component.submodules.items():
                if subcomponent.name:
                    # Only map if the names differ (e.g., ln1 -> ln_1, but attn -> attn)
                    if tl_subname != subcomponent.name:
                        attr_to_hf[tl_subname] = subcomponent.name

        # Replace only these specific attribute names in the key
        # We need to be careful to only replace whole path components, not substrings
        parts = key.split(".")
        result_parts = []

        for part in parts:
            if part in attr_to_hf:
                result_parts.append(attr_to_hf[part])
            else:
                result_parts.append(part)

        return ".".join(result_parts)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Get state dict with TransformerLens format keys.

        Converts HuggingFace format keys to TransformerLens format and filters out
        _original_component references and nested HuggingFace components.

        This returns a clean state dict with only bridge component paths converted to TL format,
        excluding nested HF components (like c_fc, c_proj, c_attn) that exist inside
        original_component modules.

        Args:
            destination: Optional dict to store state dict in
            prefix: Optional prefix to add to all keys
            keep_vars: Whether to keep variables as Variables instead of tensors

        Returns:
            Dict containing the state dict with TransformerLens format keys
        """
        if destination is not None:
            raw_state_dict = self.original_model.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )
        else:
            raw_state_dict = self.original_model.state_dict(prefix=prefix, keep_vars=keep_vars)

        # Clean _original_component references and convert to TL format
        # Also filter out nested HuggingFace components that are wrapped by bridge components
        tl_state_dict = {}

        for key, value in raw_state_dict.items():
            # Skip _original_component keys
            if key == "_original_component" or key.startswith("_original_component."):
                continue

            # Remove all _original_component from the key
            clean_key = key.replace("._original_component", "")

            # Check if this is a valid bridge path (not a nested HF component)
            if not self._is_valid_bridge_path(clean_key):
                continue

            # Normalize bridge component names to HF names for conversion
            # (e.g., 'ln1' -> 'ln_1', 'mlp.in' -> 'mlp.c_fc')
            hf_key = self._normalize_bridge_key_to_hf(clean_key)

            # Convert to TL format - this uses the adapter's component_mapping
            tl_key = self.adapter.convert_hf_key_to_tl_key(hf_key)

            # Only add if we haven't seen this TL key yet (handles duplicates)
            if tl_key not in tl_state_dict:
                tl_state_dict[tl_key] = value

        return tl_state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict into the model, handling both clean keys and original keys with _original_component references.

        Args:
            state_dict: Dictionary containing a whole state of the module
            strict: Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() function
            assign: Whether to assign items in the state dictionary to their corresponding keys in the module instead of copying them

        Returns:
            NamedTuple with missing_keys and unexpected_keys fields
        """
        current_state_dict = self.original_model.state_dict()
        clean_to_actual = {}
        actual_to_clean = {}
        for actual_key in current_state_dict.keys():
            if actual_key != "_original_component":
                clean_key = actual_key.replace("._original_component", "")
                clean_to_actual[clean_key] = actual_key
                actual_to_clean[actual_key] = clean_key
        mapped_state_dict = {}
        for input_key, value in state_dict.items():
            if input_key in current_state_dict:
                mapped_state_dict[input_key] = value
            else:
                if input_key in clean_to_actual:
                    actual_key = clean_to_actual[input_key]
                    mapped_state_dict[actual_key] = value
                else:
                    mapped_state_dict[input_key] = value
        effective_strict = strict and len(mapped_state_dict) == len(current_state_dict)
        return self.original_model.load_state_dict(
            mapped_state_dict, strict=effective_strict, assign=assign
        )

    def get_params(self):
        """Access to model parameters in the format expected by SVDInterpreter.

        For missing weights, returns zero tensors of appropriate shape instead of raising exceptions.
        This ensures compatibility across different model architectures.

        Returns:
            dict: Dictionary of parameter tensors with TransformerLens naming convention

        Raises:
            ValueError: If configuration is inconsistent (e.g., cfg.n_layers != len(blocks))
        """
        return get_bridge_params(self)

    # NOTE: list_supported_models and check_model_support are attached to this class
    # dynamically by transformer_lens.model_bridge.sources.transformers module.
    # These are HuggingFace-specific methods that belong in the transformers source module.
