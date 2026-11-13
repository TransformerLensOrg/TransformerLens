"""Activation Cache.

The :class:`ActivationCache` is at the core of Transformer Lens. It is a wrapper that stores all
important activations from a forward pass of the model, and provides a variety of helper functions
to investigate them.

Getting Started:

When reading these docs for the first time, we recommend reading the main :class:`ActivationCache`
class first, including the examples, and then skimming the available methods. You can then refer
back to these docs depending on what you need to do.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import einops
import torch
from jaxtyping import Float, Int
from typing_extensions import Literal

import transformer_lens.utilities as utils
from transformer_lens.utilities import Slice, SliceInput, warn_if_mps

if TYPE_CHECKING:
    from transformer_lens.model_protocol import TransformerLensModelWithWeights


def _normalize_projection_to_2d(
    project: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], bool]:
    """Return ``(project_2d, squeeze_at_end)`` — 1D projections are reshaped to 2D for uniform internal handling and squeezed back at the user-facing return."""
    if project is None:
        return None, False
    if project.ndim == 1:
        return project.unsqueeze(-1), True
    return project, False


class ActivationCache:
    """Activation Cache.

    A wrapper that stores all important activations from a forward pass of the model, and provides a
    variety of helper functions to investigate them.

    The :class:`ActivationCache` is at the core of Transformer Lens. It is a wrapper that stores all
    important activations from a forward pass of the model, and provides a variety of helper
    functions to investigate them. The common way to access it is to run the model with
    :meth:`transformer_lens.model_bridge.TransformerBridge.run_with_cache`.

    Examples:

    When investigating a particular behaviour of a model, a very common first step is to try and
    understand which components of the model are most responsible for that behaviour. For example,
    if you're investigating the prompt "Why did the chicken cross the" -> " road", you might want to
    understand if there is a specific sublayer (mlp or multi-head attention) that is responsible for
    the model predicting "road". This kind of analysis commonly falls under the category of "logit
    attribution" or "direct logit attribution" (DLA).

    >>> from transformer_lens.model_bridge import TransformerBridge
    >>> model = TransformerBridge.boot_transformers("roneneldan/TinyStories-1M")
    >>> model.enable_compatibility_mode()

    >>> _logits, cache = model.run_with_cache("Why did the chicken cross the")
    >>> residual_stream, labels = cache.decompose_resid(return_labels=True, mode="attn")
    >>> print(labels[0:3])
    ['embed', 'pos_embed', '0_attn_out']

    >>> answer = " road" # Note the proceeding space to match the model's tokenization
    >>> logit_attrs = cache.logit_attrs(residual_stream, answer)
    >>> print(logit_attrs.shape) # Attention layers
    torch.Size([10, 1, 7])

    >>> most_important_component_idx = torch.argmax(logit_attrs)
    >>> print(labels[most_important_component_idx])
    3_attn_out

    You can also dig in with more granularity, using :meth:`get_full_resid_decomposition` to get the
    residual stream by individual component (mlp neurons and individual attention heads). This
    creates a larger residual stack, but the approach of using :meth"`logit_attrs` remains the same.

    Equally you might want to find out if the model struggles to construct such excellent jokes
    until the very last layers, or if it is trivial and the first few layers are enough. This kind
    of analysis is called "logit lens", and you can find out more about how to do that with
    :meth:`ActivationCache.accumulated_resid`.

    Warning:

    :class:`ActivationCache` is designed to be used with
    :class:`transformer_lens.model_bridge.TransformerBridge`. Advanced helpers expect the model to
    expose the TransformerLens weight-processing interface and generally expect a complete cache;
    some internal methods may break with other models or partial caches.

    The biggest footgun and source of bugs in this code will be keeping track of indexes,
    dimensions, and the numbers of each. There are several kinds of activations:

    * Internal attn head vectors: q, k, v, z. Shape [batch, pos, head_index, d_head].
    * Internal attn pattern style results: pattern (post softmax), attn_scores (pre-softmax). Shape
      [batch, head_index, query_pos, key_pos].
    * Attn head results: result. Shape [batch, pos, head_index, d_model].
    * Internal MLP vectors: pre, post, mid (only used for solu_ln - the part between activation +
      layernorm). Shape [batch, pos, d_mlp].
    * Residual stream vectors: resid_pre, resid_mid, resid_post, attn_out, mlp_out, embed,
      pos_embed, normalized (output of each LN or LNPre). Shape [batch, pos, d_model].
    * LayerNorm Scale: scale. Shape [batch, pos, 1].

    Sometimes the batch dimension will be missing because we applied `remove_batch_dim` (used when
    batch_size=1), and as such all library functions *should* be robust to that.

    Type annotations are in the following form:

    * layers_covered is the number of layers queried in functions that stack the residual stream.
    * batch_and_pos_dims is the set of dimensions from batch and pos - by default this is ["batch",
      "pos"], but is only ["pos"] if we've removed the batch dimension and is [()] if we've removed
      batch dimension and are applying a pos slice which indexes a specific position.

    Args:
        cache_dict:
            A dictionary of cached activations from a model run.
        model:
            The model that the activations are from.
        has_batch_dim:
            Whether the activations have a batch dimension.
    """

    def __init__(
        self,
        cache_dict: Dict[str, torch.Tensor],
        model: Any,
        has_batch_dim: bool = True,
    ):
        self.cache_dict = cache_dict
        # Advanced helpers (LN folding, residual-direction projection) need the
        # weight-processing surface, which TransformerBridge exposes.
        self.model = cast("TransformerLensModelWithWeights", model)
        self.has_batch_dim = has_batch_dim
        self.has_embed = "hook_embed" in self.cache_dict
        self.has_pos_embed = "hook_pos_embed" in self.cache_dict

        # Note: model reference prevents garbage collection. Set cache.model = None if unneeded.

    def _batch_size(self) -> int:
        """The cache's batch size: the most common leading dim across entries.

        Caches may hold non-batch entries alongside genuinely batched
        activations — broadcast entries with a leading dim of 1 (e.g. the
        bridge's position-index inputs) or position-indexed entries whose
        leading dim is the sequence length (e.g. T5's relative position bias).
        The batched activations vastly outnumber both, so the mode is the
        reliable signal where max/min are not.
        """
        counts = Counter(v.size(0) for v in self.cache_dict.values() if v.ndim > 0)
        return counts.most_common(1)[0][0] if counts else 1

    def remove_batch_dim(self, batch_size: Optional[int] = None) -> ActivationCache:
        """Remove the Batch Dimension (if a single batch item).

        Args:
            batch_size:
                The true batch size, when the caller knows it. By default it is inferred from
                the most common leading dimension of the cached entries, which is wrong for a
                cache holding only flattened or position-indexed activations.

        Returns:
            The ActivationCache with the batch dimension removed.
        """
        if self.has_batch_dim:
            if batch_size is None:
                batch_size = self._batch_size()
            assert (
                batch_size == 1
            ), f"Cannot remove batch dimension from cache with batch size {batch_size}"
            for key in self.cache_dict:
                if self.cache_dict[key].ndim > 0 and self.cache_dict[key].size(0) == 1:
                    self.cache_dict[key] = self.cache_dict[key][0]
            self.has_batch_dim = False
        else:
            logging.warning("Tried removing batch dimension after already having removed it.")
        return self

    def __repr__(self) -> str:
        """Representation of the ActivationCache.

        Special method that returns a string representation of an object. It's normally used to give
        a string that can be used to recreate the object, but here we just return a string that
        describes the object.
        """
        return f"ActivationCache with keys {list(self.cache_dict.keys())}"

    def __getitem__(self, key) -> torch.Tensor:
        """Retrieve Cached Activations by Key or Shorthand.

        Enables direct access to cached activations via dictionary-style indexing using keys or
        shorthand naming conventions.

        It also supports tuples for advanced indexing, with the dimension order as (name, layer_index, layer_type).
        See :func:`transformer_lens.utilities.get_act_name` for how shorthand is converted to a full name.


        Args:
            key:
                The key or shorthand name for the activation to retrieve.

        Returns:
            The cached activation tensor corresponding to the given key.
        """
        if key in self.cache_dict:
            return self.cache_dict[key]
        elif type(key) == str:
            return self.cache_dict[utils.get_act_name(key)]
        else:
            if len(key) > 1 and key[1] is not None:
                if key[1] < 0:
                    # Supports negative indexing on the layer dimension
                    key = (key[0], self.model.cfg.n_layers + key[1], *key[2:])
            return self.cache_dict[utils.get_act_name(*key)]

    def __len__(self) -> int:
        """Length of the ActivationCache.

        Special method that returns the length of an object (in this case the number of different
        activations in the cache).
        """
        return len(self.cache_dict)

    def to(self, device: Union[str, torch.device]) -> ActivationCache:
        """Move the Cache to a Device.

        Mostly useful for moving the cache to the CPU after model computation finishes to save GPU
        memory. Note however that operations will be much slower on the CPU. Note also that some
        methods will break unless the model is also moved to the same device, eg
        `compute_head_results`.

        Args:
            device:
                The device to move the cache to (e.g. `torch.device.cpu`).

        """
        warn_if_mps(device)
        self.cache_dict = {key: value.to(device) for key, value in self.cache_dict.items()}
        return self

    def toggle_autodiff(self, mode: bool = False):
        """Toggle Autodiff Globally.

        Applies `torch.set_grad_enabled(mode)` to the global state (not just TransformerLens).

        Warning:

        This is pretty dangerous, since autodiff is global state - this turns off torch's
        ability to take gradients completely and it's easy to get a bunch of errors if you don't
        realise what you're doing.

        But autodiff consumes a LOT of GPU memory (since every intermediate activation is cached
        until all downstream activations are deleted - this means that computing the loss and
        storing it in a list will keep every activation sticking around!). So often when you're
        analysing a model's activations, and don't need to do any training, autodiff is more trouble
        than its worth.

        If you don't want to mess with global state, using torch.inference_mode as a context manager
        or decorator achieves similar effects:

        >>> with torch.inference_mode():
        ...     y = torch.Tensor([1., 2, 3])
        >>> y.requires_grad
        False
        """
        logging.warning("Changed the global state, set autodiff to %s", mode)
        torch.set_grad_enabled(mode)

    def keys(self):
        """Keys of the ActivationCache.

        Examples:

            >>> from transformer_lens.model_bridge import TransformerBridge
            >>> model = TransformerBridge.boot_transformers("roneneldan/TinyStories-1M")
            >>> model.enable_compatibility_mode()
            >>> _logits, cache = model.run_with_cache("Some prompt")
            >>> list(cache.keys())[0:8]
            ['embed.hook_in', 'hook_embed', 'embed.hook_out', 'pos_embed.hook_in',
             'hook_pos_embed', 'pos_embed.hook_out', 'blocks.0.hook_in',
             'blocks.0.hook_resid_pre']

        Returns:
            List of all keys.
        """
        return self.cache_dict.keys()

    def values(self):
        """Values of the ActivationCache.

        Returns:
            List of all values.
        """
        return self.cache_dict.values()

    def items(self):
        """Items of the ActivationCache.

        Returns:
            List of all items ((key, value) tuples).
        """
        return self.cache_dict.items()

    def __iter__(self) -> Iterator[str]:
        """ActivationCache Iterator.

        Special method that returns an iterator over the keys in the ActivationCache. Allows looping over the
        cache.

        Examples:

            >>> from transformer_lens.model_bridge import TransformerBridge
            >>> model = TransformerBridge.boot_transformers("roneneldan/TinyStories-1M")
            >>> model.enable_compatibility_mode()
            >>> _logits, cache = model.run_with_cache("Some prompt")
            >>> cache_interesting_names = []
            >>> for key in cache:
            ...     if not key.startswith("blocks.") or key.startswith("blocks.0"):
            ...         cache_interesting_names.append(key)
            >>> print(cache_interesting_names[0:8])
            ['embed.hook_in', 'hook_embed', 'embed.hook_out', 'pos_embed.hook_in',
             'hook_pos_embed', 'pos_embed.hook_out', 'blocks.0.hook_in',
             'blocks.0.hook_resid_pre']

        Returns:
            Iterator over the cache.
        """
        return self.cache_dict.__iter__()

    def apply_slice_to_batch_dim(self, batch_slice: Union[Slice, SliceInput]) -> ActivationCache:
        """Apply a Slice to the Batch Dimension.

        Args:
            batch_slice:
                The slice to apply to the batch dimension.

        Returns:
            The ActivationCache with the batch dimension sliced.
        """
        if not isinstance(batch_slice, Slice):
            batch_slice = Slice(batch_slice)
        batch_slice = cast(Slice, batch_slice)  # mypy can't seem to infer this
        assert (
            self.has_batch_dim or batch_slice.mode == "identity"
        ), "Cannot index into a cache without a batch dim"
        still_has_batch_dim = (batch_slice.mode != "int") and self.has_batch_dim
        batch_size = self._batch_size()
        # Broadcast entries (leading dim 1 when the true batch is larger) are not
        # batched — leave them untouched so slicing can't index out of bounds.
        new_cache_dict = {
            name: (
                batch_slice.apply(param, dim=0)
                if param.ndim > 0 and param.size(0) == batch_size
                else param
            )
            for name, param in self.cache_dict.items()
        }
        return ActivationCache(new_cache_dict, self.model, has_batch_dim=still_has_batch_dim)

    def _mlp_input_includes_attn(self, layer: int) -> bool:
        """Whether the target block's MLP reads the current attention contribution."""
        return layer < self.model.cfg.n_layers and bool(
            getattr(self.model.blocks[layer], "mlp_input_includes_attn", True)
        )

    def _block_modules(self, layer: int) -> Optional[Dict[str, Any]]:
        """Bridged submodules of block ``layer``, or None when the model exposes none."""
        blocks = getattr(self.model, "blocks", None)
        if blocks is None or layer >= len(blocks):
            return None
        modules = getattr(blocks[layer], "_modules", None)
        return modules if isinstance(modules, dict) else None

    def _has_attn(self, layer: int) -> bool:
        """Whether block ``layer`` has attention (and so ``hook_attn_out`` / per-head hooks)."""
        from transformer_lens.model_bridge.generalized_components.block import (
            VARIANT_SUBMODULE_NAMES,
        )

        modules = self._block_modules(layer)
        # Blocks declaring no sequence-mixer slot keep the attention default.
        return (
            modules is None
            or "attn" in modules
            or not any(name in modules for name in VARIANT_SUBMODULE_NAMES)
        )

    def _mixer_out_names(self, layer: int) -> List[str]:
        """Variant slots (``linear_attn``, ``mixer``, …) of a block without attention.

        Hybrid and SSM blocks write their sequence-mixer output in place of
        ``hook_attn_out``; empty for attention blocks.
        """
        from transformer_lens.model_bridge.generalized_components.block import (
            VARIANT_SUBMODULE_NAMES,
        )

        if self._has_attn(layer):
            return []
        modules = self._block_modules(layer) or {}
        return [name for name in VARIANT_SUBMODULE_NAMES if name in modules]

    def _has_mlp(self, layer: int) -> bool:
        """Whether block ``layer`` writes an MLP output separate from its mixer."""
        from transformer_lens.model_bridge.generalized_components.block import (
            _BLOCK_INTERNAL_MODULES,
            _NORM_PREFIXES,
            _VARIANT_SUBMODULE_SET,
        )

        if self.model.cfg.attn_only:
            return False
        modules = self._block_modules(layer)
        if modules is None or "mlp" in modules:
            return True
        # Mamba / NemotronH blocks hold only norms and a mixer. Anything else that writes to the
        # residual stream (e.g. a MoE without an `mlp` slot) keeps the `mlp_out` contract.
        return any(
            name not in _VARIANT_SUBMODULE_SET
            and name not in _BLOCK_INTERNAL_MODULES
            and not name.startswith(_NORM_PREFIXES)
            for name in modules
        )

    def _mixer_role(self, layer: int) -> Literal["attn", "mlp"]:
        """Residual slot a non-attention block's mixer fills.

        Single-mixer blocks (NemotronH) wrap attention, Mamba and MLP layers alike in one
        passthrough ``mixer``; ``cfg.layers_block_type`` says which one it is.
        """
        block_types = getattr(self.model.cfg, "layers_block_type", None)
        if isinstance(block_types, (list, tuple)) and layer < len(block_types):
            if block_types[layer] in ("mlp", "moe"):
                return "mlp"
        return "attn"

    def _mixer_label(self, layer: int, name: str) -> str:
        """``{l}_mlp_out`` / ``{l}_attn_out`` for typed passthrough mixers, else ``{l}_{name}_out``."""
        block_types = getattr(self.model.cfg, "layers_block_type", None)
        block_type = (
            block_types[layer]
            if isinstance(block_types, (list, tuple)) and layer < len(block_types)
            else None
        )
        if block_type in ("mlp", "moe"):
            return f"{layer}_mlp_out"
        if block_type in ("attention", "full_attention"):
            return f"{layer}_attn_out"
        return f"{layer}_{name}_out"

    def _mixer_stack(
        self, layer: int, pos_slice: Slice, subtract_bias: bool = False
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Residual contributions of a non-attention block's mixer, e.g. ``0_linear_attn_out``.

        Scaled by the block's ``residual_contribution_scale`` (Granite's ``residual_multiplier``
        is applied after ``hook_out``), matching the head and neuron paths. ``subtract_bias``
        drops the mixer's output bias first, for decompositions that carry it in their own
        ``bias`` row.
        """
        components: List[torch.Tensor] = []
        names = self._mixer_out_names(layer)
        block = self.model.blocks[layer]
        scale = getattr(block, "residual_contribution_scale", 1.0)
        # Same bias accumulated_bias counts for this block, so the two cancel exactly.
        bias = None
        if subtract_bias and names:
            get_bias = getattr(self.model, "_get_block_variant_bias", None)
            bias = get_bias(block) if callable(get_bias) else None
        for i, name in enumerate(names):
            out = pos_slice.apply(self[f"blocks.{layer}.{name}.hook_out"], dim=-2)
            if i == 0 and isinstance(bias, torch.Tensor):
                out = out - bias.to(out.device, out.dtype)
            if scale != 1.0:
                out = out * scale
            components.append(out)
        return components, self._mixer_labels(layer)

    def _mixer_labels(self, layer: int) -> List[str]:
        """Labels of :meth:`_mixer_stack`'s rows for block ``layer``."""
        return [self._mixer_label(layer, name) for name in self._mixer_out_names(layer)]

    def _decomposition_layer(self, layer: Optional[int]) -> int:
        """``None`` / ``-1`` mean the full stack: the residual entering the unembedding."""
        if layer is None or layer == -1:
            return int(self.model.cfg.n_layers)
        return layer

    def _resid_component_plan(
        self,
        layer: int,
        mode: Literal["all", "mlp", "attn"] = "all",
        mlp_input: bool = False,
        incl_embeds: bool = True,
    ) -> List[Tuple[List[str], Callable[[], List[torch.Tensor]]]]:
        """:meth:`decompose_resid`'s rows as ``(labels, read)`` groups.

        The labels depend only on the block layout, so they exist before any activation
        does; ``read`` fetches the group's tensors from the cache.
        """
        incl_attn = mode != "mlp"
        incl_mlp = mode != "attn" and not self.model.cfg.attn_only
        hook_dict = getattr(self.model, "hook_dict", None)
        registered = hook_dict if isinstance(hook_dict, dict) else None
        plan: List[Tuple[List[str], Callable[[], List[torch.Tensor]]]] = []

        def read(key: Any) -> Callable[[], List[torch.Tensor]]:
            return lambda: [self[key]]

        def mlp_out(l: int) -> List[torch.Tensor]:
            # Several modules can feed one MLP-side add (GraniteMoeHybrid sums its MoE and
            # shared MLP inline), leaving no hook_mlp_out; the residual difference is that write.
            if registered is not None and f"blocks.{l}.hook_mlp_out" not in registered:
                resid = (f"blocks.{l}.hook_resid_mid", f"blocks.{l}.hook_resid_post")
                if all(name in registered for name in resid):
                    return [self[("resid_post", l)] - self[("resid_mid", l)]]
            return [self[("mlp_out", l)]]

        def add_mixer(l: int) -> None:
            plan.append((self._mixer_labels(l), lambda: self._mixer_stack(l, Slice(None))[0]))

        def add_attn_slot(l: int) -> None:
            if self._has_attn(l):
                plan.append(([f"{l}_attn_out"], read(("attn_out", l))))
            elif self._mixer_role(l) == "attn":
                add_mixer(l)

        def add_mlp_slot(l: int) -> None:
            if self._has_mlp(l):
                plan.append(([f"{l}_mlp_out"], lambda: mlp_out(l)))
            elif not self._has_attn(l) and self._mixer_role(l) == "mlp":
                add_mixer(l)

        if incl_embeds:
            if self.has_embed:
                plan.append((["embed"], read("hook_embed")))
            if self.has_pos_embed:
                plan.append((["pos_embed"], read("hook_pos_embed")))
        for l in range(layer):
            if incl_attn:
                add_attn_slot(l)
            if incl_mlp:
                add_mlp_slot(l)
        if mlp_input and incl_attn and self._mlp_input_includes_attn(layer):
            add_attn_slot(layer)
        return plan

    def decompose_resid_labels(
        self,
        layer: Optional[int] = None,
        mlp_input: bool = False,
        mode: Literal["all", "mlp", "attn"] = "all",
        incl_embeds: bool = True,
    ) -> List[str]:
        """The labels :meth:`decompose_resid` returns for these arguments, read from the
        block layout alone, so a tool can check a decomposition before running the model."""
        plan = self._resid_component_plan(
            self._decomposition_layer(layer),
            mode=mode,
            mlp_input=mlp_input,
            incl_embeds=incl_embeds,
        )
        return [label for labels, _ in plan for label in labels]

    def accumulated_resid(
        self,
        layer: Optional[int] = None,
        incl_mid: bool = False,
        apply_ln: bool = False,
        pos_slice: Optional[Union[Slice, SliceInput]] = None,
        mlp_input: bool = False,
        return_labels: bool = False,
    ) -> Union[
        Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"], List[str]],
    ]:
        """Accumulated Residual Stream.

        Returns the accumulated residual stream at each layer/sub-layer. This is useful for `Logit
        Lens <https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>`
        style analysis, where it can be thought of as what the model "believes" at each point in the
        residual stream.

        :meth:`logit_lens` does the whole projection below in one call (final norm, ``W_U``,
        ``b_U`` and the adapter's output transform, chunked over the vocabulary); the recipe
        here shows what it computes.

        To project this into the vocabulary space, remember that there is a final layer norm in most
        decoder-only transformers. Therefore, you need to first apply the final layer norm (which
        can be done with `apply_ln`), and then multiply by the unembedding matrix (:math:`W_U`)
        and optionally add the unembedding bias (:math:`b_U`).

        **Note on bias terms:** There are two valid approaches for the final projection:

        1. **With bias terms:** Use `model.unembed(normalized_resid)` which applies both :math:`W_U`
           and :math:`b_U` (equivalent to `normalized_resid @ model.W_U + model.b_U`). This works
           correctly with both `fold_ln=True` and `fold_ln=False` settings, as the biases are
           handled consistently.
        2. **Without bias terms:** Use only `normalized_resid @ model.W_U`. If taking this approach,
           you should instantiate the model with `fold_ln=True`, which folds the layer norm scaling
           into :math:`W_U` and the layer norm bias into :math:`b_U`. Since `apply_ln=True` will
           apply the (now parameter-free) layer norm, and you skip :math:`b_U`, no bias terms are
           included. With `fold_ln=False`, the layer norm bias would still be applied, which is
           typically not desired when excluding bias terms.

        Both approaches are commonly used in the literature and are valid interpretability choices.

        If you instead want to look at contributions to the residual stream from each component
        (e.g. for direct logit attribution), see :meth:`decompose_resid` instead, or
        :meth:`get_full_resid_decomposition` if you want contributions broken down further into each
        MLP neuron.

        Examples:

        Logit Lens analysis can be done as follows:

        >>> from transformer_lens.model_bridge import TransformerBridge
        >>> import torch
        >>> import pandas as pd

        >>> model = TransformerBridge.boot_transformers("roneneldan/TinyStories-1M", device="cpu")
        >>> model.enable_compatibility_mode()

        >>> prompt = "Why did the chicken cross the"
        >>> answer = " road"
        >>> logits, cache = model.run_with_cache("Why did the chicken cross the")
        >>> answer_token = model.to_single_token(answer)
        >>> print(answer_token)
        2975

        >>> accum_resid, labels = cache.accumulated_resid(return_labels=True, apply_ln=True)
        >>> last_token_accum = accum_resid[:, 0, -1, :]  # layer, batch, pos, d_model
        >>> print(last_token_accum.shape)  # layer, d_model
        torch.Size([9, 64])


        >>> W_U = model.W_U
        >>> print(W_U.shape)
        torch.Size([64, 50257])

        >>> # Project to vocabulary without unembedding bias
        >>> layers_logits = last_token_accum @ W_U  # layer, d_vocab
        >>> print(layers_logits.shape)
        torch.Size([9, 50257])

        >>> # The unembedding bias can be added on top when the model carries one
        >>> # (the rank table below stays on the bias-free logits):
        >>> b_U = getattr(model, "b_U", None)
        >>> with_bias = layers_logits + b_U if b_U is not None else layers_logits
        >>> print(with_bias.shape)
        torch.Size([9, 50257])

        >>> # Get the rank of the correct answer by layer
        >>> sorted_indices = torch.argsort(layers_logits, dim=1, descending=True)
        >>> rank_answer = (sorted_indices == 2975).nonzero(as_tuple=True)[1]
        >>> print(pd.Series(rank_answer, index=labels))
        0_pre         4442
        1_pre          382
        2_pre          982
        3_pre         1160
        4_pre          408
        5_pre          145
        6_pre           78
        7_pre          387
        final_post       6
        dtype: int64

        Args:
            layer:
                The layer to take components up to - by default includes resid_pre for that layer
                and excludes resid_mid and resid_post for that layer. If set as `n_layers`, `-1` or
                `None` it will return all residual streams, including the final one (i.e.
                immediately pre logits). The indices are taken such that this gives the accumulated
                streams up to the input to layer l.
            incl_mid:
                Whether to return `resid_mid` for previous sequential-residual layers.
                Parallel-residual layers have no distinct mid stream.
            apply_ln:
                Whether to apply the final layer norm to the stack. When True, applies
                `model.ln_final`, which recomputes normalization statistics (mean and
                variance/RMS) for each intermediate state in the stack, transforming the
                activations into the format expected by the unembedding layer.
            pos_slice:
                A slice object to apply to the pos dimension. Defaults to None, do nothing.
            mlp_input:
                Target the MLP input: include the current resid_mid for sequential blocks;
                parallel blocks already expose this input as resid_pre.
            return_labels:
                Whether to return a list of labels for the residual stream components. Useful for
                labelling graphs.

        Returns:
            A tensor of the accumulated residual streams. If `return_labels` is True, also returns a
            list of labels for the components (as a tuple in the form `(components, labels)`).
        """
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers
        assert isinstance(layer, int)
        labels = []
        components_list = []
        for l in range(layer + 1):
            if l == self.model.cfg.n_layers:
                components_list.append(self[("resid_post", self.model.cfg.n_layers - 1)])
                labels.append("final_post")
                continue
            components_list.append(self[("resid_pre", l)])
            labels.append(f"{l}_pre")
            if ((incl_mid and l < layer) or (mlp_input and l == layer)) and (
                self._mlp_input_includes_attn(l)
            ):
                components_list.append(self[("resid_mid", l)])
                labels.append(f"{l}_mid")
        components_list = [pos_slice.apply(c, dim=-2) for c in components_list]
        components = torch.stack(components_list, dim=0)
        if apply_ln:
            recompute_ln = layer == self.model.cfg.n_layers
            components = self.apply_ln_to_stack(
                components,
                layer,
                pos_slice=pos_slice,
                mlp_input=mlp_input,
                has_batch_dim=self.has_batch_dim,
                recompute_ln=recompute_ln,
            )
        if return_labels:
            return components, labels
        else:
            return components

    def logit_lens(self, **kwargs: Any) -> Any:
        """One-call logit lens over this cache's accumulated residual stream.

        Delegates to :func:`transformer_lens.tools.analysis.logit_lens.logit_lens`
        with this cache; see it for the keyword arguments (``layers``,
        ``positions``, ``targets``, ``top_k``, ``vocab``, ``return_type`` ...).
        The final stack entry reproduces the model's logits through the real
        final norm, ``W_U``, ``b_U`` and the adapter's output transform.
        """
        # Lazy import: tools.analysis imports this module.
        from transformer_lens.tools.analysis.logit_lens import logit_lens

        return logit_lens(self.model, cache=self, **kwargs)

    def logit_attrs(
        self,
        residual_stack: Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
        tokens: Union[
            str,
            int,
            Int[torch.Tensor, ""],
            Int[torch.Tensor, "batch"],
            Int[torch.Tensor, "batch position"],
        ],
        incorrect_tokens: Optional[
            Union[
                str,
                int,
                Int[torch.Tensor, ""],
                Int[torch.Tensor, "batch"],
                Int[torch.Tensor, "batch position"],
            ]
        ] = None,
        pos_slice: Union[Slice, SliceInput] = None,
        batch_slice: Union[Slice, SliceInput] = None,
        has_batch_dim: bool = True,
    ) -> Float[torch.Tensor, "num_components *batch_and_pos_dims_out"]:
        """Logit Attributions.

        Takes a residual stack (typically the residual stream decomposed by components), and
        calculates how much each item in the stack "contributes" to specific tokens.

        It does this by:
            1. Getting the residual directions of the tokens (i.e. reversing the unembed)
            2. Taking the dot product of each item in the residual stack, with the token residual
               directions.

        Note that if incorrect tokens are provided, it instead takes the difference between the
        correct and incorrect tokens (to calculate the residual directions). This is useful as
        sometimes we want to know e.g. which components are most responsible for selecting the
        correct token rather than an incorrect one. For example in the `Interpretability in the Wild
        paper <https://arxiv.org/abs/2211.00593>` prompts such as "John and Mary went to the shops,
        John gave a bag to" were investigated, and it was therefore useful to calculate attribution
        for the :math:`\\text{Mary} - \\text{John}` residual direction.

        Warning:

        Choosing the correct `tokens` and `incorrect_tokens` is both important and difficult. When
        investigating specific components it's also useful to look at it's impact on all tokens
        (i.e. :math:`\\text{final_ln}(\\text{residual_stack_item}) W_U`).

        Args:
            residual_stack:
                Stack of components of residual stream to get logit attributions for.
            tokens:
                Tokens to compute logit attributions on. A [batch] tensor selects one token
                per example across all retained positions. A 1-D tensor whose length differs
                from the batch size selects one token per position, shared across examples.
                Use [1, position] to share per-position targets when batch and position sizes
                are equal; [batch, position] selects targets for each example and position.
                Token tensors should already match any batch/position slices.
            incorrect_tokens:
                If provided, compute attributions on logit difference between tokens and
                incorrect_tokens. Must have the same shape as tokens.
            pos_slice:
                The slice to apply layer norm scaling on. Defaults to None, do nothing.
            batch_slice:
                The slice to take on the batch dimension during layer norm scaling. Defaults to
                None, do nothing.
            has_batch_dim:
                Whether residual_stack has a batch dimension. Defaults to True.

        Returns:
            A tensor of the logit attributions or logit difference attributions if incorrect_tokens
            was provided.
        """
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)

        if not isinstance(batch_slice, Slice):
            batch_slice = Slice(batch_slice)

        # Convert tokens to tensor for shape checking, but pass original to tokens_to_residual_directions
        tokens_for_shape_check = tokens

        if isinstance(tokens_for_shape_check, str):
            tokens_for_shape_check = torch.as_tensor(
                self.model.to_single_token(tokens_for_shape_check)
            )
        elif isinstance(tokens_for_shape_check, int):
            tokens_for_shape_check = torch.as_tensor(tokens_for_shape_check)

        logit_directions = self.model.tokens_to_residual_directions(tokens)

        if incorrect_tokens is not None:
            # Convert incorrect_tokens to tensor for shape checking, but pass original to tokens_to_residual_directions
            incorrect_tokens_for_shape_check = incorrect_tokens

            if isinstance(incorrect_tokens_for_shape_check, str):
                incorrect_tokens_for_shape_check = torch.as_tensor(
                    self.model.to_single_token(incorrect_tokens_for_shape_check)
                )
            elif isinstance(incorrect_tokens_for_shape_check, int):
                incorrect_tokens_for_shape_check = torch.as_tensor(incorrect_tokens_for_shape_check)

            if tokens_for_shape_check.shape != incorrect_tokens_for_shape_check.shape:
                raise ValueError(
                    f"tokens and incorrect_tokens must have the same shape! \
                        (tokens.shape={tokens_for_shape_check.shape}, \
                        incorrect_tokens.shape={incorrect_tokens_for_shape_check.shape})"
                )

            # If incorrect_tokens was provided, take the logit difference
            logit_directions = logit_directions - self.model.tokens_to_residual_directions(
                incorrect_tokens
            )

        # W_U omits any constant factor the architecture applies to the unembedding
        # output (e.g. Granite's logits_scaling); include it so attributions sum to logits.
        output_logits_scale = getattr(
            getattr(self.model, "adapter", None), "output_logits_scale", None
        )
        if callable(output_logits_scale):
            logit_directions = logit_directions * output_logits_scale()

        scaled_residual_stack = self.apply_ln_to_stack(
            residual_stack,
            layer=-1,
            pos_slice=pos_slice,
            batch_slice=batch_slice,
            has_batch_dim=has_batch_dim,
        )

        if (
            has_batch_dim
            and batch_slice.mode != "int"
            and scaled_residual_stack.ndim == 4
            and logit_directions.ndim == 2
            and logit_directions.shape[0] == scaled_residual_stack.shape[1]
        ):
            # Per-example directions must broadcast over positions, not across the batch.
            logit_directions = logit_directions.unsqueeze(-2)

        logit_attrs = (scaled_residual_stack * logit_directions).sum(dim=-1)
        return logit_attrs

    def decompose_resid(
        self,
        layer: Optional[int] = None,
        mlp_input: bool = False,
        mode: Literal["all", "mlp", "attn"] = "all",
        apply_ln: bool = False,
        pos_slice: Union[Slice, SliceInput] = None,
        incl_embeds: bool = True,
        return_labels: bool = False,
    ) -> Union[
        Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"], List[str]],
    ]:
        """Decompose the Residual Stream.

        Decomposes the residual stream input to layer L into a stack of the output of previous
        layers. The sum of these is the input to layer L (plus embedding and pos embedding). This is
        useful for attributing model behaviour to different components of the residual stream

        Args:
            layer:
                The layer to take components up to - by default includes
                resid_pre for that layer and excludes resid_mid and resid_post for that layer.
                layer==n_layers means to return all layer outputs incl in the final layer, layer==0
                means just embed and pos_embed. The indices are taken such that this gives the
                accumulated streams up to the input to layer l
            mlp_input:
                Decompose the MLP input rather than the attention input. The current layer's
                attention contribution is included only for sequential-residual blocks.
            mode:
                Values are "all", "mlp" or "attn". "all" returns all
                components, "mlp" returns only the MLP components, and "attn" returns only the
                attention components. Defaults to "all". On SSM / hybrid models, blocks without
                attention contribute their sequence-mixer output (e.g. ``0_linear_attn_out``,
                ``0_mixer_out``) in the attention slot, and blocks without an MLP contribute no
                ``mlp_out``. A sequential block with no ``hook_mlp_out`` (GraniteMoeHybrid with
                experts sums its MoE and shared MLP inline) contributes
                ``resid_post - resid_mid`` as its ``mlp_out``.
            apply_ln:
                Whether to apply LayerNorm to the stack.
            pos_slice:
                A slice object to apply to the pos dimension.
                Defaults to None, do nothing.
            incl_embeds:
                Whether to include embed & pos_embed, whose hooks expose their
                contributions to the residual stream.
            return_labels:
                Whether to return a list of labels for the residual stream components.
                Useful for labelling graphs.

        Returns:
            A tensor of the accumulated residual streams. If `return_labels` is True, also returns
            a list of labels for the components (as a tuple in the form `(components, labels)`).
        """
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        pos_slice = cast(Slice, pos_slice)  # mypy can't seem to infer this
        layer = self._decomposition_layer(layer)
        plan = self._resid_component_plan(
            layer, mode=mode, mlp_input=mlp_input, incl_embeds=incl_embeds
        )
        labels = [label for entry_labels, _ in plan for label in entry_labels]
        components_list = [pos_slice.apply(c, dim=-2) for _, read in plan for c in read()]
        if components_list:
            components = torch.stack(components_list, dim=0)
        else:
            # e.g. mode="mlp" on a model whose blocks have no MLP: an empty, stackable result.
            empty_src = pos_slice.apply(self[("resid_pre", 0)], dim=-2)
            components = torch.zeros(
                0, *empty_src.shape, dtype=empty_src.dtype, device=empty_src.device
            )
        if apply_ln:
            components = self.apply_ln_to_stack(
                components, layer, pos_slice=pos_slice, mlp_input=mlp_input
            )
        if return_labels:
            return components, labels
        else:
            return components

    def compute_head_results(
        self,
    ):
        """Compute Head Results.

        Computes and caches the results for each attention head, ie the amount contributed to the
        residual stream from that head. attn_out for a layer is the sum of head results plus b_O.
        Intended use is to enable use_attn_results when running and caching the model, but this can
        be useful if you forget.

        Existing per-head results are preserved, including those without a batch dimension.
        Missing or invalid results are recomputed from cached ``hook_z`` and the model's
        ``W_O`` weights. A per-head contraction avoids materializing the broadcast product
        over both ``d_head`` and ``d_model``. As in the bridge forward pass, the contraction
        follows active autocast settings.

        TransformerBridge exposes ``blocks[i].attn.W_O`` via its component-mapping
        compatibility shim.
        """
        if self._ensure_head_results():
            logging.warning("Tried to compute head results when they were already cached")

    def _ensure_head_results(self) -> bool:
        """Recompute missing or invalid per-head results; True if all were already cached."""
        expected_ndim = 4 if self.has_batch_dim else 3
        all_cached = True
        for layer in range(self.model.cfg.n_layers):
            # SSM / linear-attention blocks have no heads.
            if not self._has_attn(layer):
                continue
            result_key = f"blocks.{layer}.attn.hook_result"
            cached_result = self.cache_dict.get(result_key)
            # Batchless per-head results are valid 3D tensors.
            if (
                isinstance(cached_result, torch.Tensor)
                and cached_result.ndim == expected_ndim
                and cached_result.shape[-2] == self.model.cfg.n_heads
            ):
                continue
            all_cached = False

            # Routed attention (JetMoE's mixture of attention heads) has no static W_O.
            routed = getattr(self.model.blocks[layer].attn, "head_results_from_cache", None)
            if callable(routed):
                self.cache_dict[result_key] = routed(self, layer)
                continue

            z = self[("z", layer, "attn")]
            weights = self.model.blocks[layer].attn.W_O
            if z.dtype != weights.dtype:
                # Preserve broadcast promotion for caches collected under autocast.
                dtype = torch.promote_types(z.dtype, weights.dtype)
                z, weights = z.to(dtype), weights.to(dtype)
            result = torch.einsum("...hd,hdm->...hm", z, weights)
            scale = getattr(self.model.blocks[layer].attn, "_result_contribution_scale", 1.0)
            self.cache_dict[result_key] = result if scale == 1.0 else result * scale
        return all_cached

    def ssm_layers(self, mixer_type: Optional[Union[type, Tuple[type, ...]]] = None) -> List[int]:
        """Return the block indices whose mixer is an SSM / recurrent mixer.

        Family-agnostic and purely structural: finds each block's *realized* SSM
        mixer in its variant slot (``.mixer`` / ``.linear_attn`` / …) via
        ``find_ssm_mixer``, which excludes a hybrid's passthrough ``.mixer`` on
        non-SSM layers (e.g. NemotronH attention/MLP/MoE) — no dependence on
        ``cfg.layers_block_type``.

        Args:
            mixer_type: Optional concrete bridge class to filter to (e.g.
                ``SSM2MixerBridge`` for only Mamba-2 layers).

        Returns:
            Ascending list of SSM block indices.
        """
        from transformer_lens.model_bridge.generalized_components.ssm_protocol import (
            find_ssm_mixer,
        )

        bridge = self.model
        layers: List[int] = []
        for i, block in enumerate(bridge.blocks):
            mixer = find_ssm_mixer(block)
            if mixer is None:
                continue
            if mixer_type is not None and not isinstance(mixer, mixer_type):
                continue
            layers.append(i)
        return layers

    def _over_ssm_layers(
        self,
        fn: Callable[[Any, int], torch.Tensor],
        mixer_type: Optional[Union[type, Tuple[type, ...]]] = None,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """Apply ``fn(mixer, layer_idx)`` over every SSM layer; stack or dict.

        The single SSM layer-enumeration used by both ``compute_ssm_state`` and
        ``compute_ssm_effective_attention``. Returns a stacked tensor (dim 0 =
        layer) when *every* block is an SSM layer, else a ``{layer_idx: result}``
        dict over the SSM layers (heterogeneous hybrids).
        """
        from transformer_lens.model_bridge.generalized_components.ssm_protocol import (
            find_ssm_mixer,
        )

        bridge = self.model
        indices = self.ssm_layers(mixer_type=mixer_type)
        if not indices:
            raise RuntimeError(
                "No SSM mixer layers found. Use an SSM / hybrid bridge and "
                "run_with_cache first (gated-delta-net needs use_cache=False)."
            )
        results: Dict[int, torch.Tensor] = {
            i: fn(cast(Any, find_ssm_mixer(bridge.blocks[i])), i) for i in indices
        }
        if indices == list(range(len(bridge.blocks))):
            return torch.stack([results[i] for i in indices], dim=0)
        return results

    def compute_ssm_effective_attention(
        self,
        layer: Optional[int] = None,
        include_dt_scaling: bool = False,
        per_state_coord: bool = False,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """Materialize SSM effective attention for one or all SSM layers, any family.

        The single discovery surface for effective attention across Mamba-1,
        Mamba-2, and gated-delta-net layers; dispatches to each layer's mixer
        ``compute_effective_attention`` regardless of family. Family-specific
        options are forwarded only to mixers whose signature accepts them.

        Args:
            layer: Specific block index, or None for every SSM layer.
            include_dt_scaling: Forwarded to Mamba-1/Mamba-2 mixers (the
                reconstruction form); gated-delta-net does not accept it.
            per_state_coord: Forwarded to Mamba-1 only (per-state-coordinate
                matrices). Setting it True for another family raises ValueError.

        Returns:
            A per-layer matrix for a single ``layer``; for ``layer=None`` a
            stacked tensor (dim 0 = layer) when every block is an SSM layer, else
            a ``{layer_idx: matrix}`` dict over the SSM layers.

        Raises:
            TypeError: If the requested ``layer`` has no SSM mixer.
            ValueError: If an option is set that the target mixer does not support.
            RuntimeError: If ``layer=None`` finds no SSM layers.
        """
        import inspect

        from transformer_lens.model_bridge.generalized_components.ssm_protocol import (
            find_ssm_mixer,
        )

        def _call(mixer: Any, layer_idx: int) -> torch.Tensor:
            fn = cast(Any, mixer).compute_effective_attention
            params = inspect.signature(fn).parameters
            kwargs: Dict[str, Any] = {}
            for name, value in (
                ("include_dt_scaling", include_dt_scaling),
                ("per_state_coord", per_state_coord),
            ):
                if name in params:
                    kwargs[name] = value
                elif value:
                    raise ValueError(
                        f"{type(mixer).__name__}.compute_effective_attention does not "
                        f"support {name}=True."
                    )
            result: torch.Tensor = fn(cache=self, layer_idx=layer_idx, **kwargs)
            return result

        if layer is not None:
            single = find_ssm_mixer(self.model.blocks[layer])
            if single is None:
                raise TypeError(f"Block {layer} has no SSM mixer (no compute_effective_attention).")
            return _call(single, layer)
        return self._over_ssm_layers(_call)

    def compute_ssm_state(
        self,
        layer: Optional[int] = None,
        time_step: Optional[int] = None,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """Reconstruct the recurrent SSM state ``S`` from this cache.

        The single discovery surface for recurrent state across families —
        ``SSMMixerBridge`` (Mamba-1), ``SSM2MixerBridge`` (Mamba-2) and
        ``GatedDeltaNetBridge`` (gated delta rule) each reconstruct it — mirroring
        ``compute_head_results``. Read-only post-hoc reconstruction from cached
        hooks (no forward re-run); requires an SSM / SSM-hybrid bridge cached via
        ``run_with_cache`` (gated-delta-net additionally needs ``use_cache=False``
        so its interior hooks fire). See the mixer's ``compute_ssm_state`` for the
        recurrence, shapes, and the ``time_step`` memory bound; state shape is
        family-specific, so ``layer=None`` returns a dict except when every block
        shares one mixer type.

        Args:
            layer: Specific block index, or None for every SSM-state layer.
            time_step: Optional single position (memory-bounded); None for all.

        Returns:
            A per-layer state tensor for a single ``layer``; for ``layer=None`` a
            stacked tensor (dim 0 = layer) when every block is an SSM-state layer,
            else a ``{layer_idx: state}`` dict over those layers.

        Raises:
            TypeError: If the requested ``layer`` has no state-reconstructing mixer.
            RuntimeError: If ``layer=None`` finds no such layers.
        """
        from transformer_lens.model_bridge.generalized_components import (
            GatedDeltaNetBridge,
            SSM2MixerBridge,
            SSMMixerBridge,
        )
        from transformer_lens.model_bridge.generalized_components.ssm_protocol import (
            find_ssm_mixer,
        )

        # Every recurrent family reconstructs S_t from its cached interior hooks.
        state_mixers = (SSMMixerBridge, SSM2MixerBridge, GatedDeltaNetBridge)

        def _call(mixer: Any, layer_idx: int) -> torch.Tensor:
            state: torch.Tensor = cast(Any, mixer).compute_ssm_state(
                self, layer_idx=layer_idx, time_step=time_step
            )
            return state

        if layer is not None:
            single = find_ssm_mixer(self.model.blocks[layer])
            if not isinstance(single, state_mixers):
                raise TypeError(
                    f"Block {layer} has no state-reconstructing SSM mixer; "
                    "compute_ssm_state supports Mamba-1 / Mamba-2 / gated-delta-net."
                )
            return _call(single, layer)
        return self._over_ssm_layers(_call, mixer_type=state_mixers)

    def _post_norm_map(
        self, layer: int, norm_name: str, pos_slice: Slice
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], bool, Optional[torch.Tensor]]]:
        """Cached-scale affine map ``(s, w, centers, b)`` of a post-sublayer norm, or None.

        With ``s`` fixed, ``norm(x) = center(x) / s * w + b`` is affine, so each component of
        the sublayer output contributes ``center(x_i) / s * w`` and the block adds ``b`` once.
        """
        norm = self.model.blocks[layer]._modules.get(norm_name)
        if norm is None:
            return None
        key = f"blocks.{layer}.{norm_name}.hook_scale"
        if norm._modules.get("hook_scale") is None:
            # Leaving it unmapped would silently return pre-norm contributions.
            raise KeyError(
                f"blocks.{layer}.{norm_name} is a post-sublayer norm with no hook_scale (not "
                f"bridged as a normalization component), so head and neuron contributions "
                f"cannot be mapped through it on this architecture."
            )
        try:
            cached_scale = self[key]
        except KeyError as e:
            raise KeyError(
                f"Post-sublayer norm scale not found at '{key}'. Head and neuron decompositions "
                f"on post-norm architectures divide by it; include it in run_with_cache's "
                f"names_filter."
            ) from e
        scale = pos_slice.apply(cached_scale, dim=-2)
        weight = getattr(norm, "weight", None)
        if weight is not None and getattr(norm, "uses_weight_offset", False):
            weight = 1.0 + weight
        centers = not getattr(norm, "uses_rms_norm", False)
        return scale, weight, centers, getattr(norm, "bias", None)

    @staticmethod
    def _apply_post_norm_map(
        x: torch.Tensor, scale: torch.Tensor, weight: Optional[torch.Tensor], centers: bool
    ) -> torch.Tensor:
        """``center(x) / scale * weight`` over the last (d_model) axis; ``scale`` broadcasts."""
        input_dtype = x.dtype
        if centers:
            x = x - x.mean(dim=-1, keepdim=True)
        x = x / scale
        x = x * weight if weight is not None else x
        # The fp32 scale would otherwise promote half-precision stacks; the norm casts back too.
        return x.to(input_dtype)

    def _post_norm_bias_correction(
        self, layer: int, mlp_input: bool, include_mlp_biases: bool, pos_slice: Slice
    ) -> Optional[torch.Tensor]:
        """Per-position ``accumulated_bias`` fix: post-norm blocks add ``norm_map(b) + norm_b``."""
        correction: Optional[torch.Tensor] = None
        sublayers = [(l, "attn") for l in range(layer)]
        if include_mlp_biases:
            sublayers += [(l, "mlp") for l in range(layer)]
        if mlp_input and layer < self.model.cfg.n_layers and self._mlp_input_includes_attn(layer):
            sublayers.append((layer, "attn"))
        for l, kind in sublayers:
            post = self._post_norm_map(l, "ln1_post" if kind == "attn" else "ln2_post", pos_slice)
            if post is None:
                continue
            scale, weight, centers, norm_bias = post
            block = self.model.blocks[l]
            if kind == "attn":
                # Same lookup as accumulated_bias, so the raw bias it added is the one replaced.
                out_bias = getattr(self.model, "_get_block_variant_bias")(block)
            else:
                out_bias = getattr(block.mlp, "b_out", None) if "mlp" in block._modules else None
            raw = torch.zeros(self.model.cfg.d_model, device=scale.device, dtype=scale.dtype)
            if out_bias is not None:
                raw = out_bias.to(device=scale.device, dtype=scale.dtype)
            mapped = self._apply_post_norm_map(raw, scale, weight, centers)
            if norm_bias is not None:
                mapped = mapped + norm_bias.to(device=scale.device)
            delta = (mapped - raw) * getattr(block, "residual_contribution_scale", 1.0)
            correction = delta if correction is None else correction + delta
        return correction

    def stack_head_results(
        self,
        layer: int = -1,
        return_labels: bool = False,
        incl_remainder: bool = False,
        pos_slice: Union[Slice, SliceInput] = None,
        apply_ln: bool = False,
    ) -> Union[
        Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"], List[str]],
    ]:
        """Stack Head Results.

        Returns a stack of all head results (ie residual stream contribution) up to layer L. A good
        way to decompose the outputs of attention layers into attribution by specific heads. Note
        that the num_components axis has length layer x n_heads ((layer head_index) in einops
        notation). On SSM / hybrid models only attention layers contribute heads; the other
        blocks' outputs are part of the remainder.

        Args:
            layer:
                Layer index - heads at all layers strictly before this are included. layer must be
                in [1, n_layers-1], or any of (n_layers, -1, None), which all mean the final layer.
            return_labels:
                Whether to also return a list of labels of the form "L0H0" for the heads.
            incl_remainder:
                Whether to return a final term which is "the rest of the residual stream".
            pos_slice:
                A slice object to apply to the pos dimension. Defaults to None, do nothing.
            apply_ln:
                Whether to apply LayerNorm to the stack.
        """
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        pos_slice = cast(Slice, pos_slice)  # mypy can't seem to infer this
        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers

        # Same contract as stack_neuron_results: layer=0 is the whole-model case (remainder reads
        # resid_post of layer -1, the final layer). Anything below it would wrap the remainder
        # target to an arbitrary late layer via __getitem__, silently returning the wrong
        # residual stream with no heads in the stack.
        if incl_remainder and layer < 0:
            raise ValueError(
                f"layer must be >= 0 with incl_remainder=True (-1/None mean n_layers); got {layer}."
            )

        # Idempotent; cleans up stale Bridge entries without the 'already cached' warning,
        # which is only meaningful for direct compute_head_results callers.
        self._ensure_head_results()

        components: Any = []
        labels = []
        for l in range(layer):
            # SSM / linear-attention blocks have no heads; their output lands in the remainder.
            if not self._has_attn(l):
                continue
            # Note that this has shape batch x pos x head_index x d_model
            result = pos_slice.apply(self[("result", l, "attn")], dim=-3)
            post = self._post_norm_map(l, "ln1_post", pos_slice)
            if post is not None:
                scale, weight, centers, _ = post
                result = self._apply_post_norm_map(result, scale.unsqueeze(-2), weight, centers)
            components.append(result)
            labels.extend([f"L{l}H{h}" for h in range(self.model.cfg.n_heads)])
        if components:
            components = torch.cat(components, dim=-2)
            components = einops.rearrange(
                components,
                "... concat_head_index d_model -> concat_head_index ... d_model",
            )
            if incl_remainder:
                remainder = pos_slice.apply(
                    self[("resid_post", layer - 1)], dim=-2
                ) - components.sum(dim=0)
                components = torch.cat([components, remainder[None]], dim=0)
                labels.append("remainder")
        elif incl_remainder:
            # There are no components, so the remainder is the entire thing.
            components = torch.cat(
                [pos_slice.apply(self[("resid_post", layer - 1)], dim=-2)[None]], dim=0
            )
            labels.append("remainder")
        else:
            # If this is called with layer 0, we return an empty tensor of the right shape to be
            # stacked correctly. This uses the shape of hook_embed, which is pretty janky since it
            # assumes embed is in the cache. But it's hard to explicitly code the shape, since it
            # depends on the pos slice, whether we have a batch dim, etc. And it's pretty messy!
            empty_src = pos_slice.apply(self["hook_embed"], dim=-2)
            components = torch.zeros(
                0,
                *empty_src.shape,
                dtype=empty_src.dtype,
                device=empty_src.device,
            )

        if apply_ln:
            components = self.apply_ln_to_stack(components, layer, pos_slice=pos_slice)

        if return_labels:
            return components, labels
        else:
            return components

    def stack_activation(
        self,
        activation_name: str,
        layer: int = -1,
        sublayer_type: Optional[str] = None,
    ) -> Float[torch.Tensor, "layers_covered ..."]:
        """Stack Activations.

        Flexible way to stack activations with a given name.

        Args:
            activation_name:
                The name of the activation to be stacked
            layer:
                'Layer index - heads' at all layers strictly before this are included. layer must be
                in [1, n_layers-1], or any of (n_layers, -1, None), which all mean the final layer.
            sublayer_type:
                The sub layer type of the activation, passed to utils.get_act_name. Can normally be
                inferred.
            incl_remainder:
                Whether to return a final term which is "the rest of the residual stream".
        """
        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers

        components = []
        for l in range(layer):
            components.append(self[(activation_name, l, sublayer_type)])

        return torch.stack(components, dim=0)

    def get_neuron_results(
        self,
        layer: int,
        neuron_slice: Union[Slice, SliceInput] = None,
        pos_slice: Union[Slice, SliceInput] = None,
        project_output_onto: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get Neuron Results.

        Get the results of for neurons in a specific layer (i.e, how much each neuron contributes to
        the residual stream). Does it for the subset of neurons specified by neuron_slice, defaults
        to all of them. Does *not* cache these because it's expensive in space and cheap to compute.

        Args:
            layer:
                Layer index.
            neuron_slice:
                Slice of the neuron. An int drops the neuron axis from the result.
            pos_slice:
                Slice of the positions.
            project_output_onto:
                Optional ``[d_model]`` or ``[d_model, num_outputs]`` projection. Contracted with
                ``W_out`` *before* the per-neuron expansion so the ``[..., d_mlp, d_model]``
                intermediate is never materialized.

        Returns:
            Last-dim is ``d_model`` (default), ``num_outputs`` (2D projection), or squeezed
            (1D projection).
        """
        if not isinstance(neuron_slice, Slice):
            neuron_slice = Slice(neuron_slice)
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)

        neuron_acts = self[("post", layer, "mlp")]
        block = self.model.blocks[layer]
        W_out = block.mlp.W_out
        contribution_scale = getattr(block, "residual_contribution_scale", 1.0)
        if pos_slice is not None:
            # Note - order is important, as Slice.apply *may* collapse a dimension, so this ensures
            # that position dimension is -2 when we apply position slice
            neuron_acts = pos_slice.apply(neuron_acts, dim=-2)
        post = self._post_norm_map(layer, "ln2_post", pos_slice)
        if post is not None:
            scale, weight, centers, _ = post
            # center(a_n W_out_n) / s * w = (a_n / s) * center(W_out_n) * w
            W_out = self._apply_post_norm_map(W_out, torch.ones(()), weight, centers)
            # Divide before slicing: an int neuron_slice drops the axis scale's trailing 1 broadcasts over.
            neuron_acts = (neuron_acts / scale).to(neuron_acts.dtype)
        if neuron_slice is not None:
            neuron_acts = neuron_slice.apply(neuron_acts, dim=-1)
            W_out = neuron_slice.apply(W_out, dim=0)
        if contribution_scale != 1.0:
            neuron_acts = neuron_acts * contribution_scale
        if project_output_onto is None:
            return neuron_acts[..., None] * W_out
        # W_out: [d_mlp, d_model]; project: [d_model] or [d_model, n_outs]
        projected = W_out @ project_output_onto
        if project_output_onto.ndim == 1:
            return neuron_acts * projected
        return neuron_acts[..., None] * projected

    def _uses_ln_centering(self, layer: Optional[int]) -> bool:
        """Use the target norm's semantics, not just the block normalization family."""
        is_final = layer is None or layer == self.model.cfg.n_layers
        return self.model.cfg.normalization_type in ["LN", "LNPre"] and not (
            is_final and getattr(self.model.cfg, "final_rms", False)
        )

    def _get_cached_ln_scale(
        self,
        layer: Optional[int],
        mlp_input: bool,
        pos_slice: Slice,
        batch_slice: Optional[Slice] = None,
    ) -> torch.Tensor:
        """Look up the cached LN scale and apply pos/batch slicing. Surfaces a clearer error
        when the expected hook isn't in the cache (some non-decoder-only architectures expose
        LN scale at a different path or not at all).
        """
        norm = "ln2" if mlp_input else "ln1"
        blocks = getattr(self.model, "blocks", None)
        block = (
            blocks[layer]
            if blocks is not None and layer is not None and layer != self.model.cfg.n_layers
            else None
        )
        if (
            mlp_input
            and block is not None
            and layer is not None
            and not self._mlp_input_includes_attn(layer)
            and getattr(block, "ln2", None) is None
            and getattr(block, "ln2_post", None) is None
        ):
            # Shared-norm parallel blocks feed the MLP from ln1, not a separate ln2.
            norm = "ln1"
        if layer == self.model.cfg.n_layers or layer is None:
            key = "ln_final.hook_scale"
        else:
            key = f"blocks.{layer}.{norm}.hook_scale"
        try:
            scale = self[key]
        except KeyError as e:
            if (
                block is not None
                and getattr(block, norm, None) is None
                and getattr(block, f"{norm}_post", None) is not None
            ):
                sublayer = "MLP" if mlp_input else "attention"
                raise KeyError(
                    f"Layer {layer} has no {norm} (pre-{sublayer} norm): this architecture "
                    f"normalizes sublayer outputs ({norm}_post) instead, so a mid-layer cached-scale "
                    f"LN is undefined. Use apply_ln=False, or apply LN at the final layer."
                ) from e
            raise KeyError(
                f"Cached LN scale not found at '{key}'. apply_ln operations require the model "
                f"to have cached this hook (some non-decoder-only architectures expose LN scale "
                f"under different module paths)."
            ) from e
        scale = pos_slice.apply(scale, dim=-2)
        if batch_slice is not None and self.has_batch_dim:
            scale = batch_slice.apply(scale)
        return scale

    def _stack_neuron_results_apply_ln_projected(
        self,
        layer: int,
        pos_slice: Slice,
        neuron_slice: Slice,
        project_2d: torch.Tensor,
        mlp_input: bool = False,
    ) -> torch.Tensor:
        """LN-applied neuron stack with projection folded in — no d_mlp×d_model intermediate.

        Analytical formula (LN models, cached scale ``s``):
            ``LN_s(a_n * W_out_n) @ p = (a_n / s) * (W_out_n @ p - mean(W_out_n) * sum_p)``
        RMS models drop the ``mean(W_out_n) * sum_p`` term (no centering). ``s`` is the
        MLP-input norm scale when ``mlp_input`` is set, else the ln1 scale.
        """
        scale = self._get_cached_ln_scale(layer, mlp_input=mlp_input, pos_slice=pos_slice)

        apply_centering = self._uses_ln_centering(layer)
        sum_p = project_2d.sum(dim=0) if apply_centering else None  # [n_outs]

        components: list = []
        for l in range(layer):
            if not self._has_mlp(l):
                continue
            block = self.model.blocks[l]
            W_out_l = block.mlp.W_out  # [d_mlp, d_model]
            contribution_scale = getattr(block, "residual_contribution_scale", 1.0)
            post = self._post_norm_map(l, "ln2_post", pos_slice)
            if post is not None:
                post_scale, post_weight, post_centers, _ = post
                W_out_l = self._apply_post_norm_map(
                    W_out_l, torch.ones(()), post_weight, post_centers
                )
            W_out_l_sliced = neuron_slice.apply(W_out_l, dim=0)
            W_proj_l = W_out_l_sliced @ project_2d  # [d_mlp, n_outs]
            if apply_centering:
                assert sum_p is not None  # set when apply_centering, narrow for mypy
                W_means_l = W_out_l_sliced.mean(dim=-1)  # [d_mlp]
                lin_form_l = W_proj_l - W_means_l[:, None] * sum_p[None, :]
            else:
                lin_form_l = W_proj_l
            if contribution_scale != 1.0:
                lin_form_l = lin_form_l * contribution_scale
            a_l = self[("post", l, "mlp")]
            a_l = pos_slice.apply(a_l, dim=-2)
            a_l = neuron_slice.apply(a_l, dim=-1)
            if post is not None:
                a_l = a_l / post_scale
            # (a_l / s)[..., None] is [..., d_mlp, 1]; broadcast with lin_form_l [d_mlp, n_outs]
            components.append((a_l / scale)[..., None] * lin_form_l)
        if not components:
            empty_src = pos_slice.apply(self["hook_embed"], dim=-2)
            return torch.zeros(
                0,
                *empty_src.shape[:-1],
                project_2d.shape[-1],
                dtype=torch.promote_types(
                    torch.promote_types(empty_src.dtype, scale.dtype), project_2d.dtype
                ),
                device=empty_src.device,
            )
        stacked = torch.cat(components, dim=-2)
        return einops.rearrange(
            stacked, "... concat_neuron_index n_outs -> concat_neuron_index ... n_outs"
        )

    def stack_neuron_results(
        self,
        layer: int,
        pos_slice: Union[Slice, SliceInput] = None,
        neuron_slice: Union[Slice, SliceInput] = None,
        return_labels: bool = False,
        incl_remainder: bool = False,
        apply_ln: bool = False,
        project_output_onto: Optional[torch.Tensor] = None,
        mlp_input: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
        """Stack Neuron Results

        Returns a stack of all neuron results (ie residual stream contribution) up to layer L - ie
        the amount each individual neuron contributes to the residual stream. Also returns a list of
        labels of the form "L0N0" for the neurons. A good way to decompose the outputs of MLP layers
        into attribution by specific neurons.

        Note that doing this for all neurons is SUPER expensive on GPU memory and only works for
        small models or short inputs. Pass ``project_output_onto`` to fold the projection into the
        per-neuron expansion and avoid the ``[..., d_mlp, d_model]`` intermediate.

        Args:
            layer:
                Layer index - heads at all layers strictly before this are included. layer must be
                in [1, n_layers]. ``layer=0`` follows the negative-index convention: the neuron
                stack is empty and with ``incl_remainder=True`` the remainder is the final layer's
                residual stream (``resid_post`` of layer ``n_layers - 1``), i.e. the whole model.
            pos_slice:
                Slice of the positions.
            neuron_slice:
                Slice of the neurons. An int or an integer-mode ``Slice(n)`` selects a single
                neuron and is treated like ``[n]``.
            return_labels:
                Whether to also return a list of labels of the form "L0N0" for the neurons.
            incl_remainder:
                Whether to return a final term which is "the rest of the residual stream".
            apply_ln:
                Whether to apply LayerNorm to the stack.
            project_output_onto:
                Optional ``[d_model]`` or ``[d_model, num_outputs]`` tensor. When set, each
                component's last d_model dim is replaced by the projection (memory-efficient for
                direction analyses; see ``get_neuron_results``). Combined with ``apply_ln=True``,
                the projection is folded into the analytical cached-scale LN so the
                ``[..., d_mlp, d_model]`` intermediate is still never materialized.
            mlp_input:
                Treat the stack as the input to ``layer``'s MLP rather than its attention, as in
                ``decompose_resid``: with ``incl_remainder=True`` the remainder fills the stack up to
                ``resid_mid`` for sequential blocks or ``resid_pre`` for parallel blocks.
                With ``apply_ln=True`` the scale of the norm feeding the MLP is used
                (ln2, or ln1 on shared-norm parallel blocks).
                The neurons stacked are unchanged.
        """
        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers

        # layer=0 is the documented whole-model case (remainder reads resid_post of layer -1, the
        # final layer). Anything below it would wrap the remainder target to an arbitrary late
        # layer via __getitem__, silently returning the wrong residual stream.
        if incl_remainder and layer < 0:
            raise ValueError(
                f"layer must be >= 0 with incl_remainder=True (-1/None mean n_layers); got {layer}."
            )

        # Layers are concatenated along the neuron axis, so it must not collapse. unwrap turns an
        # int into [n] but leaves a Slice as is, so an integer-mode Slice(n) is rebuilt as [n]
        # here. A new Slice is made, so the caller's Slice still collapses elsewhere.
        neuron_slice = Slice.unwrap(neuron_slice)
        if neuron_slice.mode == "int":
            assert isinstance(neuron_slice.slice, int)  # narrow for mypy
            neuron_slice = Slice([neuron_slice.slice])
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)

        project_2d, squeeze_projected = _normalize_projection_to_2d(project_output_onto)

        d_mlp = self.model.cfg.d_mlp
        assert d_mlp is not None, "model.cfg.d_mlp must be set"
        neuron_labels = neuron_slice.apply(torch.arange(d_mlp), dim=0)

        # Mamba / NemotronH blocks without an MLP have no neurons.
        mlp_layers = [l for l in range(layer) if self._has_mlp(l)]
        for l in mlp_layers:
            modules = self._block_modules(l)
            if modules is not None and "mlp" not in modules:
                names = ", ".join(n for n in modules if not n.startswith(("hook", "_")))
                raise NotImplementedError(
                    f"blocks.{l} writes an MLP-side output without an `mlp` submodule "
                    f"({names}), so its neurons cannot be expanded. Use "
                    "expand_neurons=False or decompose_resid instead."
                )
        labels = [f"L{l}N{h}" for l in mlp_layers for h in neuron_labels]
        if mlp_input and layer < self.model.cfg.n_layers:
            stream = "resid_mid" if self._mlp_input_includes_attn(layer) else "resid_pre"
            remainder_target = (stream, layer)
        else:
            remainder_target = ("resid_post", layer - 1)
        components: Any
        ln_folded = apply_ln and project_2d is not None
        if ln_folded:
            assert project_2d is not None  # narrow for mypy
            # Analytical LN+projection — no d_mlp×d_model intermediate.
            components = self._stack_neuron_results_apply_ln_projected(
                layer, pos_slice, neuron_slice, project_2d, mlp_input=mlp_input
            )
            if incl_remainder:
                # Linearity of cached-scale LN: remainder is LN_s(target) @ p - sum(neurons).
                target = pos_slice.apply(self[remainder_target], dim=-2)
                target_ln = self.apply_ln_to_stack(
                    target[None], layer, pos_slice=pos_slice, mlp_input=mlp_input
                )[0]
                remainder = target_ln @ project_2d
                if components.shape[0] > 0:
                    remainder = remainder - components.sum(dim=0)
                components = torch.cat([components, remainder[None]], dim=0)
                labels.append("remainder")
        else:
            per_layer: list = []
            for l in mlp_layers:
                per_layer.append(
                    self.get_neuron_results(
                        l,
                        pos_slice=pos_slice,
                        neuron_slice=neuron_slice,
                        project_output_onto=project_2d,
                    )
                )
            if per_layer:
                components = torch.cat(per_layer, dim=-2)
                components = einops.rearrange(
                    components,
                    "... concat_neuron_index d_model -> concat_neuron_index ... d_model",
                )
                if incl_remainder:
                    remainder_full = pos_slice.apply(self[remainder_target], dim=-2)
                    if project_2d is not None:
                        remainder_full = remainder_full @ project_2d
                    remainder = remainder_full - components.sum(dim=0)
                    components = torch.cat([components, remainder[None]], dim=0)
                    labels.append("remainder")
            elif incl_remainder:
                remainder_full = pos_slice.apply(self[remainder_target], dim=-2)
                if project_2d is not None:
                    remainder_full = remainder_full @ project_2d
                components = torch.cat([remainder_full[None]], dim=0)
                labels.append("remainder")
            else:
                empty_src = pos_slice.apply(self["hook_embed"], dim=-2)
                empty_shape = tuple(empty_src.shape)
                empty_dtype = empty_src.dtype
                if project_2d is not None:
                    # Shape only: a real matmul would demand the cache and direction share a dtype.
                    empty_shape = (*empty_shape[:-1], project_2d.shape[-1])
                    empty_dtype = torch.promote_types(empty_dtype, project_2d.dtype)
                components = torch.zeros(
                    0, *empty_shape, dtype=empty_dtype, device=empty_src.device
                )

            if apply_ln:
                components = self.apply_ln_to_stack(
                    components, layer, pos_slice=pos_slice, mlp_input=mlp_input
                )

        if squeeze_projected:
            components = components.squeeze(-1)

        if return_labels:
            return components, labels
        else:
            return components

    def apply_ln_to_stack(
        self,
        residual_stack: Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
        layer: Optional[int] = None,
        mlp_input: bool = False,
        pos_slice: Union[Slice, SliceInput] = None,
        batch_slice: Union[Slice, SliceInput] = None,
        has_batch_dim: bool = True,
        recompute_ln: bool = False,
    ) -> Float[torch.Tensor, "num_components *batch_and_pos_dims_out d_model"]:
        """Apply Layer Norm to a Stack.

        Takes a stack of components of the residual stream (eg outputs of decompose_resid or
        accumulated_resid), treats them as the input to a specific layer, and applies the layer norm
        scaling of that layer to them, using the cached scale factors - simulating what that
        component of the residual stream contributes to that layer's input.

        The layernorm scale is global across the entire residual stream for each layer, batch
        element and position, which is why we need to use the cached scale factors rather than just
        applying a new LayerNorm.

        When recompute_ln=True and the target layer is the final layer (unembed), each
        component is normalized using stats recomputed from that component; use this for logit lens
        analysis. When recompute_ln=False, a single cached scale is used for all components.

        If the model does not use LayerNorm or RMSNorm, it returns the residual stack unchanged.

        Args:
            residual_stack:
                A tensor, whose final dimension is d_model. The other trailing dimensions are
                assumed to be the same as the stored hook_scale - which may or may not include batch
                or position dimensions.
            layer:
                The layer we're taking the input to. In [0, n_layers], n_layers means the unembed.
                None maps to the n_layers case, ie the unembed.
            mlp_input:
                Whether the input is to the MLP or attention. The MLP uses ln2 when present,
                or ln1 on shared-norm parallel blocks. Defaults to False, ie ln1.
                If layer==n_layers, must be False, and we use ln_final
            pos_slice:
                The slice to take of positions, if residual_stack is not over the full context, None
                means do nothing. It is assumed that pos_slice has already been applied to
                residual_stack, and this is only applied to the scale. See utils.Slice for details.
                Defaults to None, do nothing.
            batch_slice:
                The slice to take on the batch dimension. Defaults to None, do nothing.
            has_batch_dim:
                Whether residual_stack has a batch dimension.
            recompute_ln:
                If True and target layer is the unembed (final layer), apply the final layer norm
                to each component with statistics recomputed from that component. Defaults to False.

        """
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        if not isinstance(batch_slice, Slice):
            batch_slice = Slice(batch_slice)

        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers

        if has_batch_dim:
            # Apply batch slice to the stack
            residual_stack = batch_slice.apply(residual_stack, dim=1)

        # Checked only after slicing so callers like logit_attrs still get their
        # batch_slice honoured on models without LayerNorm/RMSNorm.
        if self.model.cfg.normalization_type not in ["LN", "LNPre", "RMS", "RMSPre"]:
            return residual_stack

        # Logit lens: apply final layer norm to each component with recomputed statistics
        if recompute_ln and layer == self.model.cfg.n_layers and hasattr(self.model, "ln_final"):
            ln_final = self.model.ln_final
            results = []
            for i in range(residual_stack.shape[0]):
                x = residual_stack[i]
                original_shape = x.shape
                # ln_final expects (batch, pos, d_model); restore missing structural dimensions
                if not has_batch_dim:
                    x = x.unsqueeze(0)
                if x.ndim == 2:
                    x = x.unsqueeze(1)
                out = ln_final(x)
                results.append(out.reshape(original_shape))
            return torch.stack(results, dim=0)

        # Final RMSNorm must not inherit the block LayerNorm centering convention.
        if self._uses_ln_centering(layer):
            residual_stack = residual_stack - residual_stack.mean(dim=-1, keepdim=True)

        # Shape is [batch, position, 1] or [position, 1]; final dim is a dummy for broadcasting.
        scale = self._get_cached_ln_scale(layer, mlp_input, pos_slice, batch_slice)

        return residual_stack / scale

    def get_full_resid_decomposition(
        self,
        layer: Optional[int] = None,
        mlp_input: bool = False,
        expand_neurons: bool = True,
        apply_ln: bool = False,
        pos_slice: Union[Slice, SliceInput] = None,
        return_labels: bool = False,
        project_output_onto: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
        """Get the full Residual Decomposition.

        Decomposes the residual stream that is input into some layer into its
        constituent components: every attention head result, every neuron (or
        MLP layer) result, the embeddings, and the accumulated biases.

        The returned tensor stacks components along ``dim=0`` in this order:

        1. Attention head results, layer-by-layer (``L * n_heads`` rows; attention
           layers only on SSM / hybrid models)
        2. Sequence-mixer results of blocks without attention heads (SSM / hybrid
           models only; one row per mixer, e.g. ``0_linear_attn_out``, output bias
           excluded since it is in the ``bias`` row). MLP-type passthrough mixers
           (NemotronH) appear here only when ``expand_neurons=True``; otherwise they
           are part of the MLP rows below.
        3. Neuron / MLP results (only if ``cfg.attn_only=False`` and
           ``layer > 0``; ``d_mlp`` rows per block with an MLP when
           ``expand_neurons=True``, else one row per MLP output)
        4. ``embed`` (1 row, including any post-embedding runtime multiplier)
        5. ``pos_embed`` (1 row, if the model has positional embeddings)
        6. ``bias`` (1 row, the accumulated layer biases)

        ``return_labels=True`` returns a list of strings in the same order, so
        ``labels[i]`` always names ``stack[i]``. If you need to extract a
        specific component, slice by label rather than by hard-coded index —
        the row counts depend on ``layer``, ``expand_neurons``,
        ``cfg.attn_only``, and whether the model has positional embeddings.

        Args:
            layer:
                The layer we're inputting into. layer is in [0, n_layers], if layer==n_layers (or
                None) we're inputting into the unembed (the entire stream), if layer==0 then it's
                just embed and pos_embed
            mlp_input:
                Are we inputting to the MLP in that layer or the attn? Must be False for final
                layer, since that's the unembed.
            expand_neurons:
                Whether to expand the MLP outputs to give every neuron's result or just return the
                MLP layer outputs.
            apply_ln:
                Whether to apply LayerNorm to the stack.
            pos_slice:
                Slice of the positions to take.
            return_labels:
                Whether to return the labels.
            project_output_onto:
                Optional ``[d_model]`` or ``[d_model, num_outputs]`` projection. Folded in
                *before* the per-neuron expansion, so the ``[..., d_mlp, d_model]`` intermediate
                is never materialized (memory saving applies only with ``expand_neurons=True``).
                Combined with ``apply_ln=True``, the projection is fused into the analytical
                cached-scale LN so the same memory benefit holds. Output last-dim is squeezed
                for a 1D projection; ``num_outputs`` for 2D.
        """
        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers
        assert layer is not None  # keep mypy happy

        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)

        project_2d, squeeze_projected = _normalize_projection_to_2d(project_output_onto)
        # When both apply_ln and projection are requested, LN is applied per-component (in
        # d_model space for the small ones, analytically for neurons) before projection, so the
        # final apply_ln_to_stack call is skipped — last-dim is already n_outs.
        ln_folded = apply_ln and project_2d is not None

        def _ln_then_project(stack: torch.Tensor) -> torch.Tensor:
            stack = self.apply_ln_to_stack(stack, layer, pos_slice=pos_slice, mlp_input=mlp_input)
            return stack @ project_2d if project_2d is not None else stack

        head_stack, head_labels = self.stack_head_results(
            layer + int(mlp_input and self._mlp_input_includes_attn(layer)),
            pos_slice=pos_slice,
            return_labels=True,
        )
        if ln_folded:
            head_stack = _ln_then_project(head_stack)
        elif project_2d is not None:
            head_stack = head_stack @ project_2d
        labels = head_labels
        components = [head_stack]
        # Blocks without attention heads: add each mixer output whole. Its output bias is dropped
        # here because the `bias` row below already accumulates it. MLP-type mixers come from
        # the MLP path when it is not expanded into neurons.
        mixer_layers = [l for l in range(layer) if expand_neurons or self._mixer_role(l) == "attn"]
        if mlp_input and self._mlp_input_includes_attn(layer) and self._mixer_role(layer) == "attn":
            mixer_layers.append(layer)
        mixer_list: List[torch.Tensor] = []
        for l in mixer_layers:
            mixers, mixer_labels = self._mixer_stack(l, pos_slice, subtract_bias=True)
            mixer_list.extend(mixers)
            labels.extend(mixer_labels)
        if mixer_list:
            mixer_stack = torch.stack(mixer_list, dim=0)
            if ln_folded:
                mixer_stack = _ln_then_project(mixer_stack)
            elif project_2d is not None:
                mixer_stack = mixer_stack @ project_2d
            components.append(mixer_stack)
        if not self.model.cfg.attn_only and layer > 0:
            if expand_neurons:
                # Only ask stack_neuron_results to apply LN when we want the fused analytical
                # path (ln_folded). For the unfolded case the outer apply_ln_to_stack handles it.
                neuron_stack, neuron_labels = self.stack_neuron_results(
                    layer,
                    pos_slice=pos_slice,
                    return_labels=True,
                    apply_ln=ln_folded,
                    project_output_onto=project_2d,
                    mlp_input=mlp_input,
                )
                labels.extend(neuron_labels)
                components.append(neuron_stack)
            else:
                # Get the stack of just the MLP outputs
                # mlp_input included for completeness, but it doesn't actually matter, since it's
                # just for MLP outputs
                mlp_stack, mlp_labels = self.decompose_resid(
                    layer,
                    mlp_input=mlp_input,
                    pos_slice=pos_slice,
                    incl_embeds=False,
                    mode="mlp",
                    return_labels=True,
                )
                if ln_folded:
                    mlp_stack = _ln_then_project(mlp_stack)
                elif project_2d is not None:
                    mlp_stack = mlp_stack @ project_2d
                labels.extend(mlp_labels)
                components.append(mlp_stack)

        if self.has_embed:
            embed = pos_slice.apply(self["hook_embed"], -2)[None]
            if ln_folded:
                embed = _ln_then_project(embed)
            elif project_2d is not None:
                embed = embed @ project_2d
            labels.append("embed")
            components.append(embed)
        if self.has_pos_embed:
            pos_embed = pos_slice.apply(self["pos_embed"], -2)[None]
            if ln_folded:
                pos_embed = _ln_then_project(pos_embed)
            elif project_2d is not None:
                pos_embed = pos_embed @ project_2d
            labels.append("pos_embed")
            components.append(pos_embed)
        # If we didn't expand the neurons, the MLP biases are already included in the MLP outputs.
        bias_full = self.model.accumulated_bias(layer, mlp_input, include_mlp_biases=expand_neurons)
        bias_correction = self._post_norm_bias_correction(
            layer, mlp_input, include_mlp_biases=expand_neurons, pos_slice=pos_slice
        )
        if ln_folded:
            # Expand bias to per-position d_model shape so LN can center, then project.
            expand_shape: tuple = (1,) + tuple(head_stack.shape[1:-1]) + (self.model.cfg.d_model,)
            bias_d_model = bias_full.expand(expand_shape)
            if bias_correction is not None:
                bias_d_model = bias_d_model + bias_correction
            bias = _ln_then_project(bias_d_model)
        else:
            if bias_correction is not None:
                bias_full = bias_full + bias_correction
            if project_2d is not None:
                # Bias is [d_model], so project post-hoc for shape compatibility — no memory win here.
                bias_full = bias_full @ project_2d
            bias = bias_full.expand((1,) + head_stack.shape[1:])
        labels.append("bias")
        components.append(bias)
        residual_stack = torch.cat(components, dim=0)
        if apply_ln and not ln_folded:
            residual_stack = self.apply_ln_to_stack(
                residual_stack, layer, pos_slice=pos_slice, mlp_input=mlp_input
            )

        if squeeze_projected:
            residual_stack = residual_stack.squeeze(-1)

        if return_labels:
            return residual_stack, labels
        else:
            return residual_stack


StepFilter = Optional[Union[int, Iterable[int], Callable[[int], bool]]]


def _normalize_step_filter(cache_steps: StepFilter) -> Callable[[int], bool]:
    """Normalize ``cache_steps`` into a boolean predicate over 0-based denoising step indices."""
    if cache_steps is None:
        return lambda _step: True
    if isinstance(cache_steps, bool):
        raise TypeError("cache_steps must be an int, iterable of ints, callable, or None; got bool")
    if isinstance(cache_steps, int):
        if cache_steps < 0:
            raise ValueError(f"cache_steps index must be non-negative; got {cache_steps}")
        target = cache_steps
        return lambda step: step == target
    if callable(cache_steps):
        return cache_steps
    if isinstance(cache_steps, Iterable) and not isinstance(cache_steps, (str, bytes)):
        steps_set: set[int] = set()
        for item in cache_steps:
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"cache_steps iterable must contain only non-negative ints; got {item!r}"
                )
            if item < 0:
                raise ValueError(f"cache_steps index must be non-negative; got {item}")
            steps_set.add(item)
        frozen = frozenset(steps_set)
        return lambda step: step in frozen
    raise TypeError(
        f"cache_steps must be an int, iterable of ints, callable, or None; got {type(cache_steps).__name__}"
    )


class DiffusionTrajectoryCache(Mapping[int, ActivationCache]):
    """Step-indexed activation caches recorded across a diffusion denoising trajectory.

    Autoregressive :meth:`~transformer_lens.model_bridge.TransformerBridge.generate` can build
    its cache from a single post-hoc pass on the completed sequence because causal masking
    keeps earlier positions invariant as new tokens are appended. Discrete diffusion LMs
    (Dream, LLaDA, LLaDA-2, Gidd) use bidirectional attention and iteratively unmask a canvas
    over ``S`` denoising steps: step ``s``'s activations depend on the partially-masked canvas
    ``x_s`` and cannot be recovered from the final unmasked sequence ``x_S``. Attaching
    standard ``add_caching_hooks`` across ``diffusion_generate`` would overwrite ``cache[name]``
    on every forward pass, silently discarding steps ``0..S-2``.

    ``DiffusionTrajectoryCache`` holds one :class:`ActivationCache` per captured denoising step,
    along with the input token canvas ``step_tokens[s]`` fed into step ``s``.
    """

    def __init__(
        self,
        step_caches: Mapping[int, ActivationCache],
        *,
        step_tokens: Optional[Mapping[int, torch.Tensor]] = None,
        num_steps_run: int = 0,
        model: Optional[Any] = None,
    ) -> None:
        self.step_caches: Dict[int, ActivationCache] = {
            int(step): step_caches[step] for step in sorted(step_caches)
        }
        self.step_tokens: Dict[int, torch.Tensor] = (
            {int(step): step_tokens[step] for step in sorted(step_tokens)}
            if step_tokens is not None
            else {}
        )
        self.num_steps_run: int = max(
            int(num_steps_run),
            (max(self.step_caches) + 1) if self.step_caches else 0,
        )
        first_cache = next(iter(self.step_caches.values()), None)
        self.model: Optional[Any] = (
            model if model is not None else getattr(first_cache, "model", None)
        )

    @property
    def steps(self) -> Tuple[int, ...]:
        """Captured denoising step indices in ascending chronological order."""
        return tuple(self.step_caches.keys())

    def __getitem__(self, step: int) -> ActivationCache:
        if isinstance(step, bool) or not isinstance(step, int):
            raise KeyError(f"Step index must be an int; got {step!r}")
        if step not in self.step_caches:
            raise KeyError(
                f"Denoising step {step} was not captured (captured steps: {list(self.steps)})"
            )
        return self.step_caches[step]

    def __iter__(self) -> Iterator[int]:
        return iter(self.step_caches)

    def __len__(self) -> int:
        return len(self.step_caches)

    def __contains__(self, step: object) -> bool:
        return not isinstance(step, bool) and isinstance(step, int) and step in self.step_caches

    def __repr__(self) -> str:
        sample_cache = next(iter(self.step_caches.values()), None)
        n_hooks = len(sample_cache) if sample_cache is not None else 0
        return (
            f"DiffusionTrajectoryCache(steps={list(self.steps)}, "
            f"num_steps_run={self.num_steps_run}, hooks_per_step={n_hooks})"
        )

    def stack(
        self,
        key: str,
        steps: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Stack ``cache[step][key]`` across ``steps`` (default: all captured steps) along dim 0."""
        selected = tuple(self.steps if steps is None else steps)
        if not selected:
            raise ValueError("Cannot stack an empty sequence of denoising steps")
        return torch.stack([self[step][key] for step in selected], dim=0)

    def to(self, device: Union[str, torch.device]) -> "DiffusionTrajectoryCache":
        """Move every per-step :class:`ActivationCache` and ``step_tokens`` tensor to ``device``."""
        for step_cache in self.step_caches.values():
            step_cache.to(device)
        for step, tokens in list(self.step_tokens.items()):
            self.step_tokens[step] = tokens.to(device)
        return self

    def make_step_patch_hook(
        self,
        hook_name: str,
        *,
        steps: StepFilter = None,
        positions: Optional[Union[int, Sequence[int], slice]] = None,
        pos_dim: int = 1,
    ) -> Callable[..., Optional[torch.Tensor]]:
        """Build a forward hook that patches ``hook_name`` from this trajectory at selected steps.

        When attached via ``with bridge.hooks(fwd_hooks=[(hook_name, hook_fn)]):`` around
        :meth:`~transformer_lens.model_bridge.TransformerBridge.diffusion_generate`, the hook
        reads ``bridge.current_diffusion_step`` on each forward pass and substitutes the source
        activation recorded at that step. Steps not matching ``steps`` (or not present in this
        cache) pass through untouched.
        """
        step_predicate = _normalize_step_filter(steps)
        for step, step_cache in self.step_caches.items():
            if step_predicate(step) and hook_name not in step_cache:
                raise KeyError(
                    f"Hook {hook_name!r} is not present in captured step {step} "
                    f"(available keys: {list(step_cache.keys())})"
                )

        def _patch_hook(tensor: torch.Tensor, *, hook: Any) -> Optional[torch.Tensor]:
            current_step = getattr(self.model, "current_diffusion_step", None)
            if current_step is None and hook is not None:
                current_step = getattr(hook, "ctx", {}).get("diffusion_step")
            if current_step is None or not step_predicate(current_step):
                return None
            if current_step not in self.step_caches:
                return None
            source = self.step_caches[current_step][hook_name].to(
                device=tensor.device, dtype=tensor.dtype
            )
            if positions is None:
                return source.clone()
            patched = tensor.clone()
            axis = pos_dim if pos_dim >= 0 else tensor.ndim + pos_dim
            idx: List[Any] = [slice(None)] * tensor.ndim
            idx[axis] = positions
            index_tuple = tuple(idx)
            if source.shape == patched[index_tuple].shape:
                patched[index_tuple] = source
            else:
                patched[index_tuple] = source[index_tuple]
            return patched

        return _patch_hook
