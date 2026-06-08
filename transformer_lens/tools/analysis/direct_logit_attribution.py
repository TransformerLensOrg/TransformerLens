"""Direct Logit Attribution (DLA).

Direct Logit Attribution decomposes a model's output logit (or a logit
*difference* between a correct and an incorrect token) into the additive
contributions of upstream components — the embedding, each attention and MLP
sublayer, or each individual attention head. Because the unembedding is linear
and the residual stream is a sum of component outputs, the final logit is
(up to the final LayerNorm) a sum of per-component dot products with the
unembedding direction of the token of interest. DLA reads off those dot
products. See the `logit lens
<https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>`_
and `Interpretability in the Wild <https://arxiv.org/abs/2211.00593>`_ for the
canonical uses.

This module exposes a single entry point, :func:`direct_logit_attribution`,
that wraps the lower-level ``ActivationCache`` primitives
(:meth:`~transformer_lens.ActivationCache.ActivationCache.decompose_resid`,
:meth:`~transformer_lens.ActivationCache.ActivationCache.accumulated_resid`,
:meth:`~transformer_lens.ActivationCache.ActivationCache.stack_head_results`
and :meth:`~transformer_lens.ActivationCache.ActivationCache.logit_attrs`) into
one call. It works unchanged with both ``HookedTransformer`` and
``TransformerBridge`` because they share the cache API.

Example::

    from transformer_lens import HookedTransformer
    from transformer_lens.tools.analysis import direct_logit_attribution

    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    result = direct_logit_attribution(
        model,
        "The Eiffel Tower is in the city of",
        answer_tokens=" Paris",
        incorrect_tokens=" London",
        unit="component",
    )
    for label, value in zip(result.labels, result.attribution.squeeze()):
        print(f"{label:>12}: {value.item():+.3f}")
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from jaxtyping import Float

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.utilities import SliceInput

# Token-like inputs accepted for the correct/incorrect answers, mirroring
# ActivationCache.logit_attrs.
TokenInput = Union[
    str,
    int,
    torch.Tensor,
]

# Which structural unit the residual stream is decomposed into.
Unit = str  # one of: "component", "layer", "head"

_VALID_UNITS = ("component", "layer", "head")


@dataclass
class DirectLogitAttribution:
    """Result of a :func:`direct_logit_attribution` call.

    Attributes:
        attribution:
            Tensor of logit (or logit-difference) attributions with shape
            ``[component, *batch_and_pos]``. The leading axis is aligned with
            ``labels``. When ``pos`` selects a single position (the default) the
            position axis is dropped, leaving ``[component, batch]`` — or
            ``[component]`` if the cache had its batch dimension removed.
        labels:
            Human-readable name for each component, aligned with the leading
            axis of ``attribution`` (e.g. ``"embed"``, ``"0_attn_out"``,
            ``"L3H7"``).
        unit:
            The decomposition unit used ("component", "layer", or "head").
    """

    attribution: Float[torch.Tensor, "component *batch_and_pos"]
    labels: List[str]
    unit: Unit

    def top(self, k: int = 5) -> List[tuple]:
        """Return the ``k`` highest-attribution ``(label, value)`` pairs.

        Attribution is reduced to a scalar per component by meaning over any
        remaining batch/position dimensions, so this is most meaningful when a
        single position was selected.
        """
        flat = self.attribution
        if flat.ndim > 1:
            flat = flat.flatten(start_dim=1).mean(dim=-1)
        values, indices = torch.topk(flat, min(k, flat.shape[0]))
        return [(self.labels[i], values[j].item()) for j, i in enumerate(indices.tolist())]


def _residual_stack_and_labels(
    cache: ActivationCache,
    unit: Unit,
    pos_slice: SliceInput,
):
    """Decompose the residual stream into ``unit`` components plus labels.

    LayerNorm is intentionally *not* applied here — ``logit_attrs`` applies the
    final-layer scaling itself, so applying it twice would double-count.
    """
    if unit == "component":
        # embed (+ pos_embed) and each layer's attn_out / mlp_out.
        return cache.decompose_resid(apply_ln=False, pos_slice=pos_slice, return_labels=True)
    if unit == "layer":
        # Cumulative residual stream after each sublayer — logit-lens style.
        return cache.accumulated_resid(
            apply_ln=False, incl_mid=True, pos_slice=pos_slice, return_labels=True
        )
    if unit == "head":
        # Each attention head's contribution, plus the MLP/embedding remainder.
        return cache.stack_head_results(
            apply_ln=False, pos_slice=pos_slice, incl_remainder=True, return_labels=True
        )
    raise ValueError(f"unit must be one of {_VALID_UNITS}, got {unit!r}")


def direct_logit_attribution(
    model,
    input: Union[str, List[str], torch.Tensor, None] = None,
    answer_tokens: Optional[TokenInput] = None,
    incorrect_tokens: Optional[TokenInput] = None,
    *,
    unit: Unit = "component",
    pos: SliceInput = -1,
    cache: Optional[ActivationCache] = None,
) -> DirectLogitAttribution:
    """Compute Direct Logit Attribution for a prompt.

    Decomposes the contribution of model components to the logit of
    ``answer_tokens`` (or, if ``incorrect_tokens`` is given, to the logit
    *difference* ``answer - incorrect`` along the ``W_U`` direction, which is
    usually what you want for circuit analysis).

    The model is run once with caching unless a precomputed ``cache`` is passed.
    Works with both ``HookedTransformer`` and ``TransformerBridge``.

    Note that DLA attributes only the part of a logit that comes from the
    residual stream through the unembedding direction; the unembedding bias
    ``b_U`` is a per-token constant that no component produces. So a complete
    decomposition reconstructs ``logit[token] - b_U[token]`` rather than the raw
    logit.

    Args:
        model:
            A ``HookedTransformer`` or ``TransformerBridge``.
        input:
            Prompt to run — a string, list of strings, or token tensor. Optional
            only when a precomputed ``cache`` is supplied.
        answer_tokens:
            The correct token(s) to attribute, as a string, id, or tensor. A
            string is converted with ``model.to_single_token``.
        incorrect_tokens:
            Optional baseline token(s). When given, attribution is computed for
            the ``answer - incorrect`` residual direction. Must broadcast to the
            same shape as ``answer_tokens``.
        unit:
            Decomposition granularity:

            - ``"component"`` (default): embedding + each layer's attention and
              MLP output (via ``decompose_resid``).
            - ``"layer"``: cumulative residual stream after each sublayer, i.e.
              logit-lens trajectory (via ``accumulated_resid``).
            - ``"head"``: each attention head individually, plus a remainder
              term for everything else (via ``stack_head_results``).
        pos:
            Sequence position(s) to attribute. Defaults to ``-1`` (the final
            token, the usual choice for next-token DLA). Pass ``None`` to keep
            every position (the result then has a trailing position axis).
        cache:
            Optional precomputed ``ActivationCache`` to reuse instead of running
            the model again.

    Returns:
        A :class:`DirectLogitAttribution` with ``attribution`` (shape
        ``[component, *batch_and_pos]``) and aligned ``labels``.

    Raises:
        ValueError: If ``unit`` is invalid, ``answer_tokens`` is ``None``, or
            neither ``input`` nor ``cache`` is provided.
    """
    if unit not in _VALID_UNITS:
        raise ValueError(f"unit must be one of {_VALID_UNITS}, got {unit!r}")
    if answer_tokens is None:
        raise ValueError("answer_tokens is required")

    if cache is None:
        if input is None:
            raise ValueError("provide either `input` to run the model, or a precomputed `cache`")
        _, cache = model.run_with_cache(input)

    residual_stack, labels = _residual_stack_and_labels(cache, unit, pos)

    # logit_attrs applies the final LayerNorm scaling (with the same pos slice)
    # and dots each component against the (correct - incorrect) unembed direction.
    attribution = cache.logit_attrs(
        residual_stack,
        tokens=answer_tokens,
        incorrect_tokens=incorrect_tokens,
        pos_slice=pos,
        has_batch_dim=cache.has_batch_dim,
    )

    return DirectLogitAttribution(attribution=attribution, labels=labels, unit=unit)
"""Direct Logit Attribution.

Decomposes a model's logit difference into per-component contributions from
the residual stream. See :func:`dla` for usage.
"""

from typing import List, Tuple

import einops
import torch
from jaxtyping import Float, Int

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities import get_act_name

# Block variants that lack the attn_out + mlp_out structure decompose_resid expects.
_HYBRID_VARIANT_NAMES = ("mamba", "ssm", "mixer", "linear_attn")


def dla(
    bridge: TransformerBridge,
    prompts: List[str],
    answer_tokens: Int[torch.Tensor, "batch answers"],
    accumulated: bool = False,
) -> Tuple[Float[torch.Tensor, "component"], List[str]]:
    """Compute Direct Logit Attribution for a TransformerBridge model.

    Decomposes the logit (or logit difference between a correct and wrong token)
    into per-component contributions from the residual stream, averaged across
    the batch of prompts. Two modes:

    * ``accumulated=False`` (default): the contribution of each individual
      component — embedding, per-layer attention output, per-layer MLP output —
      via :meth:`transformer_lens.ActivationCache.decompose_resid`. These are
      additive: their sum reconstructs the (bias-excluded) logit difference.
    * ``accumulated=True``: the cumulative residual stream at each layer boundary
      (logit-lens style) via
      :meth:`transformer_lens.ActivationCache.accumulated_resid`. These are
      cumulative, so the full reconstruction is the *last* entry, not the sum.

    Warning:

    Returned scores sum (decompose mode) or end (accumulated mode) at
    ``actual_logit_diff - (b_U[correct] - b_U[wrong])``, not ``actual_logit_diff``
    directly. The unembedding bias :math:`b_U` is a constant offset added after
    unembedding rather than a residual-stream contribution, so it is excluded by
    convention — matching :meth:`transformer_lens.ActivationCache.logit_attrs`.

    Warning:

    This function requires the bridge to be in compatibility mode so the final
    LayerNorm weights are folded into :math:`W_U`. Without folding, the
    projection direction is wrong and the scores do not reflect actual logit
    contributions. Call ``bridge.enable_compatibility_mode()`` after loading.

    Warning:

    Hybrid architectures (Mamba, SSM, Mixer, LinearAttention) are not yet
    supported and raise :class:`NotImplementedError`; support requires extending
    :meth:`transformer_lens.ActivationCache.decompose_resid` to handle
    non-attention blocks.

    Args:
        bridge:
            A :class:`transformer_lens.model_bridge.TransformerBridge` with
            compatibility mode enabled.
        prompts:
            Prompt strings to evaluate. Length must match ``answer_tokens.shape[0]``.
        answer_tokens:
            Tensor of shape ``(batch, answers)``. ``answers == 1`` decomposes the
            single target token's logit; ``answers == 2`` treats each row as a
            ``(correct, wrong)`` pair and decomposes the logit difference.
        accumulated:
            If ``True``, return cumulative per-layer contributions (logit lens).
            If ``False`` (default), return per-component contributions.

    Returns:
        ``(scores, labels)``: ``scores`` is a 1D tensor of per-component (or
        per-layer) contributions averaged across the batch, and ``labels`` is the
        matching list of human-readable component names.

    Raises:
        ValueError: If ``len(prompts)`` does not match ``answer_tokens.shape[0]``,
            if ``answer_tokens.shape[1]`` is not ``1`` or ``2``, or if the bridge
            is not in compatibility mode.
        NotImplementedError: If the bridge contains hybrid (non attention + MLP)
            blocks.
    """

    # input validation
    if len(prompts) != answer_tokens.shape[0]:
        raise ValueError(
            "Each prompt needs a matching row of answer tokens: got "
            f"{len(prompts)} prompts but {answer_tokens.shape[0]} answer-token rows."
        )
    if answer_tokens.shape[1] not in (1, 2):
        raise ValueError(
            "answer_tokens must have 1 (single token) or 2 (correct, wrong) columns, "
            f"got {answer_tokens.shape[1]}."
        )

    # safeguard: DLA needs LayerNorm folded into W_U, which compatibility mode does
    if not getattr(bridge, "compatibility_mode", False):
        raise ValueError(
            "DLA requires the bridge to be in compatibility mode so that LayerNorm "
            "weights are folded into W_U. Call `bridge.enable_compatibility_mode()` "
            "after loading the bridge, then re-run DLA."
        )

    # safeguard: hybrid blocks (Mamba/SSM/...) have no attn_out + mlp_out to decompose
    hybrid_blocks = [
        layer_type
        for layer_type in bridge.layer_types()
        if any(part in _HYBRID_VARIANT_NAMES for part in layer_type.split("+"))
    ]
    if hybrid_blocks:
        raise NotImplementedError(
            f"DLA does not yet support hybrid architectures (found block types "
            f"{hybrid_blocks}). Only standard attention + MLP transformers (e.g. "
            f"GPT-2, LLaMA, Pythia) are supported; hybrid support requires extending "
            f"ActivationCache.decompose_resid."
        )

    # unembedding direction per prompt; single-token tensors collapse to [d_model]
    answer_residual_directions = bridge.tokens_to_residual_directions(answer_tokens).reshape(
        answer_tokens.shape[0], answer_tokens.shape[1], -1
    )
    if answer_tokens.shape[1] == 1:
        logit_diff_directions: Float[torch.Tensor, "batch d_model"] = answer_residual_directions[
            :, 0, :
        ]
    else:  # (correct, wrong) pair -> direction of the logit difference
        correct_direction, wrong_direction = answer_residual_directions.unbind(dim=1)
        logit_diff_directions = correct_direction - wrong_direction

    def residual_stack_to_logit_diff(
        residual_stack: Float[torch.Tensor, "... batch d_model"],
        cache: ActivationCache,
    ) -> Float[torch.Tensor, "..."]:
        # apply the final LayerNorm once, project onto the answer direction, average over prompts
        batch_size = residual_stack.size(-2)
        scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
        return (
            einops.einsum(
                scaled_residual_stack,
                logit_diff_directions,
                "... batch d_model, batch d_model -> ...",
            )
            / batch_size
        )

    if accumulated:
        n_layers = bridge.cfg.n_layers
        _, cache = bridge.run_with_cache(
            prompts,
            names_filter=lambda name: name == get_act_name("resid_post", n_layers - 1)
            or name == "ln_final.hook_scale"
            or name.endswith("resid_pre")
            or name.endswith("resid_mid"),
        )
        accumulated_residual, labels = cache.accumulated_resid(
            layer=-1, pos_slice=-1, incl_mid=True, return_labels=True
        )
        return residual_stack_to_logit_diff(accumulated_residual, cache), labels

    _, cache = bridge.run_with_cache(
        prompts,
        names_filter=lambda name: name == "ln_final.hook_scale"
        or name in ("hook_embed", "hook_pos_embed")
        or name.endswith("attn_out")
        or name.endswith("mlp_out"),
    )
    per_component_residual, labels = cache.decompose_resid(
        layer=-1, pos_slice=-1, return_labels=True
    )
    return residual_stack_to_logit_diff(per_component_residual, cache), labels
