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
