"""Direct Logit Attribution: residual-stream terms dotted with the answer's unembedding direction.
"layer" units are cumulative (the last is the full stream); "head" adds a non-head remainder."""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from jaxtyping import Float

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.utilities import SliceInput

TokenInput = Union[str, int, torch.Tensor]

Unit = str

_VALID_UNITS = ("component", "layer", "head")

# Refused up front: decompose_resid would otherwise fail on these with a confusing KeyError.
_HYBRID_VARIANT_NAMES = ("mamba", "ssm", "mixer", "linear_attn")


@dataclass
class DirectLogitAttribution:
    """Result of a :func:`direct_logit_attribution` call."""

    # Leading axis aligned with labels; a single position drops the position axis, None keeps it.
    attribution: Float[torch.Tensor, "component *batch_and_pos"]
    labels: List[str]
    unit: Unit

    def top(self, k: int = 5) -> List[tuple]:
        """Top ``k`` ``(label, value)`` pairs, meaning over any remaining batch/position dims."""
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
    """Residual stack and labels for ``unit``, unscaled: ``logit_attrs`` applies the final LN."""
    if unit == "component":
        return cache.decompose_resid(apply_ln=False, pos_slice=pos_slice, return_labels=True)
    if unit == "layer":
        return cache.accumulated_resid(
            apply_ln=False, incl_mid=True, pos_slice=pos_slice, return_labels=True
        )
    if unit == "head":
        return cache.stack_head_results(
            apply_ln=False, pos_slice=pos_slice, incl_remainder=True, return_labels=True
        )
    raise ValueError(f"unit must be one of {_VALID_UNITS}, got {unit!r}")


def _validate_bridge_compatibility(model) -> None:
    """Reject Bridges with an active final-norm affine or hybrid blocks; non-Bridges pass.
    Compat mode alone is not enough: ``fold_ln=False`` / ``no_processing=True`` skip the fold."""
    from transformer_lens.model_bridge import TransformerBridge

    if not isinstance(model, TransformerBridge):
        return

    if not getattr(model, "compatibility_mode", False):
        raise ValueError(
            "DLA on a TransformerBridge requires compatibility mode and an identity "
            "final norm. Call `model.enable_compatibility_mode()` with its default "
            "weight processing after loading the bridge, then re-run DLA."
        )

    final_norm = getattr(model, "ln_final", None)
    if final_norm is not None:
        weight = getattr(final_norm, "weight", None)
        if isinstance(weight, torch.Tensor):
            identity = (
                torch.zeros_like(weight)
                if getattr(model.cfg, "rmsnorm_uses_offset", False)
                else torch.ones_like(weight)
            )
            if not torch.equal(weight, identity):
                raise ValueError(
                    "DLA requires an identity final norm weight: the learned scale is not "
                    "folded into W_U. Load a fresh bridge and call "
                    "`enable_compatibility_mode()` with fold_ln=True and no_processing=False."
                )
        bias = getattr(final_norm, "bias", None)
        if isinstance(bias, torch.Tensor) and not torch.equal(bias, torch.zeros_like(bias)):
            raise ValueError(
                "DLA requires a zero final norm bias: the learned bias is not folded "
                "into b_U. Load a fresh bridge and call `enable_compatibility_mode()` "
                "with fold_ln=True and no_processing=False."
            )

    layer_types = model.layer_types()
    hybrid = [lt for lt in layer_types if any(p in _HYBRID_VARIANT_NAMES for p in lt.split("+"))]
    if hybrid:
        raise NotImplementedError(
            f"DLA does not yet support hybrid architectures (found block types {hybrid}). "
            f"Only standard attention + MLP transformers (e.g. GPT-2, LLaMA, Pythia) are "
            f"supported; hybrid support requires extending ActivationCache.decompose_resid."
        )


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
    """DLA of ``answer_tokens`` (or ``answer - incorrect``) over ``unit`` components.
    Components sum to ``logit - b_U``, not the raw logit, since no component produces ``b_U``."""
    if unit not in _VALID_UNITS:
        raise ValueError(f"unit must be one of {_VALID_UNITS}, got {unit!r}")
    if answer_tokens is None:
        raise ValueError("answer_tokens is required")

    _validate_bridge_compatibility(model)

    if cache is None:
        if input is None:
            raise ValueError("provide either `input` to run the model, or a precomputed `cache`")
        _, cache = model.run_with_cache(input)

    residual_stack, labels = _residual_stack_and_labels(cache, unit, pos)

    # logit_attrs scales by the final LN, so it needs the same pos slice as the stack.
    attribution = cache.logit_attrs(
        residual_stack,
        tokens=answer_tokens,
        incorrect_tokens=incorrect_tokens,
        pos_slice=pos,
        has_batch_dim=cache.has_batch_dim,
    )

    return DirectLogitAttribution(attribution=attribution, labels=labels, unit=unit)
