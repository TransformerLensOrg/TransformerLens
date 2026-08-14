"""Guards for weight-space code paths that cannot read quantized weights.

TransformerLens supports quantized *forward* passes: the wrapped HF module
dequantizes internally, and the bridge's forward paths deliberately skip
non-floating-point parameters when picking a compute dtype. Those paths must
keep working.

What does not work is reading a quantized ``.weight`` and doing arithmetic on it
directly — reshaping it into per-head matrices, slicing a fused projection,
folding LayerNorm into it. There the storage is packed (bitsandbytes 4-bit keeps
a ``[N, 1]`` uint8 buffer), split from its scales (FP8 keeps a separate
``weight_scale_inv``), or not a tensor at all (MXFP4 wraps a triton-kernels
object). Slicing those yields plausible-looking garbage rather than an error,
which is the failure mode this module exists to prevent.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def unreadable_weight_reason(weight: Any) -> Optional[str]:
    """Why ``weight`` cannot be read as a plain matrix (a fragment completing
    "... cannot be read because <reason>"), or None if it can.

    Readable = exactly {fp16, bf16, fp32, fp64}: every 1-byte torch dtype is
    integer storage or a scale-less narrow float.
    """
    if not isinstance(weight, torch.Tensor):
        # MXFP4 and similar hand back a wrapper object holding the packed
        # payload plus scales. It is often *named* Tensor, which makes the
        # resulting TypeError look like a torch bug — hence the qualified name,
        # which is the only thing distinguishing it from the real one.
        cls = type(weight)
        return f"it is a {cls.__module__}.{cls.__qualname__}, not a torch.Tensor"
    if weight.device.type == "meta":
        return "it is on the meta device, so no values have been materialized"
    if not weight.dtype.is_floating_point:
        # bitsandbytes int8/4-bit, GPTQ/AWQ int32, BitNet packed uint8. Note
        # that bnb's Params4bit IS a torch.Tensor subclass, so only the dtype
        # distinguishes it — and its shape is the packed [N, 1], not [out, in].
        return f"its dtype is {weight.dtype}, which is packed integer storage"
    if weight.dtype.itemsize < 2:
        # float8_e4m3fn and friends report is_floating_point=True, so a plain
        # float check lets them through. They are the only family that slices
        # and even survives nn.Parameter() silently, so omitting this branch
        # leaves the worst case uncaught.
        return (
            f"its dtype is {weight.dtype}, a narrow float whose values are "
            "meaningless without the separate scale tensor stored beside them "
            "(e.g. weight_scale_inv)"
        )
    return None


def _is_meta(weight: Any) -> bool:
    return isinstance(weight, torch.Tensor) and weight.device.type == "meta"


def quantization_method(config: Any) -> Optional[str]:
    """The ``quant_method`` declared on an HF config, or None if unquantized.

    Accepts a ``PretrainedConfig`` or its ``to_dict()`` form, and tolerates the
    nested ``quantization_config`` being either shape — both appear in the wild,
    depending on whether the config was loaded or round-tripped through JSON.
    """
    if config is None:
        return None
    if isinstance(config, dict):
        quant_config = config.get("quantization_config")
    else:
        quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        return None
    if isinstance(quant_config, dict):
        method = quant_config.get("quant_method")
    else:
        method = getattr(quant_config, "quant_method", None)
    # HF stores this as a str or a str-valued QuantizationMethod enum; require
    # that rather than str()-ing whatever turned up, or a stub object's repr
    # ends up quoted back at the user as a method name.
    method = getattr(method, "value", method)
    return method if isinstance(method, str) else None


def describe_quantization(owner: Any) -> str:
    """Best-effort name for how ``owner``'s weights are quantized.

    Resolution order: the HF config's declared ``quant_method``, then the
    weight's class name (bitsandbytes subclasses are identifiable that way),
    then a generic fallback.
    """
    for holder in (owner, getattr(owner, "original_component", None)):
        method = quantization_method(getattr(holder, "config", None))
        if method is not None:
            return method
    weight = getattr(owner, "weight", None)
    if weight is not None:
        # Exclude the two uninformative names rather than nn.Parameter itself:
        # bitsandbytes' Params4bit and Int8Params ARE Parameter subclasses, so
        # an isinstance test excluded exactly the classes this identifies.
        # transformers keys off the same class names (integrations/bitsandbytes).
        name = type(weight).__name__
        if name not in ("Tensor", "Parameter"):
            return name
    return "an unknown quantization"


_GENERIC_REMEDY = (
    "Weight-space operations need dequantized weights — reload the model "
    "without a quantization_config (or with dequantize enabled). Quantized "
    "*forward* passes remain supported."
)


def require_readable_weight(
    weight: Any, *, operation: str, owner: Any = None, remedy: Optional[str] = None
) -> torch.Tensor:
    """Return ``weight`` if it can be read as a plain matrix, else raise loudly.

    ``operation`` completes "TransformerLens cannot <operation> because ...";
    ``remedy`` replaces the generic advice — phrase it to hold for ANY
    quantization, since the caller cannot know which one it caught.
    """
    reason = unreadable_weight_reason(weight)
    if reason is None:
        assert isinstance(weight, torch.Tensor)
        return weight
    if _is_meta(weight):
        # Not a quantization problem — naming one here would send the reader
        # after the wrong cause. This is an offloaded or never-materialized load.
        raise NotImplementedError(
            f"TransformerLens cannot {operation} because {reason}. Load the "
            "model with real weights (no meta-device or disk offload) before "
            "reading them."
        )
    method = describe_quantization(owner) if owner is not None else "an unknown quantization"
    raise NotImplementedError(
        f"TransformerLens cannot {operation} because {reason} "
        f"(quantization: {method}). {remedy or _GENERIC_REMEDY}"
    )
