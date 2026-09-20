"""attn_implementation.

Shared helper for forcing eager attention on a loaded HuggingFace model.
"""

from __future__ import annotations

from typing import Any, List


def force_eager_attention(model: Any, *, per_layer: bool = False) -> None:
    """Switch a pre-loaded model to eager attention so attention hooks can fire.

    Prefers the public ``set_attn_implementation`` API; exotic wrapped models can
    reject it, so failures fall back to writing ``config._attn_implementation``
    (including nested multimodal ``text_config``). ``per_layer=True`` also stamps
    every submodule's ``self_attn.config`` — some models keep per-layer config
    copies that the top-level write never reaches.

    Best-effort by design: never raises, silently no-ops on objects exposing
    neither the public API nor a config.
    """
    handled = False
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
            handled = True
        except Exception:
            pass  # Exotic wrapped models can reject the public API; write the config instead.
    if not handled:
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
            # Nested multimodal configs carry their own attn implementation.
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                text_config._attn_implementation = "eager"
    if per_layer:
        for layer in _layer_candidates(model):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                layer.self_attn.config._attn_implementation = "eager"


def _layer_candidates(model: Any) -> List[Any]:
    """Modules that might own a per-layer attention config.

    Real models walk ``modules()`` so nested stacks (``model.language_model``,
    HRM's L/H modules) are covered; plain-object test fakes fall back to the
    conventional ``model.model.layers`` chain.
    """
    modules = getattr(model, "modules", None)
    if callable(modules):
        try:
            return list(modules())
        except TypeError:
            return []  # Mock-style modules() returning a non-iterable.
    lm = getattr(model, "model", None)
    layers = getattr(lm, "layers", None) if lm is not None else None
    return list(layers) if layers is not None else []
