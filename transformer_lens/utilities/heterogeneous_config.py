"""Safe attribute access for transformers>=5.15 heterogeneous configs.

Configs whose attention geometry varies across layers (e.g. Gemma 4) register
those fields as per-layer: reading one on the global config raises
``AmbiguousGlobalPerLayerAttributeError`` from ``__getattribute__``, which
neither ``hasattr()`` nor ``getattr(..., default)`` suppresses. These helpers
let config-probing code stay safe on any transformers version — the per-layer
machinery is simply absent pre-5.15, where every helper degrades to plain
attribute access.
"""

from typing import Any


def per_layer_attr_names(config: Any) -> frozenset:
    """Fields a heterogeneous config refuses to serve globally.

    Empty for homogeneous configs and pre-5.15 transformers.
    """
    if not getattr(config, "is_heterogeneous", False):
        return frozenset()
    return frozenset(getattr(config, "per_layer_attributes", None) or ())


def per_layer_values(config: Any, name: str) -> list:
    """Collect a per-layer attribute from a heterogeneous config's layer configs."""
    per_layer_config = config.per_layer_config
    return [getattr(per_layer_config[i], name, None) for i in range(len(per_layer_config))]


def majority_value(values: list) -> Any:
    """Most common value in a per-layer list; ties break toward the earliest layer."""
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda v: (counts[v], -values.index(v)))


def safe_config_get(config: Any, name: str, default: Any = None) -> Any:
    """getattr that resolves per-layer-registered fields to their majority value."""
    if name in per_layer_attr_names(config):
        values = [v for v in per_layer_values(config, name) if v is not None]
        return majority_value(values) if values else default
    return getattr(config, name, default)


class HetSafeConfigView:
    """Read-only getattr proxy: per-layer-registered fields resolve to their
    majority-layer value instead of raising; everything else passes through.

    Wrap a config once and downstream ``hasattr``/``getattr`` probes need no
    per-field awareness of heterogeneity.
    """

    def __init__(self, config: Any) -> None:
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_het_attrs", per_layer_attr_names(config))

    def __getattr__(self, name: str) -> Any:
        config = object.__getattribute__(self, "_config")
        if name in object.__getattribute__(self, "_het_attrs"):
            values = [v for v in per_layer_values(config, name) if v is not None]
            if not values:
                raise AttributeError(name)
            return majority_value(values)
        return getattr(config, name)


def het_safe_view(config: Any) -> Any:
    """Wrap heterogeneous configs in a :class:`HetSafeConfigView`; pass others through."""
    return HetSafeConfigView(config) if per_layer_attr_names(config) else config
