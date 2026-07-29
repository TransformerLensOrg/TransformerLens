"""Shared transformers-v5 compatibility helpers for remote-code adapters.

Modeling files loaded via ``trust_remote_code`` were mostly written against
transformers 4.x and break under v5 in recurring ways: the meta-device
load-then-materialise flow re-runs ``_init_weights`` over already-loaded
modules, ``tie_weights`` calls ``.keys()`` on list-form ``_tied_weights_keys``,
``all_tied_weights_keys`` lookups hit broken remote ``__getattr__``s, and the
``ROPE_INIT_FUNCTIONS["default"]`` entry the remote code looks up was removed.
The patch mechanics live here; each adapter's ``prepare_loading`` keeps the
WHY — what its remote code gets wrong and which classes need patching.

These paths only fire on real ``trust_remote_code`` loads (CI-invisible), so
the helpers are unit-tested directly in
``tests/unit/model_bridge/test_remote_code_compat.py``.
"""

import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import torch
from transformers import PreTrainedModel


def iter_remote_modeling_modules(*name_fragments: str) -> Iterator[ModuleType]:
    """Yield imported modules whose name contains ``modeling`` and any fragment.

    Each checkpoint revision of a remote-code repo imports its own copy of the
    modeling file (distinct ``transformers_modules.*`` entries), so class-level
    patches must be applied to every one. Matching is case-insensitive.
    """
    fragments = tuple(fragment.lower() for fragment in name_fragments)
    for key in list(sys.modules.keys()):
        key_lower = key.lower()
        if "modeling" not in key_lower:
            continue
        if not any(fragment in key_lower for fragment in fragments):
            continue
        module = sys.modules.get(key)
        if module is not None:
            yield module


def force_import_remote_class(model_name: str, dotted_ref: str, **kwargs: Any) -> type | None:
    """Import a remote-code class so its module lands in ``sys.modules`` to patch.

    ``dotted_ref`` is ``"<module_file>.<ClassName>"`` as understood by
    ``get_class_from_dynamic_module``; extra kwargs are forwarded to it.
    Returns the class, or None when the dynamic module is unavailable
    (offline, renamed, not a remote-code repo) — callers treat that as
    "nothing to patch".
    """
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module(dotted_ref, model_name, **kwargs)
        assert isinstance(cls, type)
        return cls
    except Exception:
        return None


def patch_init_weights_skip_loaded(cls: type[Any]) -> None:
    """Wrap ``cls._init_weights`` to skip modules already loaded from checkpoint.

    v5's meta-device flow calls ``_init_weights`` on modules that already hold
    checkpoint weights (remote-code modules lack ``_is_hf_initialized``),
    re-randomising them; the wrapper only lets initialisation through for
    modules still on the meta device. Idempotent via the ``_tl_patched``
    sentinel.

    Raises:
        ValueError: for ``transformers.PreTrainedModel`` itself and for classes
            without their own ``_init_weights``. Remote modules re-export the
            HF base under local names, and patching it would disable weight
            init — including HF's rotary-buffer restoration — for every model
            loaded later in the process.
    """
    if getattr(cls, "_tl_patched", False):
        return
    if cls is PreTrainedModel:
        raise ValueError(
            "Refusing to patch transformers.PreTrainedModel itself: that would "
            "disable weight init for every model loaded later in this process. "
            "Pass the remote code's own PreTrainedModel subclass."
        )
    if "_init_weights" not in cls.__dict__:
        raise ValueError(
            f"Refusing to patch {cls.__name__}: it defines no _init_weights of "
            "its own, so the wrap would capture an inherited (possibly HF base) "
            "implementation."
        )

    original_init_weights = cls._init_weights

    def safe_init_weights(self: Any, mod: Any, _original: Any = original_init_weights) -> None:
        # Only initialise modules still on meta device (pre-loading); never
        # re-randomise weights already read from the checkpoint.
        first_param = next(mod.parameters(), None)
        if first_param is not None and first_param.device.type != "meta":
            return
        _original(self, mod)

    cls._init_weights = safe_init_weights
    cls._tl_patched = True


def retie_weights_keys_v5(cls: Any, mapping: dict[str, str]) -> None:
    """Rewrite a 4.x list-form ``_tied_weights_keys`` to the v5 dict form.

    v5's ``tie_weights`` -> ``get_expanded_tied_weights_keys`` calls ``.keys()``
    on the attribute and raises ``AttributeError`` on the legacy list. No-op
    when ``cls`` is None or the attribute is already a dict (or absent).
    """
    if cls is not None and isinstance(getattr(cls, "_tied_weights_keys", None), list):
        cls._tied_weights_keys = mapping


def disable_tied_weights_lookup(cls: type[Any]) -> None:
    """Give ``cls`` an empty ``all_tied_weights_keys``.

    Some remote ``__getattr__`` implementations fail to delegate v5's
    ``all_tied_weights_keys`` lookup back to ``PreTrainedModel`` and raise;
    the affected checkpoints are untied anyway.
    """
    setattr(cls, "all_tied_weights_keys", {})


def compute_default_rope_inv_freq(
    config: Any = None,
    device: Any = None,
    seq_len: Any = None,
    **rope_kwargs: Any,
) -> tuple[torch.Tensor, float]:
    """transformers-v4 ``_compute_default_rope_parameters``, removed in v5.

    Standard (unscaled) RoPE inverse frequencies with the v4 call contract the
    remote code targets — ``(config, device) -> (inv_freq, attention_scaling)``
    — plus v4's kwargs-only fallback (``base``/``dim`` passed directly).
    Registration strategy stays with each adapter: dream re-registers the
    global ``ROPE_INIT_FUNCTIONS["default"]``, ouro deliberately patches only
    its own modeling module's copy.
    """
    if config is not None:
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", None) or 1.0
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
    else:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0
