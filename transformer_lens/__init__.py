from . import (
    components,
    conversion_utils,
    evals,
    factories,
    head_detector,
    hook_points,
    patching,
    tools,
    utilities,
)
from . import loading_from_pretrained as loading
from . import supported_models
from .ActivationCache import ActivationCache
from .cache.key_value_cache import TransformerLensKeyValueCache
from .cache.key_value_cache_entry import TransformerLensKeyValueCacheEntry
from .config import TransformerBridgeConfig
from .FactoredMatrix import FactoredMatrix

# KEPT infrastructure: HookedRootModule (with HookPoint) survives 4.0 as the
# supported way to hook arbitrary nn.Modules; it is not part of the legacy
# model-class removal below.
from .HookedRootModule import HookedRootModule

# LIT integration (optional, requires lit-nlp package)
try:
    from . import lit
except ImportError:
    # LIT is an optional dependency
    lit = None  # type: ignore

from .SVDInterpreter import SVDInterpreter


# Legacy names resolved lazily (PEP 562): the deprecated model classes and the
# train shim are deleted in 4.0, and importing transformer_lens must not load
# them eagerly. Each still warns on use via its own module; the deletion PR
# removes entries from this map and nothing else in this file.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static bindings for the lazy names below, so type checkers resolve
    # `from transformer_lens import HookedTransformer` to the class, not the
    # submodule. Runtime resolution goes through __getattr__.
    from . import train
    from .BertNextSentencePrediction import BertNextSentencePrediction
    from .config import HookedTransformerConfig
    from .HookedAudioEncoder import HookedAudioEncoder
    from .HookedEncoder import HookedEncoder
    from .HookedEncoderDecoder import HookedEncoderDecoder
    from .HookedTransformer import HookedTransformer

_LAZY_LEGACY: dict[str, tuple[str, str | None]] = {
    "HookedTransformer": (".HookedTransformer", "HookedTransformer"),
    "HookedEncoder": (".HookedEncoder", "HookedEncoder"),
    "HookedAudioEncoder": (".HookedAudioEncoder", "HookedAudioEncoder"),
    "HookedEncoderDecoder": (".HookedEncoderDecoder", "HookedEncoderDecoder"),
    "BertNextSentencePrediction": (".BertNextSentencePrediction", "BertNextSentencePrediction"),
    "HookedTransformerConfig": (".config", "HookedTransformerConfig"),
    "train": (".train", None),
}


def __getattr__(name: str):
    # Lazy: model_bridge is import-heavy and importing it eagerly here would
    # risk cycles with modules the bridge itself imports.
    if name == "TransformerBridge":
        from .model_bridge import TransformerBridge

        return TransformerBridge
    if name in _LAZY_LEGACY:
        from importlib import import_module

        module_name, attr = _LAZY_LEGACY[name]
        module = import_module(module_name, __name__)
        value = module if attr is None else getattr(module, attr)
        globals()[name] = value  # cache: subsequent access skips __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"TransformerBridge"} | set(_LAZY_LEGACY))


# Five legacy classes share their submodule's name. A submodule-path import
# (``from transformer_lens.HookedTransformer import HookedTransformer``) makes
# the import machinery bind the MODULE onto this package after the fact, and
# ``from transformer_lens import HookedTransformer`` would then return the
# module instead of the class, dependent on import order. The class override
# below de-shadows on every access, so the public name deterministically
# resolves to the class.
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

_SHADOWED_CLASS_NAMES = frozenset(
    name for name, (_mod, attr) in _LAZY_LEGACY.items() if attr == name
)


class _DeShadowingModule(_types.ModuleType):
    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        if name in _SHADOWED_CLASS_NAMES and isinstance(value, _types.ModuleType):
            value = getattr(value, name)
            setattr(self, name, value)
        return value


_sys.modules[__name__].__class__ = _DeShadowingModule


import os as _os  # noqa: E402

# Unconditional: without it, any model whose config writes an integral value for
# a float field cannot be loaded at all. See enable_hf_numeric_tower.
from .utilities.hf_utils import enable_hf_numeric_tower as _enable_hf_numeric_tower  # noqa: E402

_enable_hf_numeric_tower()

if _os.environ.get("TRANSFORMERLENS_HF_RETRY") == "1":
    from .utilities.hf_utils import enable_hf_retry as _enable_hf_retry  # noqa: E402

    _enable_hf_retry()

__all__ = [
    "HookedTransformerConfig",
    "TransformerBridge",
    "TransformerBridgeConfig",
    "FactoredMatrix",
    "ActivationCache",
    "HookedTransformer",
    "SVDInterpreter",
    "HookedEncoder",
    "HookedEncoderDecoder",
    "HookedRootModule",
    "TransformerLensKeyValueCache",
    "TransformerLensKeyValueCacheEntry",
    "components",
    "conversion_utils",
    "factories",
    "utilities",
    "tools",
]
