from . import (
    conversion_utils,
    evals,
    factories,
    head_detector,
    hook_points,
    patching,
    tools,
    utilities,
)

# Frozen HookedTransformer-era model ledger (names + aliases). Data-only; kept
# for the legacy-compatibility ledger and alias-drift tooling.
from . import supported_models
from .ActivationCache import ActivationCache
from .cache.key_value_cache import TransformerLensKeyValueCache
from .cache.key_value_cache_entry import TransformerLensKeyValueCacheEntry
from .config import TransformerBridgeConfig
from .FactoredMatrix import FactoredMatrix

# KEPT infrastructure: HookedRootModule (with HookPoint) is the supported way
# to hook arbitrary nn.Modules; it was never part of the legacy model-class
# removal.
from .HookedRootModule import HookedRootModule

# LIT integration (optional, requires lit-nlp package)
try:
    from . import lit
except ImportError:
    # LIT is an optional dependency
    lit = None  # type: ignore

from .SVDInterpreter import SVDInterpreter


def __getattr__(name: str):
    # Lazy: model_bridge is import-heavy and importing it eagerly here would
    # risk cycles with modules the bridge itself imports.
    if name == "TransformerBridge":
        from .model_bridge import TransformerBridge

        return TransformerBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"TransformerBridge"})


import os as _os  # noqa: E402

# Unconditional: without it, any model whose config writes an integral value for
# a float field cannot be loaded at all. See enable_hf_numeric_tower.
from .utilities.hf_utils import enable_hf_numeric_tower as _enable_hf_numeric_tower  # noqa: E402

_enable_hf_numeric_tower()

if _os.environ.get("TRANSFORMERLENS_HF_RETRY") == "1":
    from .utilities.hf_utils import enable_hf_retry as _enable_hf_retry  # noqa: E402

    _enable_hf_retry()

__all__ = [
    "TransformerBridge",
    "TransformerBridgeConfig",
    "FactoredMatrix",
    "ActivationCache",
    "SVDInterpreter",
    "HookedRootModule",
    "TransformerLensKeyValueCache",
    "TransformerLensKeyValueCacheEntry",
    "conversion_utils",
    "factories",
    "utilities",
    "tools",
]
