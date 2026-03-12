from . import (
    components,
    conversion_utils,
    factories,
    utilities,
)
from .ActivationCache import ActivationCache
from .cache.key_value_cache import TransformerLensKeyValueCache
from .cache.key_value_cache_entry import TransformerLensKeyValueCacheEntry
from .config import HookedTransformerConfig
from .FactoredMatrix import FactoredMatrix
from .HookedEncoder import HookedEncoder
from .HookedEncoderDecoder import HookedEncoderDecoder
from .HookedTransformer import HookedTransformer

# LIT integration (optional, requires lit-nlp package)
try:
    from . import lit
except ImportError:
    # LIT is an optional dependency
    lit = None  # type: ignore

from .SVDInterpreter import SVDInterpreter

__all__ = [
    "HookedTransformerConfig",
    "FactoredMatrix",
    "ActivationCache",
    "HookedTransformer",
    "SVDInterpreter",
    "HookedEncoder",
    "HookedEncoderDecoder",
    "TransformerLensKeyValueCache",
    "TransformerLensKeyValueCacheEntry",
    "components",
    "conversion_utils",
    "factories",
    "utilities",
]
