from . import (
    components,
    conversion_utils,
    evals,
    factories,
    head_detector,
    hook_points,
    patching,
    train,
    utilities,
)
from . import loading_from_pretrained as loading
from .ActivationCache import ActivationCache
from .BertNextSentencePrediction import BertNextSentencePrediction
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
