from . import (
    components,
    conversion_utils,
    evals,
    factories,
    head_detector,
    hook_points,
    patching,
    tools,
    train,
    utilities,
)
from . import loading_from_pretrained as loading
from . import supported_models
from .ActivationCache import ActivationCache
from .BertNextSentencePrediction import BertNextSentencePrediction
from .cache.key_value_cache import TransformerLensKeyValueCache
from .cache.key_value_cache_entry import TransformerLensKeyValueCacheEntry
from .config import HookedTransformerConfig, TransformerBridgeConfig
from .FactoredMatrix import FactoredMatrix
from .HookedEncoder import HookedEncoder
from .HookedAudioEncoder import HookedAudioEncoder
from .HookedEncoderDecoder import HookedEncoderDecoder
from .HookedTransformer import HookedTransformer
from .HookedRootModule import HookedRootModule

# LIT integration (optional, requires lit-nlp package)
try:
    from . import lit
except ImportError:
    # LIT is an optional dependency
    lit = None  # type: ignore

from .SVDInterpreter import SVDInterpreter

import os as _os  # noqa: E402

if _os.environ.get("TRANSFORMERLENS_HF_RETRY") == "1":
    from .utilities.hf_utils import enable_hf_retry as _enable_hf_retry  # noqa: E402

    _enable_hf_retry()

__all__ = [
    "HookedTransformerConfig",
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
