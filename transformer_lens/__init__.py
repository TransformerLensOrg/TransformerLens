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
from . import supported_models
from .ActivationCache import ActivationCache
from .BertNextSentencePrediction import BertNextSentencePrediction
from .cache.key_value_cache import TransformerLensKeyValueCache
from .cache.key_value_cache_entry import TransformerLensKeyValueCacheEntry
from .config import HookedTransformerConfig
from .FactoredMatrix import FactoredMatrix
from .HookedEncoder import HookedEncoder
from .HookedAudioEncoder import HookedAudioEncoder
from .HookedEncoderDecoder import HookedEncoderDecoder
from .HookedTransformer import HookedTransformer

# LIT integration (optional, requires lit-nlp package)
try:
    from . import lit
except ImportError:
    # LIT is an optional dependency
    lit = None  # type: ignore

from .SVDInterpreter import SVDInterpreter

# Opt-in: wrap transformers Auto*.from_pretrained with retry-on-429.
# Set TRANSFORMERLENS_HF_RETRY=1 in environments that hit HuggingFace rate limits
# (typically CI). Off by default so normal users see unmodified HF behavior.
# See transformer_lens.utilities.hf_utils.enable_hf_retry for details.
import os as _os  # noqa: E402

if _os.environ.get("TRANSFORMERLENS_HF_RETRY") == "1":
    from .utilities.hf_utils import enable_hf_retry as _enable_hf_retry  # noqa: E402

    _enable_hf_retry()

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
