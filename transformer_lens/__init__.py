from . import (
    conversion_utils,
    evals,
    factories,
    head_detector,
    hook_points,
    patching,
    supported_models,
    tools,
    utilities,
)
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

# Removed in 4.0: directed messages so `from transformer_lens import HookedTransformer`
# (and the other deleted top-level names) fail with a migration pointer instead of a
# bare AttributeError. Submodule-path imports (`from transformer_lens.HookedTransformer
# import ...`) raise ModuleNotFoundError before this hook runs and can't be intercepted here.
_REMOVED_IN_4_0 = {
    "HookedTransformer": "Use TransformerBridge.boot_transformers(name), then "
    "enable_compatibility_mode() for HookedTransformer-equivalent numerics.",
    "HookedEncoder": "Use TransformerBridge.boot_transformers(name) on a BERT model.",
    "HookedEncoderDecoder": "Use TransformerBridge.boot_transformers(name) on a T5 model.",
    "HookedAudioEncoder": "Use TransformerBridge.boot_transformers(name) on a HuBERT/Wav2Vec2 model.",
    "BertNextSentencePrediction": "Use TransformerBridge.boot_transformers(name, "
    "model_class=BertForNextSentencePrediction).predict_next_sentence(a, b).",
    "HookedTransformerConfig": "Use TransformerBridgeConfig.",
    "train": "Use transformer_lens.tools.training (train / TrainConfig).",
    "loading": "Model names/aliases moved to transformer_lens.supported_models; "
    "config derivation is now internal to TransformerBridge's adapters.",
    "loading_from_pretrained": "Config derivation is now internal to TransformerBridge; "
    "checkpoint labels live in transformer_lens.tools.model_registry.checkpoints.",
    "utils": "Use transformer_lens.utilities (same names).",
    "components": "The HookedTransformer component tree was removed; TransformerBridge "
    "uses transformer_lens.model_bridge.generalized_components.",
}


def __getattr__(name: str):
    # Lazy: model_bridge is import-heavy and importing it eagerly here would
    # risk cycles with modules the bridge itself imports.
    if name == "TransformerBridge":
        from .model_bridge import TransformerBridge

        return TransformerBridge
    if name in _REMOVED_IN_4_0:
        raise AttributeError(
            f"{name!r} was removed in TransformerLens 4.0. {_REMOVED_IN_4_0[name]} "
            "See docs/source/content/migrating_to_v4.md."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"TransformerBridge"})


import os as _os  # noqa: E402

# Unconditional: without it, any model whose config writes an integral value for
# a float field cannot be loaded at all. See enable_hf_numeric_tower.
from .utilities.hf_utils import (  # noqa: E402
    enable_hf_numeric_tower as _enable_hf_numeric_tower,
)

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
