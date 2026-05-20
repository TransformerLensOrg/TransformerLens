"""HuggingFace ``transformers`` source for TransformerBridge."""
from __future__ import annotations

# Re-exported so external code that patches ``AutoConfig.from_pretrained`` /
# ``AutoTokenizer.from_pretrained`` via this module path keeps working after the
# package split. Class-method monkey-patches reach the same class objects that
# ``source.py`` imports directly, so this re-export keeps tests stable.
from transformers import AutoConfig, AutoTokenizer

from transformer_lens.model_bridge.bridge import TransformerBridge

# Re-export shared HF-format utilities at the historical path for backward compatibility
# with `from transformer_lens.model_bridge.sources.transformers import ...` callers.
from transformer_lens.model_bridge.sources._hf_format import (
    determine_architecture_from_hf_config,
    map_default_transformer_lens_config,
    setup_tokenizer,
)

from .helpers import (
    _CHECKPOINT_REVISION_FORMATS,
    _resolve_checkpoint_to_revision,
    check_model_support,
    get_hf_model_class_for_architecture,
    list_supported_models,
)
from .source import boot

# Attach functions to TransformerBridge as static methods.
setattr(TransformerBridge, "boot_transformers", staticmethod(boot))
setattr(TransformerBridge, "list_supported_models", staticmethod(list_supported_models))
setattr(TransformerBridge, "check_model_support", staticmethod(check_model_support))


__all__ = [
    "AutoConfig",
    "AutoTokenizer",
    "boot",
    "check_model_support",
    "determine_architecture_from_hf_config",
    "get_hf_model_class_for_architecture",
    "list_supported_models",
    "map_default_transformer_lens_config",
    "setup_tokenizer",
    "_CHECKPOINT_REVISION_FORMATS",
    "_resolve_checkpoint_to_revision",
]
