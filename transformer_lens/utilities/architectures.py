"""Centralized architecture classification for TransformerLens.

Single source of truth for architecture type detection. Used by the bridge
loading pipeline, benchmarks, and verification tools.
"""

from typing import Optional

# Encoder-decoder models (T5, BART, etc.)
SEQ2SEQ_ARCHITECTURES: set[str] = {
    "T5ForConditionalGeneration",
    "MT5ForConditionalGeneration",
    "BartForConditionalGeneration",
    "MBartForConditionalGeneration",
    "MarianMTModel",
    "PegasusForConditionalGeneration",
    "BlenderbotForConditionalGeneration",
    "BlenderbotSmallForConditionalGeneration",
}

# Masked language models (BERT-style, no text generation)
MASKED_LM_ARCHITECTURES: set[str] = {
    "BertForMaskedLM",
    "RobertaForMaskedLM",
    "AlbertForMaskedLM",
    "DistilBertForMaskedLM",
    "ElectraForMaskedLM",
}

# Vision-language multimodal models
MULTIMODAL_ARCHITECTURES: set[str] = {
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
}

# Audio encoder models (HuBERT, wav2vec2, etc.)
AUDIO_ARCHITECTURES: set[str] = {
    "HubertForCTC",
    "HubertModel",
    "HubertForSequenceClassification",
}

# Bridge uses different hook shapes than HookedTransformer by design.
# Phase 2/3 HT comparisons are skipped; Phase 1 (HF comparison) is the gold standard.
NO_HT_COMPARISON_ARCHITECTURES: set[str] = (
    MULTIMODAL_ARCHITECTURES
    | AUDIO_ARCHITECTURES
    | {
        "Gemma3ForCausalLM",
    }
)


def classify_architecture(architecture: str) -> str:
    """Classify an architecture string into a model type.

    Returns one of: "seq2seq", "masked_lm", "multimodal", "audio", "causal_lm"
    """
    if architecture in SEQ2SEQ_ARCHITECTURES:
        return "seq2seq"
    if architecture in MASKED_LM_ARCHITECTURES:
        return "masked_lm"
    if architecture in MULTIMODAL_ARCHITECTURES:
        return "multimodal"
    if architecture in AUDIO_ARCHITECTURES:
        return "audio"
    return "causal_lm"


def get_architectures_for_config(config) -> list[str]:
    """Extract architecture strings from an HF config object."""
    architectures = []
    if hasattr(config, "original_architecture"):
        architectures.append(config.original_architecture)
    if hasattr(config, "architectures") and config.architectures:
        architectures.extend(config.architectures)
    return architectures


def classify_model_config(config) -> str:
    """Classify a model by its HF config.

    Checks config.is_encoder_decoder first, then falls back to architecture list.
    Returns one of: "seq2seq", "masked_lm", "multimodal", "audio", "causal_lm"
    """
    if getattr(config, "is_encoder_decoder", False):
        return "seq2seq"
    for arch in get_architectures_for_config(config):
        model_type = classify_architecture(arch)
        if model_type != "causal_lm":
            return model_type
    return "causal_lm"


def classify_model_name(
    model_name: str,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
) -> str:
    """Classify a model by its HuggingFace model name.

    Loads the config once, classifies from it. If token is None, reads
    HF_TOKEN from the environment automatically.
    Returns one of: "seq2seq", "masked_lm", "multimodal", "audio", "causal_lm"
    """
    try:
        from transformers import AutoConfig

        if token is None:
            from transformer_lens.utilities.hf_utils import get_hf_token

            token = get_hf_token()

        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, token=token
        )
        return classify_model_config(config)
    except Exception:
        return "causal_lm"


def is_masked_lm_model(
    model_name: str, trust_remote_code: bool = False, token: Optional[str] = None
) -> bool:
    """Check if a model is a masked language model (BERT-style)."""
    return (
        classify_model_name(model_name, trust_remote_code=trust_remote_code, token=token)
        == "masked_lm"
    )


def is_encoder_decoder_model(
    model_name: str, trust_remote_code: bool = False, token: Optional[str] = None
) -> bool:
    """Check if a model is an encoder-decoder architecture (T5, BART, etc.)."""
    return (
        classify_model_name(model_name, trust_remote_code=trust_remote_code, token=token)
        == "seq2seq"
    )


def is_multimodal_model(
    model_name: str, trust_remote_code: bool = False, token: Optional[str] = None
) -> bool:
    """Check if a model is a multimodal vision-language model (LLaVA, Gemma3)."""
    return (
        classify_model_name(model_name, trust_remote_code=trust_remote_code, token=token)
        == "multimodal"
    )


def is_audio_model(
    model_name: str, trust_remote_code: bool = False, token: Optional[str] = None
) -> bool:
    """Check if a model is an audio encoder model (HuBERT, wav2vec2)."""
    return (
        classify_model_name(model_name, trust_remote_code=trust_remote_code, token=token) == "audio"
    )
