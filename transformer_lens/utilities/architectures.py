"""Centralized architecture classification for TransformerLens.

Single source of truth for architecture type detection. Used by the bridge
loading pipeline, benchmarks, and verification tools.
"""

from typing import Optional

# Encoder-decoder models (T5, BART, etc.)
SEQ2SEQ_ARCHITECTURES: set[str] = {
    "T5ForConditionalGeneration",
    "MT5ForConditionalGeneration",
    "T5WithLMHeadModel",
    "T5GemmaForConditionalGeneration",
    "LongT5ForConditionalGeneration",
    "T5Gemma2ForConditionalGeneration",
    "BartForConditionalGeneration",
    "MBartForConditionalGeneration",
    "M2M100ForConditionalGeneration",
    "MarianMTModel",
    "PegasusForConditionalGeneration",
    "BlenderbotForConditionalGeneration",
    "BlenderbotSmallForConditionalGeneration",
    "LEDForConditionalGeneration",
    "SwitchTransformersForConditionalGeneration",
}

# Post-norm decoders: ln1/ln2 normalize each sublayer's OUTPUT before the residual
# add, so LN folding and writing-weight centering (which assume the gain sits on a
# sublayer's INPUT) are not valid algebra for them.
POST_NORM_ARCHITECTURES: set[str] = {
    "Olmo2ForCausalLM",
    "Olmo3ForCausalLM",
}

# Masked language models (BERT-style, no text generation)
MASKED_LM_ARCHITECTURES: set[str] = {
    "BertForMaskedLM",
    "RobertaForMaskedLM",
    "AlbertForMaskedLM",
    "DistilBertForMaskedLM",
    "ElectraForMaskedLM",
    "BD3LM",
}

# Vision-language multimodal models
MULTIMODAL_ARCHITECTURES: set[str] = {
    "Emu3ForConditionalGeneration",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
    "Gemma4ForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Idefics3ForConditionalGeneration",
    "Florence2ForConditionalGeneration",
    "Mistral3ForConditionalGeneration",
    "Llama4ForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "Glm4vForConditionalGeneration",
}

# Audio-conditioned text decoders (audio encoder + causal LM); load via
# AutoModelForSeq2SeqLM but behave as text decoders for classification.
AUDIO_TEXT_ARCHITECTURES: set[str] = {
    "Qwen2AudioForConditionalGeneration",
    "GlmAsrForConditionalGeneration",
    "AudioFlamingo3ForConditionalGeneration",
    "MusicFlamingoForConditionalGeneration",
}

# Audio spectrogram models for classification
AUDIO_CLASSIFICATION_ARCHITECTURES: set[str] = {
    "ASTForAudioClassification",
}

# Audio encoder models (HuBERT, wav2vec2, etc.)
AUDIO_ARCHITECTURES: set[str] = {
    "HubertForCTC",
    "HubertModel",
    "HubertForSequenceClassification",
    "Wav2Vec2ForCTC",
    "Wav2Vec2Model",
    # Pretraining checkpoints (facebook/wav2vec2-base/-large declare this class)
    # load their encoder via AutoModel -> Wav2Vec2Model.
    "Wav2Vec2ForPreTraining",
} | AUDIO_CLASSIFICATION_ARCHITECTURES

# Vision-only (non-multimodal, no text tower) encoder models. Split into the
# two HF AutoModel classes they load under: bare encoders load via AutoModel,
# classification heads load via AutoModelForImageClassification.
VISION_MODEL_ARCHITECTURES: set[str] = {
    "ViTModel",
    "DeiTModel",
}
VISION_CLASSIFICATION_ARCHITECTURES: set[str] = {
    "ViTForImageClassification",
    "DeiTForImageClassification",
}
VISION_ARCHITECTURES: set[str] = VISION_MODEL_ARCHITECTURES | VISION_CLASSIFICATION_ARCHITECTURES

# Text models whose remote code registers only under plain AutoModel
# (the class itself carries the LM head).
BASE_AUTOMODEL_ARCHITECTURES: set[str] = {
    "DreamModel",
}


def classify_architecture(architecture: str) -> str:
    """Classify an architecture string into a model type.

    Returns one of: "seq2seq", "masked_lm", "multimodal", "audio", "vision", "causal_lm"
    """
    if architecture in SEQ2SEQ_ARCHITECTURES:
        return "seq2seq"
    if architecture in MASKED_LM_ARCHITECTURES:
        return "masked_lm"
    if architecture in MULTIMODAL_ARCHITECTURES:
        return "multimodal"
    if architecture in AUDIO_ARCHITECTURES:
        return "audio"
    if architecture in VISION_ARCHITECTURES:
        return "vision"
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
    Returns one of: "seq2seq", "masked_lm", "multimodal", "audio", "vision", "causal_lm"
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
    Returns one of: "seq2seq", "masked_lm", "multimodal", "audio", "vision", "causal_lm"
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
