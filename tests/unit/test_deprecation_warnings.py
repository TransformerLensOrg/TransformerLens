import pytest

from transformer_lens import HookedAudioEncoder, HookedEncoderDecoder, HookedTransformer


def test_hooked_transformer_from_pretrained_warns_before_t5_rejection() -> None:
    with pytest.warns(
        DeprecationWarning,
        match=r"HookedTransformer\.from_pretrained is deprecated.*TransformerBridge\.boot_transformers",
    ):
        with pytest.raises(RuntimeError, match="use HookedEncoderDecoder"):
            HookedTransformer.from_pretrained("t5-small")


def test_hooked_encoder_decoder_from_pretrained_warns_before_quantization_rejection() -> None:
    with pytest.warns(
        DeprecationWarning,
        match=r"HookedEncoderDecoder\.from_pretrained is deprecated.*TransformerBridge\.boot_transformers",
    ):
        with pytest.raises(ValueError, match="Quantization not supported"):
            HookedEncoderDecoder.from_pretrained("t5-small", load_in_4bit=True)


def test_hooked_audio_encoder_from_pretrained_warns_before_quantization_rejection() -> None:
    with pytest.warns(
        DeprecationWarning,
        match=r"HookedAudioEncoder\.from_pretrained is deprecated.*TransformerBridge\.boot_transformers",
    ):
        with pytest.raises(AssertionError, match="Quantization not supported"):
            HookedAudioEncoder.from_pretrained("test-audio-model", load_in_4bit=True)
