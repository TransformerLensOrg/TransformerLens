import pytest

import transformer_lens.loading_from_pretrained as loading
from transformer_lens import HookedAudioEncoder, HookedEncoderDecoder, HookedTransformer


class _StopLoading(Exception):
    pass


def _stop_model_lookup(_: str) -> str:
    raise _StopLoading


def test_hooked_transformer_from_pretrained_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loading, "get_official_model_name", _stop_model_lookup)

    with pytest.warns(
        DeprecationWarning,
        match=r"HookedTransformer\.from_pretrained is deprecated.*"
        r"TransformerBridge\.boot_transformers",
    ), pytest.raises(_StopLoading):
        HookedTransformer.from_pretrained("test-model")


def test_hooked_encoder_decoder_from_pretrained_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loading, "get_official_model_name", _stop_model_lookup)

    with pytest.warns(
        DeprecationWarning,
        match=r"HookedEncoderDecoder\.from_pretrained is deprecated.*"
        r"TransformerBridge\.boot_transformers",
    ), pytest.raises(_StopLoading):
        HookedEncoderDecoder.from_pretrained("test-model")


def test_hooked_audio_encoder_from_pretrained_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loading, "get_official_model_name", _stop_model_lookup)

    with pytest.warns(
        DeprecationWarning,
        match=r"HookedAudioEncoder\.from_pretrained is deprecated.*"
        r"TransformerBridge\.boot_transformers",
    ), pytest.raises(_StopLoading):
        HookedAudioEncoder.from_pretrained("test-model")
