"""Regression coverage for deprecated legacy entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]


def _small_config():
    from transformer_lens import HookedTransformerConfig

    return HookedTransformerConfig(
        n_layers=1,
        d_model=16,
        d_head=4,
        n_heads=4,
        n_ctx=8,
        d_vocab=20,
        attn_only=True,
    )


def _assert_single_deprecation(constructor, class_name: str) -> None:
    with pytest.warns(DeprecationWarning, match=class_name) as caught:
        constructor()

    assert len(caught) == 1
    assert "TransformerBridge.boot_transformers" in str(caught[0].message)
    assert "4.0" in str(caught[0].message)


def test_importing_transformer_lens_emits_no_deprecation_warning():
    code = "\n".join(
        [
            "import warnings",
            "warnings.filterwarnings(",
            "    'error',",
            "    category=DeprecationWarning,",
            "    module=r'^transformer_lens(?:\\.|$)',",
            ")",
            "import transformer_lens",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_hooked_transformer_constructor_warns_once():
    from transformer_lens import HookedTransformer

    _assert_single_deprecation(lambda: HookedTransformer(_small_config()), "HookedTransformer")


def test_hooked_encoder_constructor_warns_once():
    from transformer_lens import HookedEncoder

    _assert_single_deprecation(lambda: HookedEncoder(_small_config()), "HookedEncoder")


def test_hooked_encoder_from_pretrained_warning_reaches_external_caller():
    code = "\n".join(
        [
            "import importlib",
            "hooked_encoder_module = importlib.import_module('transformer_lens.HookedEncoder')",
            "HookedEncoder = hooked_encoder_module.HookedEncoder",
            "def stop_loading(*args, **kwargs):",
            "    raise RuntimeError('stop after deprecation warning')",
            "hooked_encoder_module.loading.get_official_model_name = stop_loading",
            "try:",
            "    HookedEncoder.from_pretrained('bert-base-cased')",
            "except RuntimeError as error:",
            "    assert str(error) == 'stop after deprecation warning'",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "<string>:" in result.stderr
    assert "DeprecationWarning: HookedEncoder.from_pretrained is deprecated" in result.stderr


def test_bert_next_sentence_prediction_constructor_warns_once():
    from transformer_lens import BertNextSentencePrediction

    _assert_single_deprecation(
        lambda: BertNextSentencePrediction(object()), "BertNextSentencePrediction"
    )


def test_hooked_root_module_is_not_deprecated():
    """HookedRootModule (with HookPoint) is KEPT infrastructure — the supported
    way to hook arbitrary nn.Modules — and must construct without any
    DeprecationWarning, unlike the legacy model classes."""
    import warnings as w

    from transformer_lens import HookedRootModule

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        HookedRootModule()
    assert not [x for x in caught if issubclass(x.category, DeprecationWarning)]


def test_hooked_encoder_decoder_constructor_warns_once():
    from transformer_lens import HookedEncoderDecoder, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=16,
        d_head=4,
        n_heads=4,
        n_ctx=8,
        d_vocab=20,
        d_mlp=32,
        act_fn="relu",
        attention_dir="bidirectional",
        tie_word_embeddings=False,
        positional_embedding_type="relative_positional_bias",
        relative_attention_num_buckets=4,
        relative_attention_max_distance=8,
    )
    _assert_single_deprecation(lambda: HookedEncoderDecoder(cfg), "HookedEncoderDecoder")


def test_hooked_audio_encoder_constructor_warns_once():
    from transformer_lens import HookedAudioEncoder

    cfg = _small_config()
    _assert_single_deprecation(lambda: HookedAudioEncoder(cfg), "HookedAudioEncoder")


def test_from_pretrained_warns_exactly_once():
    """__init__ is suppressed under from_pretrained — one warning per entry point,
    attributed to the caller, not two."""
    import warnings as w

    from transformer_lens import HookedTransformer

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        HookedTransformer.from_pretrained("gpt2")
    dep = [x for x in caught if issubclass(x.category, DeprecationWarning)]
    assert len(dep) == 1, [str(d.message)[:60] for d in dep]
    assert "from_pretrained" in str(dep[0].message)
