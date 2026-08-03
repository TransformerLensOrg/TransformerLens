"""Regression coverage for deprecated legacy entry points."""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
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


def test_hooked_encoder_from_pretrained_warns_at_callsite(monkeypatch):
    hooked_encoder_module = importlib.import_module("transformer_lens.HookedEncoder")

    def stop_loading(*args, **kwargs):
        raise RuntimeError("stop after deprecation warning")

    monkeypatch.setattr(hooked_encoder_module.loading, "get_official_model_name", stop_loading)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        with pytest.raises(RuntimeError, match="stop after deprecation warning"):
            hooked_encoder_module.HookedEncoder.from_pretrained("bert-base-cased")

    deprecations = [
        warning for warning in caught if issubclass(warning.category, DeprecationWarning)
    ]
    assert len(deprecations) == 1
    assert "HookedEncoder.from_pretrained" in str(deprecations[0].message)
    assert deprecations[0].filename == __file__


def test_bert_next_sentence_prediction_constructor_warns_once():
    from transformer_lens import BertNextSentencePrediction

    _assert_single_deprecation(
        lambda: BertNextSentencePrediction(object()), "BertNextSentencePrediction"
    )


def test_direct_hooked_root_module_construction_warns_once():
    from transformer_lens import HookedRootModule

    _assert_single_deprecation(HookedRootModule, "HookedRootModule")
