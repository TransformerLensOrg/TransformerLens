"""Regression coverage for deprecated legacy entry points."""

from __future__ import annotations

import warnings

import pytest


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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import transformer_lens  # noqa: F401

    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]


def test_hooked_transformer_constructor_warns_once():
    from transformer_lens import HookedTransformer

    _assert_single_deprecation(lambda: HookedTransformer(_small_config()), "HookedTransformer")


def test_hooked_encoder_constructor_warns_once():
    from transformer_lens import HookedEncoder

    _assert_single_deprecation(lambda: HookedEncoder(_small_config()), "HookedEncoder")


def test_bert_next_sentence_prediction_constructor_warns_once():
    from transformer_lens import BertNextSentencePrediction

    _assert_single_deprecation(
        lambda: BertNextSentencePrediction(object()), "BertNextSentencePrediction"
    )


def test_direct_hooked_root_module_construction_warns_once():
    from transformer_lens import HookedRootModule

    _assert_single_deprecation(HookedRootModule, "HookedRootModule")
