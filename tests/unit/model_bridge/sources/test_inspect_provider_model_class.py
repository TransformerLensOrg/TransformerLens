"""Model-class routing in the HF-backed Inspect provider (``tl_bridge``).

The provider must pick its ``AutoModel*`` entry point config-first via
``get_hf_model_class_for_architecture`` (vision/seq2seq/masked-LM archs don't load
through ``AutoModelForCausalLM``), while archs unknown to TL's registry keep the
historical plain-causal-LM load. transformers loading is mocked; the tiny host
module gives the structural probe a real (attn/mlp-free) tree to walk.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

# The provider module subclasses inspect_ai's ModelAPI at import time; without the
# ``inspect`` extra the import fails with ModuleNotFoundError. Skip-collect here.
pytest.importorskip("inspect_ai")

from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
    TransformerLensTransformersModelAPI,
)


class _TinyHost(nn.Module):
    """Bare ``model.layers`` tree the structural probe can walk; the block has no
    attn/mlp submodules, so capability detection gates everything without a forward."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.config = SimpleNamespace(num_attention_heads=2, hidden_size=4)


def _patch_transformers(monkeypatch, hf_config):
    auto_config = MagicMock(return_value=hf_config)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", auto_config)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        MagicMock(return_value=MagicMock(name="tokenizer")),
    )
    causal = MagicMock(return_value=_TinyHost())
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", causal)
    return auto_config, causal


def test_causal_lm_loads_via_automodelforcausallm(monkeypatch):
    """Behavior for causal LMs is unchanged: routed class is AutoModelForCausalLM."""
    _, causal = _patch_transformers(
        monkeypatch, SimpleNamespace(architectures=["GPT2LMHeadModel"], model_type="gpt2")
    )
    api = TransformerLensTransformersModelAPI("fake/gpt2")
    assert causal.call_args.args[0] == "fake/gpt2"
    assert isinstance(api._hf, _TinyHost)


def test_vision_arch_routes_to_image_classification_class(monkeypatch):
    _, causal = _patch_transformers(
        monkeypatch, SimpleNamespace(architectures=["ViTForImageClassification"], model_type="vit")
    )
    vision = MagicMock(return_value=_TinyHost())
    monkeypatch.setattr("transformers.AutoModelForImageClassification.from_pretrained", vision)
    TransformerLensTransformersModelAPI("fake/vit")
    vision.assert_called_once()
    causal.assert_not_called()


def test_seq2seq_arch_routes_to_seq2seq_class(monkeypatch):
    _, causal = _patch_transformers(
        monkeypatch,
        SimpleNamespace(architectures=["T5ForConditionalGeneration"], model_type="t5"),
    )
    seq2seq = MagicMock(return_value=_TinyHost())
    monkeypatch.setattr("transformers.AutoModelForSeq2SeqLM.from_pretrained", seq2seq)
    TransformerLensTransformersModelAPI("fake/t5")
    seq2seq.assert_called_once()
    causal.assert_not_called()


def test_unknown_arch_falls_back_to_causal_lm(monkeypatch):
    """Archs outside TL's registry keep the historical standalone-provider load."""
    _, causal = _patch_transformers(
        monkeypatch,
        SimpleNamespace(architectures=["FrobnicatorForCausalLM"], model_type="frobnicator"),
    )
    TransformerLensTransformersModelAPI("fake/frob")
    causal.assert_called_once()


def test_config_fetch_gets_auth_but_not_load_kwargs(monkeypatch):
    """AutoConfig sees auth/remote-code kwargs only; torch_dtype etc. stay load-only
    (AutoConfig would stash unknown kwargs as config attributes)."""
    auto_config, causal = _patch_transformers(
        monkeypatch, SimpleNamespace(architectures=["GPT2LMHeadModel"], model_type="gpt2")
    )
    TransformerLensTransformersModelAPI(
        "fake/gpt2",
        model_kwargs={"token": "tk", "trust_remote_code": True, "torch_dtype": torch.float32},
    )
    assert auto_config.call_args.kwargs == {"token": "tk", "trust_remote_code": True}
    assert causal.call_args.kwargs["torch_dtype"] is torch.float32
