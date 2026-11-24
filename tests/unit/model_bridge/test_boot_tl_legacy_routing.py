"""boot_transformers routes legacy TransformerLens-format repos to boot_tl_legacy."""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import GPT2Config, PreTrainedTokenizerBase

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources.tl_legacy import (
    TL_LEGACY_MODEL_NAMES,
    TL_LEGACY_PREFIXES,
)
from transformer_lens.model_bridge.sources.transformers import source

GELU_1L = "NeelNanda/GELU_1L512W_C4_Code"


class _TookHFPath(Exception):
    """Raised once boot() takes the HuggingFace loading path."""


@pytest.fixture
def legacy_boot():
    sentinel = MagicMock(spec=TransformerBridge, name="legacy_bridge")
    with patch.object(source, "_boot_tl_legacy", return_value=sentinel) as mock:
        mock.sentinel = sentinel
        yield mock


@pytest.mark.parametrize("name", ["gelu-1l", GELU_1L])
def test_alias_and_repo_id_route_to_boot_tl_legacy(legacy_boot, name, caplog):
    with caplog.at_level(logging.WARNING):
        bridge = source.boot(name, device="cpu")

    assert bridge is legacy_boot.sentinel
    legacy_boot.assert_called_once_with(
        GELU_1L,
        checkpoint_index=None,
        checkpoint_value=None,
        device="cpu",
        dtype=torch.float32,
        tokenizer=None,
    )
    assert f"TransformerBridge.boot_tl_legacy('{GELU_1L}')" in caplog.text


@pytest.mark.parametrize(
    "checkpoint",
    [{"checkpoint_index": 3}, {"checkpoint_value": 1024}],
    ids=lambda checkpoint: next(iter(checkpoint)),
)
def test_supported_options_are_forwarded(legacy_boot, checkpoint):
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase, name="tokenizer")
    source.boot(
        GELU_1L,
        device="cpu",
        dtype=torch.float16,
        tokenizer=tokenizer,
        trust_remote_code=True,  # irrelevant to .pth repos, so harmlessly ignored
        **checkpoint,
    )
    legacy_boot.assert_called_once_with(
        GELU_1L,
        checkpoint_index=checkpoint.get("checkpoint_index"),
        checkpoint_value=checkpoint.get("checkpoint_value"),
        device="cpu",
        dtype=torch.float16,
        tokenizer=tokenizer,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hf_config_overrides": {"n_layer": 2}},
        {"load_weights": False},
        {"model_class": object},
        {"n_ctx": 128},
        {"revision": "main"},
        {"code_revision": "main"},
        {"device_map": "auto"},
        {"n_devices": 2},
        {"max_memory": {0: "1GB"}},
        {"offload_folder": "/tmp/offload"},
    ],
    ids=lambda kwargs: next(iter(kwargs)),
)
def test_unsupported_options_raise_instead_of_being_dropped(legacy_boot, kwargs):
    with pytest.raises(ValueError, match=f"boot_tl_legacy.*{next(iter(kwargs))}"):
        source.boot(GELU_1L, **kwargs)
    legacy_boot.assert_not_called()


# The legacy namespaces also host HF-format models, which must not be routed.
@pytest.mark.parametrize("name", ["gpt2", "Baidicoot/Llama-3-8B-Instruct-LAT"])
def test_hf_format_repos_are_not_routed(legacy_boot, name):
    with patch.object(source.AutoConfig, "from_pretrained", side_effect=_TookHFPath), pytest.raises(
        _TookHFPath
    ):
        source.boot(name, device="cpu")
    legacy_boot.assert_not_called()


def test_legacy_list_matches_the_model_registry():
    """Every legacy-namespace model TL ships is routed, and nothing else is."""
    from transformer_lens.supported_models import OFFICIAL_MODEL_NAMES

    registry = {name for name in OFFICIAL_MODEL_NAMES if name.startswith(TL_LEGACY_PREFIXES)}
    assert TL_LEGACY_MODEL_NAMES == registry


def test_preloaded_hf_model_is_not_routed(legacy_boot):
    """An hf_model is HF-format whatever its repo name; boot() reads its config."""
    hf_model = MagicMock(name="hf_model")
    hf_model.config = GPT2Config(n_layer=1)
    with patch.object(
        source, "build_bridge_config_from_hf", side_effect=_TookHFPath
    ), pytest.raises(_TookHFPath):
        source.boot(GELU_1L, hf_model=hf_model, device="cpu")
    legacy_boot.assert_not_called()
