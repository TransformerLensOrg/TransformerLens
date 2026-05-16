"""Unit tests for ArchitectureAdapterFactory — external registration and entry-point discovery."""

import pytest

from tests.mocks.architecture_adapter import MockArchitectureAdapter
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter


class OtherMockArchitectureAdapter(ArchitectureAdapter):
    """A second mock adapter class used to verify overwrite behaviour."""

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = TransformerBridgeConfig(
                d_model=512,
                d_head=64,
                n_layers=2,
                n_ctx=1024,
                d_vocab=1000,
                d_mlp=2048,
                default_prepend_bos=True,
                architecture="GPT2LMHeadModel",
            )
        super().__init__(cfg)


@pytest.fixture(autouse=True)
def _isolate_factory_state():
    """Save and restore factory state so tests don't leak into each other."""
    saved_adapters = dict(ArchitectureAdapterFactory._adapters)
    saved_discovered = ArchitectureAdapterFactory._entry_points_discovered
    yield
    ArchitectureAdapterFactory._adapters = saved_adapters
    ArchitectureAdapterFactory._entry_points_discovered = saved_discovered


def _make_cfg(**overrides) -> TransformerBridgeConfig:
    defaults = dict(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=64,
        n_heads=4,
        d_vocab=100,
        d_mlp=256,
        default_prepend_bos=True,
    )
    defaults.update(overrides)
    return TransformerBridgeConfig(**defaults)


class TestSupportedArchitectures:
    """Verify all existing hardcoded entries in SUPPORTED_ARCHITECTURES."""

    def test_has_common_architectures(self):
        common = [
            "GPT2LMHeadModel",
            "LlamaForCausalLM",
            "MistralForCausalLM",
            "Gemma2ForCausalLM",
            "Qwen2ForCausalLM",
            "BloomForCausalLM",
            "FalconForCausalLM",
        ]
        for arch in common:
            assert arch in SUPPORTED_ARCHITECTURES, f"Missing: {arch}"


class TestRegisterAdapter:
    """Verify runtime adapter registration."""

    def test_register_adds_to_adapters(self):
        key = "TestMockForCausalLM"
        ArchitectureAdapterFactory.register_adapter(key, MockArchitectureAdapter)
        assert key in ArchitectureAdapterFactory._adapters

    def test_register_overwrites_existing(self):
        key = "TestOverwriteForCausalLM"
        ArchitectureAdapterFactory.register_adapter(key, MockArchitectureAdapter)
        assert ArchitectureAdapterFactory._adapters[key] is MockArchitectureAdapter
        ArchitectureAdapterFactory.register_adapter(key, OtherMockArchitectureAdapter)
        assert ArchitectureAdapterFactory._adapters[key] is OtherMockArchitectureAdapter

    def test_select_returns_registered_adapter(self):
        key = "TestSelectForCausalLM"
        ArchitectureAdapterFactory.register_adapter(key, MockArchitectureAdapter)
        cfg = _make_cfg(architecture=key)
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, MockArchitectureAdapter)


class TestSelectErrors:
    """Verify error handling in select_architecture_adapter."""

    def test_unknown_architecture_raises(self):
        cfg = _make_cfg(architecture="NonExistentForCausalLM")
        with pytest.raises(ValueError, match="Unsupported architecture"):
            ArchitectureAdapterFactory.select_architecture_adapter(cfg)

    def test_none_architecture_raises(self):
        cfg = _make_cfg(architecture=None)
        with pytest.raises(ValueError, match="must have architecture set"):
            ArchitectureAdapterFactory.select_architecture_adapter(cfg)


class TestDiscoverEntryPoints:
    """Verify entry-point discovery behavior."""

    def test_discover_is_idempotent(self):
        ArchitectureAdapterFactory._entry_points_discovered = False
        ArchitectureAdapterFactory.discover_entry_points()
        first_run = ArchitectureAdapterFactory._entry_points_discovered
        ArchitectureAdapterFactory.discover_entry_points()
        assert ArchitectureAdapterFactory._entry_points_discovered is first_run is True

    def test_discover_does_not_remove_existing(self):
        key = "TestPreserveForCausalLM"
        ArchitectureAdapterFactory.register_adapter(key, MockArchitectureAdapter)
        ArchitectureAdapterFactory._entry_points_discovered = False
        ArchitectureAdapterFactory.discover_entry_points()
        assert key in ArchitectureAdapterFactory._adapters
