"""Shared architecture resolution + tokenizer skip in the remote boot sources.

``boot_vllm`` / ``boot_inspect`` must resolve the architecture via
``determine_architecture_from_hf_config`` (so ``architectures=None`` configs resolve
via ``model_type`` and unsupported archs fail before any engine/provider boot) and
must not load an ``AutoTokenizer`` for vision/audio adapters (those repos have none).
External boundaries (vllm, inspect_ai, HF Hub) are mocked via the shared
tests/mocks/vllm_boot.py scaffold; the resolution itself runs unmocked —
``determine_architecture_from_hf_config`` and the overlay stay real.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.mocks.vllm_boot import (
    boot_cfg,
    fake_configure_tokenizer,
    mock_hf_hub,
    mocked_vllm_boot,
    stub_adapter,
)
from transformer_lens.model_bridge.remote_bridge import RemoteBridge
from transformer_lens.model_bridge.sources.inspect.source import boot_inspect
from transformer_lens.model_bridge.sources.vllm import plugin
from transformer_lens.model_bridge.sources.vllm.source import boot_vllm


@pytest.fixture(autouse=True)
def _clean_plugin_state():
    plugin.clear_config()
    yield
    plugin.clear_config()


def _mock_inspect_boot(monkeypatch, hf_config, cfg):
    """Mock every external boundary boot_inspect crosses; return handles for assertions.

    inspect_ai is faked via sys.modules (like vllm in test_vllm_boot.py) so the tests
    run without the ``inspect`` extra; ``determine_architecture_from_hf_config`` and
    the InspectDriver/RemoteBridge construction stay real.
    """
    adapter = stub_adapter(cfg)
    _, auto_tokenizer = mock_hf_hub(monkeypatch, hf_config)

    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.inspect.source.get_hf_token",
        lambda: "fake-token",
    )
    build_cfg = MagicMock(return_value=cfg)
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.inspect.source.build_bridge_config_from_hf",
        build_cfg,
    )
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.inspect.source"
        ".ArchitectureAdapterFactory.select_architecture_adapter",
        lambda c: adapter,
    )
    configure_tok = MagicMock(side_effect=fake_configure_tokenizer)
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.inspect.source.configure_tokenizer",
        configure_tok,
    )

    # Provider self-check surface boot_inspect queries on the returned model.
    provider_api = SimpleNamespace(
        supported_kinds=lambda: frozenset({"resid_pre", "resid_post"}),
        capability_note=lambda: "",
        provides_sequence_logits=True,
    )
    get_model = MagicMock(return_value=SimpleNamespace(api=provider_api))
    inspect_ai_mod = types.ModuleType("inspect_ai")
    inspect_ai_model_mod = types.ModuleType("inspect_ai.model")
    inspect_ai_model_mod.get_model = get_model
    inspect_ai_mod.model = inspect_ai_model_mod
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_ai_mod)
    monkeypatch.setitem(sys.modules, "inspect_ai.model", inspect_ai_model_mod)
    # boot_inspect imports the provider module only for its @modelapi registration
    # side effect; a bare fake avoids the provider's own inspect_ai import chain.
    import transformer_lens.model_bridge.sources.inspect as inspect_pkg

    fake_provider = types.ModuleType(
        "transformer_lens.model_bridge.sources.inspect.transformers_provider"
    )
    monkeypatch.setitem(
        sys.modules,
        "transformer_lens.model_bridge.sources.inspect.transformers_provider",
        fake_provider,
    )
    monkeypatch.setattr(inspect_pkg, "transformers_provider", fake_provider, raising=False)
    return {
        "auto_tokenizer": auto_tokenizer,
        "build_cfg": build_cfg,
        "configure_tok": configure_tok,
        "get_model": get_model,
    }


class TestVLLMBootResolution:
    def test_architectures_none_resolves_via_model_type(self, monkeypatch):
        """A bare config (architectures=None) resolves via model_type instead of
        raising TypeError on architectures[0]."""
        hf_config = SimpleNamespace(
            architectures=None,
            model_type="llama",
            torch_dtype=torch.float16,
            hidden_size=4,
            vocab_size=16,
            num_hidden_layers=2,
        )
        handles = mocked_vllm_boot(monkeypatch, hf_config=hf_config)
        bridge = boot_vllm("any-model")
        assert isinstance(bridge, RemoteBridge)
        assert handles["build_cfg"].call_args.args[1] == "LlamaForCausalLM"

    def test_unsupported_architecture_fails_before_engine_boot(self, monkeypatch):
        """Unsupported archs must fail at config preview, not after the expensive
        LLM(...) construction (where select_architecture_adapter used to catch them)."""
        hf_config = SimpleNamespace(architectures=["FrobnicatorModel"], model_type="frobnicator")
        handles = mocked_vllm_boot(monkeypatch, hf_config=hf_config)
        with pytest.raises(ValueError, match="Could not determine supported architecture"):
            boot_vllm("any-model")
        handles["vllm_llm"].assert_not_called()

    def test_visual_adapter_skips_tokenizer(self, monkeypatch):
        """Vision repos ship no tokenizer — the load (which would raise) is skipped
        and the bridge carries tokenizer=None, as in boot_transformers."""
        hf_config = SimpleNamespace(
            architectures=["ViTForImageClassification"],
            model_type="vit",
            torch_dtype=torch.float16,
            hidden_size=4,
            vocab_size=16,
            num_hidden_layers=2,
        )
        cfg = boot_cfg(architecture="ViTForImageClassification", is_visual_model=True)
        handles = mocked_vllm_boot(monkeypatch, hf_config=hf_config, cfg=cfg)
        bridge = boot_vllm("any-model")
        handles["auto_tokenizer"].assert_not_called()
        handles["configure_tok"].assert_not_called()
        assert bridge.tokenizer is None

    def test_explicit_tokenizer_still_configured_for_visual(self, monkeypatch):
        """The skip only covers the auto-load; a caller-supplied tokenizer is honored."""
        hf_config = SimpleNamespace(
            architectures=["ViTForImageClassification"],
            model_type="vit",
            torch_dtype=torch.float16,
            hidden_size=4,
            vocab_size=16,
            num_hidden_layers=2,
        )
        cfg = boot_cfg(architecture="ViTForImageClassification", is_visual_model=True)
        handles = mocked_vllm_boot(monkeypatch, hf_config=hf_config, cfg=cfg)
        custom = MagicMock(name="custom")
        bridge = boot_vllm("any-model", tokenizer=custom)
        handles["auto_tokenizer"].assert_not_called()
        assert handles["configure_tok"].called
        assert bridge.tokenizer is custom


class TestInspectBootResolution:
    def test_architectures_none_resolves_via_model_type(self, monkeypatch):
        """A bare config (architectures=None) resolves via model_type instead of
        raising TypeError on architectures[0]."""
        hf_config = SimpleNamespace(architectures=None, model_type="gpt2")
        cfg = boot_cfg(architecture="GPT2LMHeadModel")
        handles = _mock_inspect_boot(monkeypatch, hf_config, cfg)
        bridge = boot_inspect("any-model")
        assert isinstance(bridge, RemoteBridge)
        assert handles["build_cfg"].call_args.args[1] == "GPT2LMHeadModel"
        assert handles["get_model"].call_args.args[0] == "tl_bridge/any-model"

    def test_unsupported_architecture_fails_before_provider_boot(self, monkeypatch):
        hf_config = SimpleNamespace(architectures=None, model_type="frobnicator")
        handles = _mock_inspect_boot(monkeypatch, hf_config, boot_cfg())
        with pytest.raises(ValueError, match="Could not determine supported architecture"):
            boot_inspect("any-model")
        handles["get_model"].assert_not_called()

    def test_visual_adapter_skips_tokenizer(self, monkeypatch):
        """Vision repos ship no tokenizer — the load (which would raise) is skipped
        and the bridge carries tokenizer=None, as in boot_transformers."""
        hf_config = SimpleNamespace(architectures=["ViTForImageClassification"], model_type="vit")
        cfg = boot_cfg(architecture="ViTForImageClassification", is_visual_model=True)
        handles = _mock_inspect_boot(monkeypatch, hf_config, cfg)
        bridge = boot_inspect("any-model")
        handles["auto_tokenizer"].assert_not_called()
        handles["configure_tok"].assert_not_called()
        assert bridge.tokenizer is None

    def test_audio_adapter_skips_tokenizer(self, monkeypatch):
        hf_config = SimpleNamespace(architectures=["HubertModel"], model_type="hubert")
        cfg = boot_cfg(architecture="HubertModel", is_audio_model=True)
        handles = _mock_inspect_boot(monkeypatch, hf_config, cfg)
        bridge = boot_inspect("any-model")
        handles["auto_tokenizer"].assert_not_called()
        assert bridge.tokenizer is None

    def test_text_model_still_loads_and_configures_tokenizer(self, monkeypatch):
        hf_config = SimpleNamespace(architectures=["GPT2LMHeadModel"], model_type="gpt2")
        cfg = boot_cfg(architecture="GPT2LMHeadModel")
        handles = _mock_inspect_boot(monkeypatch, hf_config, cfg)
        bridge = boot_inspect("any-model")
        assert handles["auto_tokenizer"].called
        assert handles["configure_tok"].called
        assert bridge.tokenizer is not None
