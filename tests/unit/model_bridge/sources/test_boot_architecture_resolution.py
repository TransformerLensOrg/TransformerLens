"""Shared architecture resolution + tokenizer skip in the remote boot sources.

``boot_vllm`` / ``boot_inspect`` must resolve the architecture via
``determine_architecture_from_hf_config`` (so ``architectures=None`` configs resolve
via ``model_type`` and unsupported archs fail before any engine/provider boot) and
must not load an ``AutoTokenizer`` for vision/audio adapters (those repos have none).
External boundaries (vllm, inspect_ai, HF Hub) are mocked in the style of
tests/unit/model_bridge/test_vllm_boot.py; the resolution itself runs unmocked.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.remote_bridge import RemoteBridge
from transformer_lens.model_bridge.sources.inspect.source import boot_inspect
from transformer_lens.model_bridge.sources.vllm import plugin
from transformer_lens.model_bridge.sources.vllm.source import boot_vllm


def _cfg(**overrides) -> TransformerBridgeConfig:
    params = dict(
        d_model=4,
        d_head=2,
        n_layers=2,
        n_ctx=8,
        n_heads=2,
        d_vocab=16,
        d_mlp=8,
        architecture="LlamaForCausalLM",
    )
    params.update(overrides)
    return TransformerBridgeConfig(**params)


def _adapter(cfg: TransformerBridgeConfig) -> ArchitectureAdapter:
    adapter = ArchitectureAdapter(cfg)
    adapter.component_mapping = {}
    return adapter


def _fake_configure_tokenizer(tokenizer, cfg_):
    # Real tokenizer setup/probing can't run on a MagicMock tokenizer.
    cfg_.tokenizer_prepends_bos, cfg_.tokenizer_appends_eos = (False, True)
    return tokenizer


def _mock_hf(monkeypatch, hf_config):
    """Patch the HF Hub boundary: config fetch + tokenizer fetch."""
    auto_config = MagicMock(return_value=hf_config)
    auto_tokenizer = MagicMock(return_value=MagicMock(name="tokenizer"))
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", auto_config)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", auto_tokenizer)
    return auto_config, auto_tokenizer


@pytest.fixture(autouse=True)
def _clean_plugin_state():
    plugin.clear_config()
    yield
    plugin.clear_config()


def _mock_vllm_boot(monkeypatch, hf_config, cfg):
    """Mock every external boundary boot_vllm crosses; return handles for assertions.

    ``determine_architecture_from_hf_config`` and the overlay stay real — the
    resolution path is what these tests exercise.
    """
    adapter = _adapter(cfg)
    _, auto_tokenizer = _mock_hf(monkeypatch, hf_config)

    fake_llm = MagicMock(name="llm")

    def rpc(method, *args, **kwargs):
        if method == "tl_absent_hooks":
            return [[]]
        return [torch.ones(16, 4)]  # reconstruction probe's get_param is beartype-enforced

    fake_llm.collective_rpc = MagicMock(side_effect=rpc)
    fake_vllm = MagicMock()
    fake_vllm.LLM = MagicMock(return_value=fake_llm)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.vllm.source.get_hf_token",
        lambda: "fake-token",
    )
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.vllm.source.extract_hf_config",
        lambda llm: hf_config,
    )
    build_cfg = MagicMock(return_value=cfg)
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.vllm.source.build_bridge_config_from_hf",
        build_cfg,
    )
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.vllm.source"
        ".ArchitectureAdapterFactory.select_architecture_adapter",
        lambda c: adapter,
    )
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.vllm.source.plugin.register",
        lambda: None,
    )
    configure_tok = MagicMock(side_effect=_fake_configure_tokenizer)
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.vllm.source.configure_tokenizer",
        configure_tok,
    )
    return {
        "auto_tokenizer": auto_tokenizer,
        "vllm_llm": fake_vllm.LLM,
        "build_cfg": build_cfg,
        "configure_tok": configure_tok,
    }


def _mock_inspect_boot(monkeypatch, hf_config, cfg):
    """Mock every external boundary boot_inspect crosses; return handles for assertions.

    inspect_ai is faked via sys.modules (like vllm in test_vllm_boot.py) so the tests
    run without the ``inspect`` extra; ``determine_architecture_from_hf_config`` and
    the InspectDriver/RemoteBridge construction stay real.
    """
    adapter = _adapter(cfg)
    _, auto_tokenizer = _mock_hf(monkeypatch, hf_config)

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
    configure_tok = MagicMock(side_effect=_fake_configure_tokenizer)
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
        handles = _mock_vllm_boot(monkeypatch, hf_config, _cfg())
        bridge = boot_vllm("any-model")
        assert isinstance(bridge, RemoteBridge)
        assert handles["build_cfg"].call_args.args[1] == "LlamaForCausalLM"

    def test_unsupported_architecture_fails_before_engine_boot(self, monkeypatch):
        """Unsupported archs must fail at config preview, not after the expensive
        LLM(...) construction (where select_architecture_adapter used to catch them)."""
        hf_config = SimpleNamespace(architectures=["FrobnicatorModel"], model_type="frobnicator")
        handles = _mock_vllm_boot(monkeypatch, hf_config, _cfg())
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
        cfg = _cfg(architecture="ViTForImageClassification", is_visual_model=True)
        handles = _mock_vllm_boot(monkeypatch, hf_config, cfg)
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
        cfg = _cfg(architecture="ViTForImageClassification", is_visual_model=True)
        handles = _mock_vllm_boot(monkeypatch, hf_config, cfg)
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
        cfg = _cfg(architecture="GPT2LMHeadModel")
        handles = _mock_inspect_boot(monkeypatch, hf_config, cfg)
        bridge = boot_inspect("any-model")
        assert isinstance(bridge, RemoteBridge)
        assert handles["build_cfg"].call_args.args[1] == "GPT2LMHeadModel"
        assert handles["get_model"].call_args.args[0] == "tl_bridge/any-model"

    def test_unsupported_architecture_fails_before_provider_boot(self, monkeypatch):
        hf_config = SimpleNamespace(architectures=None, model_type="frobnicator")
        handles = _mock_inspect_boot(monkeypatch, hf_config, _cfg())
        with pytest.raises(ValueError, match="Could not determine supported architecture"):
            boot_inspect("any-model")
        handles["get_model"].assert_not_called()

    def test_visual_adapter_skips_tokenizer(self, monkeypatch):
        """Vision repos ship no tokenizer — the load (which would raise) is skipped
        and the bridge carries tokenizer=None, as in boot_transformers."""
        hf_config = SimpleNamespace(architectures=["ViTForImageClassification"], model_type="vit")
        cfg = _cfg(architecture="ViTForImageClassification", is_visual_model=True)
        handles = _mock_inspect_boot(monkeypatch, hf_config, cfg)
        bridge = boot_inspect("any-model")
        handles["auto_tokenizer"].assert_not_called()
        handles["configure_tok"].assert_not_called()
        assert bridge.tokenizer is None

    def test_audio_adapter_skips_tokenizer(self, monkeypatch):
        hf_config = SimpleNamespace(architectures=["HubertModel"], model_type="hubert")
        cfg = _cfg(architecture="HubertModel", is_audio_model=True)
        handles = _mock_inspect_boot(monkeypatch, hf_config, cfg)
        bridge = boot_inspect("any-model")
        handles["auto_tokenizer"].assert_not_called()
        assert bridge.tokenizer is None

    def test_text_model_still_loads_and_configures_tokenizer(self, monkeypatch):
        hf_config = SimpleNamespace(architectures=["GPT2LMHeadModel"], model_type="gpt2")
        cfg = _cfg(architecture="GPT2LMHeadModel")
        handles = _mock_inspect_boot(monkeypatch, hf_config, cfg)
        bridge = boot_inspect("any-model")
        assert handles["auto_tokenizer"].called
        assert handles["configure_tok"].called
        assert bridge.tokenizer is not None
