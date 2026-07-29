"""Modality helpers shared by the boot sources.

``load_modality_processor`` must apply its loader table in declared order
(multimodal → audio → vision, last write wins on ``bridge.processor``), swallow
per-loader failures, and keep the torchvision-install retry an AutoProcessor-only
special case. ``skip_tokenizer_for_modality`` must skip exactly the audio/vision
flags (multimodal models keep their text tokenizer). The boot test mocks the
external boundaries in the style of test_boot_architecture_resolution.py; the
helpers themselves run real.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from torch import nn

from tests.mocks.vllm_boot import boot_cfg, stub_adapter
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    skip_tokenizer_for_modality,
)
from transformer_lens.model_bridge.sources.transformers.helpers import (
    load_modality_processor,
)

_HELPERS = "transformer_lens.model_bridge.sources.transformers.helpers"
_SOURCE = "transformer_lens.model_bridge.sources.transformers.source"


def _patch_loader(monkeypatch, class_name, calls, fail=0):
    """Install a fake auto-loader on the transformers module; records call order."""
    state = {"fails": fail}

    def from_pretrained(model_name, **kwargs):
        calls.append((class_name, model_name, kwargs))
        if state["fails"] > 0:
            state["fails"] -= 1
            raise RuntimeError(f"{class_name} unavailable")
        return f"{class_name}-processor"

    loader = MagicMock(name=class_name)
    loader.from_pretrained = MagicMock(side_effect=from_pretrained)
    monkeypatch.setattr(f"transformers.{class_name}", loader)
    return loader


def _patch_all_loaders(monkeypatch, calls, **fails):
    return {
        name: _patch_loader(monkeypatch, name, calls, fail=fails.get(name, 0))
        for name in ("AutoProcessor", "AutoFeatureExtractor", "AutoImageProcessor")
    }


class TestLoadModalityProcessor:
    def test_dual_flag_cfg_applies_table_order_last_write_wins(self, monkeypatch):
        """A cfg with every flag set runs multimodal → audio → vision; the vision
        processor is the final write (the contract future dual-flag adapters rely on)."""
        calls: list = []
        _patch_all_loaders(monkeypatch, calls)
        bridge = SimpleNamespace()
        cfg = SimpleNamespace(is_multimodal=True, is_audio_model=True, is_visual_model=True)

        load_modality_processor(bridge, cfg, "org/model", False, "tok")

        assert [c[0] for c in calls] == [
            "AutoProcessor",
            "AutoFeatureExtractor",
            "AutoImageProcessor",
        ]
        assert bridge.processor == "AutoImageProcessor-processor"

    @pytest.mark.parametrize(
        "flag, loader_name",
        [
            ("is_multimodal", "AutoProcessor"),
            ("is_audio_model", "AutoFeatureExtractor"),
            ("is_visual_model", "AutoImageProcessor"),
        ],
    )
    def test_single_flag_selects_single_loader(self, monkeypatch, flag, loader_name):
        calls: list = []
        loaders = _patch_all_loaders(monkeypatch, calls)
        bridge = SimpleNamespace()

        load_modality_processor(bridge, SimpleNamespace(**{flag: True}), "org/model", True, None)

        assert [c[0] for c in calls] == [loader_name]
        assert bridge.processor == f"{loader_name}-processor"
        for name, loader in loaders.items():
            if name != loader_name:
                loader.from_pretrained.assert_not_called()

    def test_token_and_trust_remote_code_threaded_through(self, monkeypatch):
        """The token comes in as a parameter (single derivation at the call site),
        not from a per-block env re-read."""
        calls: list = []
        _patch_all_loaders(monkeypatch, calls)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        load_modality_processor(
            SimpleNamespace(), SimpleNamespace(is_visual_model=True), "org/m", True, "hf_tok"
        )

        assert calls == [
            ("AutoImageProcessor", "org/m", {"token": "hf_tok", "trust_remote_code": True})
        ]

    def test_no_flags_loads_nothing(self, monkeypatch):
        calls: list = []
        _patch_all_loaders(monkeypatch, calls)
        bridge = SimpleNamespace()

        load_modality_processor(bridge, SimpleNamespace(), "org/model", False, None)

        assert calls == []
        assert not hasattr(bridge, "processor")

    def test_loader_failure_is_swallowed_and_later_flags_still_run(self, monkeypatch):
        """Per-block except semantics: an audio failure neither raises nor blocks vision."""
        calls: list = []
        _patch_all_loaders(monkeypatch, calls, AutoFeatureExtractor=1)
        bridge = SimpleNamespace()
        cfg = SimpleNamespace(is_audio_model=True, is_visual_model=True)

        load_modality_processor(bridge, cfg, "org/model", False, None)

        assert [c[0] for c in calls] == ["AutoFeatureExtractor", "AutoImageProcessor"]
        assert bridge.processor == "AutoImageProcessor-processor"

    def test_autoprocessor_retries_once_torchvision_available(self, monkeypatch):
        calls: list = []
        _patch_all_loaders(monkeypatch, calls, AutoProcessor=1)
        monkeypatch.setattr(f"{_HELPERS}._ensure_torchvision", lambda: True)
        bridge = SimpleNamespace()

        load_modality_processor(
            bridge, SimpleNamespace(is_multimodal=True), "org/model", False, None
        )

        assert [c[0] for c in calls] == ["AutoProcessor", "AutoProcessor"]
        assert bridge.processor == "AutoProcessor-processor"

    def test_autoprocessor_no_retry_without_torchvision(self, monkeypatch):
        calls: list = []
        _patch_all_loaders(monkeypatch, calls, AutoProcessor=1)
        monkeypatch.setattr(f"{_HELPERS}._ensure_torchvision", lambda: False)
        bridge = SimpleNamespace()

        load_modality_processor(
            bridge, SimpleNamespace(is_multimodal=True), "org/model", False, None
        )

        assert [c[0] for c in calls] == ["AutoProcessor"]
        assert not hasattr(bridge, "processor")

    def test_no_torchvision_retry_for_audio_or_vision_loaders(self, monkeypatch):
        """The install retry is an AutoProcessor-only special case."""
        calls: list = []
        _patch_all_loaders(monkeypatch, calls, AutoImageProcessor=1)
        ensure = MagicMock(return_value=True)
        monkeypatch.setattr(f"{_HELPERS}._ensure_torchvision", ensure)
        bridge = SimpleNamespace()

        load_modality_processor(
            bridge, SimpleNamespace(is_visual_model=True), "org/model", False, None
        )

        assert [c[0] for c in calls] == ["AutoImageProcessor"]
        ensure.assert_not_called()
        assert not hasattr(bridge, "processor")


class TestSkipTokenizerForModality:
    @pytest.mark.parametrize(
        "audio, visual, expected",
        [
            (False, False, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ],
    )
    def test_flag_combinations(self, audio, visual, expected):
        cfg = SimpleNamespace(is_audio_model=audio, is_visual_model=visual)
        assert skip_tokenizer_for_modality(cfg) is expected

    def test_missing_flags_default_to_no_skip(self):
        assert skip_tokenizer_for_modality(SimpleNamespace()) is False

    def test_multimodal_alone_does_not_skip(self):
        """Multimodal models keep their text tokenizer; only audio/vision skip."""
        cfg = SimpleNamespace(is_multimodal=True)
        assert skip_tokenizer_for_modality(cfg) is False


class _FakeBridge(TransformerBridge):
    """TransformerBridge subclass (boot's return hint is beartype-enforced) that skips
    the real __init__; boot only needs attribute storage plus .processor writes."""

    def __init__(self, hf_model, adapter, tokenizer, **kwargs):
        nn.Module.__init__(self)
        self.hf_model = hf_model
        self.adapter = adapter
        self.tokenizer = tokenizer


class TestBootTransformersVisualModel:
    def test_visual_boot_gets_image_processor_and_no_tokenizer(self, monkeypatch):
        """End-to-end wiring through boot(): a vision model skips the AutoTokenizer
        load and gets bridge.processor from AutoImageProcessor with the boot's token."""
        calls: list = []
        image_loader = _patch_all_loaders(monkeypatch, calls)["AutoImageProcessor"]
        monkeypatch.setenv("HF_TOKEN", "boot-token")

        cfg = boot_cfg(architecture="ViTForImageClassification", is_visual_model=True)
        adapter = stub_adapter(cfg)
        monkeypatch.setattr(f"{_SOURCE}.build_bridge_config_from_hf", MagicMock(return_value=cfg))
        monkeypatch.setattr(
            f"{_SOURCE}.ArchitectureAdapterFactory.select_architecture_adapter",
            lambda c: adapter,
        )
        auto_tokenizer = MagicMock(name="AutoTokenizer")
        monkeypatch.setattr(f"{_SOURCE}.AutoTokenizer", auto_tokenizer)
        monkeypatch.setattr(f"{_SOURCE}.TransformerBridge", _FakeBridge)
        monkeypatch.setattr(
            "transformer_lens.model_bridge.sources.transformers_driver.TransformersDriver",
            MagicMock(name="TransformersDriver"),
        )
        monkeypatch.setattr(
            "transformer_lens.utilities.multi_gpu.find_embedding_device", lambda m: None
        )
        # Pre-loaded fake model: skips the from_pretrained path entirely.
        hf_model = SimpleNamespace(
            config=SimpleNamespace(architectures=["ViTForImageClassification"], model_type="vit")
        )

        from transformer_lens.model_bridge.sources.transformers.source import boot

        bridge = boot("fake-org/tiny-vit", device="cpu", hf_model=hf_model)

        auto_tokenizer.from_pretrained.assert_not_called()
        assert bridge.tokenizer is None
        assert bridge.processor == "AutoImageProcessor-processor"
        image_loader.from_pretrained.assert_called_once_with(
            "fake-org/tiny-vit", token="boot-token", trust_remote_code=False
        )
