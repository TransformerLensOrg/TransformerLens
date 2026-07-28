"""Unit tests for Cohere2ArchitectureAdapter.

Cohere2 reuses Cohere's parallel LayerNorm/GQA/logit-scale structure but mixes
sliding-window RoPE layers with full-attention NoPE layers. These tests stay
download-free and pin the adapter-specific layer-type and NoPE logic.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch.nn as nn
from torch import allclose, float32, manual_seed, no_grad, ones, randn, zeros

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import LinearBridge
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.sources.transformers import (
    determine_architecture_from_hf_config,
)
from transformer_lens.model_bridge.supported_architectures.cohere import (
    Cohere2ArchitectureAdapter,
    CohereArchitectureAdapter,
    _Cohere2AttentionBridge,
)
from transformer_lens.tools.model_registry import (
    CANONICAL_AUTHORS_BY_ARCH,
    HF_SUPPORTED_ARCHITECTURES,
)


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 8,
    d_mlp: int = 256,
    d_vocab: int = 1000,
    n_ctx: int = 512,
    n_key_value_heads: int | None = 2,
    layer_types: list[str] | None = None,
    sliding_window_pattern: int | None = None,
) -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="Cohere2ForCausalLM",
    )
    if n_key_value_heads is not None:
        cfg.n_key_value_heads = n_key_value_heads
    setattr(cfg, "logit_scale", 1.0)
    setattr(cfg, "rope_parameters", {"rope_theta": 50000.0, "rope_type": "default"})
    if layer_types is not None:
        setattr(cfg, "layer_types", layer_types)
    if sliding_window_pattern is not None:
        setattr(cfg, "sliding_window_pattern", sliding_window_pattern)
    return cfg


class FakeCohere2Attention(nn.Module):
    """Minimal Cohere2-style attention module for bridge tests."""

    def __init__(
        self,
        cfg: TransformerBridgeConfig,
        sliding_window: int | None = 4096,
        expose_sliding_window: bool = True,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = cfg.d_head
        self.num_key_value_groups = cfg.n_heads // (cfg.n_key_value_heads or cfg.n_heads)
        self.scaling = cfg.d_head**-0.5
        self.attention_dropout = 0.0
        self.layer_idx = layer_idx
        if expose_sliding_window:
            self.sliding_window = sliding_window

        kv_width = (cfg.n_key_value_heads or cfg.n_heads) * cfg.d_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)


def _cohere2_attention_bridge(cfg: TransformerBridgeConfig) -> _Cohere2AttentionBridge:
    return _Cohere2AttentionBridge(
        name="self_attn",
        config=cfg,
        submodules={
            "q": LinearBridge(name="q_proj"),
            "k": LinearBridge(name="k_proj"),
            "v": LinearBridge(name="v_proj"),
            "o": LinearBridge(name="o_proj"),
        },
    )


class TestCohere2Config:
    def test_explicit_layer_types_are_preserved(self) -> None:
        layer_types = [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]
        adapter = Cohere2ArchitectureAdapter(_make_cfg(n_layers=4, layer_types=layer_types))

        assert adapter.cfg.layer_types == layer_types

    def test_sliding_window_pattern_generates_layer_types(self) -> None:
        adapter = Cohere2ArchitectureAdapter(_make_cfg(n_layers=8, sliding_window_pattern=4))

        assert adapter.cfg.layer_types == [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]

    def test_underscored_sliding_window_pattern_generates_layer_types(self) -> None:
        cfg = _make_cfg(n_layers=3)
        setattr(cfg, "_sliding_window_pattern", 2)

        adapter = Cohere2ArchitectureAdapter(cfg)

        assert adapter.cfg.layer_types == [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ]

    def test_layer_types_length_must_match_n_layers(self) -> None:
        cfg = _make_cfg(n_layers=2, layer_types=["sliding_attention"])

        with pytest.raises(ValueError, match="layer_types length"):
            Cohere2ArchitectureAdapter(cfg)

    def test_sliding_window_pattern_must_be_positive(self) -> None:
        cfg = _make_cfg(n_layers=2, sliding_window_pattern=0)

        with pytest.raises(ValueError, match="sliding_window_pattern must be positive"):
            Cohere2ArchitectureAdapter(cfg)

    def test_inherits_cohere_gqa_and_logit_scale_behaviour(self) -> None:
        adapter = Cohere2ArchitectureAdapter(_make_cfg(n_heads=8, n_key_value_heads=2))

        assert isinstance(adapter, CohereArchitectureAdapter)
        assert adapter.cfg.n_key_value_heads == 2
        assert getattr(adapter.cfg, "logit_scale") == 1.0
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }


class TestCohere2ComponentMapping:
    def test_attention_bridge_is_cohere2_nope_aware(self) -> None:
        adapter = Cohere2ArchitectureAdapter(_make_cfg())

        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert type(attn) is _Cohere2AttentionBridge
        assert issubclass(_Cohere2AttentionBridge, PositionEmbeddingsAttentionBridge)

    def test_parallel_cohere_block_shape_is_preserved(self) -> None:
        adapter = Cohere2ArchitectureAdapter(_make_cfg())
        blocks = adapter.component_mapping["blocks"]

        assert "ln1" in blocks.submodules
        assert "ln2" not in blocks.submodules
        assert "mlp" in blocks.submodules


class TestCohere2NoPE:
    def _record_super_forward(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        recorded: dict = {}

        def _fake_super_forward(self: Any, *args: Any, **kwargs: Any) -> str:
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return "sentinel-output"

        monkeypatch.setattr(
            PositionEmbeddingsAttentionBridge, "forward", _fake_super_forward, raising=True
        )
        return recorded

    def test_full_attention_layer_suppresses_position_embeddings_kwarg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = self._record_super_forward(monkeypatch)
        cfg = _make_cfg()
        bridge = _cohere2_attention_bridge(cfg)
        bridge.set_original_component(FakeCohere2Attention(cfg, sliding_window=None))
        hidden = randn(2, 8, 64)

        result = bridge.forward(hidden, position_embeddings=(ones(1, 8, 16), zeros(1, 8, 16)))

        assert result == "sentinel-output"
        assert recorded["kwargs"]["position_embeddings"] is None

    def test_sliding_attention_layer_passes_position_embeddings_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = self._record_super_forward(monkeypatch)
        cfg = _make_cfg()
        bridge = _cohere2_attention_bridge(cfg)
        bridge.set_original_component(FakeCohere2Attention(cfg, sliding_window=4096))
        hidden = randn(2, 8, 64)
        pos = (ones(1, 8, 16), zeros(1, 8, 16))

        bridge.forward(hidden, position_embeddings=pos)

        assert recorded["kwargs"]["position_embeddings"] is pos

    def test_full_attention_layer_suppresses_positional_position_embeddings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = self._record_super_forward(monkeypatch)
        cfg = _make_cfg()
        bridge = _cohere2_attention_bridge(cfg)
        bridge.set_original_component(FakeCohere2Attention(cfg, sliding_window=None))
        hidden = randn(2, 8, 64)
        pos = (ones(1, 8, 16), zeros(1, 8, 16))

        bridge.forward(hidden, pos, None)

        assert recorded["args"][0] is hidden
        assert recorded["args"][1] is None

    def test_layer_types_fallback_identifies_full_attention_layer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = self._record_super_forward(monkeypatch)
        cfg = _make_cfg(
            n_layers=2,
            layer_types=["sliding_attention", "full_attention"],
        )
        adapter = Cohere2ArchitectureAdapter(cfg)
        bridge = _cohere2_attention_bridge(adapter.cfg)
        bridge.set_original_component(
            FakeCohere2Attention(
                adapter.cfg,
                expose_sliding_window=False,
                layer_idx=1,
            )
        )
        hidden = randn(2, 8, 64)

        bridge.forward(hidden, position_embeddings=(ones(1, 8, 16), zeros(1, 8, 16)))

        assert recorded["kwargs"]["position_embeddings"] is None

    @staticmethod
    def _wired_bridge(
        cfg: TransformerBridgeConfig,
        sliding_window: int | None,
    ) -> _Cohere2AttentionBridge:
        fake_attn = FakeCohere2Attention(cfg, sliding_window=sliding_window)
        bridge = _cohere2_attention_bridge(cfg)
        bridge.set_original_component(fake_attn)
        for name, original in {
            "q": fake_attn.q_proj,
            "k": fake_attn.k_proj,
            "v": fake_attn.v_proj,
            "o": fake_attn.o_proj,
        }.items():
            submodule = bridge.submodules[name]
            submodule.set_original_component(original)
            bridge.add_module(name, submodule)
        bridge.setup_hook_compatibility()
        return bridge

    @staticmethod
    def _forward(bridge: _Cohere2AttentionBridge, hidden: Any, position_embeddings: Any) -> Any:
        with no_grad():
            out = bridge(hidden, position_embeddings=position_embeddings, attention_mask=None)
        return out[0] if isinstance(out, tuple) else out

    def test_full_attention_output_ignores_position_embeddings_end_to_end(self) -> None:
        manual_seed(0)
        cfg = _make_cfg(n_heads=4, n_key_value_heads=2, d_model=64)
        bridge = self._wired_bridge(cfg, sliding_window=None)
        hidden = randn(2, 8, 64)
        cos = randn(1, 8, 16)
        sin = randn(1, 8, 16)

        out_with_pos = self._forward(bridge, hidden, (cos, sin))
        out_without_pos = self._forward(bridge, hidden, None)

        assert allclose(out_with_pos, out_without_pos, atol=1e-6)

    def test_sliding_attention_output_depends_on_position_embeddings_end_to_end(self) -> None:
        manual_seed(0)
        cfg = _make_cfg(n_heads=4, n_key_value_heads=2, d_model=64)
        bridge = self._wired_bridge(cfg, sliding_window=4096)
        hidden = randn(2, 8, 64)
        cos = randn(1, 8, 16)
        sin = randn(1, 8, 16)

        out_with_pos = self._forward(bridge, hidden, (cos, sin))
        out_without_pos = self._forward(bridge, hidden, None)

        assert not allclose(out_with_pos, out_without_pos, atol=1e-6)


class TestCohere2Registration:
    def test_factory_selects_cohere2_adapter(self) -> None:
        cfg = _make_cfg()

        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)

        assert isinstance(adapter, Cohere2ArchitectureAdapter)

    def test_supported_architectures_contains_cohere2(self) -> None:
        assert SUPPORTED_ARCHITECTURES["Cohere2ForCausalLM"] is Cohere2ArchitectureAdapter

    def test_registry_contains_cohere2(self) -> None:
        assert "Cohere2ForCausalLM" in HF_SUPPORTED_ARCHITECTURES
        assert CANONICAL_AUTHORS_BY_ARCH["Cohere2ForCausalLM"] == ["CohereLabs"]

    def test_model_type_routes_to_cohere2(self) -> None:
        hf_config = SimpleNamespace(model_type="cohere2", architectures=None)

        assert determine_architecture_from_hf_config(hf_config) == "Cohere2ForCausalLM"

    def test_sliding_window_pattern_passthrough_survives_hf_config_translation(self) -> None:
        hf_config = SimpleNamespace(
            model_type="cohere2",
            architectures=["Cohere2ForCausalLM"],
            hidden_size=64,
            head_dim=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=4,
            vocab_size=100,
            max_position_embeddings=128,
            intermediate_size=256,
            hidden_act="silu",
            layer_norm_eps=1e-5,
            logit_scale=1.0,
            rope_parameters={"rope_theta": 50000.0, "rope_type": "default"},
            sliding_window_pattern=4,
        )

        cfg = build_bridge_config_from_hf(
            hf_config,
            architecture="Cohere2ForCausalLM",
            model_name="synthetic-cohere2",
            dtype=float32,
        )

        assert cfg.sliding_window_pattern == 4
