"""Unit tests for Qwen2ArchitectureAdapter.

Tests cover:
- Config attributes set by the adapter
- Component mapping structure and HF module paths
- Standard Q/K/V/O weight conversion rules, including GQA K/V head counts
- Narrow hook-shape coverage for Qwen2-style GQA attention with fake modules
- Factory registration
"""

from typing import Any

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import (
    RearrangeTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen2 import (
    Qwen2ArchitectureAdapter,
)


def _make_cfg(
    n_heads: int = 4,
    n_key_value_heads: int = 2,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 100,
    n_ctx: int = 64,
) -> TransformerBridgeConfig:
    # Keep dimensions tiny so adapter tests do not need HF downloads or real checkpoints.
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        n_key_value_heads=n_key_value_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=False,
        architecture="Qwen2ForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> Qwen2ArchitectureAdapter:
    return Qwen2ArchitectureAdapter(cfg)


class FakeQwen2Attention(nn.Module):
    """Minimal Qwen2-style attention module for adapter hook-shape tests."""

    def __init__(self, cfg: TransformerBridgeConfig) -> None:
        super().__init__()
        # PositionEmbeddingsAttentionBridge reads these HF-style attributes during forward.
        self.head_dim = cfg.d_head
        self.num_key_value_groups = cfg.n_heads // (cfg.n_key_value_heads or cfg.n_heads)
        self.scaling = cfg.d_head**-0.5
        self.attention_dropout = 0.0

        # Qwen2 uses GQA: Q has n_heads, while K/V have n_key_value_heads.
        kv_width = (cfg.n_key_value_heads or cfg.n_heads) * cfg.d_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)


class TestQwen2AdapterConfig:
    """Adapter-owned config defaults that downstream bridge code relies on."""

    def test_normalization_type_is_rms(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type_is_rotary(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms_is_true(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_gated_mlp_is_true(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_attn_only_is_false(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_default_prepend_bos_is_false(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert adapter.cfg.default_prepend_bos is False

    def test_uses_rms_norm_is_true(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_n_key_value_heads_propagated(self) -> None:
        adapter = Qwen2ArchitectureAdapter(_make_cfg(n_heads=8, n_key_value_heads=2))
        assert adapter.cfg.n_key_value_heads == 2


class TestQwen2ComponentMapping:
    """The adapter contract: TL canonical names mapped to Qwen2 HF module paths."""

    def test_top_level_keys(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_embed_path(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_path(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_path(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_path(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_unembed_path(self, adapter: Qwen2ArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_block_submodule_keys(self, adapter: Qwen2ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {"ln1", "ln2", "attn", "mlp"}

    def test_attention_submodule_keys(self, adapter: Qwen2ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}

    def test_mlp_submodule_keys(self, adapter: Qwen2ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}

    def test_bridge_types(self, adapter: Qwen2ArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        blocks = mapping["blocks"]
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(blocks, BlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["attn"], PositionEmbeddingsAttentionBridge)
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_attention_hf_paths(self, adapter: Qwen2ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.name == "self_attn"
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_mlp_hf_paths(self, adapter: Qwen2ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.name == "mlp"
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"

    def test_linear_submodule_bridge_types(self, adapter: Qwen2ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        attn = blocks.submodules["attn"]
        mlp = blocks.submodules["mlp"]
        for submodule in [*attn.submodules.values(), *mlp.submodules.values()]:
            assert isinstance(submodule, LinearBridge)


class TestQwen2GQAHookShapes:
    """Verify Qwen2 GQA q/k/v hooks use n_heads for Q and n_kv_heads for K/V."""

    N_HEADS = 4
    N_KV_HEADS = 2
    D_MODEL = 64
    D_HEAD = D_MODEL // N_HEADS
    BATCH = 2
    SEQ = 8

    @pytest.fixture
    def adapter(self) -> Qwen2ArchitectureAdapter:
        return Qwen2ArchitectureAdapter(
            _make_cfg(
                n_heads=self.N_HEADS,
                n_key_value_heads=self.N_KV_HEADS,
                d_model=self.D_MODEL,
            )
        )

    @pytest.fixture
    def wired_attn_bridge(
        self, adapter: Qwen2ArchitectureAdapter
    ) -> PositionEmbeddingsAttentionBridge:
        fake_attn = FakeQwen2Attention(adapter.cfg)
        attn_bridge = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn_bridge, PositionEmbeddingsAttentionBridge)
        attn_bridge.set_original_component(fake_attn)
        # A full TransformerBridge build materializes these child bridge modules for us.
        # This unit test wires them by hand so it can stay download-free.
        for name, original in {
            "q": fake_attn.q_proj,
            "k": fake_attn.k_proj,
            "v": fake_attn.v_proj,
            "o": fake_attn.o_proj,
        }.items():
            submodule = attn_bridge.submodules[name]
            submodule.set_original_component(original)
            attn_bridge.add_module(name, submodule)
        attn_bridge.setup_hook_compatibility()
        return attn_bridge

    def _run_and_capture(
        self, attn_bridge: PositionEmbeddingsAttentionBridge
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}

        def _capture(name: str) -> Any:
            def _hook(x: torch.Tensor, hook: Any) -> torch.Tensor:
                captured[name] = x.detach()
                return x

            return _hook

        attn_bridge.q.hook_out.add_hook(_capture("q"))
        attn_bridge.k.hook_out.add_hook(_capture("k"))
        attn_bridge.v.hook_out.add_hook(_capture("v"))

        hidden = torch.randn(self.BATCH, self.SEQ, self.D_MODEL)
        # Identity RoPE inputs keep this test focused on hook reshaping, not rotation math.
        cos = torch.ones(1, self.SEQ, self.D_HEAD)
        sin = torch.zeros(1, self.SEQ, self.D_HEAD)
        out = attn_bridge(hidden, position_embeddings=(cos, sin))
        out_tensor = out[0] if isinstance(out, tuple) else out

        return captured["q"], captured["k"], captured["v"], out_tensor

    def test_hook_q_shape(self, wired_attn_bridge: PositionEmbeddingsAttentionBridge) -> None:
        q, _, _, _ = self._run_and_capture(wired_attn_bridge)
        assert q.shape == (self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD)

    def test_hook_k_shape(self, wired_attn_bridge: PositionEmbeddingsAttentionBridge) -> None:
        _, k, _, _ = self._run_and_capture(wired_attn_bridge)
        assert k.shape == (self.BATCH, self.SEQ, self.N_KV_HEADS, self.D_HEAD)

    def test_hook_v_shape(self, wired_attn_bridge: PositionEmbeddingsAttentionBridge) -> None:
        _, _, v, _ = self._run_and_capture(wired_attn_bridge)
        assert v.shape == (self.BATCH, self.SEQ, self.N_KV_HEADS, self.D_HEAD)

    def test_attn_output_shape(self, wired_attn_bridge: PositionEmbeddingsAttentionBridge) -> None:
        _, _, _, out = self._run_and_capture(wired_attn_bridge)
        assert out.shape == (self.BATCH, self.SEQ, self.D_MODEL)


class TestQwen2WeightConversions:
    """Qwen2 uses the standard QKVO conversions, with GQA-specific K/V heads."""

    def test_has_qkvo_keys(self, adapter: Qwen2ArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        assert set(convs.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }

    def test_q_uses_n_heads(self, adapter: Qwen2ArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_kv_use_n_key_value_heads(self, adapter: Qwen2ArchitectureAdapter) -> None:
        for key in ("blocks.{i}.attn.k.weight", "blocks.{i}.attn.v.weight"):
            conv = adapter.weight_processing_conversions[key]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_key_value_heads

    def test_o_uses_n_heads(self, adapter: Qwen2ArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads


class TestQwen2FactoryRegistration:
    """Factory lookup must resolve HF's architecture string to this adapter."""

    def test_factory_key_present(self) -> None:
        assert "Qwen2ForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_factory_maps_to_correct_adapter_class(self) -> None:
        assert SUPPORTED_ARCHITECTURES["Qwen2ForCausalLM"] is Qwen2ArchitectureAdapter

    def test_factory_returns_correct_instance(self) -> None:
        cfg = _make_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, Qwen2ArchitectureAdapter)
