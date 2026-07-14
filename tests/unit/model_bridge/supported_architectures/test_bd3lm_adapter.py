"""Unit tests for BD3LMArchitectureAdapter.

Covers: config attribute propagation, component mapping bridge types and HF
path names, applicable_phases, setup patching, and factory registration.
"""

from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    DelegatedAttentionBlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    SymbolicBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.bd3lm import (
    BD3LMArchitectureAdapter,
)


def _make_cfg(
    n_layers: int = 2,
    d_model: int = 64,
    d_head: int = 8,
    n_heads: int = 8,
    d_vocab: int = 100,
    n_ctx: int = 128,
    block_size: int = 4,
    cond_dim: int = 128,
    adaln: bool = True,
    cross_attn: bool = True,
) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig for BD3LM adapter tests."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_head,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        default_prepend_bos=False,
        architecture="BD3LM",
    )
    # Inject BD3LM-specific fields
    cfg.block_size = block_size  # type: ignore[attr-defined]
    cfg.cond_dim = cond_dim  # type: ignore[attr-defined]
    cfg.adaln = adaln  # type: ignore[attr-defined]
    cfg.cross_attn = cross_attn  # type: ignore[attr-defined]
    return cfg


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> BD3LMArchitectureAdapter:
    return BD3LMArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attributes
# ---------------------------------------------------------------------------


class TestBD3LMArchitectureAdapterConfig:
    """Adapter propagates all required config attributes."""

    def test_normalization_type(self, adapter: BD3LMArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_uses_rms_norm(self, adapter: BD3LMArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is False

    def test_positional_embedding_type(self, adapter: BD3LMArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "none"

    def test_gated_mlp_false(self, adapter: BD3LMArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is False

    def test_final_rms_false(self, adapter: BD3LMArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_applicable_phases(self, adapter: BD3LMArchitectureAdapter) -> None:
        # BD3LM now supports all phases correctly
        assert adapter.applicable_phases == [1, 2, 3]

    def test_supports_generation(self, adapter: BD3LMArchitectureAdapter) -> None:
        assert adapter.supports_generation is False


# ---------------------------------------------------------------------------
# Component Mapping
# ---------------------------------------------------------------------------


class TestBD3LMArchitectureAdapterComponentMapping:
    """Adapter maps all components to the correct HF submodules."""

    def test_embed_bridge(self, adapter: BD3LMArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        embed = adapter.component_mapping["embed"]
        assert isinstance(embed, EmbeddingBridge)
        assert embed.name == "backbone.vocab_embed"

    def test_blocks_bridge(self, adapter: BD3LMArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks, DelegatedAttentionBlockBridge)
        assert blocks.name == "backbone.blocks"

    def test_blocks_submodules(self, adapter: BD3LMArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks, DelegatedAttentionBlockBridge)
        sub = blocks.submodules
        assert sub is not None

        assert isinstance(sub["ln1"], NormalizationBridge)
        assert sub["ln1"].name == "norm1"

        assert isinstance(sub["ln2"], NormalizationBridge)
        assert sub["ln2"].name == "norm2"

        assert isinstance(sub["adaln_modulation"], LinearBridge)
        assert sub["adaln_modulation"].name == "adaLN_modulation"

        attn = sub["attn"]
        assert isinstance(attn, SymbolicBridge)
        assert attn.submodules is not None
        assert isinstance(attn.submodules["qkv"], LinearBridge)
        assert attn.submodules["qkv"].name == "attn_qkv"
        assert isinstance(attn.submodules["o"], LinearBridge)
        assert attn.submodules["o"].name == "attn_out"

        mlp = sub["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert mlp.submodules is not None
        assert isinstance(mlp.submodules["in"], LinearBridge)
        assert mlp.submodules["in"].name == "0"
        assert isinstance(mlp.submodules["out"], LinearBridge)
        assert mlp.submodules["out"].name == "2"

    def test_sigma_map_bridge(self, adapter: BD3LMArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        sigma_map = adapter.component_mapping["sigma_map"]
        assert isinstance(sigma_map, MLPBridge)
        assert sigma_map.name == "backbone.sigma_map.mlp"
        assert sigma_map.submodules is not None
        assert isinstance(sigma_map.submodules["in"], LinearBridge)
        assert sigma_map.submodules["in"].name == "0"
        assert isinstance(sigma_map.submodules["out"], LinearBridge)
        assert sigma_map.submodules["out"].name == "2"

    def test_ln_final_bridge(self, adapter: BD3LMArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        ln_final = adapter.component_mapping["ln_final"]
        assert isinstance(ln_final, NormalizationBridge)
        assert ln_final.name == "backbone.output_layer.norm_final"

    def test_unembed_bridge(self, adapter: BD3LMArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        unembed = adapter.component_mapping["unembed"]
        assert isinstance(unembed, UnembeddingBridge)
        assert unembed.name == "backbone.output_layer.linear"

    def test_component_paths_resolve_to_real_submodules(
        self, adapter: BD3LMArchitectureAdapter
    ) -> None:
        # Create a mock model matching the real BD3LM structure
        import torch.nn as nn

        class MockBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(64)
                self.attn_qkv = nn.Linear(64, 64)
                self.attn_out = nn.Linear(64, 64)
                self.adaLN_modulation = nn.Linear(128, 6 * 64)
                self.norm2 = nn.LayerNorm(64)
                self.mlp = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))

        class MockSigmaMap(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))

        class MockOutputLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm_final = nn.LayerNorm(64)
                self.linear = nn.Linear(64, 100)

        class MockBackbone(nn.Module):
            def __init__(self):
                super().__init__()

                class EmbedLayer(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.embedding = nn.Parameter(torch.randn(100, 64))

                self.vocab_embed = EmbedLayer()
                self.blocks = nn.ModuleList([MockBlock() for _ in range(2)])
                self.sigma_map = MockSigmaMap()
                self.output_layer = MockOutputLayer()

        class MockBD3LM(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = MockBackbone()

        model = MockBD3LM()

        # Prepare the model with the adapter
        adapter.prepare_model(model)

        assert adapter.component_mapping is not None

        # Verify get_remote_component works for every top-level mapping
        # 1. embed
        embed_bridge = adapter.component_mapping["embed"]
        assert embed_bridge.name is not None
        embed_comp = adapter.get_remote_component(model, embed_bridge.name)
        assert hasattr(embed_comp, "weight")

        # 2. blocks
        blocks_bridge = adapter.component_mapping["blocks"]
        assert blocks_bridge.name is not None
        blocks_comp = adapter.get_remote_component(model, blocks_bridge.name)
        assert isinstance(blocks_comp, nn.ModuleList)
        assert len(blocks_comp) == 2

        # Verify block submodules resolve
        block0 = blocks_comp[0]
        assert isinstance(blocks_bridge, DelegatedAttentionBlockBridge)
        assert blocks_bridge.submodules is not None
        for sub_name, sub_bridge in blocks_bridge.submodules.items():
            if sub_name == "attn":
                assert sub_bridge.submodules is not None
                for attn_sub_name, attn_sub_bridge in sub_bridge.submodules.items():
                    assert attn_sub_bridge.name is not None
                    comp = adapter.get_remote_component(block0, attn_sub_bridge.name)
                    assert isinstance(comp, nn.Linear)
            elif sub_name == "mlp":
                assert sub_bridge.submodules is not None
                assert sub_bridge.name is not None
                in_bridge = sub_bridge.submodules["in"]
                out_bridge = sub_bridge.submodules["out"]
                assert in_bridge.name is not None
                assert out_bridge.name is not None
                comp_in = adapter.get_remote_component(
                    block0, f"{sub_bridge.name}.{in_bridge.name}"
                )
                comp_out = adapter.get_remote_component(
                    block0, f"{sub_bridge.name}.{out_bridge.name}"
                )
                assert isinstance(comp_in, nn.Linear)
                assert isinstance(comp_out, nn.Linear)
            else:
                assert sub_bridge.name is not None
                comp = adapter.get_remote_component(block0, sub_bridge.name)
                assert isinstance(comp, (nn.LayerNorm, nn.Linear))

        # 3. sigma_map
        sigma_map_bridge = adapter.component_mapping["sigma_map"]
        assert isinstance(sigma_map_bridge, MLPBridge)
        assert sigma_map_bridge.name is not None
        assert sigma_map_bridge.submodules is not None
        sigma_map_comp = adapter.get_remote_component(model, sigma_map_bridge.name)
        in_bridge = sigma_map_bridge.submodules["in"]
        out_bridge = sigma_map_bridge.submodules["out"]
        assert in_bridge.name is not None
        assert out_bridge.name is not None
        sigma_in = adapter.get_remote_component(sigma_map_comp, in_bridge.name)
        sigma_out = adapter.get_remote_component(sigma_map_comp, out_bridge.name)
        assert isinstance(sigma_in, nn.Linear)
        assert isinstance(sigma_out, nn.Linear)

        # 4. ln_final
        ln_final_bridge = adapter.component_mapping["ln_final"]
        assert ln_final_bridge.name is not None
        ln_final = adapter.get_remote_component(model, ln_final_bridge.name)
        assert isinstance(ln_final, nn.LayerNorm)

        # 5. unembed
        unembed_bridge = adapter.component_mapping["unembed"]
        assert unembed_bridge.name is not None
        unembed = adapter.get_remote_component(model, unembed_bridge.name)
        assert isinstance(unembed, nn.Linear)


# ---------------------------------------------------------------------------
# Setup and Patches
# ---------------------------------------------------------------------------


class TestBD3LMArchitectureAdapterSetup:
    """Tests the architecture setup and monkeypatches."""

    def test_setup_patches_vocab_embed(self, adapter: BD3LMArchitectureAdapter) -> None:
        # Mock model structure
        model = MagicMock()
        model.backbone = MagicMock()
        model.backbone.vocab_embed = MagicMock()
        # Pretend vocab_embed has embedding but not weight
        del model.backbone.vocab_embed.weight
        model.backbone.vocab_embed.embedding = torch.randn(10, 10)

        adapter.prepare_model(model)

        # Should have aliased weight
        assert hasattr(model.backbone.vocab_embed, "weight")
        assert model.backbone.vocab_embed.weight is model.backbone.vocab_embed.embedding

    def test_setup_patches_attn_backend_cpu(
        self, adapter: BD3LMArchitectureAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        adapter.cfg.attn_backend = "flex"  # type: ignore[attr-defined]

        model = MagicMock()
        model.backbone = MagicMock()
        model.backbone.block_size = 4
        block = MagicMock()
        block.attn_backend = "flex"
        model.backbone.blocks = [block]

        adapter.prepare_model(model)

        assert adapter.cfg.attn_backend == "sdpa"  # type: ignore[attr-defined]
        assert block.attn_backend == "sdpa"
        model.backbone.gen_mask.assert_called_once_with(
            getattr(adapter.cfg, "model_length", getattr(adapter.cfg, "n_ctx", 2048)),
            4,
            attn_backend="sdpa",
        )


# ---------------------------------------------------------------------------
# Registry Factory
# ---------------------------------------------------------------------------


class TestBD3LMArchitectureAdapterRegistry:
    def test_factory_creates_adapter(self, cfg: TransformerBridgeConfig) -> None:
        """The factory returns the correct adapter type for BD3LM."""
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, BD3LMArchitectureAdapter)
