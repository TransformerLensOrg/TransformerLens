"""Unit tests for the RecurrentGemmaArchitectureAdapter — no model downloads.

RecurrentGemma (Griffin) is a hybrid of RG-LRU recurrent layers and local
sliding-window attention layers, wrapped whole-layer (residual hooks only) like
LFM2. These tests pin the architecture-specific quirks:

- Gemma-family numerics: RMSNorm ``(1.0 + weight)`` offset, final-logit soft cap.
- Gated MLP, RMSNorm, ``final_rms``.
- The heterogeneous ``block_types`` pattern is exposed as ``layers_block_type``.
- Every layer bridges its two pre-norms, its temporal block (RG-LRU or local
  attention) as a passthrough ``mixer``, and its gated MLP.
- ``hook_embed`` is scaled by HF's bfloat16 ``normalizer``.
"""

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.recurrent_gemma import (
    RecurrentGemmaArchitectureAdapter,
    RecurrentGemmaBlockBridge,
)


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    bridge_cfg = make_bridge_cfg(
        "RecurrentGemmaForCausalLM",
        d_model=64,
        d_head=16,
        n_layers=6,
        n_ctx=128,
        n_heads=4,
        n_key_value_heads=1,
        d_vocab=256,
        d_mlp=128,
        default_prepend_bos=True,
    )
    # Mirror the real pipeline: the builder passes through HF's expanded
    # layers_block_type property, not the raw block_types pattern.
    bridge_cfg.layers_block_type = ["recurrent", "recurrent", "attention"]
    bridge_cfg.rms_norm_eps = 1e-6
    bridge_cfg.logits_soft_cap = 30.0
    return bridge_cfg


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> RecurrentGemmaArchitectureAdapter:
    return RecurrentGemmaArchitectureAdapter(cfg)


class TestRecurrentGemmaAdapterConfig:
    def test_norm_config_is_gemma_rmsnorm(self, adapter: RecurrentGemmaArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.uses_rms_norm is True
        assert adapter.cfg.final_rms is True
        # Gemma RMSNorm applies (1.0 + weight); see HF PR #29402.
        assert adapter.cfg.rmsnorm_uses_offset is True
        assert adapter.cfg.eps == 1e-6

    def test_gated_mlp_and_rotary(self, adapter: RecurrentGemmaArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.attn_only is False
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_d_mlp_is_half_the_intermediate_size(self) -> None:
        # HF's RecurrentGemmaMlp projects to intermediate_size // 2.
        bridge_cfg = make_bridge_cfg(
            "RecurrentGemmaForCausalLM",
            d_model=64,
            d_head=16,
            n_layers=3,
            n_ctx=128,
            n_heads=4,
            d_vocab=256,
            d_mlp=512,
        )
        bridge_cfg.layers_block_type = ["recurrent", "recurrent", "attention"]
        bridge_cfg.intermediate_size = 512
        assert RecurrentGemmaArchitectureAdapter(bridge_cfg).cfg.d_mlp == 256
        # Idempotent when the adapter is built again on the same config.
        assert RecurrentGemmaArchitectureAdapter(bridge_cfg).cfg.d_mlp == 256

    def test_final_logit_soft_cap(self, adapter: RecurrentGemmaArchitectureAdapter) -> None:
        assert adapter.cfg.output_logits_soft_cap == 30.0

    def test_default_prepend_bos_is_false(self, adapter: RecurrentGemmaArchitectureAdapter) -> None:
        assert adapter.cfg.default_prepend_bos is False

    def test_layer_types_exposed_for_analysis(
        self, adapter: RecurrentGemmaArchitectureAdapter
    ) -> None:
        # Canonical name used by Nemotron-H / Granite tooling, preserved from
        # the builder-provided per-layer list (not clobbered by block_types).
        # "attention" normalizes to the canonical name; "recurrent" passes through.
        assert adapter.cfg.layers_block_type == ["recurrent", "recurrent", "full_attention"]

    def test_applicable_phases_is_generation_only(
        self, adapter: RecurrentGemmaArchitectureAdapter
    ) -> None:
        assert adapter.applicable_phases == [4]


class TestRecurrentGemmaComponentMapping:
    def test_has_residual_only_top_level_mapping(
        self, adapter: RecurrentGemmaArchitectureAdapter
    ) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        assert set(mapping) == {"embed", "blocks", "ln_final", "unembed"}

    def test_component_types(self, adapter: RecurrentGemmaArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["blocks"], RecurrentGemmaBlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_hf_module_paths(self, adapter: RecurrentGemmaArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["blocks"].name == "model.layers"
        # RecurrentGemma names the decoder-final norm `final_norm`, not `norm`.
        assert mapping["ln_final"].name == "model.final_norm"
        assert mapping["unembed"].name == "lm_head"

    def test_blocks_advertise_mixer_and_mlp_aliases_but_no_attention(
        self, adapter: RecurrentGemmaArchitectureAdapter
    ) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.hook_aliases == {
            "hook_resid_pre": "hook_in",
            "hook_resid_mid": "ln2.hook_in",
            "hook_resid_post": "hook_out",
            "hook_mixer_in": "mixer.hook_in",
            "hook_mixer_out": "mixer.hook_out",
            "hook_mlp_out": "mlp.hook_out",
        }

    def test_block_submodules(self, adapter: RecurrentGemmaArchitectureAdapter) -> None:
        submodules = adapter.component_mapping["blocks"].submodules
        assert set(submodules) == {"ln1", "mixer", "ln2", "mlp"}
        assert isinstance(submodules["ln1"], RMSNormalizationBridge)
        assert submodules["ln1"].name == "temporal_pre_norm"
        # The temporal block is RG-LRU or local attention depending on the layer.
        assert type(submodules["mixer"]) is GeneralizedComponent
        assert submodules["mixer"].name == "temporal_block"
        assert isinstance(submodules["ln2"], RMSNormalizationBridge)
        assert submodules["ln2"].name == "channel_pre_norm"
        mlp = submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "mlp_block"
        assert {k: v.name for k, v in mlp.submodules.items()} == {
            "gate": "gate_proj",
            "in": "up_proj",
            "out": "down_proj",
        }
        assert all(isinstance(v, LinearBridge) for v in mlp.submodules.values())

    @pytest.mark.parametrize("d_model, scale", [(64, 8.0), (2560, 50.5), (4096, 64.0)])
    def test_embedding_scale_is_bfloat16_sqrt_d_model(self, d_model: int, scale: float) -> None:
        bridge_cfg = make_bridge_cfg(
            "RecurrentGemmaForCausalLM",
            d_model=d_model,
            d_head=16,
            n_layers=3,
            n_ctx=128,
            n_heads=4,
            d_vocab=256,
            d_mlp=128,
        )
        bridge_cfg.layers_block_type = ["recurrent", "recurrent", "attention"]
        embed = RecurrentGemmaArchitectureAdapter(bridge_cfg).component_mapping["embed"]
        assert embed._output_contribution_scale == scale

    def test_prepare_model_reads_the_normalizer_buffer(self, cfg: TransformerBridgeConfig) -> None:
        from types import SimpleNamespace

        import torch

        adapter = RecurrentGemmaArchitectureAdapter(cfg)
        hf_model = SimpleNamespace(
            model=SimpleNamespace(normalizer=torch.tensor(7.75)), config=SimpleNamespace()
        )
        adapter.prepare_model(hf_model)
        assert adapter.component_mapping["embed"]._output_contribution_scale == 7.75


class TestRecurrentGemmaFactoryRegistration:
    def test_registered_in_factory(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert (
            SUPPORTED_ARCHITECTURES["RecurrentGemmaForCausalLM"]
            is RecurrentGemmaArchitectureAdapter
        )
