"""Unit tests for MiniCPM3ArchitectureAdapter.

MiniCPM3 is the DeepSeek MLA family with all-dense gated MLPs and MiniCPM's muP
scalings. These tests pin what the subclass changes: the scaled-residual MLA
block, the dense MLP, the embedding scale staying inside HF's module, the logit
scale and the BOS policy. Numerical parity is in
tests/integration/model_bridge/test_minicpm3_adapter.py.
"""

import math

import pytest

pytest.importorskip("transformers.models.minicpm3")

from transformers import MiniCPM3Config

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    GatedMLPBridge,
    MLAAttentionBridge,
    ScaledResidualBlockBridge,
)
from transformer_lens.model_bridge.supported_architectures.deepseek_v2 import (
    DeepSeekMLAFamilyArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.minicpm3 import (
    MiniCPM3ArchitectureAdapter,
)

D_MODEL = 64
N_LAYERS = 4
SCALE_DEPTH = 1.4
DIM_MODEL_BASE = 16


def _make_cfg(**muP: float) -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        d_model=D_MODEL,
        d_head=16,
        n_layers=N_LAYERS,
        n_ctx=64,
        n_heads=4,
        d_vocab=100,
        d_mlp=96,
        architecture="MiniCPM3ForCausalLM",
    )
    # Not TransformerBridgeConfig fields; _HF_PASSTHROUGH_ATTRS sets them on real loads.
    for name, value in muP.items():
        setattr(cfg, name, value)
    return cfg


@pytest.fixture
def adapter() -> MiniCPM3ArchitectureAdapter:
    return MiniCPM3ArchitectureAdapter(
        _make_cfg(scale_emb=12.0, scale_depth=SCALE_DEPTH, dim_model_base=DIM_MODEL_BASE)
    )


def test_factory_resolves_minicpm3_arch_string() -> None:
    assert SUPPORTED_ARCHITECTURES["MiniCPM3ForCausalLM"] is MiniCPM3ArchitectureAdapter


def test_is_an_mla_family_adapter_that_prepends_bos(
    adapter: MiniCPM3ArchitectureAdapter,
) -> None:
    assert isinstance(adapter, DeepSeekMLAFamilyArchitectureAdapter)
    assert adapter.cfg.default_prepend_bos is True
    assert adapter.supports_fold_ln is False


def test_blocks_scale_residual_contributions(adapter: MiniCPM3ArchitectureAdapter) -> None:
    blocks = adapter.component_mapping["blocks"]
    assert isinstance(blocks, ScaledResidualBlockBridge)
    assert blocks.residual_contribution_scale == pytest.approx(SCALE_DEPTH / math.sqrt(N_LAYERS))
    assert isinstance(blocks.submodules["attn"], MLAAttentionBridge)


def test_blocks_drop_qkv_input_aliases(adapter: MiniCPM3ArchitectureAdapter) -> None:
    # MLA has no single tensor that becomes Q/K/V
    aliases = adapter.component_mapping["blocks"].hook_aliases
    for alias in ("hook_q_input", "hook_k_input", "hook_v_input", "hook_attn_in"):
        assert alias not in aliases


def test_every_layer_is_a_dense_gated_mlp(adapter: MiniCPM3ArchitectureAdapter) -> None:
    mlp = adapter.component_mapping["blocks"].submodules["mlp"]
    assert isinstance(mlp, GatedMLPBridge)
    assert {k: v.name for k, v in mlp.submodules.items()} == {
        "gate": "gate_proj",
        "in": "up_proj",
        "out": "down_proj",
    }


def test_embedding_scale_is_left_to_the_hf_module(adapter: MiniCPM3ArchitectureAdapter) -> None:
    # MiniCPM3ScaledWordEmbedding multiplies by scale_emb in its own forward, so
    # scaling hook_out again would double it
    embed = adapter.component_mapping["embed"]
    assert embed.name == "model.embed_tokens"
    assert embed._output_contribution_scale == 1.0


def test_output_logits_scale_divides_by_logits_scaling(
    adapter: MiniCPM3ArchitectureAdapter,
) -> None:
    assert adapter.output_logits_scale() == pytest.approx(DIM_MODEL_BASE / D_MODEL)


def test_missing_muP_fields_fall_back_to_minicpm3_config_defaults() -> None:
    adapter = MiniCPM3ArchitectureAdapter(_make_cfg())
    defaults = MiniCPM3Config()
    assert adapter.component_mapping["blocks"].residual_contribution_scale == pytest.approx(
        defaults.scale_depth / math.sqrt(N_LAYERS)
    )
    assert adapter.output_logits_scale() == pytest.approx(defaults.dim_model_base / D_MODEL)


def test_verify_models_loads_minicpm3_natively() -> None:
    # The remote-code prefixes for MiniCPM 1/2/4 must not catch MiniCPM3
    from transformer_lens.tools.model_registry import REMOTE_CODE_MODEL_PREFIXES

    assert not any(
        "openbmb/MiniCPM3-4B".startswith(prefix) for prefix in REMOTE_CODE_MODEL_PREFIXES
    )
