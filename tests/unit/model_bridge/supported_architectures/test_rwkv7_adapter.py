"""Unit tests for RWKV7ArchitectureAdapter (RWKV-7 "Goose" recurrent decoder).

Synthetic-config only — no HF Hub access, no weight loading. Covers the
architecture quirks: standard biased LayerNorm (``normalization_type == "LN"``),
no positional embeddings, ungated FFN, the empty ``applicable_phases`` (off the
transformer-shaped verify path), shape-attribute propagation, the single
``model.layers`` block list mapped via ``OpaqueBlockBridge``, the delegated
time-mixing (``attn``) and channel-mixing (``ffn``) sublayers wrapped by
``GeneralizedComponent`` with their projection submodules, the fused-signature
``ffn_norm`` likewise delegated via a plain ``GeneralizedComponent``, and four-place
registration.
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    LinearBridge,
    NormalizationBridge,
    OpaqueBlockBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.rwkv7 import (
    RWKV7ArchitectureAdapter,
)


def _make_cfg(
    d_model: int = 64,
    head_dim: int = 64,
    num_heads: int = 1,
    n_layers: int = 4,
    n_ctx: int = 128,
    d_vocab: int = 256,
    inject_shape_attrs: bool = True,
) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig for RWKV-7 adapter tests.

    ``head_dim == d_model`` keeps ``num_heads == 1`` so the tiny synthetic dims
    are legal.
    """
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=head_dim,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=num_heads,
        d_vocab=d_vocab,
        d_mlp=4 * d_model,
        default_prepend_bos=False,
        architecture="RWKV7ForCausalLM",
    )
    if inject_shape_attrs:
        # Inject RWKV-7 shape fields the adapter reads via getattr. head_dim is a
        # read-only alias of d_head on the config, so it is not set here.
        cfg.num_heads = num_heads  # type: ignore[attr-defined]
        cfg.value_dim = [d_model] * n_layers  # type: ignore[attr-defined]
        cfg.decay_low_rank_dim = 64  # type: ignore[attr-defined]
        cfg.gate_low_rank_dim = 128  # type: ignore[attr-defined]
        cfg.a_low_rank_dim = 64  # type: ignore[attr-defined]
        cfg.v_low_rank_dim = 16  # type: ignore[attr-defined]
        cfg.norm_first = True  # type: ignore[attr-defined]
        cfg.norm_bias = True  # type: ignore[attr-defined]
        cfg.fuse_norm = True  # type: ignore[attr-defined]
        cfg.attn_mode = "chunk"  # type: ignore[attr-defined]
        cfg.hidden_act = "sqrelu"  # type: ignore[attr-defined]
        cfg.norm_eps = 1e-5  # type: ignore[attr-defined]
    return cfg


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> RWKV7ArchitectureAdapter:
    return RWKV7ArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attributes
# ---------------------------------------------------------------------------


class TestRWKV7AdapterConfig:
    """Adapter sets the required norm / positional / MLP flags."""

    def test_normalization_type_ln(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_uses_rms_norm_false(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is False

    def test_positional_embedding_type_none(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "none"

    def test_final_rms_false(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_gated_mlp_false(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is False

    def test_attn_only_false(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_is_stateful_false(self, adapter: RWKV7ArchitectureAdapter) -> None:
        # fla drives decode state through its own Cache, not cache_params.
        assert adapter.cfg.is_stateful is False

    def test_applicable_phases_empty(self) -> None:
        # Off the transformer-shaped verify_models path; correctness lives in
        # the integration tests.
        assert RWKV7ArchitectureAdapter.applicable_phases == []

    def test_weight_processing_conversions_empty(self, adapter: RWKV7ArchitectureAdapter) -> None:
        # Full delegation to the fla forward — no HT-format reshaping.
        assert adapter.weight_processing_conversions == {}


class TestRWKV7ShapeAttrPropagation:
    """RWKV-7 shape attributes are surfaced on cfg."""

    def test_head_dim(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert adapter.cfg.head_dim == 64

    def test_num_heads(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert adapter.cfg.num_heads == 1

    def test_value_dim(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert adapter.cfg.value_dim == [64, 64, 64, 64]

    def test_low_rank_dims(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert adapter.cfg.decay_low_rank_dim == 64
        assert adapter.cfg.gate_low_rank_dim == 128
        assert adapter.cfg.a_low_rank_dim == 64
        assert adapter.cfg.v_low_rank_dim == 16

    def test_norm_flags(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert adapter.cfg.norm_first is True
        assert adapter.cfg.norm_bias is True
        assert adapter.cfg.fuse_norm is True

    def test_attn_mode_and_hidden_act(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert adapter.cfg.attn_mode == "chunk"
        assert adapter.cfg.hidden_act == "sqrelu"

    def test_defaults_when_absent(self) -> None:
        """Adapter falls back to RWKV-7 defaults if the attrs are missing."""
        bare = _make_cfg(inject_shape_attrs=False)
        a = RWKV7ArchitectureAdapter(bare)
        assert a.cfg.head_dim == 64
        # num_heads defaults to d_model // head_dim.
        assert a.cfg.num_heads == 1
        # value_dim defaults to [d_model] * n_layers.
        assert a.cfg.value_dim == [64, 64, 64, 64]
        assert a.cfg.decay_low_rank_dim == 64
        assert a.cfg.gate_low_rank_dim == 128
        assert a.cfg.a_low_rank_dim == 64
        assert a.cfg.v_low_rank_dim == 16
        assert a.cfg.norm_first is True
        assert a.cfg.fuse_norm is True
        assert a.cfg.hidden_act == "sqrelu"


# ---------------------------------------------------------------------------
# Top-level component mapping
# ---------------------------------------------------------------------------


class TestRWKV7TopLevelComponents:
    """component_mapping exposes embed / blocks / ln_final / unembed."""

    def test_required_keys(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_embed_is_embedding_bridge(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)
        assert adapter.component_mapping["embed"].name == "model.embeddings"

    def test_blocks_is_ssm_block_bridge(self, adapter: RWKV7ArchitectureAdapter) -> None:
        # OpaqueBlockBridge delegates the whole recurrent block so the internal
        # mixing is preserved (a standard BlockBridge assumes a pre-norm attn flow).
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks, OpaqueBlockBridge)
        assert blocks.name == "model.layers"

    def test_ln_final_is_normalization_bridge(self, adapter: RWKV7ArchitectureAdapter) -> None:
        ln_final = adapter.component_mapping["ln_final"]
        assert isinstance(ln_final, NormalizationBridge)
        assert ln_final.name == "model.norm"

    def test_unembed_is_unembedding_bridge(self, adapter: RWKV7ArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)
        assert adapter.component_mapping["unembed"].name == "lm_head"


# ---------------------------------------------------------------------------
# Block submodules
# ---------------------------------------------------------------------------


class TestRWKV7BlockSubmodules:
    """Each block wraps attn_norm / attn / ffn_norm / ffn."""

    def test_submodule_keys(self, adapter: RWKV7ArchitectureAdapter) -> None:
        block = adapter.component_mapping["blocks"]
        assert set(block.submodules.keys()) == {"attn_norm", "attn", "ffn_norm", "ffn"}

    def test_attn_norm_is_normalization_bridge(self, adapter: RWKV7ArchitectureAdapter) -> None:
        attn_norm = adapter.component_mapping["blocks"].submodules["attn_norm"]
        assert isinstance(attn_norm, NormalizationBridge)
        assert attn_norm.name == "attn_norm"

    def test_ffn_norm_is_delegating_passthrough(self, adapter: RWKV7ArchitectureAdapter) -> None:
        # Under config.fuse_norm the block calls ffn_norm(x, residual, True) ->
        # (normed, residual); the reimplementing NormalizationBridge can't express
        # that, so it is a plain delegating GeneralizedComponent, not a norm bridge.
        ffn_norm = adapter.component_mapping["blocks"].submodules["ffn_norm"]
        assert type(ffn_norm) is GeneralizedComponent
        assert not isinstance(ffn_norm, NormalizationBridge)
        assert ffn_norm.name == "ffn_norm"

    def test_attn_is_generalized_component_with_projections(
        self, adapter: RWKV7ArchitectureAdapter
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, GeneralizedComponent)
        assert attn.name == "attn"
        assert set(attn.submodules.keys()) == {"r_proj", "k_proj", "v_proj", "o_proj"}
        for proj in ("r_proj", "k_proj", "v_proj", "o_proj"):
            assert isinstance(attn.submodules[proj], LinearBridge)
            assert attn.submodules[proj].name == proj

    def test_ffn_is_generalized_component_with_projections(
        self, adapter: RWKV7ArchitectureAdapter
    ) -> None:
        ffn = adapter.component_mapping["blocks"].submodules["ffn"]
        assert isinstance(ffn, GeneralizedComponent)
        assert ffn.name == "ffn"
        # HF names the up-projection "key" and the (down) output projection "value".
        assert set(ffn.submodules.keys()) == {"key", "value"}
        assert isinstance(ffn.submodules["key"], LinearBridge)
        assert ffn.submodules["key"].name == "key"
        assert isinstance(ffn.submodules["value"], LinearBridge)
        assert ffn.submodules["value"].name == "value"


# ---------------------------------------------------------------------------
# Factory registration + model registry
# ---------------------------------------------------------------------------


class TestRWKV7FactoryRegistration:
    """RWKV7ForCausalLM is wired into the adapter factory."""

    def test_factory_returns_rwkv7_adapter(self) -> None:
        cfg = _make_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, RWKV7ArchitectureAdapter)

    def test_architecture_key_present(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "RWKV7ForCausalLM" in SUPPORTED_ARCHITECTURES
        assert SUPPORTED_ARCHITECTURES["RWKV7ForCausalLM"] is RWKV7ArchitectureAdapter


class TestRWKV7ModelRegistry:
    """RWKV7ForCausalLM is listed in the model registry constants."""

    def test_in_hf_supported(self) -> None:
        from transformer_lens.tools.model_registry import HF_SUPPORTED_ARCHITECTURES

        assert "RWKV7ForCausalLM" in HF_SUPPORTED_ARCHITECTURES

    def test_canonical_author_is_fla_hub(self) -> None:
        from transformer_lens.tools.model_registry import CANONICAL_AUTHORS_BY_ARCH

        assert CANONICAL_AUTHORS_BY_ARCH.get("RWKV7ForCausalLM") == ["fla-hub"]

    def test_has_architecture_description(self) -> None:
        from transformer_lens.tools.model_registry.generate_report import (
            ARCHITECTURE_DESCRIPTIONS,
        )

        assert "RWKV7ForCausalLM" in ARCHITECTURE_DESCRIPTIONS
