"""Unit tests for RavenArchitectureAdapter (Huginn depth-recurrent decoder).

Synthetic-config only — no HF Hub access, no weight loading. Covers the
architecture quirks: RMS/rotary/gated flags, ``supports_fold_ln = False``,
the empty ``applicable_phases`` (off the transformer-shaped verify path),
recurrence-config propagation, the three separate prelude / core_block / coda
block lists mapped via ``OpaqueBlockBridge``, the combined-QKV native attention
and combined gate+up MLP submodules, and four-place registration.
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    OpaqueBlockBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.raven import (
    RavenArchitectureAdapter,
)

# The three physical block lists Huginn stores under model.transformer.
BLOCK_LISTS = ("prelude", "core_block", "coda")


def _make_cfg(
    n_embd: int = 55,
    n_heads: int = 55,
    n_layers: int = 8,
    n_ctx: int = 128,
    d_vocab: int = 256,
    mean_recurrence: int = 32,
    n_layers_in_prelude: int = 2,
    n_layers_in_recurrent_block: int = 4,
    n_layers_in_coda: int = 2,
    injection_type: str = "linear",
    qk_bias: bool = True,
) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig for Raven adapter tests.

    n_heads == n_embd keeps d_head == 1 so the tiny synthetic dims are legal
    (Huginn is MHA: num_key_value_heads == num_attention_heads).
    """
    cfg = TransformerBridgeConfig(
        d_model=n_embd,
        d_head=n_embd // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=4 * n_embd,
        default_prepend_bos=False,
        architecture="RavenForCausalLM",
    )
    # Inject Huginn recurrence-shape fields the adapter reads via getattr.
    cfg.mean_recurrence = mean_recurrence  # type: ignore[attr-defined]
    cfg.n_layers_in_prelude = n_layers_in_prelude  # type: ignore[attr-defined]
    cfg.n_layers_in_recurrent_block = n_layers_in_recurrent_block  # type: ignore[attr-defined]
    cfg.n_layers_in_coda = n_layers_in_coda  # type: ignore[attr-defined]
    cfg.injection_type = injection_type  # type: ignore[attr-defined]
    cfg.qk_bias = qk_bias  # type: ignore[attr-defined]
    return cfg


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> RavenArchitectureAdapter:
    return RavenArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attributes
# ---------------------------------------------------------------------------


class TestRavenAdapterConfig:
    """Adapter sets the required norm / positional / MLP flags."""

    def test_normalization_type_rms(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_uses_rms_norm(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_positional_embedding_type_rotary(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms_true(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_gated_mlp_true(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_attn_only_false(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_supports_fold_ln_false(self, adapter: RavenArchitectureAdapter) -> None:
        # ln_f is reused mid-network (after recurrence, feeding the coda), so
        # it must not be folded into W_U.
        assert adapter.supports_fold_ln is False

    def test_applicable_phases_empty(self) -> None:
        # Off the transformer-shaped verify_models path; correctness lives in
        # the integration tests.
        assert RavenArchitectureAdapter.applicable_phases == []

    def test_weight_processing_conversions_empty(self, adapter: RavenArchitectureAdapter) -> None:
        # Full delegation to the HF forward — no HT-format reshaping.
        assert adapter.weight_processing_conversions == {}


class TestRavenRecurrenceConfigPropagation:
    """Recurrence-shape attributes are surfaced on cfg."""

    def test_mean_recurrence(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.cfg.mean_recurrence == 32

    def test_n_layers_in_prelude(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.cfg.n_layers_in_prelude == 2

    def test_n_layers_in_recurrent_block(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.cfg.n_layers_in_recurrent_block == 4

    def test_n_layers_in_coda(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.cfg.n_layers_in_coda == 2

    def test_injection_type(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.cfg.injection_type == "linear"

    def test_qk_bias(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.cfg.qk_bias is True

    def test_physical_layer_split_sums_to_n_layers(self, adapter: RavenArchitectureAdapter) -> None:
        """prelude + recurrent-block + coda counts the 8 physical layers."""
        total = (
            adapter.cfg.n_layers_in_prelude
            + adapter.cfg.n_layers_in_recurrent_block
            + adapter.cfg.n_layers_in_coda
        )
        assert total == adapter.cfg.n_layers

    def test_recurrence_defaults_when_absent(self) -> None:
        """Adapter falls back to Huginn defaults if the attrs are missing."""
        bare = TransformerBridgeConfig(
            d_model=55,
            d_head=1,
            n_layers=8,
            n_ctx=128,
            n_heads=55,
            d_vocab=256,
            architecture="RavenForCausalLM",
        )
        a = RavenArchitectureAdapter(bare)
        assert a.cfg.mean_recurrence == 32
        assert a.cfg.n_layers_in_prelude == 2
        assert a.cfg.n_layers_in_recurrent_block == 4
        assert a.cfg.n_layers_in_coda == 2


# ---------------------------------------------------------------------------
# Top-level component mapping
# ---------------------------------------------------------------------------


class TestRavenTopLevelComponents:
    """component_mapping exposes prelude / core_block / coda as three lists."""

    def test_required_keys(self, adapter: RavenArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "prelude",
            "core_block",
            "coda",
            "ln_final",
            "unembed",
        }

    def test_embed_is_embedding_bridge(self, adapter: RavenArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "transformer.wte"

    @pytest.mark.parametrize("key", BLOCK_LISTS)
    def test_block_list_is_opaque_block_bridge(
        self, adapter: RavenArchitectureAdapter, key: str
    ) -> None:
        # OpaqueBlockBridge delegates the whole SandwichBlock so the post-residual
        # norm placement is preserved (a standard BlockBridge would assume a
        # pre-norm flow).
        assert isinstance(adapter.component_mapping[key], OpaqueBlockBridge)

    @pytest.mark.parametrize("key", BLOCK_LISTS)
    def test_block_list_name(self, adapter: RavenArchitectureAdapter, key: str) -> None:
        assert adapter.component_mapping[key].name == f"transformer.{key}"

    def test_ln_final_is_rms_normalization_bridge(self, adapter: RavenArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], RMSNormalizationBridge)

    def test_ln_final_name(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "transformer.ln_f"

    def test_unembed_is_unembedding_bridge(self, adapter: RavenArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: RavenArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"


# ---------------------------------------------------------------------------
# SandwichBlock submodules (shared shape across all three block lists)
# ---------------------------------------------------------------------------


class TestRavenBlockSubmodules:
    """Each block list wraps a SandwichBlock: four norms, native attn, gated MLP."""

    @pytest.mark.parametrize("key", BLOCK_LISTS)
    def test_submodule_keys(self, adapter: RavenArchitectureAdapter, key: str) -> None:
        block = adapter.component_mapping[key]
        assert set(block.submodules.keys()) == {
            "norm_1",
            "attn",
            "norm_2",
            "norm_3",
            "mlp",
            "norm_4",
        }

    @pytest.mark.parametrize("key", BLOCK_LISTS)
    @pytest.mark.parametrize("norm", ["norm_1", "norm_2", "norm_3", "norm_4"])
    def test_norms_are_rms(self, adapter: RavenArchitectureAdapter, key: str, norm: str) -> None:
        block = adapter.component_mapping[key]
        assert isinstance(block.submodules[norm], RMSNormalizationBridge)
        assert block.submodules[norm].name == norm

    @pytest.mark.parametrize("key", BLOCK_LISTS)
    def test_attn_is_native_attention_with_combined_qkv(
        self, adapter: RavenArchitectureAdapter, key: str
    ) -> None:
        attn = adapter.component_mapping[key].submodules["attn"]
        assert isinstance(attn, AttentionBridge)
        assert attn.name == "attn"
        # Combined Wqkv projection + output proj (no split q/k/v).
        assert isinstance(attn.submodules["qkv"], LinearBridge)
        assert attn.submodules["qkv"].name == "Wqkv"
        assert isinstance(attn.submodules["o"], LinearBridge)
        assert attn.submodules["o"].name == "proj"
        assert set(attn.submodules.keys()) == {"qkv", "o"}

    @pytest.mark.parametrize("key", BLOCK_LISTS)
    def test_mlp_is_gated_with_combined_fc(
        self, adapter: RavenArchitectureAdapter, key: str
    ) -> None:
        mlp = adapter.component_mapping[key].submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert mlp.name == "mlp"
        # Combined gate+up "fc" and output "proj".
        assert isinstance(mlp.submodules["in"], LinearBridge)
        assert mlp.submodules["in"].name == "fc"
        assert isinstance(mlp.submodules["out"], LinearBridge)
        assert mlp.submodules["out"].name == "proj"


class TestRavenBlockListsAreIndependent:
    """The three block lists must not share bridge instances.

    Each OpaqueBlockBridge binds to a distinct HF ModuleList, so reusing a single
    submodule bridge object across lists would cross-wire them.
    """

    def test_top_level_bridges_distinct(self, adapter: RavenArchitectureAdapter) -> None:
        bridges = [adapter.component_mapping[k] for k in BLOCK_LISTS]
        assert len({id(b) for b in bridges}) == len(BLOCK_LISTS)

    def test_submodule_bridges_distinct(self, adapter: RavenArchitectureAdapter) -> None:
        attns = [adapter.component_mapping[k].submodules["attn"] for k in BLOCK_LISTS]
        assert len({id(a) for a in attns}) == len(BLOCK_LISTS)
        norms = [adapter.component_mapping[k].submodules["norm_1"] for k in BLOCK_LISTS]
        assert len({id(n) for n in norms}) == len(BLOCK_LISTS)


# ---------------------------------------------------------------------------
# Factory registration + model registry
# ---------------------------------------------------------------------------


class TestRavenFactoryRegistration:
    """RavenForCausalLM is wired into the adapter factory."""

    def test_factory_returns_raven_adapter(self) -> None:
        cfg = _make_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, RavenArchitectureAdapter)

    def test_architecture_key_present(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "RavenForCausalLM" in SUPPORTED_ARCHITECTURES
        assert SUPPORTED_ARCHITECTURES["RavenForCausalLM"] is RavenArchitectureAdapter


class TestRavenModelRegistry:
    """RavenForCausalLM is listed in the model registry constants."""

    def test_in_hf_supported(self) -> None:
        from transformer_lens.tools.model_registry import HF_SUPPORTED_ARCHITECTURES

        assert "RavenForCausalLM" in HF_SUPPORTED_ARCHITECTURES

    def test_canonical_author_is_tomg_group(self) -> None:
        from transformer_lens.tools.model_registry import CANONICAL_AUTHORS_BY_ARCH

        assert CANONICAL_AUTHORS_BY_ARCH.get("RavenForCausalLM") == ["tomg-group-umd"]

    def test_has_architecture_description(self) -> None:
        from transformer_lens.tools.model_registry.generate_report import (
            ARCHITECTURE_DESCRIPTIONS,
        )

        assert "RavenForCausalLM" in ARCHITECTURE_DESCRIPTIONS
