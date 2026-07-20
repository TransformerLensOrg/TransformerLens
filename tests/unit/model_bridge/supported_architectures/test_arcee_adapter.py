"""Unit tests for ArceeArchitectureAdapter.

Arcee (ArceeForCausalLM / AFM-4.5B) is Llama-shaped except for its MLP: an
*ungated* feed-forward block using the squared-ReLU (ReLU^2) activation. These
tests pin the architecture-specific quirks:

- config flags (RMSNorm, rotary, eager attention, ``gated_mlp = False``, GQA)
- component mapping uses the ungated ``MLPBridge`` (no ``gate`` submodule) and
  standard ``input_layernorm`` / ``post_attention_layernorm`` names with no QK-norm
- the ``"relu2"`` activation is registered and numerically correct
- ``TransformerBridgeConfig`` accepts ``act_fn = "relu2"`` (the config guard that
  asserts ``act_fn in SUPPORTED_ACTIVATIONS`` — the reason the activation must be
  registered)
- the ungated MLP exposes post-activation neurons via ``hook_post``
"""
import pytest
import torch

from transformer_lens.config import HookedTransformerConfig, TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.factories.activation_function_factory import (
    ActivationFunctionFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MLPBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.arcee import (
    ArceeArchitectureAdapter,
)
from transformer_lens.utilities.activation_functions import SUPPORTED_ACTIVATIONS


def _make_cfg(
    n_heads: int = 8,
    n_key_value_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_vocab: int = 100,
    n_ctx: int = 128,
) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig for Arcee adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        n_key_value_heads=n_key_value_heads,
        d_vocab=d_vocab,
        d_mlp=4 * d_model,
        act_fn="relu2",
        default_prepend_bos=False,
        architecture="ArceeForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> ArceeArchitectureAdapter:
    return ArceeArchitectureAdapter(cfg)


class TestArceeAdapterConfig:
    """Adapter config defaults: RMSNorm, rotary, eager attention, ungated MLP,
    and GQA propagation via n_key_value_heads."""

    def test_normalization_is_rms(self, adapter: ArceeArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.final_rms is True
        assert adapter.cfg.uses_rms_norm is True

    def test_positional_embedding_is_rotary(self, adapter: ArceeArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_mlp_is_ungated(self, adapter: ArceeArchitectureAdapter) -> None:
        """Arcee's defining feature: an ungated MLP (no gate projection)."""
        assert adapter.cfg.gated_mlp is False

    def test_attn_not_only_and_eager(self, adapter: ArceeArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False
        assert adapter.cfg.attn_implementation == "eager"

    def test_gqa_propagated(self, adapter: ArceeArchitectureAdapter) -> None:
        assert adapter.cfg.n_key_value_heads == 4


class TestArceeAdapterComponentMapping:
    """Component-mapping structure and HF module names. Key contrasts with Llama:
    the MLP is an ungated ``MLPBridge`` (no ``gate`` submodule), and attention has
    no q_norm/k_norm."""

    @staticmethod
    def _mapping(adapter: ArceeArchitectureAdapter) -> dict:
        mapping = adapter.component_mapping
        assert mapping is not None
        return mapping

    def test_embed(self, adapter: ArceeArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb(self, adapter: ArceeArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks(self, adapter: ArceeArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert mapping["blocks"].name == "model.layers"

    def test_ln_final(self, adapter: ArceeArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert mapping["ln_final"].name == "model.norm"

    def test_unembed(self, adapter: ArceeArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["unembed"].name == "lm_head"

    def test_layernorms_use_standard_names(self, adapter: ArceeArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"

    def test_attn_qkvo_names(self, adapter: ArceeArchitectureAdapter) -> None:
        attn = self._mapping(adapter)["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.name == "self_attn"
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_attn_has_no_qk_norm(self, adapter: ArceeArchitectureAdapter) -> None:
        """Unlike Qwen3/Apertus, Arcee uses plain Llama attention (no QK-norm)."""
        attn = self._mapping(adapter)["blocks"].submodules["attn"]
        assert "q_norm" not in attn.submodules
        assert "k_norm" not in attn.submodules

    def test_mlp_is_ungated_bridge(self, adapter: ArceeArchitectureAdapter) -> None:
        """The MLP must be the ungated MLPBridge, not a GatedMLPBridge, and must
        expose only up_proj / down_proj (no gate_proj)."""
        mlp = self._mapping(adapter)["blocks"].submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert not isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "mlp"
        assert isinstance(mlp.submodules["in"], LinearBridge)
        assert mlp.submodules["in"].name == "up_proj"
        assert isinstance(mlp.submodules["out"], LinearBridge)
        assert mlp.submodules["out"].name == "down_proj"
        assert "gate" not in mlp.submodules

    def test_mlp_exposes_post_activation_hook(self, adapter: ArceeArchitectureAdapter) -> None:
        """hook_post surfaces the post-activation MLP neurons (input to down_proj,
        i.e. relu2(up_proj(x))) — the sparse-activation structure the issue targets."""
        mlp = self._mapping(adapter)["blocks"].submodules["mlp"]
        assert mlp.hook_aliases["hook_post"] == "out.hook_in"


class TestArceeReLU2Activation:
    """The squared-ReLU activation: registration, numerics, and the config guard
    that requires it to be registered before an Arcee config can be constructed."""

    def test_relu2_registered(self) -> None:
        assert "relu2" in SUPPORTED_ACTIVATIONS

    def test_relu2_numerics(self) -> None:
        """relu2(x) == relu(x) ** 2: zero for negatives, x^2 for positives.

        The activation is jaxtyping-annotated ``[batch, pos, d_mlp]`` (like the
        other activations), so it is exercised with a 3-D tensor.
        """
        fn = SUPPORTED_ACTIVATIONS["relu2"]
        x = torch.tensor([-3.0, -0.5, 0.0, 0.5, 2.0]).reshape(1, 1, 5)
        expected = torch.tensor([0.0, 0.0, 0.0, 0.25, 4.0]).reshape(1, 1, 5)
        assert torch.allclose(fn(x), expected)

    def test_factory_picks_relu2(self) -> None:
        """The activation factory resolves act_fn="relu2" to the squared-ReLU fn.

        Uses a HookedTransformerConfig (the factory's declared param type), which
        also exercises the __post_init__ guard asserting act_fn in
        SUPPORTED_ACTIVATIONS."""
        htc = HookedTransformerConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=16,
            n_heads=4,
            d_vocab=100,
            d_mlp=256,
            act_fn="relu2",
        )
        fn = ActivationFunctionFactory.pick_activation_function(htc)
        x = torch.tensor([-1.0, 3.0]).reshape(1, 1, 2)
        assert torch.allclose(fn(x), torch.tensor([0.0, 9.0]).reshape(1, 1, 2))

    def test_config_accepts_relu2(self) -> None:
        """Constructing the config must not raise: the __post_init__ guard asserts
        act_fn in SUPPORTED_ACTIVATIONS."""
        cfg = _make_cfg()
        assert cfg.act_fn == "relu2"


class TestArceeAdapterWeightConversions:
    """QKVO weight conversions with GQA-aware head counts: Q uses n_heads;
    K and V use n_key_value_heads."""

    def test_four_conversion_keys(self, adapter: ArceeArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        assert len(convs) == 4

    def test_q_uses_n_heads(self, adapter: ArceeArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_k_uses_n_key_value_heads(self, adapter: ArceeArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.k.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_key_value_heads

    def test_v_uses_n_key_value_heads(self, adapter: ArceeArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.v.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_key_value_heads

    def test_o_pattern(self, adapter: ArceeArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads
