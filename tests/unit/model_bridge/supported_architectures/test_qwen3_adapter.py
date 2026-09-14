"""Unit tests for Qwen3ArchitectureAdapter.

Tests cover:
- Config attributes
- Component mapping structure and HF module names (incl. q_norm/k_norm)
- Weight conversion keys/types (GQA: k/v use n_key_value_heads)
- Gated q_proj survives weight processing on the Qwen3.5/Qwen3Next subclasses
- Factory registration
"""
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.gated_delta_net import (
    GatedDeltaNetBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen3 import (
    Qwen3ArchitectureAdapter,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig for Qwen3 adapter tests (GQA by default)."""
    overrides.setdefault("n_key_value_heads", 4)
    return make_bridge_cfg("Qwen3ForCausalLM", **overrides)


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> Qwen3ArchitectureAdapter:
    return Qwen3ArchitectureAdapter(cfg)


class TestQwen3AdapterConfig:
    """Adapter config defaults: RMSNorm, rotary, gated MLP, eager attention,
    default_prepend_bos=False, and GQA propagation via n_key_value_heads.
    """


class TestQwen3AdapterComponentMapping:
    """
    Component-mapping structure, bridge types, including the Qwen3-specific per-head q_norm / k_norm and the dense
    (non-hybrid) shape with no linear_attn submodule.
    """

    @staticmethod
    def _mapping(adapter: Qwen3ArchitectureAdapter) -> dict[str, Any]:
        mapping = adapter.component_mapping
        assert mapping is not None
        return mapping

    def test_embed_type_and_name(self, adapter: Qwen3ArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb(self, adapter: Qwen3ArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_type_and_name(self, adapter: Qwen3ArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert mapping["blocks"].name == "model.layers"

    def test_ln_final(self, adapter: Qwen3ArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert mapping["ln_final"].name == "model.norm"

    def test_unembed(self, adapter: Qwen3ArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["unembed"].name == "lm_head"

    def test_ln1(self, adapter: Qwen3ArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert blocks.submodules["ln1"].name == "input_layernorm"

    def test_ln2(self, adapter: Qwen3ArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"

    def test_attn_type_and_name(self, adapter: Qwen3ArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["attn"], PositionEmbeddingsAttentionBridge)
        assert blocks.submodules["attn"].name == "self_attn"

    def test_attn_qkvo_names(self, adapter: Qwen3ArchitectureAdapter) -> None:
        attn = self._mapping(adapter)["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_attn_qk_norms(self, adapter: Qwen3ArchitectureAdapter) -> None:
        """Qwen3-specific Q/K head norms."""
        attn = self._mapping(adapter)["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["q_norm"], RMSNormalizationBridge)
        assert attn.submodules["q_norm"].name == "q_norm"
        assert isinstance(attn.submodules["k_norm"], RMSNormalizationBridge)
        assert attn.submodules["k_norm"].name == "k_norm"

    def test_mlp(self, adapter: Qwen3ArchitectureAdapter) -> None:
        mlp = self._mapping(adapter)["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "mlp"
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"

    def test_no_linear_attn_when_dense(self, adapter: Qwen3ArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert "linear_attn" not in blocks.submodules


GATED_ADAPTERS = (
    ("Qwen3_5ForCausalLM", "qwen3_5", "Qwen3_5ArchitectureAdapter"),
    ("Qwen3_5MoeForCausalLM", "qwen3_5_moe", "Qwen3_5MoeArchitectureAdapter"),
    (
        "Qwen3_5ForConditionalGeneration",
        "qwen3_5_multimodal",
        "Qwen3_5MultimodalArchitectureAdapter",
    ),
    ("Qwen3NextForCausalLM", "qwen3_next", "Qwen3NextArchitectureAdapter"),
)


@pytest.mark.parametrize(
    "architecture,module,cls_name", GATED_ADAPTERS, ids=[a[0] for a in GATED_ADAPTERS]
)
class TestGatedQProjSurvivesWeightProcessing:
    """q_proj carries [query|gate] per head; HF and the attention bridge both split it at
    forward time and scale attn_output by sigmoid(gate). Weight processing must therefore
    hand the 2x-wide matrix back untouched — a query-half slice drops the gate and moves
    the logits, and applied twice it silently halves the weight again."""

    N_HEADS = 4
    D_HEAD = 8
    D_MODEL = 32

    def _adapter(self, architecture: str, module: str, cls_name: str) -> Any:
        import importlib

        cfg = make_bridge_cfg(
            architecture,
            n_heads=self.N_HEADS,
            d_head=self.D_HEAD,
            d_model=self.D_MODEL,
            n_key_value_heads=self.N_HEADS,
        )
        mod = importlib.import_module(
            f"transformer_lens.model_bridge.supported_architectures.{module}"
        )
        return getattr(mod, cls_name)(cfg)

    def _gated_q(self) -> torch.Tensor:
        return torch.randn(self.N_HEADS * self.D_HEAD * 2, self.D_MODEL)

    def test_gated_q_proj_returned_untouched(self, architecture, module, cls_name) -> None:
        adapter = self._adapter(architecture, module, cls_name)
        w = self._gated_q()
        # The bridge renames to TL keys before calling preprocess_weights.
        key = "blocks.1.attn.q.weight"
        out = adapter.preprocess_weights({key: w.clone()})
        assert torch.equal(out[key], w)

    def test_repeat_application_is_a_noop(self, architecture, module, cls_name) -> None:
        adapter = self._adapter(architecture, module, cls_name)
        w = self._gated_q()
        key = "blocks.1.attn.q.weight"
        state_dict = {key: w.clone()}
        for _ in range(3):
            state_dict = adapter.preprocess_weights(state_dict)
        assert torch.equal(state_dict[key], w)


class TestQwen3HybridConstructor:
    """The hybrid=True constructor branch on the base class. The Qwen3_5 /
    Qwen3Next subclasses exercise this path transitively; pinning it here
    surfaces regressions in the base contract:
      - linear_attn (GatedDeltaNetBridge) submodule appears alongside the
        full-attention branch
      - supports_fold_ln flips to False
      - weight_processing_conversions is cleared
    """

    @pytest.fixture
    def hybrid_adapter(self) -> Qwen3ArchitectureAdapter:
        return Qwen3ArchitectureAdapter(_make_cfg(), hybrid=True)

    def test_supports_fold_ln_disabled(self, hybrid_adapter: Qwen3ArchitectureAdapter) -> None:
        assert hybrid_adapter.supports_fold_ln is False

    def test_weight_processing_conversions_empty(
        self, hybrid_adapter: Qwen3ArchitectureAdapter
    ) -> None:
        assert hybrid_adapter.weight_processing_conversions == {}

    def test_linear_attn_submodule_present(self, hybrid_adapter: Qwen3ArchitectureAdapter) -> None:
        mapping = hybrid_adapter.component_mapping
        assert mapping is not None
        blocks = mapping["blocks"]
        assert "linear_attn" in blocks.submodules
        assert isinstance(blocks.submodules["linear_attn"], GatedDeltaNetBridge)
        assert blocks.submodules["linear_attn"].name == "linear_attn"

    def test_attn_submodule_still_present(self, hybrid_adapter: Qwen3ArchitectureAdapter) -> None:
        """Hybrid keeps full attention alongside linear_attn (both optional)."""
        mapping = hybrid_adapter.component_mapping
        assert mapping is not None
        blocks = mapping["blocks"]
        assert "attn" in blocks.submodules
        assert isinstance(blocks.submodules["attn"], PositionEmbeddingsAttentionBridge)

    def test_dense_default_has_conversions(self, cfg: TransformerBridgeConfig) -> None:
        """Sanity contrast: dense (hybrid=False) keeps the QKVO conversions."""
        dense = Qwen3ArchitectureAdapter(cfg)
        assert dense.weight_processing_conversions
        assert len(dense.weight_processing_conversions) == 4


class _StubAttnBlock:
    """Stand-in for a bridge block with an .attn that records set_rotary_emb."""

    def __init__(self) -> None:
        self.attn = SimpleNamespace(_rotary=None)
        # set_rotary_emb mimics the PositionEmbeddingBridgeMixin contract.
        self.attn.set_rotary_emb = lambda r: setattr(self.attn, "_rotary", r)
        # Mirror nn.Module._modules so the adapter's `"attn" in block._modules` check passes.
        self._modules = {"attn": self.attn}


class TestQwen3SetupComponentTesting:
    """
    Setup_component_testing wiring:
      - forces eager attention on both the top-level HF config and each
        per-layer self_attn.config
      - calls set_rotary_emb on each bridge block's attention
      - tolerates bridge_model=None (no-op for bridge wiring)
      - swallows get_generalized_component lookup failures on the template
        (the documented (ValueError, AttributeError, KeyError) net)
    """

    def _make_fake_attn(self, layer_idx: int) -> SimpleNamespace:
        """Per-layer self_attn with a mutable .config to assert eager flip."""
        return SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))

    def _make_fake_hf_model(self, n_layers: int = 2) -> SimpleNamespace:
        """Minimal hf_model stub exposing the attributes setup_component_testing walks."""
        layers = [SimpleNamespace(self_attn=self._make_fake_attn(i)) for i in range(n_layers)]
        sentinel_rotary = SimpleNamespace(_id="rotary-sentinel")
        return SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="sdpa"),
            model=SimpleNamespace(rotary_emb=sentinel_rotary, layers=layers),
        )

    def test_flips_top_level_attn_implementation_to_eager(
        self, adapter: Qwen3ArchitectureAdapter
    ) -> None:
        hf = self._make_fake_hf_model()
        adapter.setup_component_testing(hf)
        assert hf.config._attn_implementation == "eager"

    def test_flips_per_layer_attn_implementation_to_eager(
        self, adapter: Qwen3ArchitectureAdapter
    ) -> None:
        hf = self._make_fake_hf_model(n_layers=3)
        adapter.setup_component_testing(hf)
        for layer in hf.model.layers:
            assert layer.self_attn.config._attn_implementation == "eager"

    def test_wires_rotary_on_bridge_blocks(self, adapter: Qwen3ArchitectureAdapter) -> None:
        hf = self._make_fake_hf_model()
        bridge_blocks = [_StubAttnBlock(), _StubAttnBlock()]
        bridge_model = SimpleNamespace(blocks=bridge_blocks)
        adapter.setup_component_testing(hf, bridge_model=bridge_model)
        for block in bridge_blocks:
            assert block.attn._rotary is hf.model.rotary_emb

    def test_skips_bridge_wiring_when_bridge_model_none(
        self, adapter: Qwen3ArchitectureAdapter
    ) -> None:
        """No bridge_model → must not raise; eager flips still apply."""
        hf = self._make_fake_hf_model()
        adapter.setup_component_testing(hf, bridge_model=None)
        assert hf.config._attn_implementation == "eager"

    def test_swallows_template_lookup_failure(
        self, adapter: Qwen3ArchitectureAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_generalized_component may raise; setup_component_testing must
        not propagate (caught by the (ValueError, AttributeError, KeyError) net)."""

        def _raise(_self: Any, _path: str) -> None:
            raise KeyError("blocks.0.attn")

        monkeypatch.setattr(Qwen3ArchitectureAdapter, "get_generalized_component", _raise)
        hf = self._make_fake_hf_model()
        # Must not raise.
        adapter.setup_component_testing(hf)


class TestQwen3AdapterWeightConversions:
    """QKVO weight conversions with GQA-aware head counts:
    Q uses n_heads; K and V use n_key_value_heads."""

    def test_four_conversion_keys(self, adapter: Qwen3ArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        assert len(convs) == 4

    def test_q_uses_n_heads(self, adapter: Qwen3ArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_k_uses_n_key_value_heads(self, adapter: Qwen3ArchitectureAdapter) -> None:
        """GQA: K is split along n_key_value_heads, not n_heads."""
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.k.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_key_value_heads

    def test_v_uses_n_key_value_heads(self, adapter: Qwen3ArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.v.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_key_value_heads

    def test_o_pattern(self, adapter: Qwen3ArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads
