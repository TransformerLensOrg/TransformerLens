"""Unit tests for MiniCPMArchitectureAdapter.

Tests cover:
- Config flags and the BOS policy
- Component mapping structure (bridge types and HF module names)
- The three muP scalings (embedding, residual contribution, logits)
- The attention bridge's MiniCPM decoder-layer contract
- prepare_loading's remote-code patches, against a planted fake modeling module
"""

import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import transformers.utils.import_utils as hf_import_utils
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    ScaledResidualBlockBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures import minicpm
from transformer_lens.model_bridge.supported_architectures.minicpm import (
    MiniCPMArchitectureAdapter,
)

N_HEADS = 4
N_KV_HEADS = 2
D_MODEL = 32
N_LAYERS = 3
SCALE_EMB = 12.0
SCALE_DEPTH = 1.4
DIM_MODEL_BASE = 8


def _make_cfg(**muP: float) -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        d_model=D_MODEL,
        d_head=D_MODEL // N_HEADS,
        n_layers=N_LAYERS,
        n_ctx=64,
        n_heads=N_HEADS,
        d_vocab=100,
        d_mlp=64,
        n_key_value_heads=N_KV_HEADS,
        architecture="MiniCPMForCausalLM",
    )
    # Not TransformerBridgeConfig fields; _HF_PASSTHROUGH_ATTRS sets them on real loads.
    for name, value in muP.items():
        setattr(cfg, name, value)
    return cfg


@pytest.fixture
def adapter() -> MiniCPMArchitectureAdapter:
    return MiniCPMArchitectureAdapter(
        _make_cfg(scale_emb=SCALE_EMB, scale_depth=SCALE_DEPTH, dim_model_base=DIM_MODEL_BASE)
    )


class TestMiniCPMAdapterConfig:
    def test_llama_family_flags(self, adapter: MiniCPMArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.final_rms is True
        assert adapter.cfg.gated_mlp is True

    def test_default_prepend_bos(self, adapter: MiniCPMArchitectureAdapter) -> None:
        """The checkpoints' LlamaTokenizer sets add_bos_token=True."""
        assert adapter.cfg.default_prepend_bos is True


class TestMiniCPMAdapterComponentMapping:
    def test_top_level_components(self, adapter: MiniCPMArchitectureAdapter) -> None:
        """Rotary lives inside each attention, so there is no model-level rotary_emb."""
        mapping = adapter.component_mapping
        assert set(mapping.keys()) == {"embed", "blocks", "ln_final", "unembed"}
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["blocks"], ScaledResidualBlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_hf_paths(self, adapter: MiniCPMArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"

    def test_block_submodules(self, adapter: MiniCPMArchitectureAdapter) -> None:
        subs = adapter.component_mapping["blocks"].submodules
        assert set(subs) == {"ln1", "ln2", "attn", "mlp"}
        assert subs["ln1"].name == "input_layernorm"
        assert subs["ln2"].name == "post_attention_layernorm"
        assert subs["attn"].name == "self_attn"
        assert subs["mlp"].name == "mlp"
        assert isinstance(subs["mlp"], GatedMLPBridge)

    def test_attention_bridge(self, adapter: MiniCPMArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, minicpm._MiniCPMAttentionBridge)
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert {k: v.name for k, v in attn.submodules.items()} == {
            "q": "q_proj",
            "k": "k_proj",
            "v": "v_proj",
            "o": "o_proj",
        }
        assert all(isinstance(v, LinearBridge) for v in attn.submodules.values())

    def test_mlp_hf_paths(self, adapter: MiniCPMArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert {k: v.name for k, v in mlp.submodules.items()} == {
            "gate": "gate_proj",
            "in": "up_proj",
            "out": "down_proj",
        }

    def test_weight_conversion_keys(self, adapter: MiniCPMArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }


class TestMiniCPMScaling:
    def test_embedding_scale_is_scale_emb(self, adapter: MiniCPMArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"]._output_contribution_scale == SCALE_EMB

    def test_residual_scale_matches_hf_expression(
        self, adapter: MiniCPMArchitectureAdapter
    ) -> None:
        """Same expression as MiniCPMDecoderLayer, so the floats are identical."""
        block = adapter.component_mapping["blocks"]
        assert block.residual_contribution_scale == SCALE_DEPTH / math.sqrt(N_LAYERS)

    def test_logits_divided_by_width_ratio(self, adapter: MiniCPMArchitectureAdapter) -> None:
        logits = torch.randn(2, 3, 100)
        expected = logits / (D_MODEL / DIM_MODEL_BASE)
        assert torch.equal(adapter.apply_output_logits_transform(logits), expected)

    def test_absent_attrs_use_minicpm_config_defaults(self) -> None:
        """MiniCPMConfig defaults all three to 1 — not to 'no scaling'."""
        adapter = MiniCPMArchitectureAdapter(_make_cfg())
        assert adapter.component_mapping["embed"]._output_contribution_scale == 1.0
        block = adapter.component_mapping["blocks"]
        assert block.residual_contribution_scale == 1.0 / math.sqrt(N_LAYERS)
        logits = torch.ones(1, 1, 4)
        assert torch.equal(adapter.apply_output_logits_transform(logits), logits / D_MODEL)


class _FakeRotary(torch.nn.Module):
    """MiniCPM's rotary contract: (x, seq_len) -> the first seq_len rows of the table."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.calls: list[int] = []
        table = torch.arange(64 * dim, dtype=torch.float32).reshape(64, dim)
        self.cos_cached = table
        self.sin_cached = -table

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(seq_len)
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)


class TestMiniCPMAttentionContract:
    @pytest.fixture
    def bridge(
        self, adapter: MiniCPMArchitectureAdapter
    ) -> tuple[minicpm._MiniCPMAttentionBridge, _FakeRotary]:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        rotary = _FakeRotary(D_MODEL // N_HEADS)
        hf_attn = torch.nn.Module()
        hf_attn.rotary_emb = rotary
        # Skips set_original_component's q/k/v/o validation; only rotary_emb is read.
        attn.add_module("_original_component", hf_attn)
        return attn, rotary

    def test_position_embeddings_index_the_table_by_position_ids(self, bridge: Any) -> None:
        attn, rotary = bridge
        hidden = torch.zeros(2, 3, D_MODEL)
        position_ids = torch.tensor([[4, 5, 6], [0, 1, 2]])
        cos, sin = attn._rotary_position_embeddings(
            (), {"hidden_states": hidden, "position_ids": position_ids}
        )
        assert rotary.calls == [7]
        assert cos.shape == (2, 3, D_MODEL // N_HEADS)
        assert torch.equal(cos, rotary.cos_cached[position_ids])
        assert torch.equal(sin, rotary.sin_cached[position_ids])

    def test_missing_position_ids_default_to_arange(self, bridge: Any) -> None:
        attn, rotary = bridge
        cos, _ = attn._rotary_position_embeddings((torch.zeros(1, 5, D_MODEL),), {})
        assert rotary.calls == [5]
        assert torch.equal(cos[0], rotary.cos_cached[:5])

    def test_forward_renames_cache_and_returns_three_tuple(
        self, bridge: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attn, _ = bridge
        seen: dict[str, Any] = {}

        def fake_super_forward(self: Any, *args: Any, **kwargs: Any) -> tuple:
            seen.update(kwargs)
            return "out", "weights"

        monkeypatch.setattr(PositionEmbeddingsAttentionBridge, "forward", fake_super_forward)
        cache = object()
        result = attn.forward(
            hidden_states=torch.zeros(1, 2, D_MODEL),
            position_ids=torch.tensor([[0, 1]]),
            past_key_value=cache,
        )
        assert result == ("out", "weights", cache)
        assert seen["past_key_values"] is cache
        assert "past_key_value" not in seen
        assert seen["position_embeddings"][0].shape == (1, 2, D_MODEL // N_HEADS)


class TestMiniCPMRegistry:
    def test_verify_models_trusts_remote_code(self) -> None:
        """verify_models passes trust_remote_code only for these prefixes."""
        from transformer_lens.tools.model_registry import REMOTE_CODE_MODEL_PREFIXES

        for model_id in ("openbmb/MiniCPM4-0.5B", "openbmb/MiniCPM-1B-sft-bf16"):
            assert any(model_id.startswith(prefix) for prefix in REMOTE_CODE_MODEL_PREFIXES)


class TestMiniCPMPrepareLoading:
    def test_rejects_pretraining_tp(self, adapter: MiniCPMArchitectureAdapter) -> None:
        with pytest.raises(ValueError, match="pretraining_tp=2"):
            adapter.prepare_loading("fake/minicpm", {"config": SimpleNamespace(pretraining_tp=2)})

    def test_rejects_sparse_attention(self, adapter: MiniCPMArchitectureAdapter) -> None:
        """MiniCPM4-8B ships sparse_config=None (dense); opting into InfLLM v2 must fail."""
        config = SimpleNamespace(pretraining_tp=1, sparse_config={"kernel_size": 32})
        with pytest.raises(ValueError, match="sparse attention"):
            adapter.prepare_loading("fake/minicpm", {"config": config})

    @pytest.fixture
    def planted(self, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        """A stand-in for transformers_modules.<repo>.modeling_minicpm."""
        module = ModuleType("transformers_modules.fake.modeling_minicpm")

        class MiniCPMForCausalLM:
            _tied_weights_keys = ["lm_head.weight"]

        class MiniCPMModel:
            def __init__(self) -> None:
                # The base config stores neither; MiniCPMConfig sets both itself.
                self.config = PretrainedConfig()
                self.config.use_cache = True
                self.config.num_hidden_layers = 2

            def forward(
                self, input_ids: Any = None, past_key_values: Any = None, use_cache: Any = None
            ) -> Any:
                return past_key_values

        setattr(module, "MiniCPMForCausalLM", MiniCPMForCausalLM)
        setattr(module, "MiniCPMModel", MiniCPMModel)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(minicpm, "force_import_remote_class", lambda *a, **k: None)
        # setattr-then-delattr registers an undo that removes the backfilled helper.
        monkeypatch.setattr(hf_import_utils, "is_torch_fx_available", None, raising=False)
        monkeypatch.delattr(hf_import_utils, "is_torch_fx_available")
        return module

    def test_patches_remote_module(
        self, adapter: MiniCPMArchitectureAdapter, planted: ModuleType
    ) -> None:
        adapter.prepare_loading("fake/minicpm", {})

        assert hf_import_utils.is_torch_fx_available() is False
        assert planted.MiniCPMForCausalLM._tied_weights_keys == {
            "lm_head.weight": "model.embed_tokens.weight"
        }
        assert isinstance(planted.MiniCPMModel().forward(), DynamicCache)

    def test_idempotent(self, adapter: MiniCPMArchitectureAdapter, planted: ModuleType) -> None:
        adapter.prepare_loading("fake/minicpm", {})
        forward = planted.MiniCPMModel.forward
        adapter.prepare_loading("fake/minicpm", {})
        assert planted.MiniCPMModel.forward is forward
