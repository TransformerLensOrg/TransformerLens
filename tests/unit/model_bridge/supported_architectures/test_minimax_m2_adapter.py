"""Unit tests for the MiniMaxM2ArchitectureAdapter.

Download-free: tiny synthetic configs plus an in-memory tiny HF model.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.minimax_m2 import (
    MiniMaxM2ArchitectureAdapter,
)

try:
    from transformers import MiniMaxM2Config
    from transformers import MiniMaxM2ForCausalLM as _MiniMaxM2ForCausalLM

    _MINIMAX_M2_AVAILABLE = True
except ImportError:
    _MINIMAX_M2_AVAILABLE = False


def _make_cfg(**overrides) -> TransformerBridgeConfig:
    defaults = dict(
        d_model=64,
        d_head=32,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=32,
        d_vocab=512,
        n_key_value_heads=2,
        architecture="MiniMaxM2ForCausalLM",
    )
    defaults.update(overrides)
    return TransformerBridgeConfig(**defaults)


@pytest.fixture(scope="class")
def adapter() -> MiniMaxM2ArchitectureAdapter:
    return MiniMaxM2ArchitectureAdapter(_make_cfg())


class TestMiniMaxM2AdapterConfig:
    def test_config_flags(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.final_rms is True
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.attn_only is False
        assert adapter.cfg.uses_rms_norm is True

    def test_no_bos_prepending(self, adapter):
        """Verified against MiniMaxAI/MiniMax-M2's tokenizer."""
        assert adapter.cfg.default_prepend_bos is False

    def test_gqa_heads_propagated(self, adapter):
        assert adapter.cfg.n_key_value_heads == 2

    def test_qkvo_conversions_registered(self, adapter):
        keys = set(adapter.weight_processing_conversions)
        assert {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        } <= keys


class TestMiniMaxM2ComponentMapping:
    def test_top_level_keys(self, adapter):
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_top_level_component_types(self, adapter):
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_attention_has_full_width_qk_norms(self, adapter):
        """MiniMax-M2 normalizes Q/K over the full projection width pre-reshape."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o", "q_norm", "k_norm"}
        assert isinstance(attn.submodules["q_norm"], RMSNormalizationBridge)
        assert attn.submodules["q_norm"].name == "q_norm"
        for name, path in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")):
            sub = attn.submodules[name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == path

    def test_moe_router_hookable(self, adapter):
        """Sigmoid router logits are hookable at mlp.gate.hook_out; the tuple
        (logits, scores, indices) is re-packed for the block's unpack."""
        from transformer_lens.model_bridge.generalized_components import MoERouterBridge

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert set(mlp.submodules) == {"gate"}
        assert isinstance(mlp.submodules["gate"], MoERouterBridge)


class TestMiniMaxM2Registration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["MiniMaxM2ForCausalLM"] is MiniMaxM2ArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="minimax_m2", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "MiniMaxM2ForCausalLM"


def _make_tiny_hf_model():
    cfg = MiniMaxM2Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_local_experts=8,
        num_experts_per_tok=2,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "rope_type": "default",
        },
    )
    model = _MiniMaxM2ForCausalLM(cfg)
    model.eval()
    return model


@pytest.mark.skipif(
    not _MINIMAX_M2_AVAILABLE,
    reason="MiniMaxM2Config / MiniMaxM2ForCausalLM not available in installed transformers",
)
class TestMiniMaxM2Integration:
    @pytest.fixture(scope="class")
    def bridge_and_model(self):
        from unittest.mock import MagicMock

        from transformer_lens.model_bridge import TransformerBridge

        hf_model = _make_tiny_hf_model()
        adapter = MiniMaxM2ArchitectureAdapter(_make_cfg())
        return TransformerBridge(hf_model, adapter, tokenizer=MagicMock()), hf_model

    def test_forward_pass_consistency(self, bridge_and_model):
        import torch

        bridge, hf_model = bridge_and_model
        tokens = torch.randint(0, 512, (1, 6))
        with torch.no_grad():
            hf_logits = hf_model(tokens).logits
            bridge_logits = bridge(tokens)
        assert hf_logits.shape == bridge_logits.shape
        assert torch.allclose(
            hf_logits, bridge_logits, atol=1e-4
        ), f"Logit mismatch: max diff = {(hf_logits - bridge_logits).abs().max().item():.6f}"

    def test_moe_and_qk_norm_hooks_fire(self, bridge_and_model):
        import torch

        bridge, _ = bridge_and_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        tokens = torch.randint(0, 512, (1, 6))
        with torch.no_grad():
            bridge.run_with_hooks(
                tokens,
                fwd_hooks=[
                    ("blocks.0.mlp.hook_out", grab),
                    ("blocks.0.attn.q_norm.hook_out", grab),
                ],
            )
        assert captured["blocks.0.mlp.hook_out"] == (1, 6, 64)
        # Full projection width: n_heads * head_dim = 4 * 32.
        assert captured["blocks.0.attn.q_norm.hook_out"] == (1, 6, 128)

    def test_sigmoid_router_bias_buffer_present(self, bridge_and_model):
        """e_score_correction_bias is a buffer on the delegated HF MoE block."""
        bridge, hf_model = bridge_and_model
        moe_block = hf_model.model.layers[0].mlp
        assert hasattr(moe_block, "e_score_correction_bias")
        assert moe_block.e_score_correction_bias.shape == (8,)
