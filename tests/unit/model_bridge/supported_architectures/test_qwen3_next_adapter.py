"""Unit tests for the Qwen3Next architecture adapter.

Qwen3Next is supported via TransformerBridge.
The bridge reads HF config directly via the adapter and bypasses
transformer_lens.loading_from_pretrained, so no convert_hf_model_config tests here.
"""

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import DENSE_KEYS
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import MoEBridge


def _make_bridge_cfg(**overrides):
    """Minimal TransformerBridgeConfig for Qwen3Next adapter tests."""
    from transformer_lens.config.transformer_bridge_config import (
        TransformerBridgeConfig,
    )

    defaults = dict(
        d_model=2048,
        d_head=256,
        n_heads=8,
        n_layers=24,
        n_ctx=2048,
        d_vocab=248320,
        n_key_value_heads=2,
        architecture="Qwen3NextForCausalLM",
    )
    defaults.update(overrides)
    return TransformerBridgeConfig(**defaults)


class TestQwen3NextComponentMapping:
    """self_attn is not a block submodule (absent on linear-attn layers); only universal subs mapped."""

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
            Qwen3NextArchitectureAdapter,
        )

        cfg = _make_bridge_cfg()
        return Qwen3NextArchitectureAdapter(cfg)

    def test_component_mapping_keys(self, adapter):
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_embed_path(self, adapter):
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_path(self, adapter):
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_path(self, adapter):
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_path(self, adapter):
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_unembed_path(self, adapter):
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_block_submodules_keys(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert set(submodules.keys()) == {"ln1", "ln2", "mlp", "attn", "linear_attn"}

    def test_ln1_path(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["ln1"].name == "input_layernorm"

    def test_ln2_path(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["ln2"].name == "post_attention_layernorm"

    def test_mlp_path(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["mlp"].name == "mlp"

    def test_mlp_maps_router_shared_expert_and_dense_projections(self, adapter):
        """Qwen3NextSparseMoeBlock has 3D batched experts (delegated to HF), but
        its router and shared expert are hookable."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        # dense_* are covered by the roster in test_moe_dense_dispatch.py.
        assert set(mlp.submodules) - DENSE_KEYS == {
            "gate",
            "shared_expert",
            "shared_expert_gate",
        }
        assert all(sub.optional for sub in mlp.submodules.values())

    def test_mlp_bridge_type(self, adapter):
        """Every real checkpoint is sparse MoE."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)

    def test_ln1_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            RMSNormalizationBridge,
        )

        ln1 = adapter.component_mapping["blocks"].submodules["ln1"]
        assert isinstance(ln1, RMSNormalizationBridge)

    def test_ln2_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            RMSNormalizationBridge,
        )

        ln2 = adapter.component_mapping["blocks"].submodules["ln2"]
        assert isinstance(ln2, RMSNormalizationBridge)

    def test_blocks_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components import BlockBridge

        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_rotary_emb_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            RotaryEmbeddingBridge,
        )

        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)


class TestQwen3NextConfigAttributes:
    """cfg attributes set by the adapter."""

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
            Qwen3NextArchitectureAdapter,
        )

        return Qwen3NextArchitectureAdapter(_make_bridge_cfg())


class TestQwen3NextComponentTypes:
    """Top-level bridge classes — guards against silent type substitution."""

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
            Qwen3NextArchitectureAdapter,
        )

        return Qwen3NextArchitectureAdapter(_make_bridge_cfg())

    def test_embed_is_embedding_bridge(self, adapter):
        from transformer_lens.model_bridge.generalized_components import EmbeddingBridge

        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_ln_final_is_rms_norm_bridge(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            RMSNormalizationBridge,
        )

        assert isinstance(adapter.component_mapping["ln_final"], RMSNormalizationBridge)

    def test_unembed_is_unembedding_bridge(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            UnembeddingBridge,
        )

        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)


class TestQwen3NextAttnSubmodules:
    """Full-attention layers wire Qwen3-pattern submodules; gated q_proj half is pre-sliced."""

    @pytest.fixture
    def attn(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
            Qwen3NextArchitectureAdapter,
        )

        adapter = Qwen3NextArchitectureAdapter(_make_bridge_cfg())
        return adapter.component_mapping["blocks"].submodules["attn"]

    def test_attn_is_position_embeddings_attention(self, attn):
        from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
            PositionEmbeddingsAttentionBridge,
        )

        assert isinstance(attn, PositionEmbeddingsAttentionBridge)

    def test_attn_path(self, attn):
        assert attn.name == "self_attn"

    def test_attn_qkvo_submodule_paths(self, attn):
        from transformer_lens.model_bridge.generalized_components import LinearBridge

        for sub_name, expected_path in (
            ("q", "q_proj"),
            ("k", "k_proj"),
            ("v", "v_proj"),
            ("o", "o_proj"),
        ):
            sub = attn.submodules[sub_name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == expected_path


class TestQwen3NextArchitectureGuards:
    """Guards against drift from Qwen3 conventions."""

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
            Qwen3NextArchitectureAdapter,
        )

        return Qwen3NextArchitectureAdapter(_make_bridge_cfg())


try:
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

    _QWEN3NEXT_AVAILABLE = True
except ImportError:
    _QWEN3NEXT_AVAILABLE = False


def _make_tiny_hf_model():
    """Tiny Qwen3Next model: 8 layers (full-attn at 3, 7), sparse MoE on every layer to exercise the MoE path."""
    cfg = Qwen3NextConfig(
        hidden_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        vocab_size=512,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
    )
    model = Qwen3NextForCausalLM(cfg)
    model.eval()
    return model


def _make_tiny_bridge():
    """Build a Qwen3Next bridge from a tiny HF model."""
    from unittest.mock import MagicMock

    from transformer_lens.config.transformer_bridge_config import (
        TransformerBridgeConfig,
    )
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
        Qwen3NextArchitectureAdapter,
    )

    hf_model = _make_tiny_hf_model()

    bridge_cfg = TransformerBridgeConfig(
        d_model=128,
        d_head=32,
        n_heads=4,
        n_layers=8,
        n_ctx=2048,
        d_vocab=512,
        n_key_value_heads=2,
        architecture="Qwen3NextForCausalLM",
    )
    adapter = Qwen3NextArchitectureAdapter(bridge_cfg)
    return TransformerBridge(hf_model, adapter, tokenizer=MagicMock()), hf_model


@pytest.mark.skipif(
    not _QWEN3NEXT_AVAILABLE,
    reason="Qwen3NextForCausalLM not available in installed transformers",
)
class TestQwen3NextIntegration:
    """End-to-end tests; linear-attn falls back to torch when flash-linear-attention is absent."""

    @pytest.fixture(scope="class")
    def bridge_and_model(self):
        return _make_tiny_bridge()

    @pytest.fixture(scope="class")
    def bridge(self, bridge_and_model):
        br, _ = bridge_and_model
        return br

    @pytest.fixture(scope="class")
    def hf_model(self, bridge_and_model):
        _, hf = bridge_and_model
        return hf

    def test_hook_names_present(self, bridge):
        """blocks.0.attn.* must NOT appear — self_attn is absent on linear-attn layers."""
        hook_keys = set(bridge.hook_dict.keys())

        assert "blocks.0.hook_resid_pre" in hook_keys, "linear-attn layer must have hook_resid_pre"
        assert "blocks.3.hook_resid_pre" in hook_keys, "full-attn layer must have hook_resid_pre"

        assert any(
            "blocks.0.ln1" in k for k in hook_keys
        ), "blocks.0.ln1 submodule hooks must be present"

        assert any(
            "blocks.0.mlp" in k for k in hook_keys
        ), "blocks.0.mlp submodule hooks must be present"

        assert not any(
            "blocks.0.attn" in k for k in hook_keys
        ), "blocks.0.attn hooks must NOT be present (hybrid architecture)"

    def test_forward_pass_consistency(self, bridge, hf_model):
        import torch

        tokens = torch.randint(0, 512, (1, 4))
        with torch.no_grad():
            hf_logits = hf_model(tokens).logits
            bridge_logits = bridge(tokens)

        assert (
            hf_logits.shape == bridge_logits.shape
        ), f"Shape mismatch: HF={hf_logits.shape}, bridge={bridge_logits.shape}"
        assert torch.allclose(
            hf_logits, bridge_logits, atol=1e-4
        ), f"Logit mismatch: max diff = {(hf_logits - bridge_logits).abs().max().item():.6f}"

    def test_hook_activation_shapes(self, bridge):
        import torch

        captured: list[torch.Tensor] = []

        def capture_hook(tensor: torch.Tensor, hook: object) -> torch.Tensor:
            captured.append(tensor.detach().clone())
            return tensor

        tokens = torch.randint(0, 512, (1, 4))
        with torch.no_grad():
            bridge.run_with_hooks(tokens, fwd_hooks=[("blocks.0.mlp.hook_out", capture_hook)])

        assert len(captured) == 1, "Hook must fire exactly once per forward pass"
        output = captured[0]
        batch, seq, d_model = 1, 4, 128
        assert output.shape == (
            batch,
            seq,
            d_model,
        ), f"Expected MLP output shape ({batch}, {seq}, {d_model}), got {output.shape}"
