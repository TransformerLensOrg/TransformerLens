"""Unit tests for the Qwen3Next architecture adapter.

Qwen3Next is supported only via TransformerBridge, not HookedTransformer.
The bridge reads HF config directly via the adapter and bypasses
transformer_lens.loading_from_pretrained, so no convert_hf_model_config tests here.
"""

import pytest

from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.tools.model_registry import HF_SUPPORTED_ARCHITECTURES


class TestQwen3NextRegistration:
    """Adapter is registered in all lookup tables."""

    def test_adapter_importable(self):
        from transformer_lens.model_bridge.supported_architectures import (
            Qwen3NextArchitectureAdapter,
        )

        assert Qwen3NextArchitectureAdapter is not None

    def test_in_supported_architectures(self):
        assert "Qwen3NextForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_in_hf_supported_architectures(self):
        assert "Qwen3NextForCausalLM" in HF_SUPPORTED_ARCHITECTURES

    def test_adapter_class_correct(self):
        from transformer_lens.model_bridge.supported_architectures import (
            Qwen3NextArchitectureAdapter,
        )

        assert SUPPORTED_ARCHITECTURES["Qwen3NextForCausalLM"] is Qwen3NextArchitectureAdapter


def _make_bridge_cfg(**overrides):
    """Minimal TransformerBridgeConfig for Qwen3Next adapter tests."""
    from transformer_lens.config.TransformerBridgeConfig import TransformerBridgeConfig

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

    def test_attn_is_optional(self, adapter):
        """attn is absent on linear-attention layers."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["attn"].optional is True

    def test_linear_attn_is_optional(self, adapter):
        """linear_attn is absent on full-attention layers."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["linear_attn"].optional is True

    def test_linear_attn_bridge_type(self, adapter):
        from transformer_lens.model_bridge.generalized_components.gated_delta_net import (
            GatedDeltaNetBridge,
        )

        submodules = adapter.component_mapping["blocks"].submodules
        assert isinstance(submodules["linear_attn"], GatedDeltaNetBridge)

    def test_ln1_path(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["ln1"].name == "input_layernorm"

    def test_ln2_path(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["ln2"].name == "post_attention_layernorm"

    def test_mlp_path(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["mlp"].name == "mlp"

    def test_mlp_has_no_submodules(self, adapter):
        """Qwen3NextSparseMoeBlock has a non-Linear router and 3D batched experts; MoEBridge delegates to HF forward, so no internal subs are mapped."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules == {}

    def test_mlp_bridge_type(self, adapter):
        """Every real checkpoint is sparse MoE."""
        from transformer_lens.model_bridge.generalized_components import MoEBridge

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

    def test_weight_processing_conversions_empty(self, adapter):
        """No attention submodules mapped, so no conversions."""
        assert adapter.weight_processing_conversions == {}


class TestQwen3NextWeightConversions:
    """q_proj rows are interleaved per-head (query, gate, query, gate, ...) — naive first-half slice is wrong."""

    N_HEADS = 4
    D_HEAD = 8
    HIDDEN_SIZE = 32

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
            Qwen3NextArchitectureAdapter,
        )

        cfg = _make_bridge_cfg(
            n_heads=self.N_HEADS,
            d_head=self.D_HEAD,
            d_model=self.HIDDEN_SIZE,
            n_key_value_heads=self.N_HEADS,
        )
        return Qwen3NextArchitectureAdapter(cfg)

    def _make_q_proj_weight(self):
        import torch

        total_rows = self.N_HEADS * self.D_HEAD * 2
        w = torch.zeros(total_rows, self.HIDDEN_SIZE)
        for row_idx in range(total_rows):
            w[row_idx] = float(row_idx)
        return w

    def test_q_proj_output_shape(self, adapter):
        import torch

        w = self._make_q_proj_weight()
        state_dict = {"model.layers.3.self_attn.q_proj.weight": w}

        result = adapter.preprocess_weights(state_dict)
        out = result["model.layers.3.self_attn.q_proj.weight"]

        assert out.shape == (self.N_HEADS * self.D_HEAD, self.HIDDEN_SIZE)

    def test_q_proj_selects_query_rows_not_naive_first_half(self, adapter):
        import torch

        w = self._make_q_proj_weight()
        state_dict = {"model.layers.0.self_attn.q_proj.weight": w}

        result = adapter.preprocess_weights(state_dict)
        out = result["model.layers.0.self_attn.q_proj.weight"]

        for head_idx in range(self.N_HEADS):
            out_rows = out[head_idx * self.D_HEAD : (head_idx + 1) * self.D_HEAD]
            expected_start = head_idx * self.D_HEAD * 2
            expected_rows = w[expected_start : expected_start + self.D_HEAD]
            assert torch.equal(out_rows, expected_rows), (
                f"Head {head_idx}: output rows do not match expected query rows. "
                f"Got row values starting at {out_rows[0, 0].item()}, "
                f"expected starting at {expected_rows[0, 0].item()}"
            )

    def test_naive_slice_would_be_wrong(self, adapter):
        import torch

        w = self._make_q_proj_weight()
        state_dict = {"model.layers.0.self_attn.q_proj.weight": w}

        result = adapter.preprocess_weights(state_dict)
        correct_out = result["model.layers.0.self_attn.q_proj.weight"]

        naive_out = w[: self.N_HEADS * self.D_HEAD]

        if self.N_HEADS > 1:
            assert not torch.equal(correct_out, naive_out), (
                "Naive first-half slice gave the same result as per-head slice — "
                "test setup may be wrong"
            )

    def test_non_q_proj_weights_unchanged(self, adapter):
        import torch

        k_proj = torch.randn(self.N_HEADS * self.D_HEAD, self.HIDDEN_SIZE)
        down_proj = torch.randn(self.HIDDEN_SIZE, self.N_HEADS * self.D_HEAD)
        state_dict = {
            "model.layers.0.self_attn.k_proj.weight": k_proj.clone(),
            "model.layers.0.mlp.down_proj.weight": down_proj.clone(),
        }

        result = adapter.preprocess_weights(state_dict)

        assert torch.equal(result["model.layers.0.self_attn.k_proj.weight"], k_proj)
        assert torch.equal(result["model.layers.0.mlp.down_proj.weight"], down_proj)

    def test_multiple_layers_all_processed(self, adapter):
        import torch

        w0 = self._make_q_proj_weight()
        w3 = self._make_q_proj_weight() * 2

        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": w0,
            "model.layers.3.self_attn.q_proj.weight": w3,
        }

        result = adapter.preprocess_weights(state_dict)

        expected_shape = (self.N_HEADS * self.D_HEAD, self.HIDDEN_SIZE)
        assert result["model.layers.0.self_attn.q_proj.weight"].shape == expected_shape
        assert result["model.layers.3.self_attn.q_proj.weight"].shape == expected_shape

    def test_empty_state_dict_returns_empty(self, adapter):
        result = adapter.preprocess_weights({})
        assert result == {}

    def test_state_dict_without_q_proj_unchanged(self, adapter):
        import torch

        state_dict = {
            "model.embed_tokens.weight": torch.randn(100, self.HIDDEN_SIZE),
        }
        original_keys = set(state_dict.keys())

        result = adapter.preprocess_weights(state_dict)

        assert set(result.keys()) == original_keys

    def test_weight_processing_conversions_is_empty_dict(self, adapter):
        """q_proj slicing happens in preprocess_weights, not as a conversion."""
        assert adapter.weight_processing_conversions == {}


class TestQwen3NextConfigAttributes:
    """cfg attributes set by the adapter."""

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
            Qwen3NextArchitectureAdapter,
        )

        return Qwen3NextArchitectureAdapter(_make_bridge_cfg())

    def test_normalization_type(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter):
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms(self, adapter):
        assert adapter.cfg.final_rms is True

    def test_gated_mlp(self, adapter):
        assert adapter.cfg.gated_mlp is True

    def test_attn_only(self, adapter):
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm(self, adapter):
        assert adapter.cfg.uses_rms_norm is True

    def test_default_prepend_bos(self, adapter):
        assert adapter.cfg.default_prepend_bos is False

    def test_attn_implementation_eager(self, adapter):
        assert adapter.cfg.attn_implementation == "eager"

    def test_supports_fold_ln_false(self, adapter):
        """Hybrid layers break fold_ln."""
        assert adapter.supports_fold_ln is False

    def test_gated_q_proj_flag_set(self, adapter):
        """Flag drives preprocess_weights to slice the gated half of q_proj."""
        assert getattr(adapter.cfg, "gated_q_proj", False) is True

    def test_n_key_value_heads_propagates(self, adapter):
        assert adapter.cfg.n_key_value_heads == 2


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

    def test_attn_q_norm_k_norm_present(self, attn):
        """Qwen3 family uses per-head Q/K RMSNorm."""
        from transformer_lens.model_bridge.generalized_components import (
            RMSNormalizationBridge,
        )

        assert isinstance(attn.submodules["q_norm"], RMSNormalizationBridge)
        assert isinstance(attn.submodules["k_norm"], RMSNormalizationBridge)
        assert attn.submodules["q_norm"].name == "q_norm"
        assert attn.submodules["k_norm"].name == "k_norm"


class TestQwen3NextArchitectureGuards:
    """Guards against drift from Qwen3 conventions."""

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
            Qwen3NextArchitectureAdapter,
        )

        return Qwen3NextArchitectureAdapter(_make_bridge_cfg())

    def test_no_norm_offset_conversions(self, adapter):
        """LLaMA-style RMSNorm — no +1 offset like Gemma."""
        for key in adapter.weight_processing_conversions:
            assert "ln1" not in key
            assert "ln2" not in key
            assert "ln_final" not in key

    def test_mlp_is_moe_not_gated(self, adapter):
        """MoE, not the dense GatedMLP of Qwen3/Qwen3.5."""
        from transformer_lens.model_bridge.generalized_components import (
            GatedMLPBridge,
            MoEBridge,
        )

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert not isinstance(mlp, GatedMLPBridge)


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

    from transformer_lens.config.TransformerBridgeConfig import TransformerBridgeConfig
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

    def test_bridge_creation(self, bridge):
        from transformer_lens.model_bridge import TransformerBridge

        assert isinstance(bridge, TransformerBridge)

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
