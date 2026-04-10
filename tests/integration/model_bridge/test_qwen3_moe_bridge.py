"""Integration tests for the Qwen3MoE TransformerBridge.

All tests use a tiny programmatic Qwen3MoE config on the meta device — no
network access and no actual weights are downloaded.  The meta device means
tensor operations cannot execute, so forward-pass tests are explicitly skipped
and marked for manual execution during verification.

Fixture pattern mirrors tests/unit/model_bridge/test_gpt_oss_moe.py.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import MoEBridge
from transformer_lens.model_bridge.sources.transformers import (
    map_default_transformer_lens_config,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_moe import (
    Qwen3MoeArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Tiny programmatic model fixture (meta device, no weights)
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal stand-in so TransformerBridge(tokenizer=...) is satisfied."""

    pass


@pytest.fixture(scope="module")
def tiny_qwen3moe_config():
    """Return a small Qwen3MoeConfig (2 layers, 4 heads, 4 experts)."""
    return AutoConfig.for_model(
        "qwen3_moe",
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=256,
        max_position_embeddings=128,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


@pytest.fixture(scope="module")
def tiny_qwen3moe_model_meta(tiny_qwen3moe_config):
    """Create a Qwen3MoE model structure on meta device (no weights loaded)."""
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(tiny_qwen3moe_config)
    return model


@pytest.fixture(scope="module")
def tiny_qwen3moe_bridge(tiny_qwen3moe_config, tiny_qwen3moe_model_meta):
    """Create a TransformerBridge wrapping the tiny meta-device Qwen3MoE model."""
    tl_config = map_default_transformer_lens_config(tiny_qwen3moe_config)

    bridge_config = TransformerBridgeConfig(
        d_model=tl_config.d_model,
        d_head=tl_config.d_head,
        n_layers=tl_config.n_layers,
        n_ctx=tl_config.n_ctx,
        n_heads=tl_config.n_heads,
        n_key_value_heads=tl_config.n_key_value_heads,
        d_vocab=tl_config.d_vocab,
        architecture="Qwen3MoeForCausalLM",
    )

    adapter = Qwen3MoeArchitectureAdapter(bridge_config)

    return TransformerBridge(
        model=tiny_qwen3moe_model_meta,
        adapter=adapter,
        tokenizer=_MockTokenizer(),
    )


# ---------------------------------------------------------------------------
# HF model structure
# ---------------------------------------------------------------------------


class TestQwen3MoeModelStructure:
    def test_model_has_layers(self, tiny_qwen3moe_model_meta) -> None:
        assert hasattr(tiny_qwen3moe_model_meta, "model")
        assert hasattr(tiny_qwen3moe_model_meta.model, "layers")
        assert len(tiny_qwen3moe_model_meta.model.layers) == 2

    def test_layer_has_sparse_moe_block(self, tiny_qwen3moe_model_meta) -> None:
        layer0_mlp = tiny_qwen3moe_model_meta.model.layers[0].mlp
        # Qwen3MoeSparseMoeBlock uses batched expert parameters (not a ModuleList)
        assert hasattr(layer0_mlp, "experts")
        experts = layer0_mlp.experts
        assert hasattr(experts, "gate_up_proj")
        assert hasattr(experts, "down_proj")
        # Experts are NOT iterable — stored as batched 3D tensors
        assert not hasattr(experts, "__iter__")

    def test_layer_has_gate_router(self, tiny_qwen3moe_model_meta) -> None:
        layer0_mlp = tiny_qwen3moe_model_meta.model.layers[0].mlp
        assert hasattr(layer0_mlp, "gate")

    def test_attention_has_q_norm_k_norm(self, tiny_qwen3moe_model_meta) -> None:
        attn = tiny_qwen3moe_model_meta.model.layers[0].self_attn
        assert hasattr(attn, "q_norm")
        assert hasattr(attn, "k_norm")


# ---------------------------------------------------------------------------
# Bridge structure
# ---------------------------------------------------------------------------


class TestQwen3MoeBridgeStructure:
    def test_block_count(self, tiny_qwen3moe_bridge) -> None:
        assert len(tiny_qwen3moe_bridge.blocks) == 2

    def test_has_core_components(self, tiny_qwen3moe_bridge) -> None:
        assert hasattr(tiny_qwen3moe_bridge, "embed")
        assert hasattr(tiny_qwen3moe_bridge, "unembed")
        assert hasattr(tiny_qwen3moe_bridge, "ln_final")

    def test_cfg_final_rms_is_true(self, tiny_qwen3moe_bridge) -> None:
        """Critical Qwen3MoE config flag — differs from OLMoE which uses False."""
        assert tiny_qwen3moe_bridge.cfg.final_rms is True

    def test_cfg_n_kv_heads(self, tiny_qwen3moe_bridge) -> None:
        assert tiny_qwen3moe_bridge.cfg.n_key_value_heads == 2

    def test_cfg_positional_embedding_type(self, tiny_qwen3moe_bridge) -> None:
        assert tiny_qwen3moe_bridge.cfg.positional_embedding_type == "rotary"

    def test_cfg_normalization_type(self, tiny_qwen3moe_bridge) -> None:
        assert tiny_qwen3moe_bridge.cfg.normalization_type == "RMS"

    def test_mlp_blocks_are_moe_bridge(self, tiny_qwen3moe_bridge) -> None:
        for i, block in enumerate(tiny_qwen3moe_bridge.blocks):
            assert isinstance(
                block.mlp, MoEBridge
            ), f"Block {i} mlp is {type(block.mlp).__name__}, expected MoEBridge"

    def test_moe_bridge_has_router_scores_hook(self, tiny_qwen3moe_bridge) -> None:
        mlp = tiny_qwen3moe_bridge.blocks[0].mlp
        assert hasattr(mlp, "hook_router_scores")

    def test_block_has_ln1_and_ln2(self, tiny_qwen3moe_bridge) -> None:
        block = tiny_qwen3moe_bridge.blocks[0]
        assert hasattr(block, "ln1")
        assert hasattr(block, "ln2")

    def test_block_attn_has_q_norm_k_norm(self, tiny_qwen3moe_bridge) -> None:
        attn = tiny_qwen3moe_bridge.blocks[0].attn
        assert hasattr(attn, "q_norm")
        assert hasattr(attn, "k_norm")


# ---------------------------------------------------------------------------
# Forward-pass tests — skipped on meta device, run manually during verification
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Requires real weights — run manually during verification")
def test_forward_pass_matches_hf(tiny_qwen3moe_bridge) -> None:
    """Bridge forward should produce logits identical to the HF model.

    Run this test manually with a real (non-meta) model during Step 3
    verification.  On meta device, tensor operations raise NotImplementedError.
    """
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        bridge_out = tiny_qwen3moe_bridge(tokens)
        hf_out = tiny_qwen3moe_bridge.original_model(tokens).logits
    max_diff = (bridge_out - hf_out).abs().max().item()
    assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"


@pytest.mark.skip(reason="Requires real weights — run manually during verification")
def test_run_with_cache_captures_moe_router_scores(tiny_qwen3moe_bridge) -> None:
    """MoEBridge should capture router scores in the activation cache.

    Run manually with real weights during Step 3 verification.
    """
    tiny_qwen3moe_bridge.enable_compatibility_mode(no_processing=True)
    tokens = torch.tensor([[1, 2, 3, 4]])
    _, cache = tiny_qwen3moe_bridge.run_with_cache(tokens)
    for i in range(len(tiny_qwen3moe_bridge.blocks)):
        assert f"blocks.{i}.mlp.hook_router_scores" in cache, f"Missing router scores for block {i}"
