"""Integration tests for the Qwen3MoE TransformerBridge.

Uses a tiny programmatic config on the meta device — no network access or
weight downloads. Tensor ops can't execute on meta, so forward-pass tests are
skipped and run manually during verification. Fixture pattern mirrors
tests/unit/model_bridge/test_gpt_oss_moe.py.
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


class _MockTokenizer:
    """Stand-in to satisfy TransformerBridge(tokenizer=...)."""

    pass


@pytest.fixture(scope="module")
def tiny_qwen3moe_config():
    """Small Qwen3MoeConfig: 2 layers, 4 heads, 4 experts."""
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
    """Qwen3MoE model on meta device (no weights loaded)."""
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(tiny_qwen3moe_config)
    return model


@pytest.fixture(scope="module")
def tiny_qwen3moe_bridge(tiny_qwen3moe_config, tiny_qwen3moe_model_meta):
    """TransformerBridge wrapping the tiny meta-device Qwen3MoE model."""
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


class TestQwen3MoeModelStructure:
    def test_model_has_layers(self, tiny_qwen3moe_model_meta) -> None:
        assert hasattr(tiny_qwen3moe_model_meta, "model")
        assert hasattr(tiny_qwen3moe_model_meta.model, "layers")
        assert len(tiny_qwen3moe_model_meta.model.layers) == 2

    def test_layer_has_sparse_moe_block(self, tiny_qwen3moe_model_meta) -> None:
        # Qwen3MoeSparseMoeBlock stores experts as batched 3D tensors, not a ModuleList
        layer0_mlp = tiny_qwen3moe_model_meta.model.layers[0].mlp
        assert hasattr(layer0_mlp, "experts")
        experts = layer0_mlp.experts
        assert hasattr(experts, "gate_up_proj")
        assert hasattr(experts, "down_proj")
        assert not hasattr(experts, "__iter__")

    def test_layer_has_gate_router(self, tiny_qwen3moe_model_meta) -> None:
        layer0_mlp = tiny_qwen3moe_model_meta.model.layers[0].mlp
        assert hasattr(layer0_mlp, "gate")

    def test_attention_has_q_norm_k_norm(self, tiny_qwen3moe_model_meta) -> None:
        attn = tiny_qwen3moe_model_meta.model.layers[0].self_attn
        assert hasattr(attn, "q_norm")
        assert hasattr(attn, "k_norm")


class TestQwen3MoeBridgeStructure:
    def test_block_count(self, tiny_qwen3moe_bridge) -> None:
        assert len(tiny_qwen3moe_bridge.blocks) == 2

    def test_has_core_components(self, tiny_qwen3moe_bridge) -> None:
        assert hasattr(tiny_qwen3moe_bridge, "embed")
        assert hasattr(tiny_qwen3moe_bridge, "unembed")
        assert hasattr(tiny_qwen3moe_bridge, "ln_final")

    def test_cfg_final_rms_is_true(self, tiny_qwen3moe_bridge) -> None:
        """Qwen3MoE uses final_rms=True; OLMoE uses False."""
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


# Forward-pass tests require real weights — meta-device tensor ops raise
# NotImplementedError. Run these manually during Step 3 verification.


@pytest.mark.skip(reason="Requires real weights — run manually during verification")
def test_forward_pass_matches_hf(tiny_qwen3moe_bridge) -> None:
    """Bridge logits match the HF model."""
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        bridge_out = tiny_qwen3moe_bridge(tokens)
        hf_out = tiny_qwen3moe_bridge.original_model(tokens).logits
    max_diff = (bridge_out - hf_out).abs().max().item()
    assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"


@pytest.mark.skip(reason="Requires real weights — run manually during verification")
def test_run_with_cache_captures_moe_router_scores(tiny_qwen3moe_bridge) -> None:
    """MoEBridge captures router scores in the activation cache."""
    tiny_qwen3moe_bridge.enable_compatibility_mode(no_processing=True)
    tokens = torch.tensor([[1, 2, 3, 4]])
    _, cache = tiny_qwen3moe_bridge.run_with_cache(tokens)
    for i in range(len(tiny_qwen3moe_bridge.blocks)):
        assert f"blocks.{i}.mlp.hook_router_scores" in cache, f"Missing router scores for block {i}"
