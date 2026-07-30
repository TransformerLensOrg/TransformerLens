"""Download-free integration tests for the DeepSeek V4 architecture adapter."""

from typing import NamedTuple

import pytest
import torch
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_from_module


class DeepseekV4Case(NamedTuple):
    bridge: TransformerBridge
    tokens: torch.Tensor
    hf_logits: torch.Tensor


@pytest.fixture(scope="module")
def deepseek_v4_case() -> DeepseekV4Case:
    """Build a 40K-parameter model spanning sliding, CSA, and HCA layers."""
    torch.manual_seed(0)
    cfg = DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=16,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        max_position_embeddings=32,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        hc_mult=2,
        hc_sinkhorn_iters=2,
        mlp_layer_types=["hash_moe", "moe", "moe"],
        sliding_window=4,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        partial_rotary_factor=0.5,
        use_cache=False,
    )
    cfg._attn_implementation = "eager"
    hf_model = DeepseekV4ForCausalLM(cfg).eval()
    tokens = torch.arange(8).unsqueeze(0)

    with torch.no_grad():
        hf_logits = hf_model(tokens, use_cache=False).logits

    bridge = build_bridge_from_module(
        hf_model,
        "DeepseekV4ForCausalLM",
        hf_config=cfg,
        device="cpu",
        model_name="tiny-random-deepseek-v4",
    )
    bridge.eval()
    return DeepseekV4Case(bridge=bridge, tokens=tokens, hf_logits=hf_logits)


def test_forward_matches_hugging_face_exactly(deepseek_v4_case: DeepseekV4Case) -> None:
    with torch.no_grad():
        bridge_logits = deepseek_v4_case.bridge(deepseek_v4_case.tokens, use_cache=False)

    torch.testing.assert_close(
        bridge_logits,
        deepseek_v4_case.hf_logits,
        atol=0,
        rtol=0,
    )


def test_v4_config_metadata_is_preserved(deepseek_v4_case: DeepseekV4Case) -> None:
    cfg = deepseek_v4_case.bridge.cfg
    assert cfg.layer_types == [
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    assert cfg.compress_rates == {
        "compressed_sparse_attention": 2,
        "heavily_compressed_attention": 4,
    }
    assert cfg.hc_mult == 2
    assert cfg.mlp_layer_types == ["hash_moe", "moe", "moe"]
    assert cfg.index_topk == 2


def test_mhc_hooks_preserve_stream_and_collapsed_shapes(
    deepseek_v4_case: DeepseekV4Case,
) -> None:
    _, cache = deepseek_v4_case.bridge.run_with_cache(
        deepseek_v4_case.tokens,
        use_cache=False,
    )

    assert cache["blocks.0.hook_in"].shape == (1, 8, 2, 32)
    assert cache["blocks.0.attn_hc.hook_post"].shape == (1, 8, 2)
    assert cache["blocks.0.attn_hc.hook_comb"].shape == (1, 8, 2, 2)
    assert cache["blocks.0.attn_hc.hook_out"].shape == (1, 8, 32)
    assert cache["blocks.0.mlp_hc.hook_in"].shape == (1, 8, 2, 32)
    assert cache["blocks.0.mlp_hc.hook_out"].shape == (1, 8, 32)
    assert cache["blocks.0.hook_out"].shape == (1, 8, 2, 32)
    assert cache["hc_head.hook_in"].shape == (1, 8, 2, 32)
    assert cache["hc_head.hook_out"].shape == (1, 8, 32)


def test_compression_hooks_match_layer_types(deepseek_v4_case: DeepseekV4Case) -> None:
    _, cache = deepseek_v4_case.bridge.run_with_cache(
        deepseek_v4_case.tokens,
        use_cache=False,
    )

    assert not any(key.startswith("blocks.0.attn.compressor") for key in cache)

    for layer in (1, 2):
        compressed = cache[f"blocks.{layer}.attn.compressor.hook_out"]
        block_bias = cache[f"blocks.{layer}.attn.compressor.hook_block_bias"]
        assert compressed.shape[:2] == (1, 1)
        assert compressed.shape[-1] == 8
        assert block_bias.shape[:3] == (1, 1, 8)

    indexer = cache["blocks.1.attn.compressor.indexer.hook_out"]
    assert indexer.shape == (1, 8, 2)
    assert indexer.dtype == torch.long
    assert not any(key.startswith("blocks.2.attn.compressor.indexer") for key in cache)


def test_hash_and_topk_moe_hooks_fire(deepseek_v4_case: DeepseekV4Case) -> None:
    _, cache = deepseek_v4_case.bridge.run_with_cache(
        deepseek_v4_case.tokens,
        use_cache=False,
    )

    for layer in range(3):
        assert cache[f"blocks.{layer}.mlp.gate.hook_out"].shape == (8, 4)
        assert cache[f"blocks.{layer}.mlp.experts.hook_out"].shape == (8, 32)
        assert cache[f"blocks.{layer}.mlp.shared_experts.hook_out"].shape == (
            1,
            8,
            32,
        )


def test_collapsed_stream_hook_is_patchable(deepseek_v4_case: DeepseekV4Case) -> None:
    with torch.no_grad():
        baseline = deepseek_v4_case.bridge(deepseek_v4_case.tokens, use_cache=False)
        patched = deepseek_v4_case.bridge.run_with_hooks(
            deepseek_v4_case.tokens,
            use_cache=False,
            fwd_hooks=[
                (
                    "blocks.0.attn_hc.hook_out",
                    lambda activation, hook: torch.zeros_like(activation),
                )
            ],
        )

    assert not torch.allclose(baseline, patched)
