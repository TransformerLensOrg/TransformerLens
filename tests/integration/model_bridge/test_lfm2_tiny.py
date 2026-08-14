"""Tiny from_config Lfm2 integration tests — CI coverage without a real checkpoint.

The full LiquidAI/LFM2.5-230M parity suite (test_lfm2_adapter.py) is
``@pytest.mark.slow`` because it takes excessive time in CI for model downloads. This file
builds a tiny synthetic Lfm2 (real random CPU weights, no Hub download) that
exercises the same surfaces — wiring, bridge vs HF forward checks and hook coverage.
"""
import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.lfm2 import Lfm2Config

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    DepthwiseConv1DBridge,
    Lfm2ShortConvBridge,
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures.lfm2 import (
    Lfm2ArchitectureAdapter,
)

LAYERS = ["conv", "full_attention"]
CONV_LAYER = [0]
ATTN_LAYER = [1]


class _Tok:
    pass


@pytest.fixture(scope="module")
def bridge():
    torch.manual_seed(0)
    cfg = Lfm2Config(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        layer_types=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
    )
    cfg.architectures = ["Lfm2ForCausalLM"]
    hf = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    bridge_cfg = build_bridge_config_from_hf(
        hf.config, "Lfm2ForCausalLM", "lfm2-tiny", torch.float32
    )
    return TransformerBridge(hf, Lfm2ArchitectureAdapter(bridge_cfg), tokenizer=_Tok())


@pytest.fixture(scope="module")
def tokens():
    return torch.tensor([[1, 2, 3, 4, 5]])


@pytest.fixture(scope="module")
def cache(bridge, tokens):
    with torch.no_grad():
        _, c = bridge.run_with_cache(tokens)
    return c


class TestLfm2TinyStructure:
    def test_block_count(self, bridge):
        assert len(bridge.blocks) == len(LAYERS)

    def test_blocks_are_block_bridge(self, bridge):
        assert isinstance(bridge.blocks[0], BlockBridge)

    def test_conv_block_is_lfm2_short_conv_bridge(self, bridge):
        assert isinstance(bridge.blocks[0].conv, Lfm2ShortConvBridge)

    def test_conv_conv_block_is_depthwise_conv1d_bridge(self, bridge):
        assert isinstance(bridge.blocks[0].conv.conv, DepthwiseConv1DBridge)

    def test_attention_block_is_position_embeddings_attention_bridge(self, bridge):
        assert isinstance(bridge.blocks[1].attn, PositionEmbeddingsAttentionBridge)

    def test_layer_types_populated(self, bridge):
        assert getattr(bridge.cfg, "layer_types", None) == LAYERS


class TestLfm2TinyForwardPass:
    def test_forward_matches_hf_exactly(self, bridge, tokens):
        with torch.no_grad():
            bridge_out = bridge(tokens)
            hf_out = bridge.original_model(tokens).logits
        assert (bridge_out.float() - hf_out.float()).abs().max().item() == 0.0

    def test_no_nan_longer_sequence(self, bridge):
        with torch.no_grad():
            out = bridge(torch.arange(1, 17).unsqueeze(0))
        assert not torch.isnan(out).any()


class TestLfm2TinyHookCoverage:
    def test_block_hooks_fire(self, cache):
        for i in range(len(LAYERS)):
            assert f"blocks.{i}.hook_in" in cache
            assert f"blocks.{i}.hook_out" in cache

    def test_conv_submodule_hooks_fire(self, cache):
        for i in CONV_LAYER:
            for submod in ("in", "conv", "out"):
                assert f"blocks.{i}.conv.{submod}.hook_out" in cache

    def test_attn_submodule_hooks_fire(self, cache):
        for i in ATTN_LAYER:
            for submod in ("q", "k", "v", "o", "q_norm", "k_norm"):
                assert f"blocks.{i}.attn.{submod}.hook_out" in cache
