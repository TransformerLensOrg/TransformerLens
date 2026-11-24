"""Integration tests for Falcon architecture adapter."""

import copy

import pytest
import torch
from transformers import FalconConfig, FalconForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_from_module

MODEL = "optimum-intel-internal-testing/really-tiny-falcon-testing"


@pytest.fixture(scope="module")
def falcon_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


class TestFalconBridgeCreation:
    def test_block_count(self, falcon_bridge):
        assert len(falcon_bridge.blocks) == 2

    def test_has_core_components(self, falcon_bridge):
        assert hasattr(falcon_bridge, "embed")
        assert hasattr(falcon_bridge, "unembed")
        assert hasattr(falcon_bridge, "ln_final")


class TestFalconForwardPass:
    def test_forward_returns_logits(self, falcon_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = falcon_bridge(tokens)
        assert output.shape[0] == 1
        assert output.shape[1] == 4
        assert not torch.isnan(output).any()

    def test_forward_matches_hf(self, falcon_bridge):
        """Bridge delegates to HF native forward — output should be identical."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        hf_model = falcon_bridge.original_model
        with torch.no_grad():
            bridge_out = falcon_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"


class TestFalconParallelHooks:
    def test_no_hook_resid_mid(self, falcon_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = falcon_bridge.run_with_cache(tokens)
        assert not any("hook_resid_mid" in k for k in cache.keys())

    def test_attn_and_mlp_hooks_fire(self, falcon_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = falcon_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.attn.hook_in" in cache
            assert f"blocks.{i}.attn.hook_out" in cache
            assert f"blocks.{i}.mlp.hook_in" in cache
            assert f"blocks.{i}.mlp.hook_out" in cache

    def test_residual_hooks_fire(self, falcon_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = falcon_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.hook_resid_pre" in cache
            assert f"blocks.{i}.hook_resid_post" in cache


class TestFalconNewDecoderArchitecture:
    """Falcon-40B/180B/11B layout: fused QKV rows grouped per KV head, K/V head count
    from num_kv_heads (multi_query is ignored), and ln_attn/ln_mlp unless
    num_ln_in_parallel_attn == 1."""

    @pytest.mark.parametrize("num_ln_in_parallel_attn", [None, 1])
    def test_matches_hf_before_and_after_processing(self, num_ln_in_parallel_attn):
        torch.manual_seed(0)
        # multi_query keeps its default (True), as in the Falcon-40B config.
        config = FalconConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_kv_heads=2,
            num_hidden_layers=2,
            vocab_size=64,
            new_decoder_architecture=True,
            num_ln_in_parallel_attn=num_ln_in_parallel_attn,
            attn_implementation="eager",
        )
        # HF's decoder layer rewrites an unset num_ln_in_parallel_attn on the config it is
        # given; the bridge reads the config as loaded, before the model exists.
        bridge_config = copy.deepcopy(config)
        hf_model = FalconForCausalLM(config).eval()
        with torch.no_grad():
            for param in hf_model.parameters():
                param.add_(0.3 * torch.randn_like(param))
        tokens = torch.randint(0, 64, (2, 6))
        with torch.no_grad():
            expected = hf_model(tokens).logits
        bridge = build_bridge_from_module(
            copy.deepcopy(hf_model), "FalconForCausalLM", hf_config=bridge_config, device="cpu"
        )
        with torch.no_grad():
            max_diff = (bridge(tokens) - expected).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"
        bridge.enable_compatibility_mode()
        with torch.no_grad():
            actual = bridge(tokens).log_softmax(-1)
        max_diff = (actual - expected.log_softmax(-1)).abs().max().item()
        assert max_diff < 1e-4, f"Compatibility mode vs HF log-prob max diff = {max_diff}"
