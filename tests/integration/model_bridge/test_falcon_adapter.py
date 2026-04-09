"""Integration tests for Falcon architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "optimum-intel-internal-testing/really-tiny-falcon-testing"


@pytest.fixture(scope="module")
def falcon_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


class TestFalconBridgeCreation:
    def test_block_count(self, falcon_bridge):
        assert len(falcon_bridge.blocks) == 2

    def test_parallel_mode(self, falcon_bridge):
        assert falcon_bridge.cfg.parallel_attn_mlp is True

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
