"""Integration tests for the Seed-OSS architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "tiny-random/seed-oss"


@pytest.fixture(scope="module")
def seed_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(seed_bridge):
    torch.manual_seed(0)
    return torch.randint(0, seed_bridge.cfg.d_vocab - 10, (1, 8))


class TestSeedOssBridgeCreation:
    def test_adapter_selected(self, seed_bridge):
        from transformer_lens.model_bridge.supported_architectures.seed_oss import (
            SeedOssArchitectureAdapter,
        )

        assert isinstance(seed_bridge.adapter, SeedOssArchitectureAdapter)


class TestSeedOssForwardEquivalence:
    def test_forward_matches_hf(self, seed_bridge, sample_tokens):
        hf_model = seed_bridge.original_model
        with torch.no_grad():
            bridge_out = seed_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestSeedOssBiases:
    def test_config_gated_attention_bias_delegates(self, seed_bridge):
        """Seed-OSS gates biases via attention_bias/attention_out_bias; whatever
        the checkpoint declares must be reflected in the wrapped projections."""
        hf_config = seed_bridge.original_model.config
        q = seed_bridge.blocks[0].attn.q.original_component
        assert (q.bias is not None) == bool(getattr(hf_config, "attention_bias", False))


class TestSeedOssInferenceMode:
    def test_wrappers_adopt_eval_mode_and_forward_is_deterministic(
        self, seed_bridge, sample_tokens
    ):
        """This checkpoint ships dropout=0.1, which exposed bridge wrappers
        stuck in training mode: __setattr__ redirected self.training to the
        wrapped module, so eval() never reached the wrappers and dropout fired
        at inference."""
        assert seed_bridge.training is False
        assert seed_bridge.blocks[0].attn.training is False
        with torch.no_grad():
            first = seed_bridge(sample_tokens)
            second = seed_bridge(sample_tokens)
        assert torch.equal(first, second)


class TestSeedOssHooks:
    def test_hooks_fire(self, seed_bridge, sample_tokens):
        d_model = seed_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out"]
        with torch.no_grad():
            seed_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"
