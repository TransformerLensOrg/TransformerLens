"""Integration tests for the ERNIE 4.5 architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "baidu/ERNIE-4.5-0.3B-PT"


@pytest.fixture(scope="module")
def ernie_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(ernie_bridge):
    return ernie_bridge.tokenizer("The capital of France is", return_tensors="pt").input_ids


class TestErnie4_5BridgeCreation:
    def test_adapter_selected(self, ernie_bridge):
        from transformer_lens.model_bridge.supported_architectures.ernie4_5 import (
            Ernie4_5ArchitectureAdapter,
        )

        assert isinstance(ernie_bridge.adapter, Ernie4_5ArchitectureAdapter)

    def test_config_gated_biases_delegate(self, ernie_bridge):
        hf_config = ernie_bridge.original_model.config
        q = ernie_bridge.blocks[0].attn.q.original_component
        assert (q.bias is not None) == bool(getattr(hf_config, "use_bias", False))


class TestErnie4_5ForwardEquivalence:
    def test_forward_matches_hf(self, ernie_bridge, sample_tokens):
        hf_model = ernie_bridge.original_model
        with torch.no_grad():
            bridge_out = ernie_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestErnie4_5Hooks:
    def test_hooks_fire(self, ernie_bridge, sample_tokens):
        d_model = ernie_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out"]
        with torch.no_grad():
            ernie_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"
