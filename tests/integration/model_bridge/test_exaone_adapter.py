"""Integration tests for the EXAONE-3.x architecture adapter.

Loads the real 2.4B checkpoint with trust_remote_code (no maintained tiny
mirror exists — hyper-accel/tiny-random-exaone ships stale modeling code that
crashes on current transformers). CI-gated for download cost; run locally
with HF_TOKEN sourced.
"""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

# EXAONE 2.4B download/load; no working tiny mirror (hyper-accel/tiny-random-exaone
# ships stale remote code). Excluded from CI tiers via -m "not slow".
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(
        MODEL, device="cpu", dtype=torch.float32, trust_remote_code=True
    )


@pytest.fixture(scope="module")
def sample_tokens(bridge):
    return bridge.tokenizer("The capital of France is", return_tensors="pt").input_ids


class TestExaoneBridgeCreation:
    def test_adapter_selected(self, bridge):
        from transformer_lens.model_bridge.supported_architectures.exaone import (
            ExaoneArchitectureAdapter,
        )

        assert isinstance(bridge.adapter, ExaoneArchitectureAdapter)

    def test_gqa_config_propagated(self, bridge):
        hf_config = bridge.original_model.config
        assert bridge.cfg.n_key_value_heads == hf_config.num_key_value_heads


class TestExaoneForwardEquivalence:
    def test_forward_matches_hf(self, bridge, sample_tokens):
        hf_model = bridge.original_model
        with torch.no_grad():
            bridge_out = bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestExaoneHFDelegation:
    def test_nested_attention_is_shared_wrapper(self, bridge):
        """The dotted attn.attention path replaces the outer block wrapper; the
        bridge's original_component is the inner ExaoneAttention."""
        hf_model = bridge.original_model
        block = hf_model.transformer.h[0]
        assert bridge.blocks[0].attn is block.attn
        inner = bridge.blocks[0].attn.original_component
        assert type(inner).__name__ == "ExaoneAttention"
        assert bridge.blocks[0].attn.q is inner.q_proj
        assert bridge.blocks[0].mlp.submodules["gate"] is block.mlp.c_fc_0


class TestExaoneHooks:
    def test_hook_shapes(self, bridge, sample_tokens):
        d_model = bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out", "blocks.0.hook_out"]
        with torch.no_grad():
            bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"


class TestExaoneGeneration:
    def test_greedy_generation_is_coherent(self, bridge):
        out = bridge.generate(
            "The capital of France is", max_new_tokens=5, do_sample=False, verbose=False
        )
        assert "Paris" in out
