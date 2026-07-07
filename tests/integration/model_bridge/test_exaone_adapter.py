"""Integration tests for the EXAONE-3.x architecture adapter.

Loads the real 2.4B checkpoint with trust_remote_code (no maintained tiny
mirror exists — hyper-accel/tiny-random-exaone ships stale modeling code that
crashes on current transformers). CI-gated for download cost; run locally
with HF_TOKEN sourced.
"""

import os

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

pytestmark = pytest.mark.skipif(
    bool(os.getenv("CI")), reason="EXAONE 2.4B download too large for CI budget"
)


@pytest.fixture(scope="module")
def exaone_bridge():
    return TransformerBridge.boot_transformers(
        MODEL, device="cpu", dtype=torch.float32, trust_remote_code=True
    )


@pytest.fixture(scope="module")
def sample_tokens(exaone_bridge):
    return exaone_bridge.tokenizer("The capital of France is", return_tensors="pt").input_ids


class TestExaoneBridgeCreation:
    def test_adapter_selected(self, exaone_bridge):
        from transformer_lens.model_bridge.supported_architectures.exaone import (
            ExaoneArchitectureAdapter,
        )

        assert isinstance(exaone_bridge.adapter, ExaoneArchitectureAdapter)

    def test_gqa_config_propagated(self, exaone_bridge):
        hf_config = exaone_bridge.original_model.config
        assert exaone_bridge.cfg.n_key_value_heads == hf_config.num_key_value_heads


class TestExaoneForwardEquivalence:
    def test_forward_matches_hf(self, exaone_bridge, sample_tokens):
        hf_model = exaone_bridge.original_model
        with torch.no_grad():
            bridge_out = exaone_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestExaoneHFDelegation:
    def test_nested_attention_is_shared_wrapper(self, exaone_bridge):
        """The dotted attn.attention path replaces the outer block wrapper; the
        bridge's original_component is the inner ExaoneAttention."""
        hf_model = exaone_bridge.original_model
        block = hf_model.transformer.h[0]
        assert exaone_bridge.blocks[0].attn is block.attn
        inner = exaone_bridge.blocks[0].attn.original_component
        assert type(inner).__name__ == "ExaoneAttention"
        assert exaone_bridge.blocks[0].attn.q is inner.q_proj
        assert exaone_bridge.blocks[0].mlp.submodules["gate"] is block.mlp.c_fc_0


class TestExaoneHooks:
    def test_hook_shapes(self, exaone_bridge, sample_tokens):
        d_model = exaone_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out", "blocks.0.hook_out"]
        with torch.no_grad():
            exaone_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"


class TestExaoneGeneration:
    def test_greedy_generation_is_coherent(self, exaone_bridge):
        out = exaone_bridge.generate(
            "The capital of France is", max_new_tokens=5, do_sample=False, verbose=False
        )
        assert "Paris" in out
