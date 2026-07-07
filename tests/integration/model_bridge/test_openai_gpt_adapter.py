"""Integration tests for the OpenAI GPT-1 architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "openai-community/openai-gpt"


@pytest.fixture(scope="module")
def gpt1_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(gpt1_bridge):
    return gpt1_bridge.tokenizer("the capital of france is", return_tensors="pt").input_ids


class TestOpenAIGPTBridgeCreation:
    def test_adapter_and_structure(self, gpt1_bridge):
        from transformer_lens.model_bridge.supported_architectures.openai_gpt import (
            OpenAIGPTArchitectureAdapter,
        )

        assert isinstance(gpt1_bridge.adapter, OpenAIGPTArchitectureAdapter)
        assert len(gpt1_bridge.blocks) == 12
        assert not hasattr(gpt1_bridge, "ln_final")


class TestOpenAIGPTForwardEquivalence:
    def test_forward_matches_hf(self, gpt1_bridge, sample_tokens):
        hf_model = gpt1_bridge.original_model
        with torch.no_grad():
            bridge_out = gpt1_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestOpenAIGPTHooks:
    def test_hooks_fire(self, gpt1_bridge, sample_tokens):
        d_model = gpt1_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        # GPT-1's Block returns a list, so the block-level hook_out does not
        # fire; the post-MLP ln_2 output is the block's effective output.
        hooks = [
            "hook_embed",
            "blocks.0.attn.hook_out",
            "blocks.0.mlp.hook_out",
            "blocks.11.ln2.hook_out",
        ]
        with torch.no_grad():
            gpt1_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        assert captured["embed.hook_out"] == (1, seq, d_model)
        assert captured["blocks.0.attn.hook_out"] == (1, seq, d_model)
        assert captured["blocks.0.mlp.hook_out"] == (1, seq, d_model)
        assert captured["blocks.11.ln2.hook_out"] == (1, seq, d_model)


class TestOpenAIGPTGeneration:
    def test_greedy_generation_is_coherent(self, gpt1_bridge):
        out = gpt1_bridge.generate(
            "the capital of france is", max_new_tokens=5, do_sample=False, verbose=False
        )
        assert isinstance(out, str)
        assert len(out) > len("the capital of france is")
