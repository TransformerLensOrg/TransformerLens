"""Integration tests for the OpenAI GPT-1 architecture adapter."""

import copy

import pytest
import torch
from transformers import OpenAIGPTConfig, OpenAIGPTLMHeadModel

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_from_module

MODEL = "openai-community/openai-gpt"


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(bridge):
    return bridge.tokenizer("the capital of france is", return_tensors="pt").input_ids


class TestOpenAIGPTStopAtLayer:
    def test_stop_after_the_last_block(self, bridge, sample_tokens):
        """HF's block returns a list here; there is no ln_final, so unembed gives the logits."""
        with torch.no_grad():
            residual = bridge(sample_tokens, stop_at_layer=12)
            assert residual.shape[-1] == bridge.cfg.d_model
            assert torch.allclose(bridge.unembed(residual), bridge(sample_tokens))


class TestOpenAIGPTBridgeCreation:
    def test_adapter_and_structure(self, bridge):
        from transformer_lens.model_bridge.supported_architectures.openai_gpt import (
            OpenAIGPTArchitectureAdapter,
        )

        assert isinstance(bridge.adapter, OpenAIGPTArchitectureAdapter)
        assert len(bridge.blocks) == 12
        assert not hasattr(bridge, "ln_final")


class TestOpenAIGPTForwardEquivalence:
    def test_forward_matches_hf(self, bridge, sample_tokens):
        hf_model = bridge.original_model
        with torch.no_grad():
            bridge_out = bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestOpenAIGPTHooks:
    def test_hooks_fire(self, bridge, sample_tokens):
        d_model = bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = [
            "hook_embed",
            "blocks.0.attn.hook_out",
            "blocks.0.mlp.hook_out",
            "blocks.11.ln2.hook_out",
            "blocks.11.hook_out",
        ]
        with torch.no_grad():
            bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        assert captured["embed.hook_out"] == (1, seq, d_model)
        assert captured["blocks.0.attn.hook_out"] == (1, seq, d_model)
        assert captured["blocks.0.mlp.hook_out"] == (1, seq, d_model)
        assert captured["blocks.11.ln2.hook_out"] == (1, seq, d_model)
        assert captured["blocks.11.hook_out"] == (1, seq, d_model)

    def test_block_hook_out_is_cached(self, bridge, sample_tokens):
        """HF's block returns a list, which must still reach hook_out."""
        with torch.no_grad():
            logits, cache = bridge.run_with_cache(sample_tokens)
        assert "blocks.0.hook_out" in cache
        assert "blocks.0.hook_resid_post" in cache
        # There is no ln_final, so the last block's output is what unembed reads.
        assert torch.allclose(bridge.unembed(cache["blocks.11.hook_out"]), logits)

    def test_zeroing_the_last_block_hook_out_changes_the_logits(self, bridge, sample_tokens):
        def zero(tensor, *, hook):
            return torch.zeros_like(tensor)

        with torch.no_grad():
            clean = bridge(sample_tokens)
            edited = bridge.run_with_hooks(sample_tokens, fwd_hooks=[("blocks.11.hook_out", zero)])
        assert not torch.allclose(edited, clean)

    def test_edit_on_the_last_block_hook_out_reaches_a_stopped_run(self, bridge, sample_tokens):
        def zero(tensor, *, hook):
            return torch.zeros_like(tensor)

        with torch.no_grad():
            residual = bridge.run_with_hooks(
                sample_tokens, stop_at_layer=12, fwd_hooks=[("blocks.11.hook_out", zero)]
            )
        assert isinstance(residual, torch.Tensor)
        assert torch.count_nonzero(residual) == 0


class TestOpenAIGPTGeneration:
    def test_greedy_generation_is_coherent(self, bridge):
        out = bridge.generate(
            "the capital of france is", max_new_tokens=5, do_sample=False, verbose=False
        )
        assert isinstance(out, str)
        assert len(out) > len("the capital of france is")


@pytest.fixture(scope="module")
def tiny_models():
    """A tiny random bridge and an unwrapped copy: the bridge wraps the blocks it is given."""
    torch.manual_seed(0)
    config = OpenAIGPTConfig(vocab_size=64, n_embd=32, n_layer=2, n_head=4, n_positions=16)
    config.architectures = ["OpenAIGPTLMHeadModel"]
    hf_model = OpenAIGPTLMHeadModel(config).eval()
    reference = copy.deepcopy(hf_model)
    return build_bridge_from_module(hf_model, "OpenAIGPTLMHeadModel", hf_config=config), reference


class TestOpenAIGPTAttentionMask:
    """GPT-1 hands its blocks a [batch, 1, 1, seq] padding mask, with the causal
    mask left to the attention module."""

    PAD = 63

    def test_right_padded_batch_matches_hf(self, tiny_models):
        bridge, reference = tiny_models
        tokens = torch.tensor([[11, 12, 13, 14], [21, 22, self.PAD, self.PAD]])
        mask = (tokens != self.PAD).long()
        with torch.no_grad():
            logits = bridge(tokens, attention_mask=mask, prepend_bos=False)
            expected = reference(tokens, attention_mask=mask).logits
        assert torch.allclose(logits, expected, atol=1e-5)

    def test_left_padded_row_matches_the_row_alone(self, tiny_models):
        bridge, reference = tiny_models
        tokens = torch.tensor([[11, 12, 13, 14], [self.PAD, self.PAD, 21, 22]])
        with torch.no_grad():
            logits = bridge(tokens, attention_mask=(tokens != self.PAD).long(), prepend_bos=False)
            expected = reference(torch.tensor([[21, 22]])).logits
        assert torch.allclose(logits[1, 2:], expected[0], atol=1e-5)

    def test_mask_does_not_open_attention_to_later_tokens(self, tiny_models):
        bridge, _ = tiny_models
        tokens = torch.tensor([[11, 12, 13, 14]])
        with torch.no_grad():
            _, cache = bridge.run_with_cache(
                tokens, attention_mask=torch.ones_like(tokens), prepend_bos=False
            )
        pattern = cache["blocks.0.attn.hook_pattern"]
        assert torch.all(pattern.triu(diagonal=1) == 0)
