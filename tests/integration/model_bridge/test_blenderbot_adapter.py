"""Integration tests for the Blenderbot architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "facebook/blenderbot-400M-distill"


@pytest.fixture(scope="module")
def bb_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_inputs(bb_bridge):
    tokens = bb_bridge.tokenizer("Hello, how are you?", return_tensors="pt")
    start = bb_bridge.original_model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[start, 5, 6]])
    return tokens.input_ids, tokens.attention_mask, decoder_input_ids


class TestBlenderbotBridgeCreation:
    def test_asymmetric_stacks_wire_correctly(self, bb_bridge):
        from transformer_lens.model_bridge.supported_architectures.blenderbot import (
            BlenderbotArchitectureAdapter,
        )

        assert isinstance(bb_bridge.adapter, BlenderbotArchitectureAdapter)
        assert len(bb_bridge.encoder_blocks) == 2
        assert len(bb_bridge.decoder_blocks) == 12
        assert bb_bridge.cfg.n_layers == 12


class TestBlenderbotForwardEquivalence:
    def test_forward_matches_hf(self, bb_bridge, sample_inputs):
        input_ids, attention_mask, decoder_input_ids = sample_inputs
        hf_model = bb_bridge.original_model
        with torch.no_grad():
            bridge_out = bb_bridge(
                input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids
            )
            hf_out = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"


class TestBlenderbotHooks:
    def test_hooks_fire_on_both_stacks(self, bb_bridge, sample_inputs):
        input_ids, attention_mask, decoder_input_ids = sample_inputs
        d_model = bb_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = [
            "encoder_blocks.1.hook_out",
            "encoder_ln_final.hook_out",
            "decoder_blocks.11.hook_out",
            "decoder_blocks.0.cross_attn.hook_out",
        ]
        with torch.no_grad():
            bb_bridge.run_with_hooks(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                fwd_hooks=[(name, grab) for name in hooks],
            )
        enc_len = input_ids.shape[1]
        dec_len = decoder_input_ids.shape[1]
        assert captured["encoder_blocks.1.hook_out"] == (1, enc_len, d_model)
        assert captured["encoder_ln_final.hook_out"] == (1, enc_len, d_model)
        assert captured["decoder_blocks.11.hook_out"] == (1, dec_len, d_model)
        assert captured["decoder_blocks.0.cross_attn.hook_out"] == (1, dec_len, d_model)


class TestBlenderbotGeneration:
    def test_greedy_dialogue_is_coherent(self, bb_bridge):
        out = bb_bridge.generate(
            "Hello, how are you?", max_new_tokens=8, do_sample=False, verbose=False
        )
        assert isinstance(out, str)
        assert len(out.strip()) > 0
