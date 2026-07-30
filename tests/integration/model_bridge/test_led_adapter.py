"""Integration tests for the LED architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "allenai/led-base-16384"


@pytest.fixture(scope="module")
def led_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_inputs(led_bridge):
    enc = led_bridge.tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
    dec_ids = torch.tensor([[2, 0]])
    return enc.input_ids, dec_ids


class TestLEDBridgeCreation:
    def test_adapter_selected(self, led_bridge):
        from transformer_lens.model_bridge.supported_architectures.led import (
            LEDArchitectureAdapter,
        )

        assert isinstance(led_bridge.adapter, LEDArchitectureAdapter)


class TestLEDForwardEquivalence:
    def test_forward_matches_fresh_hf(self, led_bridge, sample_inputs):
        from transformers import AutoModelForSeq2SeqLM

        input_ids, dec_ids = sample_inputs
        fresh = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.float32)
        fresh.eval()
        with torch.no_grad():
            bridge_out = led_bridge(input_ids, decoder_input_ids=dec_ids)
            hf_out = fresh(input_ids=input_ids, decoder_input_ids=dec_ids).logits
        if isinstance(bridge_out, tuple):
            bridge_out = bridge_out[0]
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestLEDHooks:
    def test_hooks_fire(self, led_bridge, sample_inputs):
        """LED pads the encoder to the attention window (1024), so encoder
        hooks carry the padded length — faithful to HF's internal flow."""
        input_ids, dec_ids = sample_inputs
        d_model = led_bridge.cfg.d_model
        window = led_bridge.original_model.config.attention_window[0]
        expected = {
            "encoder_blocks.0.attn.hook_out": (1, window, d_model),
            "decoder_blocks.0.self_attn.hook_out": (1, dec_ids.shape[1], d_model),
            "decoder_blocks.0.cross_attn.hook_out": (1, dec_ids.shape[1], d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            led_bridge.run_with_hooks(
                input_ids,
                decoder_input_ids=dec_ids,
                fwd_hooks=[(name, grab) for name in expected],
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestLEDGeneration:
    def test_generate(self, led_bridge):
        text = led_bridge.generate(
            "The quick brown fox jumps over the lazy dog.",
            max_new_tokens=8,
            do_sample=False,
            verbose=False,
        )
        assert isinstance(text, str)
        assert len(text.strip()) > 0
