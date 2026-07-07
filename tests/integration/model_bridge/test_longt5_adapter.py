"""Integration tests for the LongT5 architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "google/long-t5-tglobal-base"


@pytest.fixture(scope="module")
def longt5_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_inputs(longt5_bridge):
    enc = longt5_bridge.tokenizer(
        "summarize: The quick brown fox jumps over the lazy dog.", return_tensors="pt"
    )
    dec_ids = torch.tensor([[0, 37, 1782]])
    return enc.input_ids, dec_ids


class TestLongT5BridgeCreation:
    def test_adapter_selected(self, longt5_bridge):
        from transformer_lens.model_bridge.supported_architectures.longt5 import (
            LongT5ArchitectureAdapter,
        )

        assert isinstance(longt5_bridge.adapter, LongT5ArchitectureAdapter)


class TestLongT5ForwardEquivalence:
    def test_forward_matches_fresh_hf(self, longt5_bridge, sample_inputs):
        from transformers import AutoModelForSeq2SeqLM

        input_ids, dec_ids = sample_inputs
        fresh = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.float32)
        fresh.eval()
        with torch.no_grad():
            bridge_out = longt5_bridge(input_ids, decoder_input_ids=dec_ids)
            hf_out = fresh(input_ids=input_ids, decoder_input_ids=dec_ids).logits
        if isinstance(bridge_out, tuple):
            bridge_out = bridge_out[0]
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestLongT5Hooks:
    def test_hooks_fire(self, longt5_bridge, sample_inputs):
        input_ids, dec_ids = sample_inputs
        d_model = longt5_bridge.cfg.d_model
        expected = {
            "encoder_blocks.0.attn.hook_out": (1, input_ids.shape[1], d_model),
            "encoder_blocks.0.mlp.hook_out": (1, input_ids.shape[1], d_model),
            "decoder_blocks.0.self_attn.hook_out": (1, dec_ids.shape[1], d_model),
            "decoder_blocks.0.cross_attn.hook_out": (1, dec_ids.shape[1], d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            longt5_bridge.run_with_hooks(
                input_ids,
                decoder_input_ids=dec_ids,
                fwd_hooks=[(name, grab) for name in expected],
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestLongT5Generation:
    def test_generate(self, longt5_bridge):
        text = longt5_bridge.generate(
            "summarize: The quick brown fox jumps over the lazy dog and runs away.",
            max_new_tokens=8,
            do_sample=False,
            verbose=False,
        )
        assert isinstance(text, str)
        assert len(text.strip()) > 0
