"""Integration tests for the Qwen2.5-VL architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "optimum-intel-internal-testing/tiny-random-qwen2.5-vl"


@pytest.fixture(scope="module")
def qwen25vl_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(qwen25vl_bridge):
    torch.manual_seed(0)
    return torch.randint(0, qwen25vl_bridge.cfg.d_vocab - 10, (1, 12))


class TestQwen2_5_VLBridgeCreation:
    def test_adapter_selected(self, qwen25vl_bridge):
        from transformer_lens.model_bridge.supported_architectures.qwen2_5_vl import (
            Qwen2_5_VLArchitectureAdapter,
        )

        assert isinstance(qwen25vl_bridge.adapter, Qwen2_5_VLArchitectureAdapter)


class TestQwen2_5_VLForwardEquivalence:
    def test_text_forward_matches_fresh_hf(self, qwen25vl_bridge, sample_tokens):
        from transformers import AutoModelForImageTextToText

        fresh = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = qwen25vl_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"

    def test_multimodal_forward_matches_fresh_hf(self, qwen25vl_bridge):
        """Image patches flow through the wrapped windowed tower + merger and
        mRoPE position streams diverge from text-only — must stay HF-exact."""
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        fresh = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        proc = AutoProcessor.from_pretrained(MODEL)
        img = Image.new("RGB", (56, 56), "red")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "Describe"}],
            }
        ]
        text = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt")
        with torch.no_grad():
            bridge_out = qwen25vl_bridge.original_model(**inputs).logits
            hf_out = fresh(**inputs).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestQwen2_5_VLHooks:
    def test_hooks_fire(self, qwen25vl_bridge, sample_tokens):
        d_model = qwen25vl_bridge.cfg.d_model
        seq = sample_tokens.shape[1]
        expected = {
            "blocks.0.attn.hook_out": (1, seq, d_model),
            "blocks.0.mlp.hook_out": (1, seq, d_model),
            "blocks.1.mlp.hook_out": (1, seq, d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            qwen25vl_bridge.run_with_hooks(
                sample_tokens, fwd_hooks=[(name, grab) for name in expected]
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestQwen2_5_VLGeneration:
    def test_generate(self, qwen25vl_bridge):
        text = qwen25vl_bridge.generate("Hello", max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")
