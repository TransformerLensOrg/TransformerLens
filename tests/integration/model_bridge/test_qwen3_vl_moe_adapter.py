"""Integration tests for the Qwen3-VL-MoE architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "tiny-random/qwen3-vl-moe"


@pytest.fixture(scope="module")
def qwen3vlmoe_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(qwen3vlmoe_bridge):
    torch.manual_seed(0)
    return torch.randint(0, qwen3vlmoe_bridge.cfg.d_vocab - 10, (1, 12))


class TestQwen3VLMoeBridgeCreation:
    def test_adapter_selected(self, qwen3vlmoe_bridge):
        from transformer_lens.model_bridge.supported_architectures.qwen3_vl_moe import (
            Qwen3VLMoeArchitectureAdapter,
        )

        assert isinstance(qwen3vlmoe_bridge.adapter, Qwen3VLMoeArchitectureAdapter)


class TestQwen3VLMoeForwardEquivalence:
    def test_text_forward_matches_fresh_hf(self, qwen3vlmoe_bridge, sample_tokens):
        from transformers import AutoModelForImageTextToText

        fresh = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = qwen3vlmoe_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"

    def test_multimodal_forward_matches_fresh_hf(self, qwen3vlmoe_bridge):
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        fresh = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        proc = AutoProcessor.from_pretrained(MODEL)
        img = Image.new("RGB", (64, 64), "red")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "Describe"}],
            }
        ]
        text = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = dict(proc(text=[text], images=[img], return_tensors="pt"))
        with torch.no_grad():
            bridge_out = qwen3vlmoe_bridge.original_model(**inputs).logits
            hf_out = fresh(**inputs).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestQwen3VLMoeHooks:
    def test_hooks_fire(self, qwen3vlmoe_bridge, sample_tokens):
        d_model = qwen3vlmoe_bridge.cfg.d_model
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
            qwen3vlmoe_bridge.run_with_hooks(
                sample_tokens, fwd_hooks=[(name, grab) for name in expected]
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestQwen3VLMoeGeneration:
    def test_generate(self, qwen3vlmoe_bridge):
        text = qwen3vlmoe_bridge.generate("Hello", max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")
