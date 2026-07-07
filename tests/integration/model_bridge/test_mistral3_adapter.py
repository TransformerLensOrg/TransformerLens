"""Integration tests for the Mistral 3 (Mistral-Small VLM) architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "tiny-random/mistral-3"


@pytest.fixture(scope="module")
def mistral3_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(mistral3_bridge):
    torch.manual_seed(0)
    return torch.randint(0, mistral3_bridge.cfg.d_vocab - 10, (1, 12))


class TestMistral3BridgeCreation:
    def test_adapter_selected(self, mistral3_bridge):
        from transformer_lens.model_bridge.supported_architectures.mistral3 import (
            Mistral3ArchitectureAdapter,
        )

        assert isinstance(mistral3_bridge.adapter, Mistral3ArchitectureAdapter)


class TestMistral3ForwardEquivalence:
    def test_text_forward_matches_fresh_hf(self, mistral3_bridge, sample_tokens):
        from transformers import AutoModelForImageTextToText

        fresh = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = mistral3_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"

    def test_multimodal_forward_matches_fresh_hf(self, mistral3_bridge):
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        fresh = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        proc = AutoProcessor.from_pretrained(MODEL)
        img = Image.new("RGB", (64, 64), "red")
        image_token = getattr(proc, "image_token", "[IMG]")
        inputs = proc(text=f"{image_token}Describe", images=img, return_tensors="pt")
        with torch.no_grad():
            bridge_out = mistral3_bridge.original_model(**inputs).logits
            hf_out = fresh(**inputs).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestMistral3Hooks:
    def test_hooks_fire(self, mistral3_bridge, sample_tokens):
        d_model = mistral3_bridge.cfg.d_model
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
            mistral3_bridge.run_with_hooks(
                sample_tokens, fwd_hooks=[(name, grab) for name in expected]
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestMistral3Generation:
    def test_generate(self, mistral3_bridge):
        text = mistral3_bridge.generate("Hello", max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")
