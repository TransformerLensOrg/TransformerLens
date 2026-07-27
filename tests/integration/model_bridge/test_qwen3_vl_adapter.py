"""Integration tests for the Qwen3-VL architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "tiny-random/qwen3-vl"


def _image_inputs(processor):
    from PIL import Image

    img = Image.new("RGB", (64, 64), "red")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Describe"}],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return dict(processor(text=[text], images=[img], return_tensors="pt"))


@pytest.fixture(scope="module")
def qwen3vl_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(qwen3vl_bridge):
    torch.manual_seed(0)
    return torch.randint(0, qwen3vl_bridge.cfg.d_vocab - 10, (1, 12))


class TestQwen3VLBridgeCreation:
    def test_adapter_selected(self, qwen3vl_bridge):
        from transformer_lens.model_bridge.supported_architectures.qwen3_vl import (
            Qwen3VLArchitectureAdapter,
        )

        assert isinstance(qwen3vl_bridge.adapter, Qwen3VLArchitectureAdapter)


class TestQwen3VLForwardEquivalence:
    def test_text_forward_matches_fresh_hf(self, qwen3vl_bridge, sample_tokens):
        from transformers import AutoModelForImageTextToText

        fresh = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = qwen3vl_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"

    def test_multimodal_forward_matches_fresh_hf(self, qwen3vl_bridge):
        """DeepStack features flow through wrapped mergers into the text
        residual stream — the full pipeline must stay HF-exact."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        fresh = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        proc = AutoProcessor.from_pretrained(MODEL)
        inputs = _image_inputs(proc)
        # Drive the bridge's own forward (input_ids positional, pixel_values/attention_mask
        # as kwargs) so the multimodal path — not just the wrapped HF model — is exercised.
        bridge_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
        with torch.no_grad():
            bridge_out = qwen3vl_bridge(inputs["input_ids"], **bridge_inputs)
            hf_out = fresh(**inputs).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestQwen3VLHooks:
    def test_deepstack_merger_hooks_fire(self, qwen3vl_bridge):
        """Each DeepStack level's merger output is hookable at the source."""
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(MODEL)
        inputs = _image_inputs(proc)
        ids = inputs.pop("input_ids")
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = [
            "vision_encoder.deepstack_mergers.0.hook_out",
            "vision_encoder.deepstack_mergers.1.hook_out",
        ]
        with torch.no_grad():
            qwen3vl_bridge.run_with_hooks(ids, fwd_hooks=[(name, grab) for name in hooks], **inputs)
        for name in hooks:
            assert name in captured, f"{name} did not fire"

    def test_text_hooks_fire(self, qwen3vl_bridge, sample_tokens):
        d_model = qwen3vl_bridge.cfg.d_model
        seq = sample_tokens.shape[1]
        expected = {
            "blocks.0.attn.hook_out": (1, seq, d_model),
            "blocks.0.mlp.hook_out": (1, seq, d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            qwen3vl_bridge.run_with_hooks(
                sample_tokens, fwd_hooks=[(name, grab) for name in expected]
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestQwen3VLGeneration:
    def test_generate(self, qwen3vl_bridge):
        text = qwen3vl_bridge.generate("Hello", max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")
