"""Integration tests for the Qwen2.5-VL architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "optimum-intel-internal-testing/tiny-random-qwen2.5-vl"


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


SAMPLE_TOKENS_LEN = 12


class TestQwen2_5_VLBridgeCreation:
    def test_adapter_selected(self, bridge):
        from transformer_lens.model_bridge.supported_architectures.qwen2_5_vl import (
            Qwen2_5_VLArchitectureAdapter,
        )

        assert isinstance(bridge.adapter, Qwen2_5_VLArchitectureAdapter)


class TestQwen2_5_VLForwardEquivalence:
    def test_text_forward_matches_fresh_hf(self, bridge, sample_tokens):
        from transformers import AutoModelForImageTextToText

        fresh = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"

    def test_multimodal_forward_matches_fresh_hf(self, bridge):
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
        # Drive the bridge's own forward (input_ids positional, the rest — pixel_values,
        # attention_mask, image_grid_thw — as kwargs) so the vision path is exercised.
        bridge_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
        with torch.no_grad():
            bridge_out = bridge(inputs["input_ids"], **bridge_inputs)
            hf_out = fresh(**inputs).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"

    def test_left_padded_multimodal_forward_matches_fresh_hf(self, bridge):
        """The bridge derives position_ids from attention_mask for left-padded
        input (#1609), but mRoPE models build their own 3-D index in
        get_rope_index and only while position_ids is None — a supplied 2-D
        tensor is silently broadcast across all three streams instead. Their
        derivation already scatters positions onto attended slots only, so left
        padding must be left entirely to HF here.
        """
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

        n_pad = 4
        ids = inputs["input_ids"]
        padded = {k: v for k, v in inputs.items() if k != "input_ids"}
        padded["attention_mask"] = torch.cat(
            [torch.zeros(1, n_pad, dtype=torch.long), torch.ones_like(ids)], dim=1
        )
        # Per-token tensors have to grow with the padding or HF's own rope index
        # rejects the mask outright.
        token_type_ids = inputs["mm_token_type_ids"]
        padded["mm_token_type_ids"] = torch.cat(
            [torch.zeros(1, n_pad, dtype=token_type_ids.dtype), token_type_ids], dim=1
        )
        padded_ids = torch.cat([torch.zeros(1, n_pad, dtype=ids.dtype), ids], dim=1)

        with torch.no_grad():
            bridge_out = bridge(padded_ids, **padded)
            hf_out = fresh(input_ids=padded_ids, **padded).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestQwen2_5_VLHooks:
    def test_hooks_fire(self, bridge, sample_tokens):
        d_model = bridge.cfg.d_model
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
            bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in expected])
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestQwen2_5_VLGeneration:
    def test_generate(self, bridge):
        text = bridge.generate("Hello", max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")
