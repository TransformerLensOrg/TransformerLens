"""Integration tests for the Florence-2 architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "florence-community/Florence-2-base-ft"


@pytest.fixture(scope="module")
def florence2_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_inputs(florence2_bridge):
    enc = florence2_bridge.tokenizer("What does the image describe?", return_tensors="pt")
    dec_ids = torch.tensor([[2, 0]])
    return enc.input_ids, dec_ids


class TestFlorence2BridgeCreation:
    def test_adapter_selected(self, florence2_bridge):
        from transformer_lens.model_bridge.supported_architectures.florence2 import (
            Florence2ArchitectureAdapter,
        )

        assert isinstance(florence2_bridge.adapter, Florence2ArchitectureAdapter)


class TestFlorence2ForwardEquivalence:
    def test_text_forward_matches_fresh_hf(self, florence2_bridge, sample_inputs):
        from transformers import AutoModelForImageTextToText

        input_ids, dec_ids = sample_inputs
        fresh = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32)
        fresh.eval()
        with torch.no_grad():
            bridge_out = florence2_bridge(input_ids, decoder_input_ids=dec_ids)
            hf_out = fresh(input_ids=input_ids, decoder_input_ids=dec_ids).logits
        if isinstance(bridge_out, tuple):
            bridge_out = bridge_out[0]
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs fresh HF max diff = {max_diff}"

    def test_multimodal_forward_matches_fresh_hf(self, florence2_bridge, sample_inputs):
        """Image features scatter into placeholder tokens through the wrapped
        vision tower; the whole pipeline must stay HF-faithful."""
        from transformers import AutoModelForImageTextToText

        input_ids, dec_ids = sample_inputs
        fresh = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32)
        fresh.eval()
        img_tok = fresh.config.image_token_id
        ids = torch.cat([torch.full((1, 577), img_tok, dtype=torch.long), input_ids], dim=1)
        torch.manual_seed(0)
        px = torch.randn(1, 3, 768, 768)
        with torch.no_grad():
            bridge_out = florence2_bridge(ids, pixel_values=px, decoder_input_ids=dec_ids)
            hf_out = fresh(input_ids=ids, pixel_values=px, decoder_input_ids=dec_ids).logits
        if isinstance(bridge_out, tuple):
            bridge_out = bridge_out[0]
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs fresh HF max diff = {max_diff}"


class TestFlorence2Hooks:
    def test_hooks_fire(self, florence2_bridge, sample_inputs):
        input_ids, dec_ids = sample_inputs
        d_model = florence2_bridge.cfg.d_model
        expected = {
            "encoder_blocks.0.attn.hook_out": (1, input_ids.shape[1], d_model),
            "encoder_blocks.0.mlp.out.hook_out": (1, input_ids.shape[1], d_model),
            "decoder_blocks.0.self_attn.hook_out": (1, dec_ids.shape[1], d_model),
            "decoder_blocks.0.cross_attn.hook_out": (1, dec_ids.shape[1], d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            florence2_bridge.run_with_hooks(
                input_ids,
                decoder_input_ids=dec_ids,
                fwd_hooks=[(name, grab) for name in expected],
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestFlorence2Generation:
    def test_generate(self, florence2_bridge):
        text = florence2_bridge.generate(
            "The tower is very", max_new_tokens=6, do_sample=False, verbose=False
        )
        assert isinstance(text, str)
        assert len(text.strip()) > 0

    def test_task_prompt_captioning(self, florence2_bridge):
        """Florence-2's real interface: <CAPTION> task prompt + image."""
        from PIL import Image, ImageDraw
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(MODEL)
        img = Image.new("RGB", (256, 256), "white")
        ImageDraw.Draw(img).rectangle([60, 60, 200, 200], fill="red")
        inputs = proc(text="<CAPTION>", images=img, return_tensors="pt")
        with torch.no_grad():
            out = florence2_bridge.original_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20,
                do_sample=False,
            )
        caption = proc.batch_decode(out, skip_special_tokens=True)[0]
        assert "red" in caption.lower(), caption
