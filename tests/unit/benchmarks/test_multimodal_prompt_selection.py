"""Image-token prompt selection for Phase 7 inputs (download-free).

Multimodal families disagree about which image token their processor expands,
and a rejected token raises rather than degrading: Gemma3 accepts only
``boi_token``, Gemma4 only ``image_token``, LLaVA has just ``image_token``.
``_prepare_test_inputs`` swallows those exceptions and returns None, which
Phase 7 reports as SKIPPED — so picking the wrong token silently drops the
whole multimodal suite. These fakes pin the fallback order per family.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from transformer_lens.benchmarks.multimodal import _prepare_test_inputs  # noqa: E402
from transformer_lens.model_bridge import TransformerBridge  # noqa: E402


class FakeProcessor:
    """Processor that expands exactly one image token and raises on the rest."""

    def __init__(self, accepted, image_token=None, boi_token=None):
        self.accepted = accepted
        if image_token is not None:
            self.image_token = image_token
        if boi_token is not None:
            self.boi_token = boi_token
        self.calls = []

    def __call__(self, text=None, images=None, return_tensors=None):
        self.calls.append(text)
        if self.accepted not in text:
            raise ValueError(f"Prompt contained 0 image tokens but received {images!r}")
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.zeros(1, 3, 8, 8),
        }


class FakeModel(torch.nn.Module):
    def __init__(self, image_token_id):
        super().__init__()
        self.config = SimpleNamespace(image_token_id=image_token_id)


def _bridge(processor, image_token_id=None):
    """A bare bridge — _prepare_test_inputs is beartype-checked against the real type."""
    bridge = TransformerBridge.__new__(TransformerBridge)
    torch.nn.Module.__init__(bridge)  # submodule assignment needs the Module state
    bridge.processor = processor
    bridge.cfg = SimpleNamespace(device="cpu")
    # original_model is a property reading self._driver.underlying_model.
    bridge._driver = SimpleNamespace(underlying_model=FakeModel(image_token_id))
    return bridge


class TestPromptSelection:
    def test_gemma3_falls_back_to_boi_token(self):
        # Gemma3's image_token is rejected by its own processor; trying it
        # first must not abort the whole preparation.
        proc = FakeProcessor(
            accepted="<start_of_image>",
            image_token="<image_soft_token>",
            boi_token="<start_of_image>",
        )
        input_ids, extra_kwargs, prompt = _prepare_test_inputs(_bridge(proc))
        assert input_ids is not None
        assert "<start_of_image>" in prompt
        assert "pixel_values" in extra_kwargs

    def test_gemma4_prefers_image_token(self):
        # Gemma4 is the mirror image: boi_token is just a marker.
        proc = FakeProcessor(accepted="<|image|>", image_token="<|image|>", boi_token="<|image>")
        input_ids, _, prompt = _prepare_test_inputs(_bridge(proc))
        assert input_ids is not None
        assert proc.calls[0].startswith("<|image|>")
        assert "<|image|>" in prompt

    def test_llava_single_token_still_works(self):
        proc = FakeProcessor(accepted="<image>", image_token="<image>")
        input_ids, _, prompt = _prepare_test_inputs(_bridge(proc))
        assert input_ids is not None
        assert "<image>" in prompt

    def test_returns_none_when_no_candidate_accepted(self):
        proc = FakeProcessor(accepted="<never-matches>", image_token="<image>")
        assert _prepare_test_inputs(_bridge(proc)) == (None, None, None)

    def test_auto_inserting_processor_uses_plain_prompt(self):
        # Processors that insert placeholders themselves are detected by the
        # probe; a manual placeholder would double-count the image.
        class AutoInserting(FakeProcessor):
            def __call__(self, text=None, images=None, return_tensors=None):
                self.calls.append(text)
                return {
                    "input_ids": torch.tensor([[1, 42, 3]]),
                    "pixel_values": torch.zeros(1, 3, 8, 8),
                }

        proc = AutoInserting(accepted="", image_token="<image>")
        _, _, prompt = _prepare_test_inputs(_bridge(proc, image_token_id=42))
        assert prompt == "Describe this image."

    def test_no_processor_returns_none(self):
        assert _prepare_test_inputs(_bridge(None)) == (None, None, None)
