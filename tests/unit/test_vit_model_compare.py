"""
Tests for HookedVisualEncoder ViT HF parity.

Covers:
- google/vit-base-patch16-224-in21k
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, ViTModel

from transformer_lens import HookedVisualEncoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


MODEL_CONFIGS = [
    {
        "checkpoint": "google/vit-base-patch16-224-in21k",
        "hf_cls": ViTModel,
        "name": "vit-base-in21k",
    },
]


def _get_hf_sequence_output(out: Any) -> torch.Tensor:
    """
    Extract HF ViTModel last_hidden_state.
    """
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, dict) and "last_hidden_state" in out:
        return out["last_hidden_state"]
    raise TypeError(f"Unsupported HF output type: {type(out)!r}")


def _get_tl_sequence_output(out: Any) -> torch.Tensor:
    """
    TL ViT forward returns a tensor directly in your implementation.
    """
    if isinstance(out, torch.Tensor):
        return out
    raise TypeError(f"Unsupported TL output type: {type(out)!r}")


@pytest.fixture(scope="module")
def hf_image() -> Image.Image:
    import requests

    image = Image.open(
        requests.get(IMAGE_URL, stream=True, timeout=30).raw
    ).convert("RGB")
    return image


@pytest.fixture(params=MODEL_CONFIGS, scope="module")
def model_cfg(request):
    return request.param


@pytest.fixture(scope="module")
def image_processor(model_cfg):
    return AutoImageProcessor.from_pretrained(model_cfg["checkpoint"])


@pytest.fixture(scope="module")
def inputs(hf_image, image_processor):
    return image_processor(images=hf_image, return_tensors="pt")


@pytest.fixture(scope="module")
def hf_model(model_cfg):
    model = model_cfg["hf_cls"].from_pretrained(model_cfg["checkpoint"])
    return model.to(DEVICE).eval()


@pytest.fixture(scope="module")
def tl_model(model_cfg):
    model = HookedVisualEncoder.from_pretrained(
        model_cfg["checkpoint"],
        device=DEVICE,
    )
    return model.eval()


class TestViTHFComparison:
    def test_last_hidden_state_matches_hf(
        self,
        model_cfg,
        hf_model,
        tl_model,
        inputs,
    ):
        """
        Compares TL output tensor against HF ViTModel.last_hidden_state.
        """
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            hf_out = hf_model(pixel_values=pixel_values)
            tl_out = tl_model(pixel_values)

            hf_seq = _get_hf_sequence_output(hf_out)
            tl_seq = _get_tl_sequence_output(tl_out)

        assert hf_seq.ndim == 3
        assert tl_seq.ndim == 3

        assert hf_seq.shape == tl_seq.shape, (
            f"[{model_cfg['name']}] shape mismatch: "
            f"HF={tuple(hf_seq.shape)} TL={tuple(tl_seq.shape)}"
        )

        torch.testing.assert_close(
            tl_seq,
            hf_seq,
            rtol=1e-3,
            atol=1e-4,
            msg=f"[{model_cfg['name']}] last_hidden_state mismatch",
        )

    def test_cls_token_matches_hf(
        self,
        model_cfg,
        hf_model,
        tl_model,
        inputs,
    ):
        """
        Compares only the CLS token embedding.
        """
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            hf_out = hf_model(pixel_values=pixel_values)
            tl_out = tl_model(pixel_values)

            hf_seq = _get_hf_sequence_output(hf_out)
            tl_seq = _get_tl_sequence_output(tl_out)

        hf_cls = hf_seq[:, 0, :]
        tl_cls = tl_seq[:, 0, :]

        assert hf_cls.shape == tl_cls.shape, (
            f"[{model_cfg['name']}] CLS shape mismatch: "
            f"HF={tuple(hf_cls.shape)} TL={tuple(tl_cls.shape)}"
        )

        torch.testing.assert_close(
            tl_cls,
            hf_cls,
            rtol=1e-3,
            atol=1e-4,
            msg=f"[{model_cfg['name']}] CLS token mismatch",
        )

    def test_global_cosine_similarity_is_high(
        self,
        model_cfg,
        hf_model,
        tl_model,
        inputs,
    ):
        """
        Useful as a softer sanity check if exact parity is a bit noisy.
        """
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            hf_out = hf_model(pixel_values=pixel_values)
            tl_out = tl_model(pixel_values)

            hf_seq = _get_hf_sequence_output(hf_out)
            tl_seq = _get_tl_sequence_output(tl_out)

        cos = F.cosine_similarity(
            hf_seq.flatten(),
            tl_seq.flatten(),
            dim=0,
        )

        assert cos.item() > 0.999, (
            f"[{model_cfg['name']}] cosine similarity too low: {cos.item()}"
        )
