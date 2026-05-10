"""Tests for HookedVisualEncoder DeiT HF parity.

Covers both:
- facebook/deit-base-patch16-224
- facebook/deit-base-distilled-patch16-224
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    DeiTForImageClassificationWithTeacher,
)

from transformer_lens import HookedVisualEncoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


# ---------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------

MODEL_CONFIGS = [
    {
        "checkpoint": "facebook/deit-base-patch16-224",
        "hf_cls": ViTForImageClassification,
        "name": "deit-base",
    },
    {
        "checkpoint": "facebook/deit-base-distilled-patch16-224",
        "hf_cls": DeiTForImageClassificationWithTeacher,
        "name": "deit-base-distilled",
    },
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _get_output_tensor(out: Any) -> torch.Tensor:
    """Extract logits tensor from model output."""
    if isinstance(out, torch.Tensor):
        return out

    if isinstance(out, dict):
        if "logits" in out:
            return out["logits"]
        if "predictions" in out:
            return out["predictions"]

    if hasattr(out, "logits"):
        return out.logits

    if hasattr(out, "predictions"):
        return out.predictions

    raise TypeError(f"Unsupported output type: {type(out)!r}")


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


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
    model = model_cfg["hf_cls"].from_pretrained(
        model_cfg["checkpoint"]
    )

    return model.to(DEVICE).eval()


@pytest.fixture(scope="module")
def tl_model(model_cfg):
    model = HookedVisualEncoder.from_pretrained(
        model_cfg["checkpoint"],
        device=DEVICE,
    )

    return model.eval()


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


class TestDeiTHFComparison:
    def test_top1_prediction_matches_hf(
        self,
        model_cfg,
        hf_model,
        tl_model,
        inputs,
    ):
        """
        Checks whether the predicted ImageNet class index matches HF.
        """
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            hf_logits = _get_output_tensor(
                hf_model(pixel_values=pixel_values)
            )

            tl_logits = _get_output_tensor(
                tl_model(pixel_values)
            )

        assert hf_logits.ndim == 2
        assert tl_logits.ndim == 2

        assert hf_logits.shape == tl_logits.shape, (
            f"[{model_cfg['name']}] "
            f"Shape mismatch: "
            f"HF={tuple(hf_logits.shape)} "
            f"TL={tuple(tl_logits.shape)}"
        )

        hf_pred = hf_logits.argmax(dim=-1)
        tl_pred = tl_logits.argmax(dim=-1)

        assert torch.equal(hf_pred, tl_pred), (
            f"[{model_cfg['name']}] "
            f"Top-1 mismatch: "
            f"HF={hf_pred.item()} "
            f"TL={tl_pred.item()}"
        )

    def test_logits_are_close(
        self,
        model_cfg,
        hf_model,
        tl_model,
        inputs,
    ):
        """
        Checks cosine similarity between logits.
        """
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            hf_logits = _get_output_tensor(
                hf_model(pixel_values=pixel_values)
            )

            tl_logits = _get_output_tensor(
                tl_model(pixel_values)
            )

        hf_vec = hf_logits[0]
        tl_vec = tl_logits[0]

        cos = F.cosine_similarity(hf_vec, tl_vec, dim=0)

        assert cos.item() > 0.999, (
            f"[{model_cfg['name']}] "
            f"Cosine similarity too low: {cos.item()}"
        )

    def test_softmax_distribution_matches_reasonably(
        self,
        model_cfg,
        hf_model,
        tl_model,
        inputs,
    ):
        """
        Compares probability distributions using KL divergence.
        """
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            hf_logits = _get_output_tensor(
                hf_model(pixel_values=pixel_values)
            )

            tl_logits = _get_output_tensor(
                tl_model(pixel_values)
            )

            hf_prob = hf_logits.softmax(dim=-1)
            tl_prob = tl_logits.softmax(dim=-1)

        kl = F.kl_div(
            tl_prob.log(),
            hf_prob,
            reduction="batchmean",
        )

        assert kl.item() < 1e-3, (
            f"[{model_cfg['name']}] "
            f"KL divergence too large: {kl.item()}"
        )
