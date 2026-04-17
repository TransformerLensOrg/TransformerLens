"""Tests for HookedVisualEncoder basic functionality and HF ViT comparison."""

from __future__ import annotations

import io
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

from transformer_lens import HookedVisualEncoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_CHECKPOINT = "google/vit-large-patch16-224"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


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


@pytest.fixture(scope="module")
def hf_image():
    # Keep the test self-contained; you can replace with a local image fixture if preferred.
    import requests

    image = Image.open(requests.get(IMAGE_URL, stream=True, timeout=30).raw).convert("RGB")
    return image


@pytest.fixture(scope="module")
def hf_model():
    return ViTForImageClassification.from_pretrained(HF_CHECKPOINT).to(DEVICE).eval()


@pytest.fixture(scope="module")
def tl_model():
    return HookedVisualEncoder.from_pretrained(HF_CHECKPOINT, device=DEVICE).eval()


@pytest.fixture(scope="module")
def feature_extractor():
    return ViTFeatureExtractor.from_pretrained(HF_CHECKPOINT)


@pytest.fixture(scope="module")
def inputs(hf_image, feature_extractor):
    return feature_extractor(images=hf_image, return_tensors="pt")


class TestViTHFComparison:
    def test_top1_prediction_matches_hf(self, hf_model, tl_model, inputs):
        """
        Checks whether the predicted ImageNet class index matches HF's output.
        This is the most important end-to-end test for classification parity.
        """
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            hf_logits = _get_output_tensor(hf_model(pixel_values=pixel_values))
            tl_logits = _get_output_tensor(tl_model(pixel_values))

        assert hf_logits.ndim == 2, f"HF logits should be [B, num_labels], got {hf_logits.shape}"
        assert tl_logits.ndim == 2, f"TL logits should be [B, num_labels], got {tl_logits.shape}"
        assert hf_logits.shape == tl_logits.shape, (
            f"Shape mismatch: HF {tuple(hf_logits.shape)} vs TL {tuple(tl_logits.shape)}"
        )

        hf_pred = hf_logits.argmax(dim=-1)
        tl_pred = tl_logits.argmax(dim=-1)

        assert torch.equal(
            hf_pred, tl_pred
        ), f"Top-1 class mismatch: HF={hf_pred.item()} TL={tl_pred.item()}"

    def test_logits_are_close(self, hf_model, tl_model, inputs):
        """
        Measures numerical closeness of the full logits.
        Cosine similarity is usually more stable than exact allclose for end-to-end tests.
        """
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            hf_logits = _get_output_tensor(hf_model(pixel_values=pixel_values))
            tl_logits = _get_output_tensor(tl_model(pixel_values))

        # Remove batch dimension for a single-image test.
        hf_vec = hf_logits[0]
        tl_vec = tl_logits[0]

        cos = F.cosine_similarity(hf_vec, tl_vec, dim=0)
        assert cos.item() > 0.999, f"Logits cosine similarity too low: {cos.item()}"

    def test_softmax_distribution_matches_reasonably(self, hf_model, tl_model, inputs):
        """
        Optional: compares probability distributions.
        This is stricter than top-1 but more forgiving than exact equality.
        """
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            hf_logits = _get_output_tensor(hf_model(pixel_values=pixel_values))
            tl_logits = _get_output_tensor(tl_model(pixel_values))

            hf_prob = hf_logits.softmax(dim=-1)
            tl_prob = tl_logits.softmax(dim=-1)

        # KL(HF || TL); lower is better.
        kl = F.kl_div(tl_prob.log(), hf_prob, reduction="batchmean")
        assert kl.item() < 1e-3, f"KL divergence too large: {kl.item()}"
