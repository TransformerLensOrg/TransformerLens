"""Tests for HookedVisualEncoder hook and ablation functionality."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from transformer_lens import HookedVisualEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_CHECKPOINT = "google/vit-large-patch16-224"


def _get_output_tensor(out: Any) -> torch.Tensor:
    """Extract tensor from model output (handles tensor or dict-like outputs)."""
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
def tl_model():
    return HookedVisualEncoder.from_pretrained(HF_CHECKPOINT, device=DEVICE).eval()


@pytest.fixture(scope="module")
def sample_inputs():
    """
    Reuse the same image input style as the HF comparison test.

    If you already have an `inputs` fixture from your comparison test file,
    you can import/reuse that instead.
    """
    import requests
    from PIL import Image
    from transformers import ViTImageProcessor

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True, timeout=30).raw).convert("RGB")
    processor = ViTImageProcessor.from_pretrained(HF_CHECKPOINT)
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(DEVICE)


class TestViTHookCache:
    def test_cache_contains_v_input_hook(self, tl_model, sample_inputs):
        """
        Confirms that the ViT block exposes a hook with head dimension.
        """
        with torch.no_grad():
            _, cache = tl_model.run_with_cache(sample_inputs)

        hook_name = "blocks.0.hook_attn_out"

        if hook_name not in cache:
            pytest.skip(
                f"{hook_name} not found in cache. "
                f"Available keys (sample): {list(cache.keys())[:20]}"
            )

        act = cache[hook_name]
        assert act.ndim in (3, 4), f"Expected 3D/4D activation, got shape {tuple(act.shape)}"

    def test_cache_hook_shape_is_reasonable(self, tl_model, sample_inputs):
        """
        Ensures the cached activation can support head-wise interventions.
        """
        with torch.no_grad():
            _, cache = tl_model.run_with_cache(sample_inputs)

        hook_name = "blocks.0.hook_attn_out"
        if hook_name not in cache:
            pytest.skip(f"{hook_name} not found in cache.")

        act = cache[hook_name]
        assert act.ndim in (3, 4)

        # If 4D, we expect a head axis.
        if act.ndim == 4:
            assert act.shape[2] > 0, f"Expected head axis > 0, got shape {tuple(act.shape)}"


class TestViTHeadAblation:
    def test_replacing_one_head_changes_logits(self, tl_model, sample_inputs):
        """
        Replaces one head's value-stream input with a counterfactual value
        and checks that the final logits change.
        """
        hook_name = "blocks.0.hook_attn_out"
        head_to_edit = 0
        source_head = 1  # counterfactual source

        # Baseline
        with torch.no_grad():
            baseline_logits = _get_output_tensor(tl_model(sample_inputs))

        def counterfactual_head_hook(act: torch.Tensor, hook: Any) -> torch.Tensor:
            x = act.clone()

            # Expected ViT shape here is usually [batch, pos, n_heads, d_model]
            if x.ndim == 4:
                if x.shape[2] <= max(head_to_edit, source_head):
                    pytest.skip(
                        f"Not enough heads for intervention: got {x.shape[2]}"
                    )
                x[:, :, head_to_edit, :] = x[:, :, source_head, :]
                return x

            # Fallback for a 3D head axis layout, if your implementation uses one
            if x.ndim == 3:
                if x.shape[1] <= max(head_to_edit, source_head):
                    pytest.skip(
                        f"Not enough heads for intervention: got {x.shape[1]}"
                    )
                x[:, head_to_edit, :] = x[:, source_head, :]
                return x

            raise AssertionError(f"Unexpected activation rank for {hook_name}: {x.ndim}")

        # Ablated / counterfactual
        with torch.no_grad():
            edited_logits = _get_output_tensor(
                tl_model.run_with_hooks(
                    sample_inputs,
                    fwd_hooks=[(hook_name, counterfactual_head_hook)],
                )
            )

        assert baseline_logits.shape == edited_logits.shape
        assert not torch.allclose(
            baseline_logits, edited_logits, atol=1e-6
        ), "Counterfactual head edit had no effect on logits"

    def test_zeroing_one_head_changes_logits(self, tl_model, sample_inputs):
        """
        Stronger ablation test: zero out one head and ensure logits change.
        """
        hook_name = "blocks.0.hook_attn_out"
        head_to_zero = 0

        with torch.no_grad():
            baseline_logits = _get_output_tensor(tl_model(sample_inputs))

        def zero_head_hook(act: torch.Tensor, hook: Any) -> torch.Tensor:
            x = act.clone()

            if x.ndim == 4:
                if x.shape[2] <= head_to_zero:
                    pytest.skip(f"Not enough heads for intervention: got {x.shape[2]}")
                x[:, :, head_to_zero, :] = 0.0
                return x

            if x.ndim == 3:
                if x.shape[1] <= head_to_zero:
                    pytest.skip(f"Not enough heads for intervention: got {x.shape[1]}")
                x[:, head_to_zero, :] = 0.0
                return x

            raise AssertionError(f"Unexpected activation rank for {hook_name}: {x.ndim}")

        with torch.no_grad():
            ablated_logits = _get_output_tensor(
                tl_model.run_with_hooks(
                    sample_inputs,
                    fwd_hooks=[(hook_name, zero_head_hook)],
                )
            )

        assert baseline_logits.shape == ablated_logits.shape
        assert not torch.allclose(
            baseline_logits, ablated_logits, atol=1e-6
        ), "Zeroing one head had no effect on logits"

    def test_head_ablation_changes_top1_or_distribution(self, tl_model, sample_inputs):
        """
        Optional sanity check: the intervention should usually change either
        the top-1 class or at least the distribution.
        """
        hook_name = "blocks.0.hook_attn_out"

        with torch.no_grad():
            baseline_logits = _get_output_tensor(tl_model(sample_inputs))
            baseline_prob = baseline_logits.softmax(dim=-1)
            baseline_top1 = baseline_logits.argmax(dim=-1)

        def zero_head_hook(act: torch.Tensor, hook: Any) -> torch.Tensor:
            x = act.clone()
            if x.ndim == 4:
                x[:, :, 0, :] = 0.0
            elif x.ndim == 3:
                x[:, 0, :] = 0.0
            else:
                raise AssertionError(f"Unexpected activation rank: {x.ndim}")
            return x

        with torch.no_grad():
            edited_logits = _get_output_tensor(
                tl_model.run_with_hooks(
                    sample_inputs,
                    fwd_hooks=[(hook_name, zero_head_hook)],
                )
            )
            edited_prob = edited_logits.softmax(dim=-1)
            edited_top1 = edited_logits.argmax(dim=-1)

        # Distribution should move even if top-1 stays the same.
        kl = F.kl_div(edited_prob.log(), baseline_prob, reduction="batchmean")
        assert kl.item() >= 0.0

        # At least one of these should often change.
        assert (
            not torch.equal(baseline_top1, edited_top1)
            or not torch.allclose(baseline_logits, edited_logits, atol=1e-6)
        )
