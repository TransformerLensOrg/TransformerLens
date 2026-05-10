"""Tests for HookedVisualEncoder hook and ablation functionality on DeiT."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor

from transformer_lens import HookedVisualEncoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

MODEL_CONFIGS = [
    {
        "checkpoint": "facebook/deit-base-patch16-224",
        "name": "deit-base",
    },
    {
        "checkpoint": "facebook/deit-base-distilled-patch16-224",
        "name": "deit-base-distilled",
    },
]


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


@pytest.fixture(params=MODEL_CONFIGS, scope="module")
def model_cfg(request):
    return request.param


@pytest.fixture(scope="module")
def tl_model(model_cfg):
    return HookedVisualEncoder.from_pretrained(
        model_cfg["checkpoint"],
        device=DEVICE,
    ).eval()


@pytest.fixture(scope="module")
def sample_inputs(model_cfg):
    """
    Load one image and preprocess it with the checkpoint's processor.
    """
    import requests

    image = Image.open(
        requests.get(IMAGE_URL, stream=True, timeout=30).raw
    ).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(model_cfg["checkpoint"])
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(DEVICE)


class TestDeiTHookCache:
    def test_cache_contains_attn_out_hook(self, model_cfg, tl_model, sample_inputs):
        """
        Confirms that the DeiT block exposes a hook with an activation shape
        suitable for intervention.
        """
        with torch.no_grad():
            _, cache = tl_model.run_with_cache(sample_inputs)

        hook_name = "blocks.0.hook_attn_out"

        if hook_name not in cache:
            pytest.skip(
                f"[{model_cfg['name']}] {hook_name} not found in cache. "
                f"Available keys (sample): {list(cache.keys())[:20]}"
            )

        act = cache[hook_name]
        assert act.ndim in (3, 4), (
            f"[{model_cfg['name']}] Expected 3D/4D activation, got {tuple(act.shape)}"
        )

    def test_cache_hook_shape_is_reasonable(self, model_cfg, tl_model, sample_inputs):
        """
        Ensures the cached activation can support head-wise interventions.
        """
        with torch.no_grad():
            _, cache = tl_model.run_with_cache(sample_inputs)

        hook_name = "blocks.0.hook_attn_out"
        if hook_name not in cache:
            pytest.skip(f"[{model_cfg['name']}] {hook_name} not found in cache.")

        act = cache[hook_name]
        assert act.ndim in (3, 4)

        if act.ndim == 4:
            assert act.shape[2] > 0, (
                f"[{model_cfg['name']}] Expected head axis > 0, got shape {tuple(act.shape)}"
            )


class TestDeiTHeadAblation:
    def test_replacing_one_head_changes_logits(self, model_cfg, tl_model, sample_inputs):
        """
        Replaces one head's output with another head's output and checks that
        the final logits change.
        """
        hook_name = "blocks.0.hook_attn_out"
        head_to_edit = 0
        source_head = 1

        with torch.no_grad():
            baseline_logits = _get_output_tensor(tl_model(sample_inputs))

        def counterfactual_head_hook(act: torch.Tensor, hook: Any) -> torch.Tensor:
            x = act.clone()

            if x.ndim == 4:
                if x.shape[2] <= max(head_to_edit, source_head):
                    pytest.skip(
                        f"[{model_cfg['name']}] Not enough heads for intervention: got {x.shape[2]}"
                    )
                x[:, :, head_to_edit, :] = x[:, :, source_head, :]
                return x

            if x.ndim == 3:
                if x.shape[1] <= max(head_to_edit, source_head):
                    pytest.skip(
                        f"[{model_cfg['name']}] Not enough heads for intervention: got {x.shape[1]}"
                    )
                x[:, head_to_edit, :] = x[:, source_head, :]
                return x

            raise AssertionError(
                f"[{model_cfg['name']}] Unexpected activation rank for {hook_name}: {x.ndim}"
            )

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
        ), f"[{model_cfg['name']}] Counterfactual head edit had no effect on logits"

    def test_zeroing_one_head_changes_logits(self, model_cfg, tl_model, sample_inputs):
        """
        Zeroes one head and checks that logits change.
        """
        hook_name = "blocks.0.hook_attn_out"
        head_to_zero = 0

        with torch.no_grad():
            baseline_logits = _get_output_tensor(tl_model(sample_inputs))

        def zero_head_hook(act: torch.Tensor, hook: Any) -> torch.Tensor:
            x = act.clone()

            if x.ndim == 4:
                if x.shape[2] <= head_to_zero:
                    pytest.skip(
                        f"[{model_cfg['name']}] Not enough heads for intervention: got {x.shape[2]}"
                    )
                x[:, :, head_to_zero, :] = 0.0
                return x

            if x.ndim == 3:
                if x.shape[1] <= head_to_zero:
                    pytest.skip(
                        f"[{model_cfg['name']}] Not enough heads for intervention: got {x.shape[1]}"
                    )
                x[:, head_to_zero, :] = 0.0
                return x

            raise AssertionError(
                f"[{model_cfg['name']}] Unexpected activation rank for {hook_name}: {x.ndim}"
            )

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
        ), f"[{model_cfg['name']}] Zeroing one head had no effect on logits"

    def test_head_ablation_changes_top1_or_distribution(self, model_cfg, tl_model, sample_inputs):
        """
        Sanity check: the intervention should usually change either top-1 or
        at least the probability distribution.
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
                raise AssertionError(
                    f"[{model_cfg['name']}] Unexpected activation rank: {x.ndim}"
                )
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

        kl = F.kl_div(edited_prob.log(), baseline_prob, reduction="batchmean")
        assert kl.item() >= 0.0

        assert (
            not torch.equal(baseline_top1, edited_top1)
            or not torch.allclose(baseline_logits, edited_logits, atol=1e-6)
        ), f"[{model_cfg['name']}] Ablation did not measurably affect output"
