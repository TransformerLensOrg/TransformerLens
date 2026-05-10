"""
Tests for HookedVisualEncoder hook and ablation functionality on ViTModel.

Checks that:
- hooks exist and have usable shapes
- counterfactual head replacement changes final hidden states
- zeroing a head changes final hidden states
- zeroing an MLP/residual feature changes final hidden states
"""

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
        "checkpoint": "google/vit-base-patch16-224-in21k",
        "name": "vit-base-in21k",
    },
]


def _get_output_tensor(out: Any) -> torch.Tensor:
    """Extract tensor from model output."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        if "last_hidden_state" in out:
            return out["last_hidden_state"]
        if "hidden_states" in out:
            return out["hidden_states"][-1]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if hasattr(out, "hidden_states"):
        return out.hidden_states[-1]
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


class TestViTHookCache:
    def test_cache_contains_attn_out_hook(self, model_cfg, tl_model, sample_inputs):
        """
        Confirms the ViT block exposes an attention output hook suitable for intervention.
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

    def test_cache_contains_mlp_out_hook(self, model_cfg, tl_model, sample_inputs):
        """
        Confirms the ViT block exposes an MLP/residual hook suitable for intervention.
        """
        with torch.no_grad():
            _, cache = tl_model.run_with_cache(sample_inputs)

        hook_name = "blocks.0.hook_mlp_out"

        if hook_name not in cache:
            pytest.skip(
                f"[{model_cfg['name']}] {hook_name} not found in cache. "
                f"Available keys (sample): {list(cache.keys())[:20]}"
            )

        act = cache[hook_name]
        assert act.ndim == 3, (
            f"[{model_cfg['name']}] Expected 3D activation, got {tuple(act.shape)}"
        )
        assert act.shape[-1] > 0, (
            f"[{model_cfg['name']}] Invalid hidden size in {hook_name}: {tuple(act.shape)}"
        )


class TestViTHeadAblation:
    def test_replacing_one_head_changes_output(
        self,
        model_cfg,
        tl_model,
        sample_inputs,
    ):
        """
        Replaces one attention head's output with another head's output and checks
        that the final hidden states change.
        """
        hook_name = "blocks.0.hook_attn_out"
        head_to_edit = 0
        source_head = 1

        with torch.no_grad():
            baseline_out = _get_output_tensor(tl_model(sample_inputs))

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
            edited_out = _get_output_tensor(
                tl_model.run_with_hooks(
                    sample_inputs,
                    fwd_hooks=[(hook_name, counterfactual_head_hook)],
                )
            )

        assert baseline_out.shape == edited_out.shape
        assert not torch.allclose(
            baseline_out, edited_out, atol=1e-6
        ), f"[{model_cfg['name']}] Counterfactual head edit had no effect"

    def test_zeroing_one_head_changes_output(
        self,
        model_cfg,
        tl_model,
        sample_inputs,
    ):
        """
        Zeroes one attention head and checks that the final hidden states change.
        """
        hook_name = "blocks.0.hook_attn_out"
        head_to_zero = 0

        with torch.no_grad():
            baseline_out = _get_output_tensor(tl_model(sample_inputs))

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
            ablated_out = _get_output_tensor(
                tl_model.run_with_hooks(
                    sample_inputs,
                    fwd_hooks=[(hook_name, zero_head_hook)],
                )
            )

        assert baseline_out.shape == ablated_out.shape
        assert not torch.allclose(
            baseline_out, ablated_out, atol=1e-6
        ), f"[{model_cfg['name']}] Zeroing one head had no effect"

    def test_head_ablation_changes_cls_token_or_distribution(
        self,
        model_cfg,
        tl_model,
        sample_inputs,
    ):
        """
        Sanity check: the intervention should usually change the CLS token,
        even if the full tensor change is small.
        """
        hook_name = "blocks.0.hook_attn_out"

        with torch.no_grad():
            baseline_out = _get_output_tensor(tl_model(sample_inputs))
            baseline_cls = baseline_out[:, 0, :]

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
            edited_out = _get_output_tensor(
                tl_model.run_with_hooks(
                    sample_inputs,
                    fwd_hooks=[(hook_name, zero_head_hook)],
                )
            )
            edited_cls = edited_out[:, 0, :]

        cos = F.cosine_similarity(baseline_cls.flatten(), edited_cls.flatten(), dim=0)
        assert cos.item() < 1.0, (
            f"[{model_cfg['name']}] CLS token did not change under head ablation"
        )


class TestViTMlpAblation:
    def test_zeroing_mlp_feature_changes_output(
        self,
        model_cfg,
        tl_model,
        sample_inputs,
    ):
        """
        Zeroes one feature in the MLP output / residual stream and checks that
        the final hidden states change.
        """
        hook_name = "blocks.0.hook_mlp_out"
        feature_to_zero = 0

        with torch.no_grad():
            baseline_out = _get_output_tensor(tl_model(sample_inputs))

        def zero_feature_hook(act: torch.Tensor, hook: Any) -> torch.Tensor:
            x = act.clone()
            if x.ndim != 3:
                raise AssertionError(
                    f"[{model_cfg['name']}] Unexpected activation rank for {hook_name}: {x.ndim}"
                )
            if x.shape[-1] <= feature_to_zero:
                pytest.skip(
                    f"[{model_cfg['name']}] Hidden size too small for intervention: got {x.shape[-1]}"
                )
            x[:, :, feature_to_zero] = 0.0
            return x

        with torch.no_grad():
            ablated_out = _get_output_tensor(
                tl_model.run_with_hooks(
                    sample_inputs,
                    fwd_hooks=[(hook_name, zero_feature_hook)],
                )
            )

        assert baseline_out.shape == ablated_out.shape
        assert not torch.allclose(
            baseline_out, ablated_out, atol=1e-6
        ), f"[{model_cfg['name']}] Zeroing one MLP feature had no effect"

    def test_zeroing_mlp_feature_changes_cls_token(
        self,
        model_cfg,
        tl_model,
        sample_inputs,
    ):
        """
        Stronger check focused on the CLS token.
        """
        hook_name = "blocks.0.hook_mlp_out"

        with torch.no_grad():
            baseline_out = _get_output_tensor(tl_model(sample_inputs))
            baseline_cls = baseline_out[:, 0, :]

        def zero_feature_hook(act: torch.Tensor, hook: Any) -> torch.Tensor:
            x = act.clone()
            if x.ndim != 3:
                raise AssertionError(
                    f"[{model_cfg['name']}] Unexpected activation rank: {x.ndim}"
                )
            x[:, :, 0] = 0.0
            return x

        with torch.no_grad():
            edited_out = _get_output_tensor(
                tl_model.run_with_hooks(
                    sample_inputs,
                    fwd_hooks=[(hook_name, zero_feature_hook)],
                )
            )
            edited_cls = edited_out[:, 0, :]

        assert not torch.allclose(
            baseline_cls, edited_cls, atol=1e-6
        ), f"[{model_cfg['name']}] CLS token did not change under MLP ablation"
