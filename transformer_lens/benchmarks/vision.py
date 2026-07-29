"""Vision benchmarks for TransformerBridge.

Tests that vision-only encoder models (ViT, DeiT) correctly handle image inputs
through forward() and run_with_cache(). No generation benchmark — vision
encoders don't generate.
"""

import torch

from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    compare_tensors,
)
from transformer_lens.model_bridge import TransformerBridge

# Bridge wraps the real HF forward (hooked submodules substituted in place), so
# parity should be near-exact; 1e-4 matches the ViT adapter integration tests.
_VISION_PARITY_ATOL = 1e-4


def _prepare_pixel_values(bridge: TransformerBridge) -> torch.Tensor:
    """Synthetic pixel_values batch sized from the HF config's image geometry."""
    hf_config = getattr(bridge.original_model, "config", None)
    image_size = getattr(hf_config, "image_size", 224)
    if isinstance(image_size, (list, tuple)):
        img_h, img_w = image_size[0], image_size[1]
    else:
        img_h = img_w = image_size
    num_channels = getattr(hf_config, "num_channels", 3)
    dtype = getattr(bridge.cfg, "dtype", torch.float32)
    return torch.randn(1, num_channels, img_h, img_w, device=bridge.cfg.device, dtype=dtype)


def benchmark_vision_forward(
    bridge: TransformerBridge,
    reference_model=None,
) -> BenchmarkResult:
    """Benchmark forward() parity against the raw HF model for vision encoders.

    Feeds the same synthetic pixel_values to the bridge and the raw HF model and
    compares logits (classifier heads) or last_hidden_state (bare encoders).

    Args:
        bridge: TransformerBridge model to test.
        reference_model: Optional raw HF model; defaults to bridge.original_model.

    Returns:
        BenchmarkResult with forward parity details.
    """
    if not getattr(bridge.cfg, "is_visual_model", False):
        return BenchmarkResult(
            name="vision_forward",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: model is not a vision model",
        )

    try:
        pixel_values = _prepare_pixel_values(bridge)
        hf_model = reference_model if reference_model is not None else bridge.original_model
        with torch.no_grad():
            bridge_out = bridge.forward(pixel_values, return_type="logits")
            hf_out = hf_model(pixel_values)
        # Classifier heads expose .logits; bare encoders only last_hidden_state.
        hf_logits = getattr(hf_out, "logits", None)
        hf_ref = hf_logits if hf_logits is not None else hf_out.last_hidden_state

        if bridge_out is None:
            return BenchmarkResult(
                name="vision_forward",
                severity=BenchmarkSeverity.DANGER,
                message="Forward pass returned None",
                passed=False,
            )

        has_nan = torch.isnan(bridge_out).any().item()
        has_inf = torch.isinf(bridge_out).any().item()
        if has_nan or has_inf:
            return BenchmarkResult(
                name="vision_forward",
                severity=BenchmarkSeverity.DANGER,
                message=f"Output contains NaN={has_nan}, Inf={has_inf}",
                details={"shape": list(bridge_out.shape)},
                passed=False,
            )

        # Shared comparator: handles the shape check plus device/dtype
        # normalization a custom-device reference would otherwise mis-compare on.
        return compare_tensors(
            bridge_out,
            hf_ref,
            atol=_VISION_PARITY_ATOL,
            rtol=0.0,
            name="vision_forward",
        )

    except Exception as e:
        return BenchmarkResult(
            name="vision_forward",
            severity=BenchmarkSeverity.ERROR,
            message=f"Vision forward benchmark failed: {str(e)}",
            passed=False,
        )


def benchmark_vision_cache(
    bridge: TransformerBridge,
    reference_model=None,
) -> BenchmarkResult:
    """Benchmark run_with_cache() with pixel input for vision encoders.

    Tests that key hooks (embed input, block-0 attention projections, and the
    classifier head or final norm) fired and captured finite activations.

    Args:
        bridge: TransformerBridge model to test.
        reference_model: Not used, kept for API compatibility.

    Returns:
        BenchmarkResult with cache details.
    """
    if not getattr(bridge.cfg, "is_visual_model", False):
        return BenchmarkResult(
            name="vision_cache",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: model is not a vision model",
        )

    try:
        pixel_values = _prepare_pixel_values(bridge)
        with torch.no_grad():
            _, cache = bridge.run_with_cache(pixel_values)

        if cache is None or len(cache) == 0:
            return BenchmarkResult(
                name="vision_cache",
                severity=BenchmarkSeverity.DANGER,
                message="run_with_cache() returned empty cache",
                passed=False,
            )

        required_hooks = [
            "embed.hook_in",
            "blocks.0.attn.q.hook_out",
            "blocks.0.attn.o.hook_out",
            # Classifier heads cache the head output; bare encoders stop at ln_final.
            "unembed.hook_out" if hasattr(bridge, "unembed") else "ln_final.hook_normalized",
        ]
        missing = [k for k in required_hooks if k not in cache]
        if missing:
            return BenchmarkResult(
                name="vision_cache",
                severity=BenchmarkSeverity.DANGER,
                message=f"Cache missing expected hooks: {missing}",
                details={"missing_hooks": missing, "total_cache_entries": len(cache)},
                passed=False,
            )

        non_finite = [k for k in required_hooks if not torch.isfinite(cache[k]).all()]
        if non_finite:
            return BenchmarkResult(
                name="vision_cache",
                severity=BenchmarkSeverity.DANGER,
                message=f"Cached activations contain NaN/Inf: {non_finite}",
                details={"non_finite_hooks": non_finite},
                passed=False,
            )

        cache_keys = list(cache.keys()) if hasattr(cache, "keys") else []
        return BenchmarkResult(
            name="vision_cache",
            severity=BenchmarkSeverity.INFO,
            message=(
                f"Vision cache populated: {len(cache_keys)} entries, " f"key hooks fired and finite"
            ),
            details={
                "total_cache_entries": len(cache_keys),
                "checked_hooks": required_hooks,
                "sample_keys": cache_keys[:10],
            },
        )

    except Exception as e:
        return BenchmarkResult(
            name="vision_cache",
            severity=BenchmarkSeverity.ERROR,
            message=f"Vision cache test failed: {str(e)}",
            passed=False,
        )
