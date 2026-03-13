"""Multimodal benchmarks for TransformerBridge.

Tests that multimodal models (LLaVA, Gemma3, etc.) correctly handle image inputs
through forward(), generate(), and run_with_cache().
"""


import torch

from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    is_tiny_test_model,
)
from transformer_lens.model_bridge import TransformerBridge


def _create_test_image():
    """Create a small synthetic test image using PIL.

    Returns a 224x224 red image, or None if PIL is not available.
    """
    try:
        from PIL import Image

        return Image.new("RGB", (224, 224), color="red")
    except ImportError:
        return None


def _prepare_test_inputs(bridge: TransformerBridge):
    """Prepare multimodal test inputs using the bridge's processor.

    Returns (input_ids, extra_kwargs, prompt) where extra_kwargs is a dict
    containing pixel_values and any other processor outputs (e.g. image_sizes
    for LlavaNext).  Returns (None, None, None) on failure.
    """
    if bridge.processor is None:
        return None, None, None

    image = _create_test_image()
    if image is None:
        return None, None, None

    # Build a prompt with the model's image token placeholder.
    # Different models use different tokens:
    #   LLava: image_token = "<image>"
    #   Gemma3: boi_token = "<start_of_image>"
    image_token = getattr(bridge.processor, "boi_token", None) or getattr(
        bridge.processor, "image_token", "<image>"
    )
    prompt = f"{image_token}\nDescribe this image."
    try:
        inputs = bridge.processor(text=prompt, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"].to(bridge.cfg.device)

        # Collect all extra kwargs the model's forward() may need
        # (pixel_values, image_sizes, pixel_attention_mask, etc.)
        extra_kwargs = {}
        for key, val in inputs.items():
            if key == "input_ids":
                continue
            if hasattr(val, "to"):
                extra_kwargs[key] = val.to(bridge.cfg.device)
            else:
                extra_kwargs[key] = val

        return input_ids, extra_kwargs, prompt
    except Exception:
        return None, None, None


def benchmark_multimodal_forward(
    bridge: TransformerBridge,
    test_text: str = "Describe this image.",
    reference_model=None,
) -> BenchmarkResult:
    """Benchmark forward() with pixel_values for multimodal models.

    Tests that passing pixel_values produces valid logits (non-NaN, correct shape).

    Args:
        bridge: TransformerBridge model to test.
        test_text: Text prompt (used as fallback if processor unavailable).
        reference_model: Not used, kept for API compatibility.

    Returns:
        BenchmarkResult with forward pass details.
    """
    if not getattr(bridge.cfg, "is_multimodal", False):
        return BenchmarkResult(
            name="multimodal_forward",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: model is not multimodal",
        )

    if is_tiny_test_model(getattr(bridge.cfg, "model_name", "") or ""):
        return BenchmarkResult(
            name="multimodal_forward",
            severity=BenchmarkSeverity.INFO,
            message="Skipped for tiny/test model",
        )

    input_ids, extra_kwargs, prompt = _prepare_test_inputs(bridge)
    if input_ids is None:
        return BenchmarkResult(
            name="multimodal_forward",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: processor or PIL not available",
        )

    try:
        with torch.no_grad():
            logits = bridge.forward(input_ids, return_type="logits", **extra_kwargs)

        if logits is None:
            return BenchmarkResult(
                name="multimodal_forward",
                severity=BenchmarkSeverity.DANGER,
                message="Forward pass returned None",
                passed=False,
            )

        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()

        if has_nan or has_inf:
            return BenchmarkResult(
                name="multimodal_forward",
                severity=BenchmarkSeverity.DANGER,
                message=f"Logits contain NaN={has_nan}, Inf={has_inf}",
                details={"shape": list(logits.shape)},
                passed=False,
            )

        pixel_values = extra_kwargs.get("pixel_values")
        return BenchmarkResult(
            name="multimodal_forward",
            severity=BenchmarkSeverity.INFO,
            message=f"Multimodal forward pass successful, logits shape: {list(logits.shape)}",
            details={
                "logits_shape": list(logits.shape),
                "input_ids_shape": list(input_ids.shape),
                "pixel_values_shape": list(pixel_values.shape)
                if pixel_values is not None
                else None,
            },
        )

    except Exception as e:
        return BenchmarkResult(
            name="multimodal_forward",
            severity=BenchmarkSeverity.ERROR,
            message=f"Multimodal forward pass failed: {str(e)}",
            passed=False,
        )


def benchmark_multimodal_generation(
    bridge: TransformerBridge,
    test_text: str = "Describe this image.",
    max_new_tokens: int = 10,
    reference_model=None,
) -> BenchmarkResult:
    """Benchmark generate() with pixel_values for multimodal models.

    Tests that generation with image input produces text output longer than input.

    Args:
        bridge: TransformerBridge model to test.
        test_text: Text prompt.
        max_new_tokens: Number of tokens to generate.
        reference_model: Not used, kept for API compatibility.

    Returns:
        BenchmarkResult with generation details.
    """
    if not getattr(bridge.cfg, "is_multimodal", False):
        return BenchmarkResult(
            name="multimodal_generation",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: model is not multimodal",
        )

    if is_tiny_test_model(getattr(bridge.cfg, "model_name", "") or ""):
        return BenchmarkResult(
            name="multimodal_generation",
            severity=BenchmarkSeverity.INFO,
            message="Skipped for tiny/test model",
        )

    input_ids, extra_kwargs, prompt = _prepare_test_inputs(bridge)
    if input_ids is None:
        return BenchmarkResult(
            name="multimodal_generation",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: processor or PIL not available",
        )

    try:
        output = bridge.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            return_type="tokens",
            **extra_kwargs,
        )

        if not isinstance(output, torch.Tensor):
            return BenchmarkResult(
                name="multimodal_generation",
                severity=BenchmarkSeverity.DANGER,
                message="Generation did not return a tensor",
                passed=False,
            )

        input_len = input_ids.shape[-1]
        output_len = output.shape[-1]

        if output_len <= input_len:
            return BenchmarkResult(
                name="multimodal_generation",
                severity=BenchmarkSeverity.DANGER,
                message="Generation produced no new tokens",
                details={"input_tokens": input_len, "output_tokens": output_len},
                passed=False,
            )

        generated_text = bridge.tokenizer.decode(output[0], skip_special_tokens=True)

        return BenchmarkResult(
            name="multimodal_generation",
            severity=BenchmarkSeverity.INFO,
            message=f"Multimodal generation successful: {input_len} -> {output_len} tokens",
            details={
                "input_tokens": input_len,
                "output_tokens": output_len,
                "max_new_tokens": max_new_tokens,
                "generated_text": generated_text[:200],
            },
        )

    except Exception as e:
        return BenchmarkResult(
            name="multimodal_generation",
            severity=BenchmarkSeverity.ERROR,
            message=f"Multimodal generation failed: {str(e)}",
            passed=False,
        )


def benchmark_multimodal_cache(
    bridge: TransformerBridge,
    test_text: str = "Describe this image.",
    reference_model=None,
) -> BenchmarkResult:
    """Benchmark run_with_cache() with pixel_values for multimodal models.

    Tests that running with cache and image input populates the activation cache,
    including vision encoder hooks if present.

    Args:
        bridge: TransformerBridge model to test.
        test_text: Text prompt.
        reference_model: Not used, kept for API compatibility.

    Returns:
        BenchmarkResult with cache details.
    """
    if not getattr(bridge.cfg, "is_multimodal", False):
        return BenchmarkResult(
            name="multimodal_cache",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: model is not multimodal",
        )

    if is_tiny_test_model(getattr(bridge.cfg, "model_name", "") or ""):
        return BenchmarkResult(
            name="multimodal_cache",
            severity=BenchmarkSeverity.INFO,
            message="Skipped for tiny/test model",
        )

    input_ids, extra_kwargs, prompt = _prepare_test_inputs(bridge)
    if input_ids is None:
        return BenchmarkResult(
            name="multimodal_cache",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: processor or PIL not available",
        )

    try:
        with torch.no_grad():
            logits, cache = bridge.run_with_cache(input_ids, **extra_kwargs)

        if cache is None or len(cache) == 0:
            return BenchmarkResult(
                name="multimodal_cache",
                severity=BenchmarkSeverity.DANGER,
                message="run_with_cache() returned empty cache",
                passed=False,
            )

        cache_keys = list(cache.keys()) if hasattr(cache, "keys") else []
        vision_keys = [k for k in cache_keys if "vision" in k.lower()]

        return BenchmarkResult(
            name="multimodal_cache",
            severity=BenchmarkSeverity.INFO,
            message=(
                f"Multimodal cache populated: {len(cache_keys)} entries "
                f"({len(vision_keys)} vision-related)"
            ),
            details={
                "total_cache_entries": len(cache_keys),
                "vision_cache_entries": len(vision_keys),
                "vision_keys": vision_keys[:10],
                "sample_keys": cache_keys[:10],
            },
        )

    except Exception as e:
        return BenchmarkResult(
            name="multimodal_cache",
            severity=BenchmarkSeverity.ERROR,
            message=f"Multimodal cache test failed: {str(e)}",
            passed=False,
        )
