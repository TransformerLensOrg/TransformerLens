"""Vision benchmarks for TransformerBridge (Phase 9).

Tests that vision encoder models (ViT, DeiT) correctly handle pixel inputs
through forward(), run_with_cache(), and produce stable representations —
the hook/cache coverage that Phase 1 (HF parity on one forward) doesn't give
non-text models. The audio analog is Phase 8 (audio.py).
"""

from typing import List, Optional

import torch

from transformer_lens.benchmarks.encoder_common import (
    benchmark_encoder_cache,
    benchmark_encoder_forward,
    benchmark_encoder_representation_stability,
)
from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    build_modality_input,
    is_tiny_test_model,
)
from transformer_lens.model_bridge import TransformerBridge

# Top-level component-mapping names whose hook_out must be cached when the
# architecture declares them (unembed = classifier head, absent on bare encoders).
_CRITICAL_VISION_COMPONENTS = ("embed", "ln_final", "unembed")


def benchmark_vision_forward(
    bridge: TransformerBridge,
    test_pixels: torch.Tensor,
    reference_model: Optional[torch.nn.Module] = None,
) -> BenchmarkResult:
    """Benchmark forward pass with pixel input.

    Compares bridge output against the HF native model on the same pixels.
    Bare encoders (ViTModel) compare last_hidden_state; classification heads
    compare logits.

    Args:
        bridge: TransformerBridge model to test
        test_pixels: Pixel tensor [batch, channels, height, width]
        reference_model: Optional HF reference model for comparison
    """
    return benchmark_encoder_forward(
        bridge,
        test_pixels,
        name="vision_forward",
        ref_input_key="pixel_values",
        reference_model=reference_model,
    )


def benchmark_vision_cache(
    bridge: TransformerBridge,
    test_pixels: torch.Tensor,
) -> BenchmarkResult:
    """Benchmark run_with_cache() for vision models.

    Verifies that critical vision hooks fire and produce valid tensors: the
    patch embeddings, final layernorm, classifier head (when present), and the
    first and last block.

    Args:
        bridge: TransformerBridge model to test
        test_pixels: Pixel tensor [batch, channels, height, width]
    """
    return benchmark_encoder_cache(
        bridge,
        test_pixels,
        name="vision_cache",
        critical_components=_CRITICAL_VISION_COMPONENTS,
    )


def benchmark_vision_representation_stability(
    bridge: TransformerBridge,
    test_pixels: torch.Tensor,
) -> BenchmarkResult:
    """Benchmark representation stability under small pixel perturbations.

    Args:
        bridge: TransformerBridge model to test
        test_pixels: Pixel tensor [batch, channels, height, width]
    """
    return benchmark_encoder_representation_stability(
        bridge,
        test_pixels,
        name="vision_representation_stability",
    )


def benchmark_vision_embeddings(
    bridge: TransformerBridge,
    test_pixels: torch.Tensor,
) -> BenchmarkResult:
    """Verify patch-embedding hook outputs.

    Checks that embed.hook_out produces [batch, seq, d_model] tensors with
    non-degenerate values — the vision analog of the audio feature-extractor
    check. Seq length is architecture-dependent (ViT: patches + CLS; DeiT:
    patches + CLS + distillation token), so only lower-bounded here.

    Args:
        bridge: TransformerBridge model to test
        test_pixels: Pixel tensor [batch, channels, height, width]
    """
    try:
        if "embed" not in (bridge.adapter.component_mapping or {}):
            return BenchmarkResult(
                name="vision_embeddings",
                severity=BenchmarkSeverity.SKIPPED,
                message="Skipped: architecture declares no embed component",
            )

        with torch.no_grad():
            _, cache = bridge.run_with_cache(test_pixels)

        hook_key = "embed.hook_out"
        if hook_key not in cache:
            return BenchmarkResult(
                name="vision_embeddings",
                severity=BenchmarkSeverity.DANGER,
                message=f"Hook '{hook_key}' not found in cache",
                passed=False,
            )

        embeddings = cache[hook_key]

        if embeddings.dim() != 3:
            return BenchmarkResult(
                name="vision_embeddings",
                severity=BenchmarkSeverity.DANGER,
                message=f"Expected 3D tensor [batch, seq, d_model], got {embeddings.dim()}D",
                passed=False,
                details={"shape": str(embeddings.shape)},
            )

        d_model = bridge.cfg.d_model
        if embeddings.shape[0] != test_pixels.shape[0] or embeddings.shape[-1] != d_model:
            return BenchmarkResult(
                name="vision_embeddings",
                severity=BenchmarkSeverity.DANGER,
                message=f"Embedding shape {tuple(embeddings.shape)} does not match "
                f"[batch={test_pixels.shape[0]}, seq, d_model={d_model}]",
                passed=False,
                details={"shape": str(embeddings.shape), "d_model": d_model},
            )

        is_all_zeros = embeddings.abs().max().item() == 0
        has_nan = torch.isnan(embeddings).any().item()
        has_inf = torch.isinf(embeddings).any().item()

        if is_all_zeros or has_nan or has_inf:
            issues = []
            if is_all_zeros:
                issues.append("all zeros")
            if has_nan:
                issues.append("NaN")
            if has_inf:
                issues.append("Inf")
            return BenchmarkResult(
                name="vision_embeddings",
                severity=BenchmarkSeverity.DANGER,
                message=f"Degenerate embedding values: {', '.join(issues)}",
                passed=False,
                details={"shape": str(embeddings.shape), "issues": issues},
            )

        return BenchmarkResult(
            name="vision_embeddings",
            severity=BenchmarkSeverity.INFO,
            message=f"Patch embeddings OK: shape={embeddings.shape}, "
            f"mean={embeddings.mean().item():.4f}, std={embeddings.std().item():.4f}",
            details={
                "shape": str(embeddings.shape),
                "mean": embeddings.mean().item(),
                "std": embeddings.std().item(),
            },
        )

    except Exception as e:
        return BenchmarkResult(
            name="vision_embeddings",
            severity=BenchmarkSeverity.ERROR,
            message=f"Patch embedding check failed: {str(e)}",
            passed=False,
        )


def benchmark_vision_classification_decode(
    bridge: TransformerBridge,
) -> BenchmarkResult:
    """Benchmark image-classification decoding on a real image.

    Loads the cats-image fixture, preprocesses it with the bridge's image
    processor, and reports the top predicted labels — the vision analog of the
    audio CTC decode, exercising the processor wiring that synthetic pixel
    tensors don't. Skipped for bare encoders (no classifier head), tiny-random
    models, and when no processor/datasets are available.

    Args:
        bridge: TransformerBridge model to test
    """
    model_name = getattr(bridge.cfg, "model_name", "")
    if is_tiny_test_model(model_name):
        return BenchmarkResult(
            name="vision_classification_decode",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped for tiny-random model (untrained classifier head)",
        )

    # Bare encoders return hidden states from return_type="logits", so gate on
    # the adapter-declared classifier head rather than the output shape.
    if "unembed" not in (bridge.adapter.component_mapping or {}):
        return BenchmarkResult(
            name="vision_classification_decode",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: bare encoder (no classifier head)",
        )

    processor = getattr(bridge, "processor", None)
    if processor is None:
        return BenchmarkResult(
            name="vision_classification_decode",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: no image processor available on the bridge",
        )

    try:
        from datasets import load_dataset

        ds = load_dataset("huggingface/cats-image", split="test", trust_remote_code=True)
        image = ds[0]["image"]

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(bridge.cfg.device)

        with torch.no_grad():
            # Classifier heads return the logits tensor directly; bare encoders
            # fall through to a BaseModelOutput with no logits attribute.
            output = bridge(pixel_values, return_type="logits")

        logits = output if isinstance(output, torch.Tensor) else getattr(output, "logits", None)
        if logits is None or logits.ndim != 2:
            return BenchmarkResult(
                name="vision_classification_decode",
                severity=BenchmarkSeverity.DANGER,
                message="Classifier model produced no [batch, num_labels] logits",
                passed=False,
            )

        k = min(5, logits.shape[-1])
        top = torch.topk(logits[0], k=k)
        id2label = getattr(getattr(bridge, "original_model", None), "config", None)
        id2label = getattr(id2label, "id2label", None) or {}
        top_labels = [id2label.get(int(i), str(int(i))) for i in top.indices]

        return BenchmarkResult(
            name="vision_classification_decode",
            severity=BenchmarkSeverity.INFO,
            message=f"Classification decode successful: top-1 = {top_labels[0]!r}",
            details={
                "top_labels": top_labels,
                "top_logits": [round(float(v), 4) for v in top.values],
                "logits_shape": str(logits.shape),
            },
        )

    except ImportError:
        return BenchmarkResult(
            name="vision_classification_decode",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: 'datasets' package not available",
        )
    except Exception as e:
        return BenchmarkResult(
            name="vision_classification_decode",
            severity=BenchmarkSeverity.ERROR,
            message=f"Classification decode failed: {str(e)}",
            passed=False,
        )


def run_vision_benchmarks(
    bridge: TransformerBridge,
    test_pixels: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> List[BenchmarkResult]:
    """Run all vision benchmarks.

    Args:
        bridge: TransformerBridge model to test
        test_pixels: Optional pixel tensor. If None, generates a synthetic input
            shaped for this architecture from the HF config.
        verbose: Whether to print progress

    Returns:
        List of BenchmarkResult objects
    """
    if test_pixels is None:
        test_pixels = build_modality_input(bridge, device=bridge.cfg.device, dtype=bridge.cfg.dtype)
    if test_pixels is None:
        return [
            BenchmarkResult(
                name="vision_forward",
                severity=BenchmarkSeverity.ERROR,
                message="Could not build a pixel input for this model",
                passed=False,
            )
        ]

    results = []

    if verbose:
        print("1. Vision Forward Pass")
    results.append(benchmark_vision_forward(bridge, test_pixels))

    if verbose:
        print("2. Vision Cache Verification")
    results.append(benchmark_vision_cache(bridge, test_pixels))

    if verbose:
        print("3. Representation Stability")
    results.append(benchmark_vision_representation_stability(bridge, test_pixels))

    if verbose:
        print("4. Patch Embedding Verification")
    results.append(benchmark_vision_embeddings(bridge, test_pixels))

    if verbose:
        print("5. Classification Decoding")
    results.append(benchmark_vision_classification_decode(bridge))

    return results
