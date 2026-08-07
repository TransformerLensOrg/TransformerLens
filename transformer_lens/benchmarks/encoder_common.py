"""Shared benchmark implementations for non-text encoder bridges.

Audio and vision encoders take one raw tensor (waveform, spectrogram, or pixels)
where text models take token ids; given that tensor, the forward, run_with_cache,
and perturbation-stability checks are modality-independent. The modality modules
(audio.py, vision.py) wrap these with their result names, HF reference-forward
kwarg, and critical component lists.
"""

from typing import Optional, Sequence

import torch

from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    compare_tensors,
    is_tiny_test_model,
)
from transformer_lens.model_bridge import TransformerBridge


def extract_encoder_states(out) -> Optional[torch.Tensor]:
    """Pull the hidden-state/logit tensor out of a tensor, BaseModelOutput, or head output."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    return None


def benchmark_encoder_forward(
    bridge: TransformerBridge,
    test_input: torch.Tensor,
    name: str,
    ref_input_key: str,
    reference_model: Optional[torch.nn.Module] = None,
) -> BenchmarkResult:
    """Forward-pass benchmark on a raw modality tensor.

    Compares bridge output against the HF native model on the same input when a
    reference is given. Bare encoders compare last_hidden_state; head models
    (CTC, classification) compare logits.

    Args:
        bridge: TransformerBridge model to test
        test_input: Raw modality tensor (waveform, spectrogram, or pixels)
        name: Result name (e.g. "audio_forward", "vision_forward")
        ref_input_key: Kwarg the HF reference takes the tensor under
            ("input_values" for audio, "pixel_values" for vision)
        reference_model: Optional HF reference model for comparison
    """
    try:
        with torch.no_grad():
            # Use return_type="logits" — for bare encoders without logits, this
            # returns the BaseModelOutput object (bridge falls through to logits=output).
            bridge_output_raw = bridge(test_input, return_type="logits")

        # Extract the output tensor
        if isinstance(bridge_output_raw, torch.Tensor):
            bridge_output = bridge_output_raw
            output_key = "logits"
        elif hasattr(bridge_output_raw, "logits") and bridge_output_raw.logits is not None:
            bridge_output = bridge_output_raw.logits
            output_key = "logits"
        elif hasattr(bridge_output_raw, "last_hidden_state"):
            bridge_output = bridge_output_raw.last_hidden_state
            output_key = "last_hidden_state"
        else:
            return BenchmarkResult(
                name=name,
                severity=BenchmarkSeverity.DANGER,
                message="Bridge produced no recognizable output (no logits or last_hidden_state)",
                passed=False,
            )

        if bridge_output.numel() == 0:
            return BenchmarkResult(
                name=name,
                severity=BenchmarkSeverity.DANGER,
                message="Bridge output is empty",
                passed=False,
            )

        if torch.isnan(bridge_output).any() or torch.isinf(bridge_output).any():
            return BenchmarkResult(
                name=name,
                severity=BenchmarkSeverity.DANGER,
                message="Bridge output contains NaN or Inf values",
                passed=False,
            )

        # Compare against HF reference if available
        if reference_model is not None:
            with torch.no_grad():
                ref_output_raw = reference_model(**{ref_input_key: test_input})
                if output_key == "logits":
                    ref_output = ref_output_raw.logits
                else:
                    ref_output = ref_output_raw.last_hidden_state

            return compare_tensors(
                bridge_output,
                ref_output,
                atol=1e-3,
                rtol=3e-2,
                name=name,
            )

        return BenchmarkResult(
            name=name,
            severity=BenchmarkSeverity.INFO,
            message=f"Forward pass successful ({output_key} shape: {bridge_output.shape})",
            details={"output_shape": str(bridge_output.shape), "output_key": output_key},
        )

    except Exception as e:
        return BenchmarkResult(
            name=name,
            severity=BenchmarkSeverity.ERROR,
            message=f"Forward pass failed: {str(e)}",
            passed=False,
        )


def benchmark_encoder_cache(
    bridge: TransformerBridge,
    test_input: torch.Tensor,
    name: str,
    critical_components: Sequence[str],
    min_found: int = 3,
) -> BenchmarkResult:
    """run_with_cache() benchmark on a raw modality tensor.

    Verifies that critical hooks fire and produce valid tensors: the
    ``critical_components`` this architecture actually declares in its
    component mapping, plus the first and last block.

    Args:
        bridge: TransformerBridge model to test
        test_input: Raw modality tensor (waveform, spectrogram, or pixels)
        name: Result name (e.g. "audio_cache", "vision_cache")
        critical_components: Component-mapping names whose hook_out must be
            cached when the architecture declares them
        min_found: Minimum critical hooks present to still pass with a warning
    """
    try:
        with torch.no_grad():
            _, cache = bridge.run_with_cache(test_input)

        cache_keys = list(cache.keys())
        if len(cache_keys) == 0:
            return BenchmarkResult(
                name=name,
                severity=BenchmarkSeverity.DANGER,
                message="run_with_cache returned empty cache",
                passed=False,
            )

        component_mapping = bridge.adapter.component_mapping or {}
        critical_hooks = [
            f"{comp}.hook_out" for comp in critical_components if comp in component_mapping
        ]
        # Also check at least the first and last block
        n_layers = bridge.cfg.n_layers
        critical_hooks.append("blocks.0.hook_out")
        critical_hooks.append(f"blocks.{n_layers - 1}.hook_out")

        missing = [h for h in critical_hooks if h not in cache_keys]
        found = len(critical_hooks) - len(missing)

        # Check for NaN/Inf in cached values
        nan_hooks = []
        for key in cache_keys[:20]:  # Sample first 20 hooks
            val = cache[key]
            if isinstance(val, torch.Tensor) and (torch.isnan(val).any() or torch.isinf(val).any()):
                nan_hooks.append(key)

        if missing:
            return BenchmarkResult(
                name=name,
                severity=BenchmarkSeverity.WARNING,
                message=f"Missing {len(missing)} critical hooks: {missing[:3]}",
                passed=found >= min_found,
                details={
                    "total_cached": len(cache_keys),
                    "critical_found": found,
                    "critical_expected": len(critical_hooks),
                    "missing": missing,
                },
            )

        if nan_hooks:
            return BenchmarkResult(
                name=name,
                severity=BenchmarkSeverity.DANGER,
                message=f"NaN/Inf found in {len(nan_hooks)} cached hooks",
                passed=False,
                details={"nan_hooks": nan_hooks[:5]},
            )

        return BenchmarkResult(
            name=name,
            severity=BenchmarkSeverity.INFO,
            message=f"Cache successful: {len(cache_keys)} hooks captured, "
            f"{found}/{len(critical_hooks)} critical hooks present",
            details={
                "total_cached": len(cache_keys),
                "critical_found": found,
                "critical_expected": len(critical_hooks),
            },
        )

    except Exception as e:
        return BenchmarkResult(
            name=name,
            severity=BenchmarkSeverity.ERROR,
            message=f"Cache benchmark failed: {str(e)}",
            passed=False,
        )


def benchmark_encoder_representation_stability(
    bridge: TransformerBridge,
    test_input: torch.Tensor,
    name: str,
) -> BenchmarkResult:
    """Representation stability under small input perturbations.

    Similar inputs should produce similar hidden states. Skipped for
    tiny-random models (random weights won't produce stable representations).

    Args:
        bridge: TransformerBridge model to test
        test_input: Raw modality tensor (waveform, spectrogram, or pixels)
        name: Result name (e.g. "audio_representation_stability")
    """
    model_name = getattr(bridge.cfg, "model_name", "")
    if is_tiny_test_model(model_name):
        return BenchmarkResult(
            name=name,
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped for tiny-random model (random weights won't produce stable representations)",
        )

    try:
        # Create a slightly perturbed version
        noise = torch.randn_like(test_input) * 0.01
        perturbed_input = test_input + noise

        with torch.no_grad():
            output_orig = bridge(test_input, return_type="logits")
            output_pert = bridge(perturbed_input, return_type="logits")

        orig_states = extract_encoder_states(output_orig)
        pert_states = extract_encoder_states(output_pert)

        if orig_states is None or pert_states is None:
            return BenchmarkResult(
                name=name,
                severity=BenchmarkSeverity.WARNING,
                message="Could not extract hidden states for stability check",
                passed=False,
            )

        # Compute cosine similarity (flatten to 2D: [batch, features])
        orig_flat = orig_states.reshape(orig_states.shape[0], -1)
        pert_flat = pert_states.reshape(pert_states.shape[0], -1)
        cosine_sim = (
            torch.nn.functional.cosine_similarity(orig_flat, pert_flat, dim=-1).mean().item()
        )

        passed = cosine_sim > 0.95
        return BenchmarkResult(
            name=name,
            severity=BenchmarkSeverity.INFO if passed else BenchmarkSeverity.WARNING,
            message=f"Representation stability: cosine_similarity={cosine_sim:.4f} "
            f"(threshold: 0.95)",
            passed=passed,
            details={"cosine_similarity": cosine_sim, "noise_std": 0.01},
        )

    except Exception as e:
        return BenchmarkResult(
            name=name,
            severity=BenchmarkSeverity.ERROR,
            message=f"Representation stability check failed: {str(e)}",
            passed=False,
        )
