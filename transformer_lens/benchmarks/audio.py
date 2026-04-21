"""Audio benchmarks for TransformerBridge.

Tests that audio encoder models (HuBERT, wav2vec2, etc.) correctly handle
audio waveform inputs through forward(), run_with_cache(), and produce
stable representations.
"""

from typing import List, Optional

import torch

from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    compare_tensors,
    is_tiny_test_model,
)
from transformer_lens.model_bridge import TransformerBridge


def benchmark_audio_forward(
    bridge: TransformerBridge,
    test_audio: torch.Tensor,
    reference_model: Optional[torch.nn.Module] = None,
) -> BenchmarkResult:
    """Benchmark forward pass with audio input.

    Compares bridge output against HF native model on the same waveform.
    For bare encoder models, compares last_hidden_state. For CTC models,
    compares logits.

    Args:
        bridge: TransformerBridge model to test
        test_audio: Audio waveform tensor [batch, num_samples]
        reference_model: Optional HF reference model for comparison
    """
    try:
        with torch.no_grad():
            # Use return_type="logits" — for audio encoders without logits, this
            # returns the BaseModelOutput object (bridge falls through to logits=output).
            bridge_output_raw = bridge(test_audio, return_type="logits")

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
                name="audio_forward",
                severity=BenchmarkSeverity.DANGER,
                message="Bridge produced no recognizable output (no logits or last_hidden_state)",
                passed=False,
            )

        if bridge_output.numel() == 0:
            return BenchmarkResult(
                name="audio_forward",
                severity=BenchmarkSeverity.DANGER,
                message="Bridge output is empty",
                passed=False,
            )

        if torch.isnan(bridge_output).any() or torch.isinf(bridge_output).any():
            return BenchmarkResult(
                name="audio_forward",
                severity=BenchmarkSeverity.DANGER,
                message="Bridge output contains NaN or Inf values",
                passed=False,
            )

        # Compare against HF reference if available
        if reference_model is not None:
            with torch.no_grad():
                ref_output_raw = reference_model(input_values=test_audio)
                if output_key == "logits":
                    ref_output = ref_output_raw.logits
                else:
                    ref_output = ref_output_raw.last_hidden_state

            return compare_tensors(
                bridge_output,
                ref_output,
                atol=1e-3,
                rtol=3e-2,
                name="audio_forward",
            )

        return BenchmarkResult(
            name="audio_forward",
            severity=BenchmarkSeverity.INFO,
            message=f"Audio forward pass successful ({output_key} shape: {bridge_output.shape})",
            details={"output_shape": str(bridge_output.shape), "output_key": output_key},
        )

    except Exception as e:
        return BenchmarkResult(
            name="audio_forward",
            severity=BenchmarkSeverity.ERROR,
            message=f"Audio forward pass failed: {str(e)}",
            passed=False,
        )


def benchmark_audio_cache(
    bridge: TransformerBridge,
    test_audio: torch.Tensor,
) -> BenchmarkResult:
    """Benchmark run_with_cache() for audio models.

    Verifies that critical audio-specific hooks fire and produce valid tensors.

    Args:
        bridge: TransformerBridge model to test
        test_audio: Audio waveform tensor [batch, num_samples]
    """
    try:
        with torch.no_grad():
            _, cache = bridge.run_with_cache(test_audio)

        cache_keys = list(cache.keys())
        if len(cache_keys) == 0:
            return BenchmarkResult(
                name="audio_cache",
                severity=BenchmarkSeverity.DANGER,
                message="run_with_cache returned empty cache",
                passed=False,
            )

        # Check for critical audio-specific hooks
        critical_hooks = [
            "audio_feature_extractor.hook_out",
            "conv_pos_embed.hook_out",
            "embed_ln.hook_out",
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
                name="audio_cache",
                severity=BenchmarkSeverity.WARNING,
                message=f"Missing {len(missing)} critical hooks: {missing[:3]}",
                passed=found >= 3,  # Pass if at least 3 of 5 critical hooks present
                details={
                    "total_cached": len(cache_keys),
                    "critical_found": found,
                    "critical_expected": len(critical_hooks),
                    "missing": missing,
                },
            )

        if nan_hooks:
            return BenchmarkResult(
                name="audio_cache",
                severity=BenchmarkSeverity.DANGER,
                message=f"NaN/Inf found in {len(nan_hooks)} cached hooks",
                passed=False,
                details={"nan_hooks": nan_hooks[:5]},
            )

        return BenchmarkResult(
            name="audio_cache",
            severity=BenchmarkSeverity.INFO,
            message=f"Audio cache successful: {len(cache_keys)} hooks captured, "
            f"{found}/{len(critical_hooks)} critical hooks present",
            details={
                "total_cached": len(cache_keys),
                "critical_found": found,
                "critical_expected": len(critical_hooks),
            },
        )

    except Exception as e:
        return BenchmarkResult(
            name="audio_cache",
            severity=BenchmarkSeverity.ERROR,
            message=f"Audio cache failed: {str(e)}",
            passed=False,
        )


def benchmark_audio_representation_stability(
    bridge: TransformerBridge,
    test_audio: torch.Tensor,
) -> BenchmarkResult:
    """Benchmark representation stability under small input perturbations.

    Verifies that the model produces stable representations: similar audio
    inputs should produce similar hidden states. Skip for tiny-random models
    (random weights won't produce stable representations).

    Args:
        bridge: TransformerBridge model to test
        test_audio: Audio waveform tensor [batch, num_samples]
    """
    model_name = getattr(bridge.cfg, "model_name", "")
    if is_tiny_test_model(model_name):
        return BenchmarkResult(
            name="audio_representation_stability",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped for tiny-random model (random weights won't produce stable representations)",
        )

    try:
        # Create a slightly perturbed version
        noise = torch.randn_like(test_audio) * 0.01
        perturbed_audio = test_audio + noise

        with torch.no_grad():
            output_orig = bridge(test_audio, return_type="logits")
            output_pert = bridge(perturbed_audio, return_type="logits")

        # Extract hidden states — handle tensor, BaseModelOutput, or CTC output
        def _extract_states(out):
            if isinstance(out, torch.Tensor):
                return out
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state
            if hasattr(out, "logits") and out.logits is not None:
                return out.logits
            return None

        orig_states = _extract_states(output_orig)
        pert_states = _extract_states(output_pert)

        if orig_states is None or pert_states is None:
            return BenchmarkResult(
                name="audio_representation_stability",
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
            name="audio_representation_stability",
            severity=BenchmarkSeverity.INFO if passed else BenchmarkSeverity.WARNING,
            message=f"Representation stability: cosine_similarity={cosine_sim:.4f} "
            f"(threshold: 0.95)",
            passed=passed,
            details={"cosine_similarity": cosine_sim, "noise_std": 0.01},
        )

    except Exception as e:
        return BenchmarkResult(
            name="audio_representation_stability",
            severity=BenchmarkSeverity.ERROR,
            message=f"Representation stability check failed: {str(e)}",
            passed=False,
        )


def benchmark_audio_feature_extractor(
    bridge: TransformerBridge,
    test_audio: torch.Tensor,
) -> BenchmarkResult:
    """Verify CNN feature extractor hook outputs.

    Checks that the audio_feature_extractor.hook_out produces tensors with
    correct shape and non-degenerate values.

    Args:
        bridge: TransformerBridge model to test
        test_audio: Audio waveform tensor [batch, num_samples]
    """
    try:
        with torch.no_grad():
            _, cache = bridge.run_with_cache(test_audio)

        hook_key = "audio_feature_extractor.hook_out"
        if hook_key not in cache:
            return BenchmarkResult(
                name="audio_feature_extractor",
                severity=BenchmarkSeverity.DANGER,
                message=f"Hook '{hook_key}' not found in cache",
                passed=False,
            )

        features = cache[hook_key]

        # Check shape: should be [batch, conv_dim, num_frames]
        if features.dim() != 3:
            return BenchmarkResult(
                name="audio_feature_extractor",
                severity=BenchmarkSeverity.DANGER,
                message=f"Expected 3D tensor [batch, conv_dim, frames], got {features.dim()}D",
                passed=False,
                details={"shape": str(features.shape)},
            )

        # Check for degenerate values
        is_all_zeros = features.abs().max().item() == 0
        has_nan = torch.isnan(features).any().item()
        has_inf = torch.isinf(features).any().item()

        if is_all_zeros or has_nan or has_inf:
            issues = []
            if is_all_zeros:
                issues.append("all zeros")
            if has_nan:
                issues.append("NaN")
            if has_inf:
                issues.append("Inf")
            return BenchmarkResult(
                name="audio_feature_extractor",
                severity=BenchmarkSeverity.DANGER,
                message=f"Degenerate feature values: {', '.join(issues)}",
                passed=False,
                details={"shape": str(features.shape), "issues": issues},
            )

        return BenchmarkResult(
            name="audio_feature_extractor",
            severity=BenchmarkSeverity.INFO,
            message=f"Feature extractor OK: shape={features.shape}, "
            f"mean={features.mean().item():.4f}, std={features.std().item():.4f}",
            details={
                "shape": str(features.shape),
                "mean": features.mean().item(),
                "std": features.std().item(),
            },
        )

    except Exception as e:
        return BenchmarkResult(
            name="audio_feature_extractor",
            severity=BenchmarkSeverity.ERROR,
            message=f"Feature extractor check failed: {str(e)}",
            passed=False,
        )


def benchmark_audio_ctc_decode(
    bridge: TransformerBridge,
) -> BenchmarkResult:
    """Benchmark CTC decoding for HubertForCTC models.

    Loads a small sample from librispeech_asr_dummy, decodes via greedy CTC,
    and reports the decoded text. Skipped for bare encoder models (no CTC head)
    and tiny-random models.

    Args:
        bridge: TransformerBridge model to test
    """
    model_name = getattr(bridge.cfg, "model_name", "")
    if is_tiny_test_model(model_name):
        return BenchmarkResult(
            name="audio_ctc_decode",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped for tiny-random model (untrained CTC head)",
        )

    try:
        from datasets import load_dataset

        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            trust_remote_code=True,
        )
        audio = ds[0]["audio"]
        reference_text = ds[0]["text"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
        waveform = waveform.to(bridge.cfg.device)

        with torch.no_grad():
            output = bridge(waveform, return_type=None)

        if not hasattr(output, "logits") or output.logits is None:
            return BenchmarkResult(
                name="audio_ctc_decode",
                severity=BenchmarkSeverity.SKIPPED,
                message="Skipped: model output has no logits (bare encoder)",
            )

        # Greedy CTC decode
        predicted_ids = torch.argmax(output.logits, dim=-1)

        # Try to decode with processor
        processor = getattr(bridge, "processor", None)
        if processor is not None and hasattr(processor, "decode"):
            decoded_text = processor.decode(predicted_ids[0])
        elif processor is not None and hasattr(processor, "batch_decode"):
            decoded_text = processor.batch_decode(predicted_ids)[0]
        else:
            decoded_text = str(predicted_ids[0].tolist()[:20]) + "..."

        return BenchmarkResult(
            name="audio_ctc_decode",
            severity=BenchmarkSeverity.INFO,
            message=f"CTC decode successful",
            details={
                "decoded_text": decoded_text[:200],
                "reference_text": reference_text[:200],
                "logits_shape": str(output.logits.shape),
            },
        )

    except ImportError:
        return BenchmarkResult(
            name="audio_ctc_decode",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: 'datasets' package not available",
        )
    except Exception as e:
        return BenchmarkResult(
            name="audio_ctc_decode",
            severity=BenchmarkSeverity.ERROR,
            message=f"CTC decode failed: {str(e)}",
            passed=False,
        )


def run_audio_benchmarks(
    bridge: TransformerBridge,
    test_audio: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> List[BenchmarkResult]:
    """Run all audio benchmarks.

    Args:
        bridge: TransformerBridge model to test
        test_audio: Optional audio waveform tensor. If None, generates synthetic audio.
        verbose: Whether to print progress

    Returns:
        List of BenchmarkResult objects
    """
    if test_audio is None:
        device = bridge.cfg.device
        dtype = bridge.cfg.dtype
        test_audio = torch.randn(1, 16000, device=device, dtype=dtype)

    results = []

    if verbose:
        print("1. Audio Forward Pass")
    results.append(benchmark_audio_forward(bridge, test_audio))

    if verbose:
        print("2. Audio Cache Verification")
    results.append(benchmark_audio_cache(bridge, test_audio))

    if verbose:
        print("3. Representation Stability")
    results.append(benchmark_audio_representation_stability(bridge, test_audio))

    if verbose:
        print("4. Feature Extractor Verification")
    results.append(benchmark_audio_feature_extractor(bridge, test_audio))

    if verbose:
        print("5. CTC Decoding")
    results.append(benchmark_audio_ctc_decode(bridge))

    return results
