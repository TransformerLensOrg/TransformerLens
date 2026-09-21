"""Audio benchmarks for TransformerBridge.

Tests that audio encoder models (HuBERT, wav2vec2, etc.) correctly handle
audio waveform inputs through forward(), run_with_cache(), and produce
stable representations.
"""

from typing import Any, List, Optional

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

# Component-mapping names whose hook_out must be cached when the architecture
# declares them: waveform encoders (HuBERT, wav2vec2) have a conv feature
# extractor, while spectrogram encoders (AST) patch-embed the spectrogram directly.
_CRITICAL_AUDIO_COMPONENTS = ("audio_feature_extractor", "conv_pos_embed", "embed_ln", "embed")


def _prepare_audio_encoder_input(
    bridge: Any, test_audio: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Model-ready audio input via the bridge's feature extractor when available.

    Non-wav2vec2-style architectures (e.g. AST) consume feature-extractor
    outputs (spectrograms), not raw waveforms, and declare their own sampling
    rate — so input prep must go through ``bridge.processor`` whenever the
    boot attached one. Falls back to a raw 16 kHz waveform otherwise.
    """
    processor = getattr(bridge, "processor", None)
    fe = getattr(processor, "feature_extractor", processor)
    sampling_rate = int(getattr(fe, "sampling_rate", 16000) or 16000)

    device = bridge.cfg.device
    dtype = bridge.cfg.dtype
    if test_audio is None:
        test_audio = torch.randn(1, sampling_rate, device=device, dtype=dtype)

    if fe is not None and callable(fe):
        try:
            waveforms = [w for w in test_audio.detach().cpu().float().numpy()]
            out = fe(waveforms, sampling_rate=sampling_rate, return_tensors="pt")
            prepared = out.get("input_values", out.get("input_features"))
            if prepared is not None:
                return prepared.to(device=device, dtype=dtype)
        except Exception:
            pass  # fall through to the raw waveform
    return test_audio


def _prepare_audio_text_inputs(bridge: TransformerBridge):
    """Build audio-conditioned inputs (synthetic waveform + audio token) for an
    audio-text decoder; ``(None, None)`` if the processor has no audio path."""
    processor = getattr(bridge, "processor", None)
    audio_token = getattr(processor, "audio_token", None) if processor is not None else None
    if processor is None or audio_token is None:
        return None, None
    import numpy as np

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    audio = (0.1 * np.sin(2 * np.pi * (200 + 400 * t) * t)).astype(np.float32)
    prompt = f"{audio_token}\nTranscribe this audio."
    try:
        inputs = processor(text=prompt, audio=audio, sampling_rate=sr, return_tensors="pt")
        input_ids = inputs["input_ids"].to(bridge.cfg.device)
        extra = {
            k: (v.to(bridge.cfg.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
            if k != "input_ids"
        }
        return input_ids, extra
    except Exception:
        return None, None


def benchmark_audio_text_forward(bridge: TransformerBridge) -> BenchmarkResult:
    """Benchmark the audio-conditioned forward (input_features -> finite logits) of
    an audio-text decoder -- the audio path that the image and encoder benchmarks do not cover."""
    if not getattr(bridge.cfg, "is_multimodal", False):
        return BenchmarkResult(
            name="audio_text_forward",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: model is not multimodal",
        )
    if is_tiny_test_model(getattr(bridge.cfg, "model_name", "") or ""):
        return BenchmarkResult(
            name="audio_text_forward",
            severity=BenchmarkSeverity.INFO,
            message="Skipped for tiny/test model",
        )

    input_ids, extra = _prepare_audio_text_inputs(bridge)
    if input_ids is None:
        return BenchmarkResult(
            name="audio_text_forward",
            severity=BenchmarkSeverity.SKIPPED,
            message="Skipped: processor could not build audio inputs (no audio_token?)",
        )

    try:
        with torch.no_grad():
            out = bridge(input_ids, return_type="logits", **extra)
        logits = out if isinstance(out, torch.Tensor) else getattr(out, "logits", None)
        if logits is None:
            return BenchmarkResult(
                name="audio_text_forward",
                severity=BenchmarkSeverity.DANGER,
                message="Audio-conditioned forward returned no logits",
                passed=False,
            )
        d_vocab = getattr(bridge.cfg, "d_vocab", None)
        shape_ok = logits.ndim == 3 and (d_vocab is None or logits.shape[-1] == d_vocab)
        finite = bool(torch.isfinite(logits).all())
        if not finite or not shape_ok:
            return BenchmarkResult(
                name="audio_text_forward",
                severity=BenchmarkSeverity.DANGER,
                message=f"Audio-conditioned forward produced invalid logits (finite={finite}, shape={tuple(logits.shape)})",
                details={"logits_shape": list(logits.shape), "all_finite": finite},
                passed=False,
            )
        return BenchmarkResult(
            name="audio_text_forward",
            severity=BenchmarkSeverity.INFO,
            message=f"Audio-conditioned forward OK: finite logits {tuple(logits.shape)}",
            details={
                "logits_shape": list(logits.shape),
                "audio_feature_keys": [k for k in extra if "feature" in k or "audio" in k],
            },
        )
    except Exception as e:
        return BenchmarkResult(
            name="audio_text_forward",
            severity=BenchmarkSeverity.ERROR,
            message=f"Audio-conditioned forward failed: {str(e)}",
            passed=False,
        )


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
    return benchmark_encoder_forward(
        bridge,
        test_audio,
        name="audio_forward",
        ref_input_key="input_values",
        reference_model=reference_model,
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
    return benchmark_encoder_cache(
        bridge,
        test_audio,
        name="audio_cache",
        critical_components=_CRITICAL_AUDIO_COMPONENTS,
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
    return benchmark_encoder_representation_stability(
        bridge,
        test_audio,
        name="audio_representation_stability",
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
        if "audio_feature_extractor" not in (bridge.adapter.component_mapping or {}):
            return BenchmarkResult(
                name="audio_feature_extractor",
                severity=BenchmarkSeverity.SKIPPED,
                message="Skipped: architecture has no conv feature extractor (spectrogram input)",
            )

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
        test_audio: Optional audio input tensor. If None, generates a synthetic input
            shaped for this architecture (waveform or spectrogram).
        verbose: Whether to print progress

    Returns:
        List of BenchmarkResult objects
    """
    if test_audio is None:
        test_audio = build_modality_input(bridge, device=bridge.cfg.device, dtype=bridge.cfg.dtype)
    if test_audio is None:
        return [
            BenchmarkResult(
                name="audio_forward",
                severity=BenchmarkSeverity.ERROR,
                message="Could not build an audio input for this model",
                passed=False,
            )
        ]

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
