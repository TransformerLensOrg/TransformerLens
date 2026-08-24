"""Forward pass benchmarks for TransformerBridge."""

from typing import Optional, Union

import torch

from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    bridge_self_target_loss,
    compare_scalars,
    compare_tensors,
)
from transformer_lens.model_bridge import TransformerBridge


def _compute_self_target_loss(bridge: TransformerBridge, test_text: str) -> torch.Tensor:
    """Compute loss with the tokenized input supplied as explicit labels."""
    return bridge_self_target_loss(bridge, test_text)


def _is_encoder_decoder(model: torch.nn.Module) -> bool:
    """Check if a model is an encoder-decoder architecture."""
    config = getattr(model, "config", None)
    if config is None:
        return False
    return getattr(config, "is_encoder_decoder", False)


def _get_decoder_input_ids(model: torch.nn.Module, batch_size: int = 1) -> torch.Tensor:
    """Get decoder_input_ids for encoder-decoder models.

    Args:
        model: The model to get decoder_start_token_id from
        batch_size: Batch size for the decoder_input_ids

    Returns:
        Tensor of shape [batch_size, 1] with decoder_start_token_id
    """
    config = getattr(model, "config", None)
    decoder_start_token_id = getattr(config, "decoder_start_token_id", None) if config else None
    if decoder_start_token_id is None:
        # HF fallback chain: bos, then eos (MBart-family checkpoints leave
        # decoder_start unset and start from EOS).
        decoder_start_token_id = getattr(config, "bos_token_id", None) if config else None
    if decoder_start_token_id is None:
        decoder_start_token_id = getattr(config, "eos_token_id", None) if config else None
    if isinstance(decoder_start_token_id, (list, tuple)):
        decoder_start_token_id = decoder_start_token_id[0]
    if decoder_start_token_id is None:
        decoder_start_token_id = 0
    return torch.tensor([[decoder_start_token_id]] * batch_size)


def benchmark_forward_pass(
    bridge: TransformerBridge,
    test_input: Union[str, torch.Tensor],
    reference_model: Optional[torch.nn.Module] = None,
    reference_logits: Optional[torch.Tensor] = None,
    atol: float = 1e-3,
    rtol: float = 3e-2,
) -> BenchmarkResult:
    """Benchmark forward pass between TransformerBridge and reference model.

    Args:
        bridge: TransformerBridge model to test
        test_input: Input text string or audio waveform tensor for testing
        reference_model: Optional live HF reference model (audio / encoder-decoder paths)
        reference_logits: Optional pre-computed reference logits/hidden states tensor
            (e.g., saved from a prior HF forward pass to avoid needing both models in memory)
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        BenchmarkResult with comparison details
    """
    try:
        _is_audio = getattr(bridge.cfg, "is_audio_model", False)

        is_enc_dec = _is_encoder_decoder(bridge.original_model)

        # Prepare extra kwargs for encoder-decoder models
        extra_kwargs = {}
        if is_enc_dec and isinstance(test_input, str):
            tokens = bridge.to_tokens(test_input)
            batch_size = tokens.shape[0]
            decoder_input_ids = _get_decoder_input_ids(bridge.original_model, batch_size)
            decoder_input_ids = decoder_input_ids.to(tokens.device)
            extra_kwargs["decoder_input_ids"] = decoder_input_ids

        # Run bridge forward pass (use no_grad to match HF reference context —
        # MPS SDPA can produce different results with vs without gradient tracking)
        with torch.no_grad():
            if _is_audio and isinstance(test_input, torch.Tensor):
                # Audio models: pass waveform, extract tensor from output
                bridge_output_raw = bridge(test_input, return_type="logits")
                if isinstance(bridge_output_raw, torch.Tensor):
                    bridge_output = bridge_output_raw
                elif hasattr(bridge_output_raw, "logits") and bridge_output_raw.logits is not None:
                    bridge_output = bridge_output_raw.logits
                elif hasattr(bridge_output_raw, "last_hidden_state"):
                    bridge_output = bridge_output_raw.last_hidden_state
                else:
                    bridge_output = bridge_output_raw
            else:
                bridge_output = bridge(test_input, return_type="logits", **extra_kwargs)

        if reference_model is None and reference_logits is None:
            # No reference model or logits - just verify output shape and validity
            if not isinstance(bridge_output, torch.Tensor):
                return BenchmarkResult(
                    name="forward_pass",
                    severity=BenchmarkSeverity.DANGER,
                    message="Bridge output is not a tensor",
                    passed=False,
                )

            if bridge_output.numel() == 0:
                return BenchmarkResult(
                    name="forward_pass",
                    severity=BenchmarkSeverity.DANGER,
                    message="Bridge output is empty",
                    passed=False,
                )

            return BenchmarkResult(
                name="forward_pass",
                severity=BenchmarkSeverity.INFO,
                message=f"Bridge forward pass successful (shape: {bridge_output.shape})",
                details={"output_shape": str(bridge_output.shape)},
            )

        # Get reference logits from a pre-computed tensor or a live HF model
        if reference_logits is not None:
            reference_output = reference_logits.to(bridge_output.device)
        elif _is_audio and isinstance(test_input, torch.Tensor):
            # Audio HF reference model: pass the prepared audio input positionally
            # (input_values for wav2vec2-style, input_features for AST-style)
            assert reference_model is not None
            with torch.no_grad():
                hf_output = reference_model(test_input)
                if hasattr(hf_output, "logits") and hf_output.logits is not None:
                    reference_output = hf_output.logits
                else:
                    reference_output = hf_output.last_hidden_state
        else:
            # reference_model is non-None here: the both-None case returns early above
            assert reference_model is not None
            assert isinstance(test_input, str), "Text model requires string input"
            tokens = bridge.to_tokens(test_input)
            with torch.no_grad():
                if is_enc_dec:
                    # Encoder-decoder models need decoder_input_ids
                    batch_size = tokens.shape[0]
                    decoder_input_ids = _get_decoder_input_ids(reference_model, batch_size)
                    decoder_input_ids = decoder_input_ids.to(tokens.device)
                    hf_output = reference_model(tokens, decoder_input_ids=decoder_input_ids)
                else:
                    hf_output = reference_model(tokens)
                reference_output = hf_output.logits

        return compare_tensors(
            bridge_output,
            reference_output,
            atol=atol,
            rtol=rtol,
            name="forward_pass_logits",
        )

    except Exception as e:
        return BenchmarkResult(
            name="forward_pass",
            severity=BenchmarkSeverity.ERROR,
            message=f"Forward pass failed: {str(e)}",
            passed=False,
        )


def benchmark_loss_equivalence(
    bridge: TransformerBridge,
    test_text: str,
    reference_loss: Optional[float] = None,
    atol: float = 1e-3,
) -> BenchmarkResult:
    """Benchmark loss computation against a pre-computed reference value.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_loss: Optional pre-computed reference loss value (e.g., the HF
            loss captured in Phase 1, or a golden fixture). Self-check only if None.
        atol: Absolute tolerance for comparison

    Returns:
        BenchmarkResult with comparison details
    """
    try:
        bridge_loss = _compute_self_target_loss(bridge, test_text)

        if reference_loss is None:
            # No reference - just verify loss is valid
            if not isinstance(bridge_loss, torch.Tensor):
                return BenchmarkResult(
                    name="loss_equivalence",
                    severity=BenchmarkSeverity.DANGER,
                    message="Bridge loss is not a tensor",
                    passed=False,
                )

            loss_value = bridge_loss.item()
            if torch.isnan(bridge_loss) or torch.isinf(bridge_loss):
                return BenchmarkResult(
                    name="loss_equivalence",
                    severity=BenchmarkSeverity.DANGER,
                    message=f"Bridge loss is invalid: {loss_value}",
                    passed=False,
                )

            return BenchmarkResult(
                name="loss_equivalence",
                severity=BenchmarkSeverity.INFO,
                message=f"Bridge loss computed successfully: {loss_value:.6f}",
                details={"loss": loss_value},
            )

        return compare_scalars(
            bridge_loss.item(),
            reference_loss,
            atol=atol,
            name="loss_equivalence",
        )

    except Exception as e:
        return BenchmarkResult(
            name="loss_equivalence",
            severity=BenchmarkSeverity.ERROR,
            message=f"Loss computation failed: {str(e)}",
            passed=False,
        )


def benchmark_logits_equivalence(
    bridge: TransformerBridge,
    test_text: str,
    reference_logits: Optional[torch.Tensor] = None,
    atol: float = 3e-2,
    rtol: float = 3e-2,
) -> BenchmarkResult:
    """Benchmark logits output against a pre-computed reference tensor.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_logits: Optional pre-computed reference logits tensor (e.g., the
            HF logits captured in Phase 1, or a golden fixture). Self-check only if None.
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        BenchmarkResult with comparison details
    """
    try:
        bridge_logits = bridge(test_text, return_type="logits")

        if reference_logits is None:
            # No reference - just verify logits shape and validity
            if not isinstance(bridge_logits, torch.Tensor):
                return BenchmarkResult(
                    name="logits_equivalence",
                    severity=BenchmarkSeverity.DANGER,
                    message="Bridge logits is not a tensor",
                    passed=False,
                )

            if bridge_logits.numel() == 0:
                return BenchmarkResult(
                    name="logits_equivalence",
                    severity=BenchmarkSeverity.DANGER,
                    message="Bridge logits is empty",
                    passed=False,
                )

            return BenchmarkResult(
                name="logits_equivalence",
                severity=BenchmarkSeverity.INFO,
                message=f"Bridge logits computed successfully (shape: {bridge_logits.shape})",
                details={"output_shape": str(bridge_logits.shape)},
            )

        ref_logits = reference_logits.to(bridge_logits.device)

        return compare_tensors(
            bridge_logits,
            ref_logits,
            atol=atol,
            rtol=rtol,
            name="logits_equivalence",
        )

    except Exception as e:
        return BenchmarkResult(
            name="logits_equivalence",
            severity=BenchmarkSeverity.ERROR,
            message=f"Logits computation failed: {str(e)}",
            passed=False,
        )
