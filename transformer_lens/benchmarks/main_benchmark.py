"""Main benchmark runner for TransformerBridge.

This module provides the main benchmark suite that compares TransformerBridge
against reference implementations in an optimized multi-phase approach:
Phase 1: HF + Bridge (unprocessed) - Compare against raw HuggingFace model
Phase 2: Bridge (unprocessed) + HT (unprocessed) - Compare unprocessed models
Phase 3: Bridge (processed) + HT (processed) - Full compatibility mode testing
Phase 4: Text Quality - Perplexity-based legibility scoring via GPT-2 Medium
Phase 5: Granular Weight Processing Tests (optional, individual flags)
Phase 6: Granular Weight Processing Tests (optional, combined flags)
Phase 7: Multimodal Tests (only for multimodal models with pixel_values support)
"""

import gc
import os
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def _hf_token() -> Optional[str]:
    """Get HuggingFace token from environment for gated model access."""
    return os.environ.get("HF_TOKEN", "") or None


from transformer_lens import HookedTransformer
from transformer_lens.benchmarks.activation_cache import (
    benchmark_activation_cache,
    benchmark_run_with_cache,
)
from transformer_lens.benchmarks.backward_gradients import (
    benchmark_backward_hooks,
    benchmark_critical_backward_hooks,
    benchmark_gradient_computation,
)
from transformer_lens.benchmarks.component_benchmark import benchmark_all_components
from transformer_lens.benchmarks.forward_pass import (
    benchmark_forward_pass,
    benchmark_logits_equivalence,
    benchmark_loss_equivalence,
)
from transformer_lens.benchmarks.generation import (
    benchmark_generation,
    benchmark_generation_with_kv_cache,
    benchmark_multiple_generation_calls,
)
from transformer_lens.benchmarks.hook_registration import (
    benchmark_critical_forward_hooks,
    benchmark_forward_hooks,
    benchmark_hook_functionality,
    benchmark_hook_registry,
)
from transformer_lens.benchmarks.text_quality import benchmark_text_quality
from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    PhaseReferenceData,
    compare_tensors,
    format_results,
)
from transformer_lens.benchmarks.weight_processing import (
    benchmark_attention_output_centering,
    benchmark_layer_norm_folding,
    benchmark_mlp_output_centering,
    benchmark_no_nan_inf,
    benchmark_unembed_centering,
    benchmark_value_bias_folding,
    benchmark_weight_magnitudes,
    benchmark_weight_modification,
    benchmark_weight_processing,
    benchmark_weight_sharing,
)
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge import TransformerBridge

# Architecture names that indicate encoder-decoder models
ENCODER_DECODER_ARCHITECTURES = [
    "T5ForConditionalGeneration",
    "BartForConditionalGeneration",
    "MBartForConditionalGeneration",
    "MT5ForConditionalGeneration",
    "PegasusForConditionalGeneration",
    "BlenderbotForConditionalGeneration",
    "MarianMTModel",
]

# Architecture names that indicate masked language models (not suited for text generation)
MASKED_LM_ARCHITECTURES = [
    "BertForMaskedLM",
    "RobertaForMaskedLM",
    "AlbertForMaskedLM",
    "DistilBertForMaskedLM",
    "ElectraForMaskedLM",
]

# Architectures where the Bridge intentionally uses different hook shapes than
# HookedTransformer. These models skip Phase 2/3 (HT comparison) because the
# Bridge preserves HuggingFace's native tensor layouts for interpretability
# (e.g., 4D [batch, heads, seq, d_head] QK norm hooks) rather than matching
# HT's flattened convention. Phase 1 (HF comparison) remains the gold standard.
NO_HT_COMPARISON_ARCHITECTURES = [
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration",
]


def is_masked_lm_model(model_name: str, trust_remote_code: bool = False) -> bool:
    """Check if a model is a masked language model (not suited for text generation).

    Args:
        model_name: The HuggingFace model name or path
        trust_remote_code: Whether to trust remote code for custom architectures.

    Returns:
        True if the model is a masked LM (like BERT), False otherwise
    """
    try:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, token=_hf_token()
        )
        architectures = getattr(config, "architectures", []) or []
        return any(arch in MASKED_LM_ARCHITECTURES for arch in architectures)
    except Exception:
        return False


def is_encoder_decoder_model(model_name: str, trust_remote_code: bool = False) -> bool:
    """Check if a model is an encoder-decoder architecture.

    Args:
        model_name: The HuggingFace model name or path
        trust_remote_code: Whether to trust remote code for custom architectures.

    Returns:
        True if the model is encoder-decoder (like T5), False otherwise
    """
    try:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, token=_hf_token()
        )
        # Check config attribute first
        if getattr(config, "is_encoder_decoder", False):
            return True
        # Fallback to architecture check
        architectures = getattr(config, "architectures", []) or []
        return any(arch in ENCODER_DECODER_ARCHITECTURES for arch in architectures)
    except Exception:
        return False


def should_skip_ht_comparison(model_name: str, trust_remote_code: bool = False) -> bool:
    """Check if a model's architecture should skip HookedTransformer comparison.

    Some architectures (e.g., Gemma3) intentionally use different hook tensor
    layouts in the Bridge than HookedTransformer. For these models, Phase 1
    (Bridge vs HuggingFace) is the gold standard; Phase 2/3 HT comparisons
    are skipped since shape differences are by design, not bugs.

    Args:
        model_name: The HuggingFace model name or path
        trust_remote_code: Whether to trust remote code for custom architectures.

    Returns:
        True if the model should skip HT comparison, False otherwise
    """
    try:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, token=_hf_token()
        )
        architectures = getattr(config, "architectures", []) or []
        return any(arch in NO_HT_COMPARISON_ARCHITECTURES for arch in architectures)
    except Exception:
        return False


def _is_multimodal_model(model_name: str, trust_remote_code: bool = False) -> bool:
    """Check if a model is a multimodal (vision-language) model."""
    MULTIMODAL_ARCHITECTURES = [
        "LlavaForConditionalGeneration",
        "LlavaNextForConditionalGeneration",
        "LlavaOnevisionForConditionalGeneration",
        "Gemma3ForConditionalGeneration",
    ]
    try:
        config = AutoConfig.from_pretrained(
            model_name, token=_hf_token(), trust_remote_code=trust_remote_code
        )
        architectures = getattr(config, "architectures", []) or []
        return any(arch in MULTIMODAL_ARCHITECTURES for arch in architectures)
    except Exception:
        return False


def get_auto_model_class(model_name: str, trust_remote_code: bool = False):
    """Determine the correct AutoModel class for a given model.

    Some models (like T5) are encoder-decoder and need AutoModelForSeq2SeqLM
    instead of AutoModelForCausalLM. Multimodal models (LLaVA, Gemma3) need
    AutoModel to load the full vision+language architecture.

    Args:
        model_name: The HuggingFace model name or path

    Returns:
        The appropriate AutoModel class
    """
    if is_encoder_decoder_model(model_name, trust_remote_code=trust_remote_code):
        return AutoModelForSeq2SeqLM
    if _is_multimodal_model(model_name, trust_remote_code=trust_remote_code):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def _fixup_custom_model(hf_model) -> None:
    """Apply post-load fixups for models with custom code (e.g., OpenELM).

    Recomputes non-persistent buffers (inv_freq, causal_mask) that may be
    zeroed during HuggingFace's meta-device loading.
    """
    # OpenELM fixups
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "layers"):
        # Ensure use_cache is set (OpenELM custom config omits it)
        if not hasattr(hf_model.config, "use_cache") or "use_cache" not in hf_model.config.__dict__:
            hf_model.config.use_cache = False

        # Fix 1: Always recompute causal_mask (non-persistent buffer).
        # After meta→real materialization, the buffer may contain garbage values
        # rather than clean zeros, so we always recompute.
        if hasattr(hf_model.transformer, "causal_mask"):
            cm = hf_model.transformer.causal_mask
            if cm is not None and cm.numel() > 0:
                seq_len = cm.shape[-1]
                correct_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=cm.dtype, device=cm.device),
                    diagonal=1,
                )
                hf_model.transformer.causal_mask = correct_mask

        # Fix 2: Always recompute RoPE inv_freq and sin/cos (non-persistent buffers).
        rope_max = getattr(hf_model.config, "rope_max_length", None)
        if rope_max is not None:
            for layer in hf_model.transformer.layers:
                if hasattr(layer, "attn") and hasattr(layer.attn, "pos_embedding"):
                    rope = layer.attn.pos_embedding
                    if hasattr(rope, "inv_freq"):
                        correct_inv_freq = 1.0 / (
                            rope.freq_constant
                            ** (
                                torch.arange(0, rope.model_dim, 2, dtype=torch.float32)
                                / rope.model_dim
                            )
                        )
                        rope.inv_freq = correct_inv_freq.to(rope.inv_freq.device)
                    # Force-recompute sin/cos
                    rope._cached_cos = None
                    rope._cached_sin = None
                    rope._compute_sin_cos_embeddings(rope_max)

        # Create synthetic lm_head for weight-tied models (share_input_output_layers)
        if getattr(hf_model, "lm_head", None) is None:
            embed = hf_model.transformer.token_embeddings
            lm_head = torch.nn.Linear(embed.embedding_dim, embed.num_embeddings, bias=False)
            lm_head.weight = embed.weight
            hf_model.lm_head = lm_head


def run_comparison_benchmarks(
    bridge_model: TransformerBridge,
    reference_model: Optional[HookedTransformer],
    test_text: str,
    phase_name: str,
    is_processed: bool,
    verbose: bool = True,
    phase1_reference: Optional[PhaseReferenceData] = None,
    restore_dtype_after_equivalence: Optional[torch.dtype] = None,
) -> List[BenchmarkResult]:
    """Run standardized comparison benchmarks between Bridge and reference model.

    This function runs the same comprehensive test suite for both unprocessed (Phase 2)
    and processed (Phase 3) modes to ensure parity in testing coverage.

    Args:
        bridge_model: TransformerBridge model to test
        reference_model: HookedTransformer reference (same architecture) or None
        test_text: Input text for testing
        phase_name: Name of the phase ("Phase 2" or "Phase 3") for logging
        is_processed: Whether models have processed weights (for weight-specific tests)
        verbose: Whether to print detailed results
        phase1_reference: Optional saved Phase 1 HF reference data for equivalence testing
        restore_dtype_after_equivalence: If set, downcast bridge_model to this dtype after
            the equivalence comparison but before hook/cache/gradient tests. Used when the
            bridge was upcast to float32 for precise equivalence testing.

    Returns:
        List of BenchmarkResult objects
    """
    results: List[BenchmarkResult] = []

    def add_result(result: BenchmarkResult) -> None:
        """Add a result and optionally print it immediately."""
        results.append(result)
        if verbose:
            result.print_immediate()

    # Check if we have a same-architecture reference
    ht_available = reference_model is not None

    # ========================================================================
    # 1. Weight Processing Benchmarks (only for processed mode)
    # MOST BASIC: Check weights are valid before testing anything else
    # ========================================================================
    if is_processed:
        if verbose:
            print("1. Weight Processing Benchmarks (Foundation)")
        try:
            # Critical weight validation tests (run first - most basic)
            add_result(benchmark_no_nan_inf(bridge_model, test_text))
            add_result(benchmark_weight_magnitudes(bridge_model, test_text))

            # Detailed weight processing validation benchmarks (don't need reference model)
            add_result(benchmark_layer_norm_folding(bridge_model, test_text))
            add_result(benchmark_attention_output_centering(bridge_model, test_text))
            add_result(benchmark_mlp_output_centering(bridge_model, test_text))
            add_result(benchmark_unembed_centering(bridge_model, test_text))
            add_result(benchmark_value_bias_folding(bridge_model, test_text))

            # Weight comparison tests (require reference model)
            if ht_available:
                add_result(
                    benchmark_weight_processing(
                        bridge_model, test_text, reference_model=reference_model
                    )
                )
                add_result(
                    benchmark_weight_sharing(
                        bridge_model, test_text, reference_model=reference_model
                    )
                )
            else:
                if verbose:
                    print("⏭️ weight_processing and weight_sharing skipped (no HT reference)")
                for benchmark_name in ["weight_processing", "weight_sharing"]:
                    add_result(
                        BenchmarkResult(
                            name=benchmark_name,
                            severity=BenchmarkSeverity.SKIPPED,
                            message="Skipped (HookedTransformer not available for this model)",
                            passed=True,
                        )
                    )

            # weight_modification doesn't need reference model
            add_result(benchmark_weight_modification(bridge_model, test_text))
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Weight processing benchmark failed: {e}\n")

    # ========================================================================
    # 2. Model Equivalence Benchmarks (Forward Pass)
    # Tests basic forward computation - depends on weights being correct
    # ========================================================================
    if verbose:
        print("2. Model Equivalence Benchmarks (Forward Pass)")

    has_phase1_ref = phase1_reference is not None and phase1_reference.hf_logits is not None

    if ht_available:
        try:
            add_result(
                benchmark_logits_equivalence(
                    bridge_model, test_text, reference_model=reference_model
                )
            )
            add_result(
                benchmark_loss_equivalence(bridge_model, test_text, reference_model=reference_model)
            )
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Equivalence benchmark failed: {e}\n")
    elif has_phase1_ref:
        # Compare processed bridge against unprocessed Phase 1 reference.
        # We use log_softmax because center_unembed shifts raw logits by a
        # softmax-invariant constant. Both passes run in float32 (no bf16 round-trip).
        try:
            if verbose:
                print("Using saved Phase 1 bridge reference for equivalence comparison")

            assert phase1_reference is not None
            assert phase1_reference.hf_logits is not None

            # Compare log_softmax (centering-invariant) instead of raw logits.
            bridge_logits = bridge_model(test_text, return_type="logits")
            ref_logits = phase1_reference.hf_logits.to(bridge_logits.device)
            bridge_log_probs = torch.nn.functional.log_softmax(bridge_logits, dim=-1)
            ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

            # Both passes in float32 — remaining error is float32 non-associativity
            # in weight processing (~0.006 max_diff on 24-layer Qwen2).
            logits_atol = 0.01
            logits_rtol = 1e-4
            loss_atol = 1e-3

            add_result(
                compare_tensors(
                    bridge_log_probs,
                    ref_log_probs,
                    atol=logits_atol,
                    rtol=logits_rtol,
                    name="logits_equivalence",
                )
            )
            if phase1_reference.hf_loss is not None:
                add_result(
                    benchmark_loss_equivalence(
                        bridge_model,
                        test_text,
                        reference_loss=phase1_reference.hf_loss,
                        atol=loss_atol,
                    )
                )
            else:
                add_result(
                    BenchmarkResult(
                        name="loss_equivalence",
                        severity=BenchmarkSeverity.SKIPPED,
                        message="Skipped (no Phase 1 loss reference available)",
                        passed=True,
                    )
                )
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Phase 1 reference comparison failed: {e}\n")
    else:
        if verbose:
            print("⏭️ Skipped (no HookedTransformer reference)\n")
        for benchmark_name in ["logits_equivalence", "loss_equivalence"]:
            add_result(
                BenchmarkResult(
                    name=benchmark_name,
                    severity=BenchmarkSeverity.SKIPPED,
                    message="Skipped (HookedTransformer not available for this model)",
                    passed=True,
                )
            )

    # Restore native dtype so remaining tests run in the model's real dtype.
    # Both bridge and reference must be downcast so hook comparisons use the
    # same precision — otherwise bridge activations (bfloat16) are compared
    # against reference activations (float32), producing spurious mismatches.
    if restore_dtype_after_equivalence is not None:
        try:
            bridge_model.to(restore_dtype_after_equivalence)
            if reference_model is not None:
                reference_model.to(restore_dtype_after_equivalence)
            if verbose:
                print(f"  (restored to {restore_dtype_after_equivalence} for remaining tests)\n")
        except Exception as e:
            if verbose:
                print(f"⚠ Could not restore dtype: {e}\n")

    # ========================================================================
    # 3. Hook Registration Benchmarks
    # Tests hooks exist and are registered - depends on model structure
    # ========================================================================
    if verbose:
        print("3. Hook Registration Benchmarks")

    if ht_available:
        try:
            add_result(benchmark_hook_registry(bridge_model, reference_model=reference_model))
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Hook registry benchmark failed: {e}\n")
    else:
        try:
            add_result(benchmark_hook_registry(bridge_model))
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Hook registry benchmark failed: {e}\n")

    # ========================================================================
    # 4. Forward Hook Functionality Benchmarks
    # Tests hooks fire and produce correct values - depends on forward pass + hooks
    # ========================================================================
    if verbose:
        print("4. Forward Hook Functionality Benchmarks")

    if ht_available:
        try:
            add_result(
                benchmark_hook_functionality(
                    bridge_model, test_text, reference_model=reference_model
                )
            )
            add_result(
                benchmark_critical_forward_hooks(
                    bridge_model, test_text, reference_model=reference_model
                )
            )
            add_result(
                benchmark_forward_hooks(bridge_model, test_text, reference_model=reference_model)
            )
            # Reset hooks to prevent handle leaks
            if hasattr(bridge_model, "reset_hooks"):
                bridge_model.reset_hooks()
            if reference_model is not None and hasattr(reference_model, "reset_hooks"):
                reference_model.reset_hooks()
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Forward hook benchmark failed: {e}\n")
    else:
        try:
            add_result(benchmark_hook_functionality(bridge_model, test_text))
            add_result(benchmark_critical_forward_hooks(bridge_model, test_text))
            add_result(benchmark_forward_hooks(bridge_model, test_text))
            # Reset hooks to prevent handle leaks
            if hasattr(bridge_model, "reset_hooks"):
                bridge_model.reset_hooks()
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Forward hook benchmark failed: {e}\n")

    # ========================================================================
    # 5. Activation Cache Benchmarks
    # Tests caching mechanism - depends on forward pass + hooks working
    # ========================================================================
    if verbose:
        print("5. Activation Cache Benchmarks")

    if ht_available:
        try:
            add_result(
                benchmark_run_with_cache(bridge_model, test_text, reference_model=reference_model)
            )
            add_result(
                benchmark_activation_cache(bridge_model, test_text, reference_model=reference_model)
            )
            # Reset hooks to prevent handle leaks
            if hasattr(bridge_model, "reset_hooks"):
                bridge_model.reset_hooks()
            if reference_model is not None and hasattr(reference_model, "reset_hooks"):
                reference_model.reset_hooks()
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Activation cache benchmark failed: {e}\n")
    else:
        try:
            add_result(benchmark_run_with_cache(bridge_model, test_text))
            add_result(benchmark_activation_cache(bridge_model, test_text))
            # Reset hooks to prevent handle leaks
            if hasattr(bridge_model, "reset_hooks"):
                bridge_model.reset_hooks()
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Activation cache benchmark failed: {e}\n")

    # ========================================================================
    # 6. Backward Gradient Benchmarks
    # MOST COMPLEX: Tests gradients and backward hooks - depends on everything above
    # ========================================================================
    if verbose:
        print("6. Backward Gradient Benchmarks")

    # MPS does not support bfloat16 autograd. Upcast to float32 for gradient tests if needed.
    bridge_grad_dtype = bridge_model.cfg.dtype if hasattr(bridge_model, "cfg") else None
    bridge_device = next(bridge_model.parameters()).device
    mps_bf16_upcast = str(bridge_device).startswith("mps") and bridge_grad_dtype == torch.bfloat16
    if mps_bf16_upcast:
        try:
            bridge_model.to(torch.float32)
            if reference_model is not None:
                reference_model.to(torch.float32)
        except Exception:
            mps_bf16_upcast = False  # Upcast failed; proceed as-is

    if ht_available:
        try:
            add_result(
                benchmark_gradient_computation(
                    bridge_model, test_text, reference_model=reference_model
                )
            )
            add_result(
                benchmark_critical_backward_hooks(
                    bridge_model, test_text, reference_model=reference_model
                )
            )
            add_result(
                benchmark_backward_hooks(bridge_model, test_text, reference_model=reference_model)
            )
            # Reset hooks to prevent handle leaks
            if hasattr(bridge_model, "reset_hooks"):
                bridge_model.reset_hooks()
            if reference_model is not None and hasattr(reference_model, "reset_hooks"):
                reference_model.reset_hooks()
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Gradient benchmark failed: {e}\n")
    else:
        try:
            add_result(benchmark_gradient_computation(bridge_model, test_text))
            add_result(benchmark_critical_backward_hooks(bridge_model, test_text))
            add_result(benchmark_backward_hooks(bridge_model, test_text))
            # Reset hooks to prevent handle leaks
            if hasattr(bridge_model, "reset_hooks"):
                bridge_model.reset_hooks()
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Gradient benchmark failed: {e}\n")

    if mps_bf16_upcast and bridge_grad_dtype is not None:
        try:
            bridge_model.to(bridge_grad_dtype)
            if reference_model is not None:
                reference_model.to(bridge_grad_dtype)
        except Exception:
            pass

    return results


def run_benchmark_suite(
    model_name: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    test_text: Optional[str] = None,
    use_hf_reference: bool = True,
    use_ht_reference: bool = True,
    enable_compatibility_mode: bool = True,
    verbose: bool = True,
    track_memory: bool = False,
    test_weight_processing_individually: bool = False,
    phases: list[int] | None = None,
    trust_remote_code: bool = False,
    conserve_memory: bool = False,
    scoring_model: PreTrainedModel | None = None,
    scoring_tokenizer: PreTrainedTokenizerBase | None = None,
) -> List[BenchmarkResult]:
    """Run comprehensive benchmark suite for TransformerBridge.

    This function implements an optimized multi-phase approach to minimize model reloading:
    Phase 1: HF + Bridge (unprocessed) - Compare against raw HuggingFace model
    Phase 2: Bridge (unprocessed) + HT (unprocessed) - Compare unprocessed models
    Phase 3: Bridge (processed) + HT (processed) - Full compatibility mode testing
    Phase 4: Text Quality - Perplexity-based legibility scoring via GPT-2
    Phase 5: Individual Weight Processing Flags (optional)
    Phase 6: Combined Weight Processing Flags (optional)

    When test_weight_processing_individually=True, Phases 5 & 6 run after
    Phase 3, testing each weight processing flag individually and in combinations.

    Args:
        model_name: Name of the model to benchmark (e.g., "gpt2")
        device: Device to run on ("cpu" or "cuda")
        dtype: Precision for model loading (default: torch.float32). Use
            torch.bfloat16 to halve memory for larger models. Phase 2/3
            comparisons automatically upcast to float32 for precision.
        test_text: Optional test text (default: standard test prompt)
        use_hf_reference: Whether to compare against HuggingFace model
        use_ht_reference: Whether to compare against HookedTransformer
        enable_compatibility_mode: Whether to enable compatibility mode on bridge
        verbose: Whether to print results to console
        track_memory: Whether to track and report memory usage (requires psutil)
        test_weight_processing_individually: Whether to run granular weight processing
            tests that check each processing flag individually (default: False)
        phases: Optional list of phase numbers to run (e.g., [1, 2, 3]). If None, runs all phases.
        trust_remote_code: Whether to trust remote code for custom architectures.
        conserve_memory: When True, Phase 1 avoids loading a separate HF model
            and instead uses bridge.original_model for component benchmarks and
            forward pass comparison. This halves Phase 1 peak memory (1.0x vs 2.0x)
            at the cost of losing the independent HF loading cross-check (~5%
            weakening). Default is False (full dual-load for maximum test coverage).
        scoring_model: Optional pre-loaded GPT-2 scoring model for Phase 4. When
            provided with scoring_tokenizer, avoids reloading for each model in batch.
        scoring_tokenizer: Optional pre-loaded tokenizer for Phase 4 scoring model.

    Returns:
        List of BenchmarkResult objects
    """
    if test_text is None:
        test_text = (
            "Natural language processing tasks, such as question answering, "
            "machine translation, reading comprehension, and summarization, "
            "are typically approached with supervised learning."
        )

    results: List[BenchmarkResult] = []

    # Memory tracking setup
    memory_tracker = None
    if track_memory:
        try:
            import psutil

            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            def get_memory_mb():
                return process.memory_info().rss / 1024 / 1024

            memory_tracker = {"initial": initial_memory, "checkpoints": []}
            if verbose:
                print(f"Memory tracking enabled (initial: {initial_memory:.1f} MB)")
        except ImportError:
            if verbose:
                print("⚠ psutil not available - memory tracking disabled")
            track_memory = False

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running TransformerBridge Benchmark Suite")
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"{'='*80}\n")

    # Auto-skip HT comparison for architectures with intentionally different hook shapes
    if use_ht_reference and should_skip_ht_comparison(model_name, trust_remote_code):
        use_ht_reference = False
        if verbose:
            print(
                "Note: Skipping HookedTransformer comparison (architecture uses "
                "different hook shapes by design). Phase 1 is the gold standard.\n"
            )

    # Early exit if only running Phase 5/6 (they load their own models independently)
    if phases is not None and all(p in [5, 6] for p in phases):
        if verbose:
            print(f"Skipping Phase 1-4 (only running Phase {', '.join(map(str, sorted(phases)))})")
            print("Phase 5/6 load their own models independently\n")

        from transformer_lens.benchmarks.granular_weight_processing import (
            run_granular_weight_processing_benchmarks,
        )

        if 5 in phases and test_weight_processing_individually and enable_compatibility_mode:
            phase5_results = run_granular_weight_processing_benchmarks(
                model_name=model_name,
                device=device,
                test_text=test_text,
                verbose=verbose,
                phase=5,
            )
            for config_name, config_results in phase5_results.items():
                for result in config_results:
                    result.phase = 5
                    results.append(result)
                    if verbose:
                        result.print_immediate()

        if 6 in phases and test_weight_processing_individually and enable_compatibility_mode:
            phase6_results = run_granular_weight_processing_benchmarks(
                model_name=model_name,
                device=device,
                test_text=test_text,
                verbose=verbose,
                phase=6,
            )
            for config_name, config_results in phase6_results.items():
                for result in config_results:
                    result.phase = 6
                    results.append(result)
                    if verbose:
                        result.print_immediate()

        return results

    # Track current phase for result tagging
    current_phase: List[Optional[int]] = [None]  # Use list to allow modification in nested function

    def should_run_phase(phase_num: int) -> bool:
        """Check if a phase should run based on the phases filter."""
        return phases is None or phase_num in phases

    def add_result(result: BenchmarkResult) -> None:
        """Add a result and optionally print it immediately."""
        # Tag result with current phase
        if current_phase[0] is not None and result.phase is None:
            result.phase = current_phase[0]
        results.append(result)
        if verbose:
            result.print_immediate()

    def cleanup_tensors(*tensors) -> None:
        """Free memory from tensors and caches."""
        for tensor in tensors:
            if tensor is not None:
                # If it's an ActivationCache, clear all tensors
                if hasattr(tensor, "cache_dict"):
                    for key in list(tensor.cache_dict.keys()):
                        val = tensor.cache_dict[key]
                        if val is not None and isinstance(val, torch.Tensor):
                            del val
                        tensor.cache_dict[key] = None
                    tensor.cache_dict.clear()
                # If it's a regular tensor, just delete it
                elif isinstance(tensor, torch.Tensor):
                    del tensor
        # Force cleanup
        gc.collect()
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.synchronize()
            torch.mps.empty_cache()

    def cleanup_model(model, model_name_str: str):
        """Free up memory by deleting a model and forcing garbage collection."""
        import gc

        if verbose:
            print(f"Cleaning up {model_name_str}...")

        # Track memory before cleanup
        if track_memory and memory_tracker is not None:
            memory_before = get_memory_mb()

        # Move model to CPU first to free GPU memory immediately
        if device != "cpu" and hasattr(model, "cpu"):
            try:
                model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
            except Exception:
                pass

        # Explicitly remove all hooks to prevent memory leaks
        if hasattr(model, "modules"):
            try:
                for module in model.modules():
                    # Clear PyTorch hooks
                    if hasattr(module, "_forward_hooks"):
                        module._forward_hooks.clear()
                    if hasattr(module, "_backward_hooks"):
                        module._backward_hooks.clear()
                    if hasattr(module, "_forward_pre_hooks"):
                        module._forward_pre_hooks.clear()
                    if hasattr(module, "_backward_pre_hooks"):
                        module._backward_pre_hooks.clear()
                    if hasattr(module, "_state_dict_hooks"):
                        module._state_dict_hooks.clear()
                    if hasattr(module, "_state_dict_pre_hooks"):
                        module._state_dict_pre_hooks.clear()
                    if hasattr(module, "_load_state_dict_pre_hooks"):
                        module._load_state_dict_pre_hooks.clear()
                    if hasattr(module, "_load_state_dict_post_hooks"):
                        module._load_state_dict_post_hooks.clear()

                    # Clear TransformerLens-specific hooks
                    if hasattr(module, "remove_all_hooks"):
                        module.remove_all_hooks()

                    # Clear gradients
                    if hasattr(module, "zero_grad"):
                        try:
                            module.zero_grad(set_to_none=True)
                        except Exception:
                            pass
            except Exception:
                # If hook cleanup fails, continue anyway
                pass

        # Clear top-level hooks
        if hasattr(model, "_forward_hooks"):
            model._forward_hooks.clear()
        if hasattr(model, "_backward_hooks"):
            model._backward_hooks.clear()
        if hasattr(model, "_forward_pre_hooks"):
            model._forward_pre_hooks.clear()

        # Clear top-level gradients
        if hasattr(model, "zero_grad"):
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass

        # Break circular references to help GC
        if hasattr(model, "_modules"):
            # Clear each submodule's __dict__ to break circular references
            for name, submodule in list(model._modules.items()):
                if submodule is not None:
                    # Clear submodule hooks
                    if hasattr(submodule, "_forward_hooks"):
                        submodule._forward_hooks.clear()
                    if hasattr(submodule, "_backward_hooks"):
                        submodule._backward_hooks.clear()
                    # Break reference
                    model._modules[name] = None
            model._modules.clear()

        # Clear parameters dict
        if hasattr(model, "_parameters"):
            for param_name in list(model._parameters.keys()):
                param = model._parameters[param_name]
                if param is not None:
                    del param
                model._parameters[param_name] = None
            model._parameters.clear()

        # Clear buffers dict
        if hasattr(model, "_buffers"):
            for buffer_name in list(model._buffers.keys()):
                buffer = model._buffers[buffer_name]
                if buffer is not None:
                    del buffer
                model._buffers[buffer_name] = None
            model._buffers.clear()

        del model

        # Aggressive garbage collection (multiple passes to break circular references)
        for _ in range(3):
            gc.collect()

        # Clear GPU cache
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.synchronize()
            torch.mps.empty_cache()

        # Track memory after cleanup
        if track_memory and memory_tracker is not None:
            memory_after = get_memory_mb()
            freed_mb = memory_before - memory_after
            memory_tracker["checkpoints"].append(
                {
                    "label": f"Cleanup: {model_name_str}",
                    "memory_mb": memory_after,
                    "freed_mb": freed_mb,
                }
            )
            if verbose and freed_mb > 0:
                print(f"  Freed {freed_mb:.1f} MB")

    # ========================================================================
    # PHASE 1: HuggingFace + Bridge (unprocessed)
    # ========================================================================
    current_phase[0] = 1
    if verbose:
        print(f"\n{'='*80}")
        print("PHASE 1: HuggingFace + TransformerBridge (unprocessed)")
        print(f"{'='*80}\n")

    bridge_unprocessed = None
    hf_model = None
    phase1_reference = PhaseReferenceData()

    # Load bridge without weights first to detect attn_implementation and dtype
    if verbose:
        print("Detecting model configuration...")
    bridge_dtype = dtype
    attn_implementation = None
    try:
        # Load a lightweight version without weights to get config
        bridge_config_only = TransformerBridge.boot_transformers(model_name, device=device, dtype=bridge_dtype, load_weights=False, trust_remote_code=trust_remote_code)  # type: ignore[attr-defined]
        # Match bridge's attn_implementation: check adapter config first, then
        # default to "eager" (bridge uses output_attentions=True which forces eager).
        if hasattr(bridge_config_only.adapter.cfg, "attn_implementation"):
            attn_implementation = bridge_config_only.adapter.cfg.attn_implementation
        if attn_implementation is None:
            attn_implementation = "eager"
        if verbose:
            print(f"✓ Detected attn_implementation={attn_implementation}")
        # Clean up config-only bridge immediately to free memory
        del bridge_config_only
        gc.collect()  # Force garbage collection immediately
    except Exception as e:
        if verbose:
            print(f"⚠ Could not detect config (will use defaults): {str(e)}")
        # Config-only bridge failed; apply architecture patches directly to prevent
        # _init_weights from re-randomizing loaded weights.
        if trust_remote_code:
            try:
                from transformer_lens.model_bridge.sources.transformers import (
                    determine_architecture_from_hf_config,
                    map_default_transformer_lens_config,
                )

                hf_cfg = AutoConfig.from_pretrained(
                    model_name, trust_remote_code=True, token=_hf_token()
                )
                tl_cfg = map_default_transformer_lens_config(hf_cfg)
                arch = determine_architecture_from_hf_config(hf_cfg)
                bridge_cfg = TransformerBridgeConfig.from_dict(tl_cfg.__dict__)
                bridge_cfg.architecture = arch
                bridge_cfg.model_name = model_name
                adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_cfg)
                adapter.prepare_loading(model_name, {})
                if verbose:
                    print("✓ Applied architecture patches for custom code model")
                del adapter, bridge_cfg, tl_cfg, hf_cfg
            except Exception as patch_err:
                if verbose:
                    print(f"⚠ Could not apply architecture patches: {patch_err}")

    # ----------------------------------------------------------------
    # Phase 1 memory strategy (controlled by conserve_memory flag):
    #
    # conserve_memory=False (default):
    #   Load separate HF model, capture logits to CPU, load Bridge,
    #   run component benchmark with both models (brief 2.0x), delete
    #   HF immediately after, forward pass uses saved logits (1.0x).
    #
    # conserve_memory=True:
    #   Skip separate HF model entirely.  Load Bridge only (1.0x
    #   throughout).  Component benchmark uses bridge.original_model
    #   as the HF reference.  Forward pass compares bridge output
    #   against bridge.original_model logits.
    # ----------------------------------------------------------------
    hf_saved_logits = None
    hf_saved_loss = None

    if use_hf_reference and not conserve_memory and should_run_phase(1):
        try:
            if verbose:
                print("Loading HuggingFace reference model...")
            # Match bridge loading path: no device_map, explicit .to(device),
            # and matching torch_dtype.  When dtype=float32, loading in float32
            # ensures non-persistent buffers (e.g., Gemma3's embed_scale) are
            # computed at full precision.  When dtype=bfloat16, both HF and
            # Bridge load in bfloat16 so comparisons are apples-to-apples.
            hf_kwargs: dict[str, object] = {
                "low_cpu_mem_usage": True,  # Reduce memory spikes during loading
                "torch_dtype": dtype,
            }
            if _hf_token():
                hf_kwargs["token"] = _hf_token()
            if attn_implementation is not None:
                hf_kwargs["attn_implementation"] = attn_implementation
                if verbose:
                    print(f"Using attn_implementation={attn_implementation}")
            # Use appropriate AutoModel class (e.g., AutoModelForSeq2SeqLM for T5)
            auto_model_class = get_auto_model_class(model_name, trust_remote_code=trust_remote_code)
            if verbose and auto_model_class != AutoModelForCausalLM:
                print(f"Using {auto_model_class.__name__} for encoder-decoder model")
            # Ensure pad_token_id exists (some models crash without it during init).
            hf_config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, token=_hf_token()
            )
            if not hasattr(hf_config, "pad_token_id") or "pad_token_id" not in hf_config.__dict__:
                hf_config.pad_token_id = getattr(hf_config, "eos_token_id", None)
                hf_kwargs["config"] = hf_config
            if trust_remote_code:
                hf_kwargs["trust_remote_code"] = True
            hf_model = auto_model_class.from_pretrained(model_name, **hf_kwargs)  # type: ignore[arg-type]
            hf_model = hf_model.to(device)
            # Post-load fixup for custom code models (e.g., OpenELM).
            # Must run AFTER .to(device) so non-persistent buffers (RoPE sin/cos,
            # causal_mask) are recomputed on the target device, matching the bridge
            # which also recomputes after .to(device).
            _fixup_custom_model(hf_model)
            hf_model.eval()
            # Detect dtype from HF model
            try:
                bridge_dtype = next(hf_model.parameters()).dtype
                if verbose:
                    print(f"Detected dtype={bridge_dtype}")
            except StopIteration:
                pass
            # When float32 was requested but the model natively uses reduced
            # precision, upcast for maximum benchmark accuracy.  When dtype was
            # explicitly set to bfloat16/float16 (e.g., to fit larger models in
            # memory), respect it — both HF and Bridge will run in that precision.
            if dtype == torch.float32 and bridge_dtype in (torch.float16, torch.bfloat16):
                if verbose:
                    print(f"⚠ {bridge_dtype} detected, upcasting to float32 for benchmarking...")
                hf_model.to(torch.float32)
                bridge_dtype = torch.float32
                if verbose:
                    print("✓ Upcast to float32 in-place")
            elif bridge_dtype != dtype:
                bridge_dtype = dtype  # Trust the requested dtype
            if verbose:
                print("✓ HuggingFace model loaded")

            # HF reference logits will be captured AFTER the bridge is
            # loaded so we can use bridge.to_tokens() for consistent
            # tokenization (e.g. BOS prepending).  This happens right
            # after the component benchmark, while both models are still
            # in memory, before the HF model is deleted.

        except Exception as e:
            if verbose:
                print(f"✗ Could not load HuggingFace model: {str(e)}\n")

    # Now load the full bridge with correct dtype (GPU is mostly free)
    if verbose:
        print("Loading TransformerBridge (unprocessed)...")
    try:
        bridge_unprocessed = TransformerBridge.boot_transformers(model_name, device=device, dtype=bridge_dtype, trust_remote_code=trust_remote_code)  # type: ignore[attr-defined]
        if verbose:
            print("✓ TransformerBridge loaded (unprocessed)\n")
        # Apply the adapter's prepare_model() to the HF reference model so
        # both bridge and reference have the same fixups (e.g., weight tying).
        # This keeps model-specific logic in the adapter, not the benchmark.
        if hf_model is not None and hasattr(bridge_unprocessed, "adapter"):
            bridge_unprocessed.adapter.prepare_model(hf_model)
    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        add_result(
            BenchmarkResult(
                name="load_bridge_unprocessed",
                severity=BenchmarkSeverity.ERROR,
                message=f"Failed to load unprocessed TransformerBridge: {str(e)}",
                passed=False,
            )
        )
        if verbose:
            print(f"✗ Failed to load TransformerBridge: {str(e)}")
            print(f"\nStack trace:\n{error_trace}")
        return results

    # Run Phase 1 benchmarks
    if should_run_phase(1) and bridge_unprocessed:
        if verbose:
            mode_label = " [conserve-memory]" if conserve_memory else ""
            print(f"Running Phase 1 benchmarks{mode_label}...\n")

        # Component-level benchmarks
        if verbose:
            print("1. Component-Level Benchmarks")
        if conserve_memory:
            # conserve_memory mode: use bridge.original_model as the HF
            # reference (no separate HF load, 1.0x peak throughout).
            try:
                component_result = benchmark_all_components(
                    bridge_unprocessed, bridge_unprocessed.original_model
                )
                add_result(component_result)
                if verbose:
                    status = "✓" if component_result.passed else "✗"
                    print(f"{status} {component_result.message}")
                    print("  (reference: bridge.original_model)\n")
            except Exception as e:
                if verbose:
                    print(f"✗ Component benchmark failed: {e}\n")
        elif hf_model is not None:
            # Full mode: component benchmark with independent HF model (brief 2.0x)
            try:
                component_result = benchmark_all_components(bridge_unprocessed, hf_model)
                add_result(component_result)
                if verbose:
                    status = "✓" if component_result.passed else "✗"
                    print(f"{status} {component_result.message}\n")
                gc.collect()
                if device != "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
            except Exception as e:
                if verbose:
                    print(f"✗ Component benchmark failed: {e}\n")

            # Capture HF reference logits using bridge.to_tokens() for
            # consistent tokenization (BOS prepending, etc.).  Both models
            # are still in memory so this is still within the 2.0x window.
            if verbose:
                print("Capturing HF reference outputs to CPU...")
            try:
                hf_tokens = bridge_unprocessed.to_tokens(test_text)
                is_enc_dec = is_encoder_decoder_model(
                    model_name, trust_remote_code=trust_remote_code
                )
                with torch.no_grad():
                    if is_enc_dec:
                        decoder_start_id = getattr(
                            getattr(hf_model, "config", None),
                            "decoder_start_token_id",
                            0,
                        )
                        dec_ids = torch.tensor([[decoder_start_id]]).to(hf_tokens.device)
                        hf_out = hf_model(hf_tokens, decoder_input_ids=dec_ids)
                    else:
                        hf_out = hf_model(hf_tokens)
                    hf_saved_logits = hf_out.logits.detach().cpu().clone()

                    # Compute causal LM loss (shift logits and labels)
                    if not is_enc_dec and hf_saved_logits.shape[1] > 1:
                        shift_logits = hf_out.logits[..., :-1, :].contiguous()
                        shift_labels = hf_tokens[..., 1:].contiguous()
                        loss_fn = torch.nn.CrossEntropyLoss()
                        hf_saved_loss = loss_fn(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        ).item()

                if verbose:
                    loss_str = f"{hf_saved_loss:.4f}" if hf_saved_loss is not None else "N/A"
                    print(f"✓ Captured HF logits {hf_saved_logits.shape}, " f"loss={loss_str}\n")
                del hf_tokens
            except Exception as e:
                if verbose:
                    print(f"⚠ Could not capture HF reference outputs: {e}\n")

            # Delete HF model immediately after component benchmark + logit capture.
            # From here on, Phase 1 runs at 1.0x using saved HF tensors.
            cleanup_model(hf_model, "HuggingFace model")
            hf_model = None
        else:
            if verbose:
                print("⏭️ Skipped (no HF reference model available)\n")

        # Forward pass benchmarks
        if verbose:
            print("2. Forward Pass Benchmarks")

        # Widen tolerance for reduced-precision benchmarking — MPS bfloat16
        # matmul non-determinism can exceed the float32 default of 1e-3
        p1_atol = 1e-3 if dtype == torch.float32 else 5e-3

        if conserve_memory:
            # conserve_memory mode: capture reference logits from
            # bridge.original_model (same tokenization as bridge).
            try:
                tokens = bridge_unprocessed.to_tokens(test_text)
                with torch.no_grad():
                    hf_out = bridge_unprocessed.original_model(tokens)
                    ref_logits = hf_out.logits.detach()
                add_result(
                    benchmark_forward_pass(
                        bridge_unprocessed,
                        test_text,
                        reference_logits=ref_logits,
                        atol=p1_atol,
                    )
                )
                del ref_logits
            except Exception as e:
                if verbose:
                    print(f"✗ Forward pass benchmark failed: {e}\n")
        elif hf_saved_logits is not None:
            # Full mode: use pre-captured HF logits (bridge only, 1.0x)
            try:
                add_result(
                    benchmark_forward_pass(
                        bridge_unprocessed,
                        test_text,
                        reference_logits=hf_saved_logits.to(device),
                        atol=p1_atol,
                    )
                )
            except Exception as e:
                if verbose:
                    print(f"✗ Forward pass benchmark failed: {e}\n")
        else:
            try:
                add_result(benchmark_forward_pass(bridge_unprocessed, test_text, atol=p1_atol))
            except Exception as e:
                if verbose:
                    print(f"✗ Forward pass benchmark failed: {e}\n")

        # Capture Phase 1 reference for Phase 3 equivalence comparison.
        # When dtype==float32 (default) and the model natively uses reduced
        # precision, upcast for maximum accuracy.  When the user explicitly
        # requested a non-float32 dtype, run the reference pass in that dtype
        # so the entire pipeline honours the requested precision.
        if bridge_unprocessed is not None:
            try:
                original_dtype = bridge_unprocessed.cfg.dtype
                needs_upcast = dtype == torch.float32 and original_dtype not in (
                    torch.float32,
                    torch.float64,
                )
                # Snapshot registered buffers before the round-trip.  HF's
                # RotaryEmbedding recomputes inv_freq during the float32 forward
                # pass, and the downcast back to bfloat16 would produce different
                # values than the original, corrupting the model for Phase 2.
                saved_buffers = {}
                if needs_upcast:
                    for bname, buf in bridge_unprocessed.named_buffers():
                        saved_buffers[bname] = buf.data.clone()
                    bridge_unprocessed.to(torch.float32)
                with torch.no_grad():
                    bridge_logits = bridge_unprocessed(test_text, return_type="logits")
                    phase1_reference.hf_logits = bridge_logits.detach().cpu().clone()
                    bridge_loss = bridge_unprocessed(test_text, return_type="loss")
                    phase1_reference.hf_loss = bridge_loss.item()
                    phase1_reference.test_text = test_text
                if needs_upcast:
                    bridge_unprocessed.to(original_dtype)
                    # Restore buffers that were corrupted by the round-trip.
                    # Use direct assignment (not copy_) to preserve original dtype.
                    # HF's RotaryEmbedding keeps inv_freq in float32 even when the
                    # model is bfloat16.  After to(bfloat16), the buffer becomes
                    # bfloat16, and copy_() would truncate the float32 saved values.
                    for bname, buf in bridge_unprocessed.named_buffers():
                        if bname in saved_buffers:
                            buf.data = saved_buffers[bname]
                if verbose:
                    dtype_note = " (upcast to float32)" if needs_upcast else ""
                    print(
                        f"✓ Saved Phase 1 reference data "
                        f"(logits: {phase1_reference.hf_logits.shape}){dtype_note}"
                    )
            except Exception as e:
                if verbose:
                    print(f"⚠ Could not save Phase 1 reference data: {e}")

    # Free saved HF tensors now that Phase 1 is done
    del hf_saved_logits, hf_saved_loss

    # Save bridge_dtype before potential cleanup (needed for Phase 3)
    saved_bridge_dtype = bridge_dtype

    # Clean up HF model if still alive (e.g., Phase 1 was skipped)
    if hf_model is not None:
        cleanup_model(hf_model, "HuggingFace model")
        hf_model = None

    # ========================================================================
    # PHASE 2: Bridge (unprocessed) + HookedTransformer (unprocessed)
    # ========================================================================
    current_phase[0] = 2
    if verbose:
        print(f"\n{'='*80}")
        print("PHASE 2: TransformerBridge (unprocessed) + HookedTransformer (unprocessed)")
        print(f"{'='*80}\n")

    # OPTIMIZATION: Run generation benchmarks first (only bridge in memory)
    # Then cleanup bridge before loading HT to reduce peak memory
    if should_run_phase(2) and bridge_unprocessed:
        if verbose:
            print("Running Phase 2 benchmarks...\n")

        # Generation benchmarks (unprocessed only) - RUN FIRST
        # Skip for encoder-decoder models (T5, etc.) which require different generation API
        is_enc_dec = is_encoder_decoder_model(model_name)
        if verbose:
            print("1. Generation Benchmarks (unprocessed)")
        if is_enc_dec:
            if verbose:
                print("⏭️ Skipped (encoder-decoder model - requires decoder_input_ids)\n")
            add_result(
                BenchmarkResult(
                    name="generation",
                    severity=BenchmarkSeverity.INFO,
                    passed=True,
                    message="Skipped (encoder-decoder model)",
                )
            )
            add_result(
                BenchmarkResult(
                    name="generation_with_kv_cache",
                    severity=BenchmarkSeverity.INFO,
                    passed=True,
                    message="Skipped (encoder-decoder model)",
                )
            )
            add_result(
                BenchmarkResult(
                    name="multiple_generation_calls",
                    severity=BenchmarkSeverity.INFO,
                    passed=True,
                    message="Skipped (encoder-decoder model)",
                )
            )
            add_result(
                BenchmarkResult(
                    name="text_quality",
                    severity=BenchmarkSeverity.INFO,
                    passed=True,
                    message="Skipped (encoder-decoder model)",
                )
            )
        else:
            try:
                add_result(benchmark_generation(bridge_unprocessed, test_text, max_new_tokens=10))
                add_result(
                    benchmark_generation_with_kv_cache(
                        bridge_unprocessed, test_text, max_new_tokens=10
                    )
                )
                add_result(
                    benchmark_multiple_generation_calls(
                        bridge_unprocessed,
                        test_prompts=[
                            "The quick brown fox",
                            "Hello world",
                            "Machine learning is",
                        ],
                        max_new_tokens=5,
                    )
                )
                gc.collect()  # Force cleanup after generation benchmarks
            except Exception as e:
                if verbose:
                    print(f"✗ Generation benchmark failed: {e}\n")

    # Match bridge's default_prepend_bos setting in HookedTransformer.
    ht_prepend_bos = None
    if bridge_unprocessed is not None and hasattr(bridge_unprocessed, "cfg"):
        bridge_bos = getattr(bridge_unprocessed.cfg, "default_prepend_bos", None)
        if bridge_bos is not None:
            ht_prepend_bos = bridge_bos

    # Load HookedTransformer for comparison (after generation benchmarks)
    ht_model_unprocessed = None
    if should_run_phase(2) and use_ht_reference:
        try:
            if verbose:
                print("Loading HookedTransformer (unprocessed) for comparison...")
            ht_model_unprocessed = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                dtype=bridge_dtype,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                fold_value_biases=False,
                refactor_factored_attn_matrices=False,
                default_prepend_bos=ht_prepend_bos,
            )
            if verbose:
                print("✓ HookedTransformer loaded (unprocessed)\n")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load unprocessed HookedTransformer: {str(e)}\n")

    # Run Phase 2 comparison benchmarks using unified function
    if should_run_phase(2) and bridge_unprocessed:
        if verbose:
            print("2. Running Unprocessed Model Comparison Benchmarks\n")

        # When dtype==float32 (default) but the model natively loaded in
        # reduced precision, upcast for maximum benchmark accuracy.  When the
        # user explicitly requested bfloat16/float16, honour that — run the
        # entire comparison in the requested precision.
        phase2_restore_dtype = None
        if dtype == torch.float32 and bridge_dtype in (torch.bfloat16, torch.float16):
            try:
                bridge_unprocessed.to(torch.float32)
                if ht_model_unprocessed is not None:
                    ht_model_unprocessed.to(torch.float32)
                phase2_restore_dtype = bridge_dtype
                if verbose:
                    print(f"  (upcast from {bridge_dtype} to float32 for comparison)\n")
            except Exception:
                phase2_restore_dtype = None  # Upcast failed; proceed as-is

        phase2_results = run_comparison_benchmarks(
            bridge_model=bridge_unprocessed,
            reference_model=ht_model_unprocessed,
            test_text=test_text,
            phase_name="Phase 2",
            is_processed=False,  # Unprocessed mode - skip weight processing tests
            verbose=verbose,
            restore_dtype_after_equivalence=phase2_restore_dtype,
        )
        # Tag all phase 2 results with phase number
        for result in phase2_results:
            if result.phase is None:
                result.phase = 2
        results.extend(phase2_results)

        # Generation benchmarks already run above (before loading HT)

    # Clean up unprocessed HT model - no longer needed
    if ht_model_unprocessed is not None:
        cleanup_model(ht_model_unprocessed, "HookedTransformer (unprocessed)")
        ht_model_unprocessed = None
    # bridge_unprocessed is kept alive for Phase 3 and Phase 4 — reusing the
    # same instance avoids non-deterministic loading in some architectures
    # (e.g., OpenELM).

    # ========================================================================
    # PHASE 4: Text Quality (GPT-2 perplexity scoring)
    # Runs before Phase 3 so it can reuse bridge_unprocessed (Phase 3
    # destructively processes the weights, consuming the bridge).
    # ========================================================================
    current_phase[0] = 4

    if (
        should_run_phase(4)
        and bridge_unprocessed is not None
        and not is_masked_lm_model(model_name, trust_remote_code=trust_remote_code)
    ):
        if verbose:
            print(f"\n{'='*80}")
            print("PHASE 2.5: Text Quality (GPT-2 perplexity scoring)")
            print(f"{'='*80}\n")

        try:
            text_quality_result = benchmark_text_quality(
                bridge_unprocessed,
                test_text,
                max_new_tokens=50,
                scoring_model_name="gpt2",
                pass_threshold=85.0,
                device=device,
                scoring_model=scoring_model,
                scoring_tokenizer=scoring_tokenizer,
            )
            text_quality_result.phase = 4
            add_result(text_quality_result)
        except Exception as e:
            if verbose:
                print(f"✗ Text quality benchmark failed: {e}\n")

    # ========================================================================
    # Phase 7: Multimodal Tests (only for multimodal models)
    # Runs before Phase 3 so we can reuse bridge_unprocessed before cleanup.
    # ========================================================================
    if (
        bridge_unprocessed is not None
        and getattr(bridge_unprocessed.cfg, "is_multimodal", False)
        and should_run_phase(7)
    ):
        current_phase[0] = 7
        if verbose:
            print("\n" + "=" * 80)
            print("PHASE 7: MULTIMODAL TESTS")
            print("=" * 80)
            print("Testing multimodal forward pass, generation, and caching with images.")
            print("=" * 80 + "\n")

        try:
            from transformer_lens.benchmarks.multimodal import (
                benchmark_multimodal_cache,
                benchmark_multimodal_forward,
                benchmark_multimodal_generation,
            )

            mm_results = [
                benchmark_multimodal_forward(bridge_unprocessed, test_text=test_text),
                benchmark_multimodal_generation(bridge_unprocessed, test_text=test_text),
                benchmark_multimodal_cache(bridge_unprocessed, test_text=test_text),
            ]
            for result in mm_results:
                result.phase = 7
                results.append(result)
                if verbose:
                    print(result)

            if verbose:
                print("\n" + "=" * 80)
                print("PHASE 7 COMPLETE")
                print("=" * 80)

        except Exception as e:
            if verbose:
                print(f"\n⚠ Multimodal tests failed: {e}\n")
            results.append(
                BenchmarkResult(
                    name="multimodal_suite",
                    passed=False,
                    severity=BenchmarkSeverity.ERROR,
                    message=f"Failed to run multimodal tests: {str(e)}",
                    details={"error": str(e)},
                    phase=7,
                )
            )

    # ========================================================================
    # PHASE 3: Bridge (processed) + HookedTransformer (processed)
    # ========================================================================
    current_phase[0] = 3

    def _cleanup_bridge_unprocessed():
        """Clean up the kept-alive bridge_unprocessed if Phase 3 is skipped."""
        nonlocal bridge_unprocessed
        if bridge_unprocessed is not None:
            cleanup_model(bridge_unprocessed, "TransformerBridge (unprocessed)")
            bridge_unprocessed = None

    _skip_phase3 = False
    if not enable_compatibility_mode:
        _cleanup_bridge_unprocessed()
        _skip_phase3 = True
        if verbose:
            print("\n⚠ Compatibility mode disabled - skipping Phase 3\n")
    elif not should_run_phase(3):
        _cleanup_bridge_unprocessed()
        _skip_phase3 = True
        if verbose:
            print("\n⚠ Phase 3 skipped (not in phases list)\n")
    elif is_encoder_decoder_model(model_name):
        _cleanup_bridge_unprocessed()
        _skip_phase3 = True
        if verbose:
            print("\n⚠ Phase 3 skipped (encoder-decoder model - weight processing not supported)\n")

    bridge_processed = None
    ht_model_processed = None

    if not _skip_phase3:
        if verbose:
            print(f"\n{'='*80}")
            print("PHASE 3: TransformerBridge (processed) + HookedTransformer (processed)")
            print(f"{'='*80}\n")

    if not _skip_phase3:
        # Reuse the Phase 1 bridge instance and process weights in-place.
        # When dtype==float32 (default) and the model natively uses reduced
        # precision, upcast before processing to avoid bf16 quantization
        # round-trips.  When the user explicitly requested bfloat16/float16,
        # process weights in the requested precision — no upcast.
        phase3_native_dtype = None  # Set if we upcast; used to restore later
        if bridge_unprocessed is not None:
            try:
                if verbose:
                    print("Processing weights on existing bridge (reusing Phase 1 instance)...")
                bridge_processed = bridge_unprocessed
                bridge_unprocessed = None  # Transfer ownership
                phase3_native_dtype = bridge_processed.cfg.dtype
                if dtype == torch.float32 and phase3_native_dtype not in (
                    torch.float32,
                    torch.float64,
                ):
                    bridge_processed.to(torch.float32)
                    if verbose:
                        print(f"  (upcast from {phase3_native_dtype} to float32 before processing)")
                else:
                    phase3_native_dtype = None  # No restore needed
                bridge_processed.enable_compatibility_mode(disable_warnings=True)
                if verbose:
                    print("✓ TransformerBridge compatibility mode enabled (processed)\n")
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                add_result(
                    BenchmarkResult(
                        name="process_bridge_weights",
                        severity=BenchmarkSeverity.ERROR,
                        message=f"Failed to process bridge weights: {str(e)}",
                        passed=False,
                        details={"error": str(e), "traceback": error_trace},
                    )
                )
                if verbose:
                    print(f"✗ Failed to process bridge weights: {str(e)}")
                    print(f"\nStack trace:\n{error_trace}")
        else:
            # Fallback: load a fresh bridge if Phase 1 bridge was not available
            try:
                if verbose:
                    print("Loading TransformerBridge (processed)...")
                bridge_dtype = saved_bridge_dtype
                if verbose:
                    print(f"Using dtype={bridge_dtype} from Phase 1")
                bridge_processed = TransformerBridge.boot_transformers(model_name, device=device, dtype=bridge_dtype, trust_remote_code=trust_remote_code)  # type: ignore[attr-defined]
                bridge_processed.enable_compatibility_mode(disable_warnings=True)
                if verbose:
                    print("✓ TransformerBridge compatibility mode enabled (processed)\n")
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                add_result(
                    BenchmarkResult(
                        name="load_bridge_processed",
                        severity=BenchmarkSeverity.ERROR,
                        message=f"Failed to load processed TransformerBridge: {str(e)}",
                        passed=False,
                        details={"error": str(e), "traceback": error_trace},
                    )
                )
                if verbose:
                    print(f"✗ Failed to load processed TransformerBridge: {str(e)}")
                    print(f"\nStack trace:\n{error_trace}")

        if bridge_processed is None:
            # Add failure results for all Phase 3 tests
            phase3_tests = [
                "no_nan_inf",
                "weight_magnitudes",
                "layer_norm_folding",
                "attention_output_centering",
                "mlp_output_centering",
                "unembed_centering",
                "value_bias_folding",
                "weight_processing",
                "weight_sharing",
                "weight_modification",
                "logits_equivalence",
                "loss_equivalence",
                "hook_registry",
                "hook_functionality",
                "critical_forward_hooks",
                "forward_hooks",
                "run_with_cache",
                "activation_cache",
                "gradient_computation",
                "critical_backward_hooks",
                "backward_hooks",
            ]

            for test_name in phase3_tests:
                add_result(
                    BenchmarkResult(
                        name=test_name,
                        severity=BenchmarkSeverity.ERROR,
                        message=f"Skipped due to weight processing failure",
                        passed=False,
                        details={"reason": "bridge_processing_failed"},
                    )
                )

            if verbose:
                print("\n" + format_results(results))

        # Load HT in the same dtype that was requested for the benchmark.
        # This ensures a fair comparison — both bridge and HT operate in
        # the same precision throughout.
        phase3_ht_dtype = dtype

        if use_ht_reference:
            try:
                if verbose:
                    print("Loading HookedTransformer (processed)...")
                ht_model_processed = HookedTransformer.from_pretrained(
                    model_name,
                    device=device,
                    dtype=phase3_ht_dtype,
                    fold_ln=True,
                    center_writing_weights=True,
                    center_unembed=True,
                    fold_value_biases=True,
                    refactor_factored_attn_matrices=False,
                    default_prepend_bos=ht_prepend_bos,
                )
                if verbose:
                    print("✓ HookedTransformer loaded (processed)\n")
            except Exception as e:
                if verbose:
                    print(f"✗ Could not load processed HookedTransformer: {str(e)}\n")

        # Run Phase 3 benchmarks using unified function
        if bridge_processed:
            if verbose:
                print("Running Phase 3 benchmarks...\n")

            # Phase 3 runs in the requested dtype end-to-end.  Both bridge and HT
            # operate in the same precision — no dtype restoration needed.
            phase3_results = run_comparison_benchmarks(
                bridge_model=bridge_processed,
                reference_model=ht_model_processed,
                test_text=test_text,
                phase_name="Phase 3",
                is_processed=True,  # Processed mode - include weight processing tests
                verbose=verbose,
                phase1_reference=phase1_reference,  # Saved HF logits/loss for equivalence testing
            )
            # Tag all phase 3 results with phase number
            for result in phase3_results:
                if result.phase is None:
                    result.phase = 3
            results.extend(phase3_results)

        # Clean up Phase 3 models
        if bridge_processed is not None:
            cleanup_model(bridge_processed, "TransformerBridge (processed)")
            bridge_processed = None
        if ht_model_processed is not None:
            cleanup_model(ht_model_processed, "HookedTransformer (processed)")
            ht_model_processed = None

    # ========================================================================
    # Phase 5/6: Granular Weight Processing Tests (Optional)
    # ========================================================================
    if test_weight_processing_individually and enable_compatibility_mode:
        if verbose:
            print("\n" + "=" * 80)
            print("PHASE 5/6: GRANULAR WEIGHT PROCESSING TESTS")
            print("=" * 80)
            print("Testing each weight processing flag individually and in combinations")
            print("to isolate which specific processing steps cause issues.")
            print("=" * 80 + "\n")

        try:
            from transformer_lens.benchmarks.granular_weight_processing import (
                run_granular_weight_processing_benchmarks,
            )

            granular_results = run_granular_weight_processing_benchmarks(
                model_name=model_name,
                device=device,
                test_text=test_text,
                verbose=verbose,
            )

            # Convert granular results to BenchmarkResult format and add to main results
            for config_name, config_results in granular_results.items():
                for result in config_results:
                    # Prefix the name with the config for clarity
                    result.name = f"granular_{config_name}_{result.name}"
                    results.append(result)

            if verbose:
                print("\n" + "=" * 80)
                print("PHASE 5/6 COMPLETE")
                print("=" * 80)

        except Exception as e:
            if verbose:
                print(f"\n⚠ Granular weight processing tests failed: {e}\n")
            results.append(
                BenchmarkResult(
                    name="granular_weight_processing_suite",
                    passed=False,
                    severity=BenchmarkSeverity.ERROR,
                    message=f"Failed to run granular weight processing tests: {str(e)}",
                    details={"error": str(e)},
                )
            )

    # Print summary (individual results already printed immediately)
    if verbose:
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Group results by phase
        results_by_phase: Dict[Union[int, str], List[BenchmarkResult]] = {}
        for r in results:
            phase = r.phase if r.phase is not None else "Other"
            if phase not in results_by_phase:
                results_by_phase[phase] = []
            results_by_phase[phase].append(r)

        # Print phase-by-phase summary
        for phase in sorted(
            results_by_phase.keys(), key=lambda x: x if isinstance(x, int) else 999
        ):
            phase_results = results_by_phase[phase]
            phase_name = f"Phase {phase}" if isinstance(phase, int) else phase

            phase_passed = sum(
                1 for r in phase_results if r.passed and r.severity != BenchmarkSeverity.SKIPPED
            )
            phase_failed = sum(
                1 for r in phase_results if not r.passed and r.severity != BenchmarkSeverity.SKIPPED
            )
            phase_skipped = sum(1 for r in phase_results if r.severity == BenchmarkSeverity.SKIPPED)
            phase_total = len(phase_results)
            phase_run = phase_total - phase_skipped

            print(f"\n{phase_name}: {phase_run} tests run")
            if phase_run > 0:
                print(f"  Passed: {phase_passed}/{phase_run} ({phase_passed/phase_run*100:.1f}%)")
                print(f"  Failed: {phase_failed}/{phase_run} ({phase_failed/phase_run*100:.1f}%)")
            if phase_skipped > 0:
                print(f"  Skipped: {phase_skipped}")

        # Overall summary
        passed = sum(1 for r in results if r.passed and r.severity != BenchmarkSeverity.SKIPPED)
        failed = sum(1 for r in results if not r.passed and r.severity != BenchmarkSeverity.SKIPPED)
        skipped = sum(1 for r in results if r.severity == BenchmarkSeverity.SKIPPED)
        total = len(results)
        run_tests = total - skipped

        print(f"\nOverall:")
        print(f"Total: {total} tests")
        if skipped > 0:
            print(f"Run: {run_tests} tests")
            print(f"Skipped: {skipped} tests")
        if run_tests > 0:
            print(f"Passed: {passed}/{run_tests} ({passed/run_tests*100:.1f}%)")
            print(f"Failed: {failed}/{run_tests} ({failed/run_tests*100:.1f}%)")
        print("=" * 80)

    # Print memory summary
    if track_memory and memory_tracker is not None:
        final_memory = get_memory_mb()
        total_increase = final_memory - memory_tracker["initial"]

        if verbose:
            print("\n" + "=" * 80)
            print("MEMORY USAGE SUMMARY")
            print("=" * 80)
            print(f"Initial memory:  {memory_tracker['initial']:>8.1f} MB")
            print(f"Final memory:    {final_memory:>8.1f} MB")
            print(f"Net increase:    {total_increase:>+8.1f} MB")

            if memory_tracker["checkpoints"]:
                print("\nCleanup operations:")
                for cp in memory_tracker["checkpoints"]:
                    if cp.get("freed_mb", 0) > 0:
                        print(
                            f"  {cp['label']:<40} freed {cp['freed_mb']:>7.1f} MB "
                            f"(after: {cp['memory_mb']:.1f} MB)"
                        )
            print("=" * 80)

    return results


def update_model_registry(model_name: str, results: List[BenchmarkResult]) -> bool:
    """Update the model registry with benchmark results.

    Args:
        model_name: The model that was benchmarked
        results: List of benchmark results

    Returns:
        True if registry was updated successfully
    """
    from transformer_lens.tools.model_registry.registry_io import (
        STATUS_VERIFIED,
        add_verification_record,
        update_model_status,
    )

    # Calculate phase scores (percentage of passed tests per phase)
    phase_results: Dict[int, List[bool]] = {1: [], 2: [], 3: []}
    for result in results:
        if result.phase in phase_results and result.severity != BenchmarkSeverity.SKIPPED:
            phase_results[result.phase].append(result.passed)

    phase_scores: Dict[int, Optional[float]] = {}
    for phase, passed_list in phase_results.items():
        if passed_list:
            phase_scores[phase] = round(sum(passed_list) / len(passed_list) * 100, 1)
        else:
            phase_scores[phase] = None

    # Try to determine architecture
    architecture_id = "Unknown"
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, token=_hf_token())
        archs = getattr(config, "architectures", []) or []
        if archs:
            architecture_id = archs[0]
    except Exception:
        pass

    updated = update_model_status(
        model_id=model_name,
        arch_id=architecture_id,
        status=STATUS_VERIFIED,
        phase_scores=phase_scores,
    )

    add_verification_record(
        model_id=model_name,
        arch_id=architecture_id,
        notes="Benchmark passed",
        verified_by="main_benchmark",
    )

    print(
        f"Updated registry for {model_name}: "
        f"P1={phase_scores.get(1)}%, P2={phase_scores.get(2)}%, P3={phase_scores.get(3)}%"
    )
    return updated


def main():
    """Run benchmarks from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run TransformerBridge benchmarks")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name to benchmark (default: gpt2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--no-hf-reference",
        action="store_true",
        help="Disable HuggingFace reference comparison",
    )
    parser.add_argument(
        "--no-ht-reference",
        action="store_true",
        help="Disable HookedTransformer reference comparison",
    )
    parser.add_argument(
        "--no-compat",
        action="store_true",
        help="Disable compatibility mode",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--update-registry",
        action="store_true",
        help="Update model registry with benchmark results (default: false)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for custom architectures (e.g., OpenELM)",
    )
    parser.add_argument(
        "--conserve-memory",
        action="store_true",
        help="Reduce Phase 1 peak memory from 2.0x to 1.0x by using "
        "bridge.original_model instead of loading a separate HF model",
    )

    args = parser.parse_args()

    results = run_benchmark_suite(
        model_name=args.model,
        device=args.device,
        use_hf_reference=not args.no_hf_reference,
        use_ht_reference=not args.no_ht_reference,
        enable_compatibility_mode=not args.no_compat,
        verbose=not args.quiet,
        trust_remote_code=args.trust_remote_code,
        conserve_memory=args.conserve_memory,
    )

    if args.update_registry:
        update_model_registry(args.model, results)


if __name__ == "__main__":
    main()
