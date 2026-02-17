"""Main benchmark runner for TransformerBridge.

This module provides the main benchmark suite that compares TransformerBridge
against reference implementations in an optimized 4-phase approach:
Phase 1: HF + Bridge (unprocessed) - Compare against raw HuggingFace model
Phase 2: Bridge (unprocessed) + HT (unprocessed) - Compare unprocessed models
Phase 3: Bridge (processed) + HT (processed) - Full compatibility mode testing
Phase 4: Granular Weight Processing Tests (optional)
"""

import gc
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM

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
from transformer_lens.benchmarks.hook_structure import (
    benchmark_activation_cache_structure,
)
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


def is_encoder_decoder_model(model_name: str, trust_remote_code: bool = False) -> bool:
    """Check if a model is an encoder-decoder architecture.

    Args:
        model_name: The HuggingFace model name or path
        trust_remote_code: Whether to trust remote code for custom architectures.

    Returns:
        True if the model is encoder-decoder (like T5), False otherwise
    """
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        # Check config attribute first
        if getattr(config, "is_encoder_decoder", False):
            return True
        # Fallback to architecture check
        architectures = getattr(config, "architectures", []) or []
        return any(arch in ENCODER_DECODER_ARCHITECTURES for arch in architectures)
    except Exception:
        return False


def get_auto_model_class(model_name: str, trust_remote_code: bool = False):
    """Determine the correct AutoModel class for a given model.

    Some models (like T5) are encoder-decoder and need AutoModelForSeq2SeqLM
    instead of AutoModelForCausalLM.

    Args:
        model_name: The HuggingFace model name or path

    Returns:
        The appropriate AutoModel class (AutoModelForCausalLM or AutoModelForSeq2SeqLM)
    """
    if is_encoder_decoder_model(model_name, trust_remote_code=trust_remote_code):
        return AutoModelForSeq2SeqLM
    return AutoModelForCausalLM


def _fixup_custom_model(hf_model) -> None:
    """Apply post-load fixups for models with custom code.

    Some custom models (e.g., OpenELM) have non-persistent buffers (inv_freq,
    causal_mask) that may be zeroed during HuggingFace's meta-device loading.
    This function recomputes broken buffers to minimize forward pass divergence
    against the bridge model.

    Note: The bridge model goes through a more thorough initialization via the
    adapter's prepare_loading() + prepare_model() lifecycle hooks. Any remaining
    forward pass divergence is an inherent consequence of different loading paths
    for custom-code models, not a bridge correctness issue (all individual
    components produce identical output, and hooks have zero numerical impact).
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
    gpt2_reference: Optional[HookedTransformer] = None,
    phase1_reference: Optional[PhaseReferenceData] = None,
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
        gpt2_reference: Optional GPT-2 reference for cross-model validation
        phase1_reference: Optional saved Phase 1 HF reference data for equivalence testing

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
    has_cross_model_ref = gpt2_reference is not None

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

    has_phase1_ref = (
        phase1_reference is not None and phase1_reference.hf_logits is not None
    )

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
        # Use saved Phase 1 bridge logits/loss as ground truth.
        # Weight processing should be mathematically equivalent, so the processed
        # bridge should produce the same output as the unprocessed bridge.
        #
        # Important: center_unembed intentionally shifts raw logits by a per-position
        # constant (softmax-invariant). We compare log_softmax to be invariant to this.
        try:
            if verbose:
                print("Using saved Phase 1 bridge reference for equivalence comparison")

            # Compare log_softmax instead of raw logits to be centering-invariant.
            # center_unembed shifts all vocab logits at each position by a constant,
            # which changes raw logits but preserves log-probabilities.
            bridge_logits = bridge_model(test_text, return_type="logits")
            ref_logits = phase1_reference.hf_logits.to(bridge_logits.device)
            bridge_log_probs = torch.nn.functional.log_softmax(bridge_logits, dim=-1)
            ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            add_result(
                compare_tensors(
                    bridge_log_probs,
                    ref_log_probs,
                    atol=1e-4,
                    rtol=1e-4,
                    name="logits_equivalence",
                )
            )
            if phase1_reference.hf_loss is not None:
                add_result(
                    benchmark_loss_equivalence(
                        bridge_model,
                        test_text,
                        reference_loss=phase1_reference.hf_loss,
                        atol=1e-3,
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
    elif has_cross_model_ref:
        # Use GPT-2 for cross-model validation with dimensional matching
        try:
            if verbose:
                print("Using GPT-2 for cross-model validation (dimensional matching)")
            add_result(
                benchmark_hook_registry(
                    bridge_model, reference_model=gpt2_reference, cross_model=True
                )
            )
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Hook registry benchmark failed: {e}\n")
    else:
        if verbose:
            print("⏭️ Skipped (no HookedTransformer reference)\n")
        add_result(
            BenchmarkResult(
                name="hook_registry",
                severity=BenchmarkSeverity.SKIPPED,
                message="Skipped (HookedTransformer not available for this model)",
                passed=True,
            )
        )

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
    elif has_cross_model_ref:
        # Use GPT-2 for cross-model validation with dimensional matching
        try:
            if verbose:
                print("Using GPT-2 for cross-model validation (dimensional matching)")
            add_result(
                benchmark_hook_functionality(
                    bridge_model,
                    test_text,
                    reference_model=gpt2_reference,
                    cross_model=True,
                )
            )
            add_result(
                benchmark_critical_forward_hooks(
                    bridge_model,
                    test_text,
                    reference_model=gpt2_reference,
                    cross_model=True,
                )
            )
            add_result(
                benchmark_forward_hooks(
                    bridge_model,
                    test_text,
                    reference_model=gpt2_reference,
                    cross_model=True,
                )
            )
            # Reset hooks
            if hasattr(bridge_model, "reset_hooks"):
                bridge_model.reset_hooks()
            if gpt2_reference is not None and hasattr(gpt2_reference, "reset_hooks"):
                gpt2_reference.reset_hooks()
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Forward hook benchmark failed: {e}\n")
    else:
        if verbose:
            print("⏭️ Skipped (no HookedTransformer reference)\n")
        for benchmark_name in [
            "hook_functionality",
            "critical_forward_hooks",
            "forward_hooks",
        ]:
            add_result(
                BenchmarkResult(
                    name=benchmark_name,
                    severity=BenchmarkSeverity.SKIPPED,
                    message="Skipped (HookedTransformer not available for this model)",
                    passed=True,
                )
            )

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
    elif has_cross_model_ref:
        # Use GPT-2 for structural validation of cache
        try:
            if verbose:
                print("Using GPT-2 for cache structural validation")
            # Structure-only benchmark with cross-model comparison
            add_result(
                benchmark_activation_cache_structure(
                    bridge_model,
                    test_text,
                    reference_model=gpt2_reference,
                    cross_model=True,
                )
            )
            # Value benchmarks are skipped
            if verbose:
                print("⏭️ Cache value comparison skipped (requires same-model HT reference)\n")
            for benchmark_name in ["run_with_cache_values", "activation_cache_values"]:
                add_result(
                    BenchmarkResult(
                        name=benchmark_name,
                        severity=BenchmarkSeverity.SKIPPED,
                        message="Skipped (cache value comparison requires same-model HT reference)",
                        passed=True,
                    )
                )
            # Reset hooks
            if hasattr(bridge_model, "reset_hooks"):
                bridge_model.reset_hooks()
            if gpt2_reference is not None and hasattr(gpt2_reference, "reset_hooks"):
                gpt2_reference.reset_hooks()
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Activation cache structure benchmark failed: {e}\n")
    else:
        if verbose:
            print("⏭️ Skipped (no HookedTransformer reference)\n")
        for benchmark_name in ["run_with_cache", "activation_cache"]:
            add_result(
                BenchmarkResult(
                    name=benchmark_name,
                    severity=BenchmarkSeverity.SKIPPED,
                    message="Skipped (HookedTransformer not available for this model)",
                    passed=True,
                )
            )

    # ========================================================================
    # 6. Backward Gradient Benchmarks
    # MOST COMPLEX: Tests gradients and backward hooks - depends on everything above
    # ========================================================================
    if verbose:
        print("6. Backward Gradient Benchmarks")

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
    elif has_cross_model_ref:
        # Use GPT-2 for cross-model validation with dimensional matching
        try:
            if verbose:
                print("Using GPT-2 for backward hook cross-model validation (dimensional matching)")
            add_result(
                benchmark_gradient_computation(
                    bridge_model, test_text, reference_model=gpt2_reference
                )
            )
            add_result(
                benchmark_critical_backward_hooks(
                    bridge_model,
                    test_text,
                    reference_model=gpt2_reference,
                    cross_model=True,
                )
            )
            add_result(
                benchmark_backward_hooks(
                    bridge_model,
                    test_text,
                    reference_model=gpt2_reference,
                    cross_model=True,
                )
            )
            # Reset hooks
            if hasattr(bridge_model, "reset_hooks"):
                bridge_model.reset_hooks()
            if gpt2_reference is not None and hasattr(gpt2_reference, "reset_hooks"):
                gpt2_reference.reset_hooks()
            gc.collect()
        except Exception as e:
            if verbose:
                print(f"✗ Backward hooks benchmark failed: {e}\n")
    else:
        if verbose:
            print("⏭️ Skipped (no HookedTransformer reference)\n")
        for benchmark_name in [
            "gradient_computation",
            "critical_backward_hooks",
            "backward_hooks",
        ]:
            add_result(
                BenchmarkResult(
                    name=benchmark_name,
                    severity=BenchmarkSeverity.SKIPPED,
                    message="Skipped (HookedTransformer not available for this model)",
                    passed=True,
                )
            )

    return results


def run_benchmark_suite(
    model_name: str,
    device: str = "cpu",
    test_text: Optional[str] = None,
    use_hf_reference: bool = True,
    use_ht_reference: bool = True,
    enable_compatibility_mode: bool = True,
    verbose: bool = True,
    track_memory: bool = False,
    test_weight_processing_individually: bool = False,
    phases: list[int] | None = None,
    trust_remote_code: bool = False,
) -> List[BenchmarkResult]:
    """Run comprehensive benchmark suite for TransformerBridge.

    This function implements an optimized 5-phase approach to minimize model reloading:
    Phase 1: HF + Bridge (unprocessed) - Compare against raw HuggingFace model
    Phase 2: Bridge (unprocessed) + HT (unprocessed) - Compare unprocessed models
    Phase 3: Bridge (processed) + HT (processed) - Full compatibility mode testing
    Phase 4: Individual Weight Processing Flags (optional)
    Phase 5: Combined Weight Processing Flags (optional)

    When test_weight_processing_individually=True, Phases 4 & 5 run after
    Phase 3, testing each weight processing flag individually and in combinations.

    Args:
        model_name: Name of the model to benchmark (e.g., "gpt2")
        device: Device to run on ("cpu" or "cuda")
        test_text: Optional test text (default: standard test prompt)
        use_hf_reference: Whether to compare against HuggingFace model
        use_ht_reference: Whether to compare against HookedTransformer
        enable_compatibility_mode: Whether to enable compatibility mode on bridge
        verbose: Whether to print results to console
        track_memory: Whether to track and report memory usage (requires psutil)
        test_weight_processing_individually: Whether to run granular weight processing
            tests that check each processing flag individually (default: False)
        phases: Optional list of phase numbers to run (e.g., [1, 2, 3]). If None, runs all phases.

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

    # Early exit if only running Phase 4/5 (they load their own models independently)
    if phases is not None and all(p in [4, 5] for p in phases):
        if verbose:
            print(f"Skipping Phase 1-3 (only running Phase {', '.join(map(str, sorted(phases)))})")
            print("Phase 4/5 load their own models independently\n")

        # Jump directly to Phase 4/5
        # Jump to granular testing
        from transformer_lens.benchmarks.granular_weight_processing import (
            run_granular_weight_processing_benchmarks,
        )

        if 4 in phases and test_weight_processing_individually and enable_compatibility_mode:
            phase4_results = run_granular_weight_processing_benchmarks(
                model_name=model_name,
                device=device,
                test_text=test_text,
                verbose=verbose,
                phase=4,
            )
            for config_name, config_results in phase4_results.items():
                for result in config_results:
                    result.phase = 4
                    results.append(result)
                    if verbose:
                        result.print_immediate()

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

    def cleanup_model(model, model_name_str: str):
        """Free up memory by deleting a model and forcing garbage collection."""
        import gc

        if verbose:
            print(f"Cleaning up {model_name_str}...")

        # Track memory before cleanup
        if track_memory and memory_tracker is not None:
            memory_before = get_memory_mb()

        # NEW: Move model to CPU first to free GPU memory immediately
        if device != "cpu" and hasattr(model, "cpu"):
            try:
                model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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

                    # NEW: Clear gradients
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

        # NEW: Clear top-level gradients
        if hasattr(model, "zero_grad"):
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass

        # OPTIMIZATION: Break circular references more aggressively
        # Clear all submodule references to help GC
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
                    # NEW: Delete parameter tensor
                    del param
                model._parameters[param_name] = None
            model._parameters.clear()

        # Clear buffers dict
        if hasattr(model, "_buffers"):
            for buffer_name in list(model._buffers.keys()):
                buffer = model._buffers[buffer_name]
                if buffer is not None:
                    # NEW: Delete buffer tensor
                    del buffer
                model._buffers[buffer_name] = None
            model._buffers.clear()

        del model

        # Aggressive garbage collection (multiple passes to break circular references)
        for _ in range(3):
            gc.collect()

        # Clear CUDA cache if using GPU
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            # NEW: Synchronize to ensure GPU operations complete
            torch.cuda.synchronize()

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
    bridge_dtype = torch.float32
    attn_implementation = None
    try:
        # Load a lightweight version without weights to get config
        bridge_config_only = TransformerBridge.boot_transformers(model_name, device=device, dtype=bridge_dtype, load_weights=False, trust_remote_code=trust_remote_code)  # type: ignore[attr-defined]
        # Extract attn_implementation for HF model loading.
        # First check if adapter explicitly sets it (e.g. qwen3, gemma3).
        if hasattr(bridge_config_only.adapter.cfg, "attn_implementation"):
            attn_implementation = bridge_config_only.adapter.cfg.attn_implementation
        # TransformerBridge always loads HF models with output_attentions=True
        # (see sources/transformers.py), which causes HF to fall back from SDPA
        # to eager attention. We must match this in the reference model.
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
        # For custom code models, the config-only bridge may fail. We still need to
        # apply architecture-specific patches (e.g., OpenELM _init_weights fix) before
        # loading any model, otherwise _init_weights may re-randomize loaded weights.
        if trust_remote_code:
            try:
                from transformer_lens.model_bridge.sources.transformers import (
                    determine_architecture_from_hf_config,
                    map_default_transformer_lens_config,
                )

                hf_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
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

    # Load HF model with matching attn_implementation
    if use_hf_reference:
        try:
            if verbose:
                print("Loading HuggingFace reference model...")
            # Match loading path to TransformerBridge: no device_map, explicit .to(device)
            # Using device_map causes different weight materialization than .to(device),
            # which produces numerical divergence for bfloat16 models.
            hf_kwargs = {
                "low_cpu_mem_usage": True,  # Reduce memory spikes during loading
            }
            if attn_implementation is not None:
                hf_kwargs["attn_implementation"] = attn_implementation  # type: ignore[assignment]
                if verbose:
                    print(f"Using attn_implementation={attn_implementation}")
            # Use appropriate AutoModel class (e.g., AutoModelForSeq2SeqLM for T5)
            auto_model_class = get_auto_model_class(model_name, trust_remote_code=trust_remote_code)
            if verbose and auto_model_class != AutoModelForCausalLM:
                print(f"Using {auto_model_class.__name__} for encoder-decoder model")
            # Ensure pad_token_id exists on HF config. Transformers v5 raises
            # AttributeError for missing config attributes, which crashes models
            # like StableLM that access config.pad_token_id during __init__.
            hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            if not hasattr(hf_config, "pad_token_id") or "pad_token_id" not in hf_config.__dict__:
                hf_config.pad_token_id = getattr(hf_config, "eos_token_id", None)
                hf_kwargs["config"] = hf_config
            if trust_remote_code:
                hf_kwargs["trust_remote_code"] = True
            hf_model = auto_model_class.from_pretrained(model_name, **hf_kwargs)  # type: ignore[arg-type]
            # Post-load fixup for custom code models (e.g., OpenELM).
            # NOTE: We intentionally use _fixup_custom_model instead of the adapter's
            # prepare_model here. The adapter's prepare_model unconditionally recomputes
            # non-persistent buffers (causal_mask, inv_freq) which is needed for the
            # bridge path (meta-device loading), but the reference model loads normally
            # on CPU with correct buffers. Recomputing them can introduce numeric drift.
            _fixup_custom_model(hf_model)
            hf_model = hf_model.to(device)
            hf_model.eval()
            # Detect dtype from HF model
            try:
                bridge_dtype = next(hf_model.parameters()).dtype
                if verbose:
                    print(f"Detected dtype={bridge_dtype}")
            except StopIteration:
                pass
            if verbose:
                print("✓ HuggingFace model loaded\n")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load HuggingFace model: {str(e)}\n")

    # Now load the full bridge with correct dtype
    if verbose:
        print("Loading TransformerBridge (unprocessed)...")
    try:
        bridge_unprocessed = TransformerBridge.boot_transformers(model_name, device=device, dtype=bridge_dtype, trust_remote_code=trust_remote_code)  # type: ignore[attr-defined]
        if verbose:
            print("✓ TransformerBridge loaded (unprocessed)\n")
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
    if should_run_phase(1) and hf_model and bridge_unprocessed:
        if verbose:
            print("Running Phase 1 benchmarks...\n")

        # Component-level benchmarks
        if verbose:
            print("1. Component-Level Benchmarks")
        try:
            component_result = benchmark_all_components(bridge_unprocessed, hf_model)
            add_result(component_result)
            if verbose:
                status = "✓" if component_result.passed else "✗"
                print(f"{status} {component_result.message}\n")
            # Clean up after component testing
            gc.collect()
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            if verbose:
                print(f"✗ Component benchmark failed: {e}\n")

        # Forward pass benchmarks
        if verbose:
            print("2. Forward Pass Benchmarks")
        try:
            add_result(
                benchmark_forward_pass(bridge_unprocessed, test_text, reference_model=hf_model)
            )
        except Exception as e:
            if verbose:
                print(f"✗ Forward pass benchmark failed: {e}\n")

        # Capture unprocessed bridge reference data for Phase 3 reuse.
        # We save the BRIDGE's logits/loss (not the HF model's), because the bridge
        # forward path may differ slightly from HF. Phase 3 tests whether weight
        # processing preserves the bridge's own output — comparing processed bridge
        # vs unprocessed bridge.
        if bridge_unprocessed is not None:
            try:
                with torch.no_grad():
                    bridge_logits = bridge_unprocessed(test_text, return_type="logits")
                    phase1_reference.hf_logits = bridge_logits.detach().cpu().clone()
                    bridge_loss = bridge_unprocessed(test_text, return_type="loss")
                    phase1_reference.hf_loss = bridge_loss.item()
                    phase1_reference.test_text = test_text
                if verbose:
                    print(
                        f"✓ Saved Phase 1 reference data "
                        f"(logits: {phase1_reference.hf_logits.shape})"
                    )
            except Exception as e:
                if verbose:
                    print(f"⚠ Could not save Phase 1 reference data: {e}")

    # Save bridge_dtype before cleaning up HF model (needed for Phase 3)
    saved_bridge_dtype = bridge_dtype

    # Clean up HF model - no longer needed
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

        phase2_results = run_comparison_benchmarks(
            bridge_model=bridge_unprocessed,
            reference_model=ht_model_unprocessed,
            test_text=test_text,
            phase_name="Phase 2",
            is_processed=False,  # Unprocessed mode - skip weight processing tests
            verbose=verbose,
            gpt2_reference=None,  # No cross-model ref for Phase 2
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
    # NOTE: bridge_unprocessed is intentionally kept alive for Phase 3.
    # Instead of loading a fresh bridge (which can produce non-deterministic
    # outputs for some architectures like OpenELM), we reuse the same instance
    # and process its weights in-place. This ensures Phase 3 tests purely
    # measure the effect of weight processing, not loading variability.

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

    if not enable_compatibility_mode:
        _cleanup_bridge_unprocessed()
        if verbose:
            print("\n⚠ Compatibility mode disabled - skipping Phase 3\n")
        if verbose:
            print("\n" + format_results(results))
        return results

    if not should_run_phase(3):
        _cleanup_bridge_unprocessed()
        if verbose:
            print("\n⚠ Phase 3 skipped (not in phases list)\n")
        return results

    # Skip Phase 3 for encoder-decoder models - weight processing is designed for decoder-only models
    if is_encoder_decoder_model(model_name):
        _cleanup_bridge_unprocessed()
        if verbose:
            print("\n⚠ Phase 3 skipped (encoder-decoder model - weight processing not supported)\n")
            print("\n" + format_results(results))
        return results

    if verbose:
        print(f"\n{'='*80}")
        print("PHASE 3: TransformerBridge (processed) + HookedTransformer (processed)")
        print(f"{'='*80}\n")

    bridge_processed = None
    ht_model_processed = None

    # Reuse the Phase 1 bridge instance for Phase 3 instead of loading a fresh one.
    # This avoids non-deterministic loading issues (some architectures like OpenELM
    # produce different outputs across separate from_pretrained calls despite
    # identical parameters and buffers). Processing weights in-place on the same
    # instance ensures Phase 3 purely measures weight processing equivalence.
    if bridge_unprocessed is not None:
        try:
            if verbose:
                print("Processing weights on existing bridge (reusing Phase 1 instance)...")
            bridge_processed = bridge_unprocessed
            bridge_unprocessed = None  # Transfer ownership
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
        return results

    if use_ht_reference:
        try:
            if verbose:
                print("Loading HookedTransformer (processed)...")
            ht_model_processed = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                dtype=bridge_dtype,
                fold_ln=True,
                center_writing_weights=True,
                center_unembed=True,
                fold_value_biases=True,
                refactor_factored_attn_matrices=False,
            )
            if verbose:
                print("✓ HookedTransformer loaded (processed)\n")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load processed HookedTransformer: {str(e)}\n")

    # Automatically load GPT-2 as cross-model reference if HT not available for this model
    gpt2_reference = None
    if use_ht_reference and ht_model_processed is None and model_name.lower() != "gpt2":
        try:
            if verbose:
                print("HookedTransformer not available for this model.")
                print("Loading GPT-2 as cross-model reference for hook validation...\n")
            gpt2_reference = HookedTransformer.from_pretrained(
                "gpt2",
                device=device,
                fold_ln=True,
                center_writing_weights=True,
                center_unembed=True,
                fold_value_biases=True,
                refactor_factored_attn_matrices=False,
            )
            if verbose:
                print("✓ GPT-2 cross-model reference loaded\n")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load GPT-2 cross-model reference: {str(e)}\n")

    # Run Phase 3 benchmarks using unified function
    if bridge_processed:
        if verbose:
            print("Running Phase 3 benchmarks...\n")

        phase3_results = run_comparison_benchmarks(
            bridge_model=bridge_processed,
            reference_model=ht_model_processed,
            test_text=test_text,
            phase_name="Phase 3",
            is_processed=True,  # Processed mode - include weight processing tests
            verbose=verbose,
            gpt2_reference=gpt2_reference,  # Use GPT-2 cross-model ref if no same-arch HT
            phase1_reference=phase1_reference,  # Saved HF logits/loss for equivalence testing
        )
        # Tag all phase 3 results with phase number
        for result in phase3_results:
            if result.phase is None:
                result.phase = 3
        results.extend(phase3_results)

    # Clean up Phase 3 models before reporting memory
    if bridge_processed is not None:
        cleanup_model(bridge_processed, "TransformerBridge (processed)")
        bridge_processed = None
    if ht_model_processed is not None:
        cleanup_model(ht_model_processed, "HookedTransformer (processed)")
        ht_model_processed = None

    # ========================================================================
    # Phase 4: Granular Weight Processing Tests (Optional)
    # ========================================================================
    if test_weight_processing_individually and enable_compatibility_mode:
        if verbose:
            print("\n" + "=" * 80)
            print("PHASE 4: GRANULAR WEIGHT PROCESSING TESTS")
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
                print("PHASE 4 COMPLETE")
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
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for custom architectures (e.g., OpenELM)",
    )

    args = parser.parse_args()

    run_benchmark_suite(
        model_name=args.model,
        device=args.device,
        use_hf_reference=not args.no_hf_reference,
        use_ht_reference=not args.no_ht_reference,
        enable_compatibility_mode=not args.no_compat,
        verbose=not args.quiet,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
