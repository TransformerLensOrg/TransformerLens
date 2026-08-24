"""Main benchmark runner for TransformerBridge.

This module provides the main benchmark suite that compares TransformerBridge
against reference implementations in an optimized multi-phase approach:
Phase 1: HF + Bridge (unprocessed) - Compare against raw HuggingFace model
Phase 2: Bridge (unprocessed) + HT (unprocessed) - Compare unprocessed models
Phase 3: Bridge (processed) + HT (processed) - Full compatibility mode testing
Phase 4: Text Quality - profile prompts scored by a pinned judge's perplexity ratio
Phase 5: Granular Weight Processing Tests (optional, individual flags)
Phase 6: Granular Weight Processing Tests (optional, combined flags)
Phase 7: Multimodal Tests (only for multimodal models with pixel_values support)
Phase 8: Audio Tests (only for audio encoder models / audio-conditioned decoders)
Phase 9: Vision Tests (only for vision-only encoder models, e.g. ViT/DeiT)
"""

import gc
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from transformer_lens.benchmarks.activation_cache import (
    benchmark_activation_cache,
    benchmark_run_with_cache,
)
from transformer_lens.benchmarks.backward_gradients import (
    benchmark_backward_hooks,
    benchmark_critical_backward_hooks,
    benchmark_gradient_computation,
    needs_fp32_gradients,
)
from transformer_lens.benchmarks.component_benchmark import benchmark_all_components
from transformer_lens.benchmarks.forward_pass import (
    _compute_self_target_loss,
    benchmark_forward_pass,
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
    benchmark_gated_hooks_fire,
    benchmark_hook_functionality,
    benchmark_hook_registry,
)
from transformer_lens.benchmarks.text_quality import benchmark_text_quality
from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    PhaseReferenceData,
    build_modality_input,
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
)
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.model_registry.registry_io import TEXT_PHASES

# Architecture classification — single source of truth in utilities.architectures
from transformer_lens.utilities.architectures import (
    get_architectures_for_config,
    is_audio_model,
    is_encoder_decoder_model,
    is_masked_lm_model,
)
from transformer_lens.utilities.hf_utils import get_hf_token as _hf_token


def _adapter_applicable_phases(model_name: str, trust_remote_code: bool = False) -> list[int]:
    """Text phases (1-4) the model's adapter declares applicable (default all)."""
    from transformer_lens.factories.architecture_adapter_factory import (
        SUPPORTED_ARCHITECTURES,
    )

    try:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, token=_hf_token()
        )
        for arch in get_architectures_for_config(config):
            adapter_cls = SUPPORTED_ARCHITECTURES.get(arch)
            if adapter_cls is not None:
                return getattr(adapter_cls, "applicable_phases", list(TEXT_PHASES))
    except Exception:
        pass
    return list(TEXT_PHASES)


def _phase_enabled(
    phase_num: int, phases: Optional[List[int]], applicable_phases: List[int]
) -> bool:
    """Phase gating shared by run_benchmark_suite's should_run_phase.

    An adapter's ``applicable_phases`` declares which text phases (1-4) it covers.
    Phases 7/8/9 are gated separately by ``is_multimodal``/``is_audio_model``/
    ``is_visual_model`` at their call sites, so they are never filtered out here
    (mirrors verify_models._phases_to_run).
    """
    if phases is not None and phase_num not in phases:
        return False
    return phase_num not in TEXT_PHASES or phase_num in applicable_phases


def get_auto_model_class(model_name: str, trust_remote_code: bool = False):
    """Delegates to the bridge's architecture detection for consistency."""
    from transformer_lens.model_bridge.sources.transformers import (
        determine_architecture_from_hf_config,
        get_hf_model_class_for_architecture,
    )

    try:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, token=_hf_token()
        )
        architecture = determine_architecture_from_hf_config(config)
        return get_hf_model_class_for_architecture(architecture)
    except Exception:
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

    # Rotary tables destroyed by meta-device loading, for ANY architecture: the
    # reference needs the same repair the adapter applies to the bridge, or
    # Phase 1 compares a correct model against a corrupt one. Only tables that
    # fail a validity check are touched, so scaled RoPE is never clobbered.
    from transformer_lens.model_bridge.buffer_restore import restore_rotary_inv_freq

    restore_rotary_inv_freq(hf_model)

    if type(hf_model).__name__ == "GiddForDiffusionLM":
        from transformer_lens.model_bridge.supported_architectures.gidd import (
            restore_frequencies,
        )

        restore_frequencies(hf_model)


def _hf_forward_with_mask_fallback(hf_model, tokens):
    """Run an HF decoder forward, retrying with a 2D then 4D mask for models that
    dereference ``attention_mask`` unconditionally (e.g. LLaDA2) -- else the Phase-1
    capture raises, gets swallowed, and silently degrades to a shape-only check."""
    try:
        return hf_model(tokens)
    except (AttributeError, ValueError):
        b, s = tokens.shape[0], tokens.shape[-1]
        for mask in (
            torch.ones(b, s, dtype=torch.long, device=tokens.device),
            torch.ones(b, 1, s, s, dtype=torch.long, device=tokens.device),
        ):
            try:
                return hf_model(tokens, attention_mask=mask)
            except (AttributeError, ValueError):
                continue
        raise


def run_comparison_benchmarks(
    bridge_model: TransformerBridge,
    test_text: str,
    phase_name: str,
    is_processed: bool,
    verbose: bool = True,
    phase1_reference: Optional[PhaseReferenceData] = None,
    restore_dtype_after_equivalence: Optional[torch.dtype] = None,
) -> List[BenchmarkResult]:
    """Run standardized runtime benchmarks on the bridge.

    This function runs the same comprehensive test suite for both unprocessed (Phase 2)
    and processed (Phase 3) modes: HF-anchored logits/loss equivalence (via the saved
    Phase 1 reference) plus reference-free structural self-checks for hooks, cache,
    and gradients.

    Args:
        bridge_model: TransformerBridge model to test
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

            # weight_modification doesn't need a reference model
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

    if has_phase1_ref:
        # Compare the bridge against the saved Phase 1 HF reference.
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
            print("⏭️ Skipped (no Phase 1 HF reference)\n")
        for benchmark_name in ["logits_equivalence", "loss_equivalence"]:
            add_result(
                BenchmarkResult(
                    name=benchmark_name,
                    severity=BenchmarkSeverity.SKIPPED,
                    message="Skipped (no Phase 1 HF reference available)",
                    passed=True,
                )
            )

    # Restore native dtype so remaining tests run in the model's real dtype.
    if restore_dtype_after_equivalence is not None:
        try:
            bridge_model.to(restore_dtype_after_equivalence)
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

    try:
        add_result(benchmark_hook_functionality(bridge_model, test_text))
        add_result(benchmark_critical_forward_hooks(bridge_model, test_text))
        add_result(benchmark_forward_hooks(bridge_model, test_text))
        add_result(benchmark_gated_hooks_fire(bridge_model, test_text))
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

    # Gradient comparisons are graded against fp32-calibrated thresholds
    # (REL_L2_TOLERANCE): bf16's rounding floor alone is ~2e-3 rel_l2, inside the
    # measured bug band, so reduced-precision gradients cannot be graded at all.
    # Upcast for the gradient section on every device (MPS additionally lacks
    # bf16 autograd), then restore below.
    bridge_grad_dtype = bridge_model.cfg.dtype if hasattr(bridge_model, "cfg") else None
    grad_fp32_upcast = needs_fp32_gradients(bridge_grad_dtype)
    if grad_fp32_upcast:
        try:
            bridge_model.to(torch.float32)
        except Exception:
            grad_fp32_upcast = False  # Upcast failed; proceed as-is

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

    if grad_fp32_upcast and bridge_grad_dtype is not None:
        try:
            bridge_model.to(bridge_grad_dtype)
        except Exception:
            pass

    return results


def run_benchmark_suite(
    model_name: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    test_text: Optional[str] = None,
    use_hf_reference: bool = True,
    enable_compatibility_mode: bool = True,
    verbose: bool = True,
    track_memory: bool = False,
    phases: list[int] | None = None,
    trust_remote_code: bool = False,
    judge_model: PreTrainedModel | None = None,
    judge_tokenizer: PreTrainedTokenizerBase | None = None,
    prompt_profile: str | None = None,
) -> List[BenchmarkResult]:
    """Run comprehensive benchmark suite for TransformerBridge.

    This function implements an optimized multi-phase approach to minimize model reloading:
    Phase 1: HF + Bridge (unprocessed) - Compare against raw HuggingFace model
    Phase 2: Bridge (unprocessed) - Runtime self-checks + HF logits/loss equivalence
    Phase 3: Bridge (processed) - Compatibility mode + HF logits/loss equivalence
    Phase 4: Text Quality - profile prompts scored by a pinned judge's perplexity ratio

    Args:
        model_name: Name of the model to benchmark (e.g., "gpt2")
        device: Device to run on ("cpu" or "cuda")
        dtype: Precision for model loading (default: torch.float32). Use
            torch.bfloat16 to halve memory for larger models. Phase 2/3
            comparisons automatically upcast to float32 for precision.
        test_text: Optional test text (default: standard test prompt)
        use_hf_reference: Whether to compare against HuggingFace model
        enable_compatibility_mode: Whether to enable compatibility mode on bridge
        verbose: Whether to print results to console
        track_memory: Whether to track and report memory usage (requires psutil)
        phases: Optional list of phase numbers to run (e.g., [1, 2, 3]). If None, runs all phases.
        trust_remote_code: Whether to trust remote code for custom architectures.
        judge_model: Optional pre-loaded Phase-4 judge. When provided with
            judge_tokenizer, avoids reloading for each model in batch.
        judge_tokenizer: Optional pre-loaded tokenizer for the Phase-4 judge.
        prompt_profile: Optional Phase-4 prompt profile (e.g. "chat",
            "task:translation@en-de"). Resolved from curation + the registry
            when None.

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

    # Track current phase for result tagging
    current_phase: List[Optional[int]] = [None]  # Use list to allow modification in nested function

    adapter_applicable = _adapter_applicable_phases(model_name, trust_remote_code)

    def should_run_phase(phase_num: int) -> bool:
        """Check if a phase should run based on the phases filter and adapter applicability."""
        return _phase_enabled(phase_num, phases, adapter_applicable)

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
        gc.collect()
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

    hf_saved_logits = None
    hf_saved_loss = None

    if use_hf_reference and should_run_phase(1):
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
                print(f"Using {auto_model_class.__name__}")
            # Ensure pad_token_id exists (some models crash without it during init).
            hf_config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, token=_hf_token()
            )
            if not hasattr(hf_config, "pad_token_id") or "pad_token_id" not in hf_config.__dict__:
                eos = getattr(hf_config, "eos_token_id", None)
                hf_config.pad_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos
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

    # Detect audio/vision models once for use across all phases
    _is_audio = bridge_unprocessed is not None and getattr(
        bridge_unprocessed.cfg, "is_audio_model", False
    )
    _is_visual = bridge_unprocessed is not None and getattr(
        bridge_unprocessed.cfg, "is_visual_model", False
    )
    # Shared non-text input (spectrogram, waveform, or pixels) — the same tensor is used
    # for the HF reference capture and the bridge forward so they stay comparable.
    _test_modality_input = (
        build_modality_input(bridge_unprocessed, device=device, dtype=dtype)
        if (_is_audio or _is_visual)
        else None
    )

    # Run Phase 1 benchmarks
    if should_run_phase(1) and bridge_unprocessed:
        if verbose:
            print("Running Phase 1 benchmarks...\n")

        # Component-level benchmarks
        if verbose:
            print("1. Component-Level Benchmarks")
        if hf_model is not None:
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

            # Capture HF reference outputs. Both models are still in memory (2.0x window).
            if verbose:
                print("Capturing HF reference outputs to CPU...")
            try:
                if _test_modality_input is not None:
                    # Audio/vision models: use the shared non-text input for HF vs bridge
                    with torch.no_grad():
                        if _is_visual:
                            hf_out = hf_model(pixel_values=_test_modality_input)
                        else:
                            hf_out = hf_model(input_values=_test_modality_input)
                        # Bare encoders output last_hidden_state, not logits
                        if hasattr(hf_out, "logits") and hf_out.logits is not None:
                            hf_saved_logits = hf_out.logits.detach().cpu().clone()
                        else:
                            hf_saved_logits = hf_out.last_hidden_state.detach().cpu().clone()
                        # No loss computation — there are no next-token labels here
                    if verbose:
                        kind = "vision" if _is_visual else "audio"
                        print(
                            f"✓ Captured HF {kind} output {hf_saved_logits.shape}, "
                            f"loss=N/A (no token labels)\n"
                        )
                else:
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
                            hf_out = _hf_forward_with_mask_fallback(hf_model, hf_tokens)
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
                        print(
                            f"✓ Captured HF logits {hf_saved_logits.shape}, " f"loss={loss_str}\n"
                        )
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

        # For audio/vision models, reuse the input from HF reference capture
        _p1_input: Union[str, torch.Tensor] = test_text
        if _test_modality_input is not None:
            _p1_input = _test_modality_input

        if hf_saved_logits is not None:
            # Full mode: use pre-captured HF logits (bridge only, 1.0x)
            try:
                add_result(
                    benchmark_forward_pass(
                        bridge_unprocessed,
                        _p1_input,
                        reference_logits=hf_saved_logits.to(device),
                        atol=p1_atol,
                    )
                )
            except Exception as e:
                if verbose:
                    print(f"✗ Forward pass benchmark failed: {e}\n")
        else:
            try:
                add_result(benchmark_forward_pass(bridge_unprocessed, _p1_input, atol=p1_atol))
            except Exception as e:
                if verbose:
                    print(f"✗ Forward pass benchmark failed: {e}\n")

        # Capture Phase 1 reference for Phase 3 equivalence comparison.
        # Skip for audio/vision models (Phase 3 won't run — weight processing
        # unsupported — and the capture below feeds text, which they cannot accept).
        # When dtype==float32 (default) and the model natively uses reduced
        # precision, upcast for maximum accuracy.  When the user explicitly
        # requested a non-float32 dtype, run the reference pass in that dtype
        # so the entire pipeline honours the requested precision.
        if bridge_unprocessed is not None and not _is_audio and not _is_visual:
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
                    bridge_loss = _compute_self_target_loss(bridge_unprocessed, test_text)
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
    # PHASE 2: Bridge (unprocessed) — runtime self-checks + HF equivalence
    # ========================================================================
    current_phase[0] = 2

    # OPTIMIZATION: Run generation benchmarks first (only bridge in memory)
    # Then cleanup bridge before loading HT to reduce peak memory
    if should_run_phase(2) and bridge_unprocessed:
        if verbose:
            print(f"\n{'='*80}")
            print("PHASE 2: TransformerBridge (unprocessed) — runtime self-checks + HF equivalence")
            print(f"{'='*80}\n")
        if verbose:
            print("Running Phase 2 benchmarks...\n")

        # Generation benchmarks (unprocessed only) - RUN FIRST
        # Skip for encoder-decoder and audio models (no text generation capability)
        # Diffusion LMs generate through a native sampler; the benchmarks route
        # to it, so only architectures with neither path are skipped.
        _adapter = getattr(bridge_unprocessed, "adapter", None)
        _no_generate = not getattr(_adapter, "supports_generation", True) and not getattr(
            _adapter, "native_sampler", None
        )
        _skip_generation = (
            is_encoder_decoder_model(model_name)
            or getattr(bridge_unprocessed.cfg, "is_audio_model", False)
            or _no_generate
        )
        _skip_reason = (
            "Skipped (model does not support generation)"
            if _no_generate
            else "Skipped (encoder-decoder model)"
        )
        if verbose:
            print("1. Generation Benchmarks (unprocessed)")
        if _skip_generation:
            if verbose:
                print(f"⏭️ {_skip_reason}\n")
            add_result(
                BenchmarkResult(
                    name="generation",
                    severity=BenchmarkSeverity.INFO,
                    passed=True,
                    message=_skip_reason,
                )
            )
            add_result(
                BenchmarkResult(
                    name="generation_with_kv_cache",
                    severity=BenchmarkSeverity.INFO,
                    passed=True,
                    message=_skip_reason,
                )
            )
            add_result(
                BenchmarkResult(
                    name="multiple_generation_calls",
                    severity=BenchmarkSeverity.INFO,
                    passed=True,
                    message=_skip_reason,
                )
            )
            add_result(
                BenchmarkResult(
                    name="text_quality",
                    severity=BenchmarkSeverity.INFO,
                    passed=True,
                    message=_skip_reason,
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

    # Run Phase 2 runtime benchmarks using unified function
    if should_run_phase(2) and bridge_unprocessed:
        if verbose:
            print("2. Running Unprocessed Model Runtime Benchmarks\n")

        # When dtype==float32 (default) but the model natively loaded in
        # reduced precision, upcast for maximum benchmark accuracy.  When the
        # user explicitly requested bfloat16/float16, honour that — run the
        # entire comparison in the requested precision.
        phase2_restore_dtype = None
        if dtype == torch.float32 and bridge_dtype in (torch.bfloat16, torch.float16):
            try:
                bridge_unprocessed.to(torch.float32)
                phase2_restore_dtype = bridge_dtype
                if verbose:
                    print(f"  (upcast from {bridge_dtype} to float32 for comparison)\n")
            except Exception:
                phase2_restore_dtype = None  # Upcast failed; proceed as-is

        phase2_results = run_comparison_benchmarks(
            bridge_model=bridge_unprocessed,
            test_text=test_text,
            phase_name="Phase 2",
            is_processed=False,  # Unprocessed mode - skip weight processing tests
            verbose=verbose,
            phase1_reference=phase1_reference,  # Saved HF logits/loss for equivalence testing
            restore_dtype_after_equivalence=phase2_restore_dtype,
        )
        # Tag all phase 2 results with phase number
        for result in phase2_results:
            if result.phase is None:
                result.phase = 2
        results.extend(phase2_results)

    # bridge_unprocessed is kept alive for Phase 3 and Phase 4 — reusing the
    # same instance avoids non-deterministic loading in some architectures
    # (e.g., OpenELM).

    # ========================================================================
    # PHASE 4: Text Quality (profile prompts, judge perplexity-ratio scoring)
    # Runs before Phase 3 so it can reuse bridge_unprocessed (Phase 3
    # destructively processes the weights, consuming the bridge).
    # ========================================================================
    current_phase[0] = 4

    if (
        should_run_phase(4)
        and bridge_unprocessed is not None
        # applicable_phases and supports_generation are independent switches;
        # without this check a disagreement surfaces as an ERROR, not a skip.
        # Native-sampler architectures generate too, just not autoregressively.
        and (
            getattr(bridge_unprocessed.adapter, "supports_generation", True)
            or getattr(bridge_unprocessed.adapter, "native_sampler", None) is not None
        )
        and not is_masked_lm_model(model_name, trust_remote_code=trust_remote_code)
        and not is_audio_model(model_name, trust_remote_code=trust_remote_code)
    ):
        if prompt_profile is None:
            from transformer_lens.benchmarks.text_quality_profiles import (
                resolve_profile,
            )
            from transformer_lens.tools.model_registry.registry_io import (
                registry_prompt_profile,
            )

            config = getattr(bridge_unprocessed, "original_model", None)
            archs = getattr(getattr(config, "config", None), "architectures", None) or []
            prompt_profile = str(
                resolve_profile(
                    model_name, archs[0] if archs else None, registry_prompt_profile(model_name)
                )
            )

        if verbose:
            print(f"\n{'='*80}")
            print(f"PHASE 2.5: Text Quality (profile {prompt_profile}, judge ratio scoring)")
            print(f"{'='*80}\n")

        try:
            text_quality_result = benchmark_text_quality(
                bridge_unprocessed,
                prompt_profile,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                model_name=model_name,
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
    # Phase 8: Audio Tests (only for audio encoder models)
    # Runs before Phase 3 so we can reuse bridge_unprocessed before cleanup.
    # ========================================================================
    if (
        bridge_unprocessed is not None
        and getattr(bridge_unprocessed.cfg, "is_audio_model", False)
        and should_run_phase(8)
    ):
        current_phase[0] = 8
        if verbose:
            print("\n" + "=" * 80)
            print("PHASE 8: AUDIO TESTS")
            print("=" * 80)
            print("Testing audio forward pass, caching, representation stability, and features.")
            print("=" * 80 + "\n")

        try:
            from transformer_lens.benchmarks.audio import run_audio_benchmarks

            audio_results = run_audio_benchmarks(
                bridge_unprocessed,
                test_audio=_test_modality_input,
                verbose=verbose,
            )
            for result in audio_results:
                result.phase = 8
                results.append(result)
                if verbose:
                    print(result)

            if verbose:
                print("\n" + "=" * 80)
                print("PHASE 8 COMPLETE")
                print("=" * 80)

        except Exception as e:
            if verbose:
                print(f"\n⚠ Audio tests failed: {e}\n")
            results.append(
                BenchmarkResult(
                    name="audio_suite",
                    passed=False,
                    severity=BenchmarkSeverity.ERROR,
                    message=f"Failed to run audio tests: {str(e)}",
                    details={"error": str(e)},
                    phase=8,
                )
            )

    # ========================================================================
    # PHASE 8 (audio-text): audio-conditioned forward for audio decoders
    # (Qwen2Audio etc.) — is_multimodal with an audio processor, not an encoder.
    # Image Phase 7 feeds pixel_values and encoder Phase 8 feeds a raw waveform;
    # neither exercises these models' processed-feature audio path.
    # ========================================================================
    _audio_text = (
        bridge_unprocessed is not None
        and getattr(bridge_unprocessed.cfg, "is_multimodal", False)
        and not getattr(bridge_unprocessed.cfg, "is_audio_model", False)
        and getattr(getattr(bridge_unprocessed, "processor", None), "audio_token", None) is not None
    )
    if _audio_text and should_run_phase(8):
        current_phase[0] = 8
        if verbose:
            print("\n" + "=" * 80 + "\nPHASE 8: AUDIO-TEXT FORWARD\n" + "=" * 80 + "\n")
        from transformer_lens.benchmarks.audio import benchmark_audio_text_forward

        result = benchmark_audio_text_forward(bridge_unprocessed)
        result.phase = 8
        add_result(result)

    # ========================================================================
    # Phase 9: Vision Tests (only for vision encoder models — ViT/DeiT, not
    # vision+text multimodal models, which Phase 7 covers)
    # Runs before Phase 3 so we can reuse bridge_unprocessed before cleanup.
    # ========================================================================
    if (
        bridge_unprocessed is not None
        and getattr(bridge_unprocessed.cfg, "is_visual_model", False)
        and not getattr(bridge_unprocessed.cfg, "is_multimodal", False)
        and should_run_phase(9)
    ):
        current_phase[0] = 9
        if verbose:
            print("\n" + "=" * 80)
            print("PHASE 9: VISION TESTS")
            print("=" * 80)
            print("Testing pixel forward pass, caching, representation stability, and decoding.")
            print("=" * 80 + "\n")

        try:
            from transformer_lens.benchmarks.vision import run_vision_benchmarks

            vision_results = run_vision_benchmarks(
                bridge_unprocessed,
                test_pixels=_test_modality_input,
                verbose=verbose,
            )
            for result in vision_results:
                result.phase = 9
                results.append(result)
                if verbose:
                    print(result)

            if verbose:
                print("\n" + "=" * 80)
                print("PHASE 9 COMPLETE")
                print("=" * 80)

        except Exception as e:
            if verbose:
                print(f"\n⚠ Vision tests failed: {e}\n")
            results.append(
                BenchmarkResult(
                    name="vision_suite",
                    passed=False,
                    severity=BenchmarkSeverity.ERROR,
                    message=f"Failed to run vision tests: {str(e)}",
                    details={"error": str(e)},
                    phase=9,
                )
            )

    # ========================================================================
    # PHASE 3: Bridge (processed/compatibility mode) — HF equivalence
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
            print("\n⚠ Phase 3 skipped (excluded by phases filter or adapter applicable_phases)\n")
    elif is_encoder_decoder_model(model_name):
        _cleanup_bridge_unprocessed()
        _skip_phase3 = True
        if verbose:
            print("\n⚠ Phase 3 skipped (encoder-decoder model - weight processing not supported)\n")

    bridge_processed = None

    if not _skip_phase3:
        if verbose:
            print(f"\n{'='*80}")
            print("PHASE 3: TransformerBridge (processed/compatibility mode) — HF equivalence")
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

        # Run Phase 3 benchmarks using unified function
        if bridge_processed:
            if verbose:
                print("Running Phase 3 benchmarks...\n")

            # Phase 3 runs in the requested dtype end-to-end.  Both bridge and HT
            # operate in the same precision — no dtype restoration needed.
            phase3_results = run_comparison_benchmarks(
                bridge_model=bridge_processed,
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


def update_model_registry(
    model_name: str, results: List[BenchmarkResult], use_hf_reference: bool = False
) -> bool:
    """Update the model registry with benchmark results.

    Args:
        model_name: The model that was benchmarked
        results: List of benchmark results
        use_hf_reference: Whether the run numerically compared against an HF
            reference. Defaults to False so an unstated reference state records
            a passing run as PROVISIONAL, never VERIFIED.

    Returns:
        True if registry was updated successfully
    """
    from transformer_lens.tools.model_registry.registry_io import (
        STATUS_FAILED,
        STATUS_PROVISIONAL,
        add_verification_record,
        extract_phase_scores,
        pass_status,
        update_model_status,
    )

    # Threshold/note logic shared with verify_models so the two paths can't drift.
    from transformer_lens.tools.model_registry.verify_models import (
        _build_verified_note,
        _check_phase_scores,
        _extract_prompt_profile,
        _sanitize_note,
    )

    phase_scores = extract_phase_scores(results)

    score_error = _check_phase_scores(phase_scores, results)
    if score_error:
        status = STATUS_FAILED
        note = score_error
    else:
        status = pass_status(use_hf_reference)
        note = _build_verified_note(phase_scores, results)
        if status == STATUS_PROVISIONAL:
            note = f"Structural only (no HF reference): {note}"

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
        status=status,
        phase_scores=phase_scores,
        note=note,
        sanitize_fn=_sanitize_note,
        prompt_profile=_extract_prompt_profile(results),
    )

    # No history record for provisional runs — VerificationHistory.is_verified()
    # treats any record as verified, which would bypass the provisional gate.
    if status != STATUS_PROVISIONAL:
        add_verification_record(
            model_id=model_name,
            arch_id=architecture_id,
            notes=note,
            verified_by="main_benchmark",
            sanitize_fn=_sanitize_note,
        )

    label = {STATUS_FAILED: "FAILED", STATUS_PROVISIONAL: "PROVISIONAL"}.get(status, "VERIFIED")
    score_parts = ", ".join(f"P{p}={s}%" for p, s in sorted(phase_scores.items()))
    print(f"Updated registry for {model_name} ({label}): {score_parts or 'no phase results'}")
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
    args = parser.parse_args()

    results = run_benchmark_suite(
        model_name=args.model,
        device=args.device,
        use_hf_reference=not args.no_hf_reference,
        enable_compatibility_mode=not args.no_compat,
        verbose=not args.quiet,
        trust_remote_code=args.trust_remote_code,
    )

    if args.update_registry:
        # Same requested-reference state verify_models feeds pass_status(): a
        # --no-hf-reference run can only mint PROVISIONAL, never VERIFIED.
        update_model_registry(args.model, results, use_hf_reference=not args.no_hf_reference)


if __name__ == "__main__":
    main()
