#!/usr/bin/env python3
"""Run benchmarks for all supported architectures with the smallest available models.

This utility runs the TransformerBridge benchmark suite against each architecture
adapter using the smallest model that fits in available memory. Results are
collected and summarized at the end.

Usage:
    python utilities/run_all_benchmarks.py [--skip-large] [--only MODEL_KEY]
"""

import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ModelSpec:
    """Specification for a model to benchmark."""
    architecture: str
    model_name: str
    approx_params_m: int  # Approximate parameter count in millions
    trust_remote_code: bool = False
    has_hooked_transformer: bool = True
    notes: str = ""


@dataclass
class BenchmarkRunResult:
    """Result of a single benchmark run."""
    model_spec: ModelSpec
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list = field(default_factory=list)
    duration_s: float = 0.0
    status: str = "not_run"  # not_run, success, failure, error, skipped_memory
    failure_details: list = field(default_factory=list)


# Define all models to benchmark, sorted by size
# Memory budget: 24GB RAM, need 3x model size (HF + Bridge + HT) in fp32
# Safe limit: ~1B params (4GB per instance * 3 = 12GB, leaving 12GB headroom)
BENCHMARK_MODELS = [
    ModelSpec("NeelSoluOld", "NeelNanda/SoLU_1L512W_C4_Code", 3, notes="Tiny 1-layer model"),
    ModelSpec("Pythia", "EleutherAI/pythia-14m", 14, notes="Smallest Pythia variant"),
    ModelSpec("T5", "google-t5/t5-small", 60, notes="Encoder-decoder, Phase 3 skipped"),
    ModelSpec("GPT2", "gpt2", 124, notes="Baseline reference architecture"),
    ModelSpec("BERT", "google-bert/bert-base-uncased", 110, notes="Encoder-only"),
    ModelSpec("Neo", "EleutherAI/gpt-neo-125M", 125),
    ModelSpec("OPT", "facebook/opt-125m", 125),
    ModelSpec("OpenELM", "apple/OpenELM-270M", 270, trust_remote_code=True,
             has_hooked_transformer=False, notes="New architecture - no HT support"),
    ModelSpec("Qwen2", "Qwen/Qwen2-0.5B", 500),
    ModelSpec("Bloom", "bigscience/bloom-560m", 560),
    ModelSpec("Qwen3", "Qwen/Qwen3-0.6B", 600, trust_remote_code=True),
    ModelSpec("Llama", "meta-llama/Llama-3.2-1B", 1000,
             notes="Gated model - requires HF auth"),
]

# Models too large for 24GB RAM (3x model in fp32)
TOO_LARGE_MODELS = [
    ModelSpec("Phi", "microsoft/phi-1", 1300, trust_remote_code=True,
             notes="~15.6GB for 3 instances"),
    ModelSpec("Gpt2LmHeadCustom", "bigcode/santacoder", 1600, trust_remote_code=True,
             notes="~19.2GB for 3 instances"),
    ModelSpec("Qwen", "Qwen/Qwen-1_8B", 1800, trust_remote_code=True,
             notes="~21.6GB for 3 instances"),
    ModelSpec("Gemma1", "google/gemma-2b", 2000,
             notes="~24GB for 3 instances - too tight"),
    ModelSpec("Gemma2", "google/gemma-2-2b", 2000,
             notes="~24GB for 3 instances - too tight"),
    ModelSpec("Gemma3", "google/gemma-3-270m", 270,
             notes="Needs gated access and special tokenizer"),
    ModelSpec("Olmo", "allenai/OLMo-1B-hf", 1000, trust_remote_code=True,
             notes="1B but trust_remote_code adds overhead"),
    ModelSpec("Olmo2", "allenai/OLMo-2-0425-1B", 1000, trust_remote_code=True,
             notes="1B but trust_remote_code adds overhead"),
    ModelSpec("StableLM", "stabilityai/stablelm-base-alpha-3b", 3000,
             notes="~36GB for 3 instances"),
    ModelSpec("Phi3", "microsoft/Phi-3-mini-4k-instruct", 3800, trust_remote_code=True,
             notes="~45.6GB for 3 instances"),
    ModelSpec("GPTJ", "EleutherAI/gpt-j-6B", 6000,
             notes="~72GB for 3 instances"),
    ModelSpec("Mistral", "mistralai/Mistral-7B-v0.1", 7000,
             notes="~84GB for 3 instances"),
    ModelSpec("Olmo3", "allenai/OLMo-3-7B-Instruct", 7000, trust_remote_code=True,
             notes="~84GB for 3 instances"),
    ModelSpec("OlmoE", "allenai/OLMoE-1B-7B-0924", 7000, trust_remote_code=True,
             notes="MoE - ~84GB for 3 instances"),
    ModelSpec("Neox", "EleutherAI/gpt-neox-20b", 20000,
             notes="~240GB for 3 instances"),
    ModelSpec("Mixtral", "mistralai/Mixtral-8x7B-v0.1", 46700,
             notes="MoE - ~560GB for 3 instances"),
]

# Not testable (custom models only, no public weights)
NOT_TESTABLE = [
    ModelSpec("NanoGPT", "N/A", 0, notes="Custom models only - no public weights"),
    ModelSpec("MinGPT", "N/A", 0, notes="Custom models only - no public weights"),
    ModelSpec("GPTOSS", "N/A", 0, notes="No official public models"),
]


def run_single_benchmark(spec: ModelSpec, device: str = "cpu") -> BenchmarkRunResult:
    """Run the benchmark suite for a single model."""
    result = BenchmarkRunResult(model_spec=spec)
    start_time = time.time()

    try:
        from transformer_lens.benchmarks.main_benchmark import run_benchmark_suite

        print(f"\n{'#'*80}")
        print(f"# BENCHMARKING: {spec.architecture} ({spec.model_name})")
        print(f"# Approx size: {spec.approx_params_m}M params")
        if spec.notes:
            print(f"# Notes: {spec.notes}")
        print(f"{'#'*80}\n")

        benchmark_results = run_benchmark_suite(
            model_name=spec.model_name,
            device=device,
            use_hf_reference=True,
            use_ht_reference=spec.has_hooked_transformer,
            enable_compatibility_mode=True,
            verbose=True,
            trust_remote_code=spec.trust_remote_code,
        )

        # Analyze results
        from transformer_lens.benchmarks.utils import BenchmarkSeverity

        for br in benchmark_results:
            result.total_tests += 1
            if br.severity == BenchmarkSeverity.SKIPPED:
                result.skipped += 1
            elif br.passed:
                result.passed += 1
            else:
                result.failed += 1
                result.failure_details.append({
                    "name": br.name,
                    "severity": br.severity.value if hasattr(br.severity, 'value') else str(br.severity),
                    "message": br.message,
                    "phase": br.phase,
                })

        result.status = "success" if result.failed == 0 else "failure"

    except MemoryError:
        result.status = "skipped_memory"
        result.errors.append("Out of memory")
        print(f"\nMEMORY ERROR: {spec.model_name} exceeded available memory")
    except Exception as e:
        result.status = "error"
        result.errors.append(f"{type(e).__name__}: {str(e)}")
        print(f"\nERROR running {spec.model_name}: {e}")
        traceback.print_exc()
    finally:
        result.duration_s = time.time() - start_time
        # Force cleanup
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    return result


def print_summary(results: list, too_large: list, not_testable: list):
    """Print a comprehensive summary of all benchmark results."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}\n")

    # Tested models
    print("TESTED MODELS:")
    print(f"{'Architecture':<20} {'Model':<40} {'Status':<10} {'Pass/Fail/Skip':<20} {'Time':<10}")
    print("-" * 100)

    total_pass = 0
    total_fail = 0
    total_skip = 0
    total_error = 0

    for r in results:
        s = r.model_spec
        if r.status == "success":
            status = "PASS"
            total_pass += 1
        elif r.status == "failure":
            status = "FAIL"
            total_fail += 1
        elif r.status == "error":
            status = "ERROR"
            total_error += 1
        elif r.status == "skipped_memory":
            status = "OOM"
            total_error += 1
        else:
            status = "N/A"

        pfs = f"{r.passed}/{r.failed}/{r.skipped}"
        duration = f"{r.duration_s:.1f}s"
        print(f"{s.architecture:<20} {s.model_name:<40} {status:<10} {pfs:<20} {duration:<10}")

        if r.failure_details:
            for fd in r.failure_details:
                phase_str = f"P{fd['phase']}" if fd.get('phase') else "?"
                print(f"  [{phase_str}] FAIL: {fd['name']} - {fd['message'][:80]}")

    print(f"\nTested: {len(results)} architectures")
    print(f"  All passing: {total_pass}")
    print(f"  Failures: {total_fail}")
    print(f"  Errors: {total_error}")

    # Too large models
    if too_large:
        print(f"\n\nMODELS TOO LARGE FOR 24GB RAM (not tested):")
        print(f"{'Architecture':<20} {'Smallest Model':<40} {'Size':<10} {'Notes'}")
        print("-" * 100)
        for s in too_large:
            size = f"{s.approx_params_m}M"
            print(f"{s.architecture:<20} {s.model_name:<40} {size:<10} {s.notes}")

    # Not testable
    if not_testable:
        print(f"\n\nNOT TESTABLE (no public models):")
        for s in not_testable:
            print(f"  {s.architecture}: {s.notes}")

    print(f"\n{'='*80}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all architecture benchmarks")
    parser.add_argument("--skip-large", action="store_true",
                       help="Skip models > 500M params")
    parser.add_argument("--only", type=str, default=None,
                       help="Run only a specific architecture (e.g., 'GPT2')")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on (default: cpu)")
    args = parser.parse_args()

    models_to_run = BENCHMARK_MODELS

    if args.only:
        models_to_run = [m for m in BENCHMARK_MODELS if m.architecture.lower() == args.only.lower()]
        if not models_to_run:
            print(f"No model found for architecture '{args.only}'")
            print(f"Available: {', '.join(m.architecture for m in BENCHMARK_MODELS)}")
            sys.exit(1)

    if args.skip_large:
        models_to_run = [m for m in models_to_run if m.approx_params_m <= 500]

    results = []
    for spec in models_to_run:
        result = run_single_benchmark(spec, device=args.device)
        results.append(result)

        # Print intermediate status
        status = "PASS" if result.status == "success" else result.status.upper()
        print(f"\n>>> {spec.architecture}: {status} "
              f"({result.passed} pass, {result.failed} fail, {result.skipped} skip) "
              f"in {result.duration_s:.1f}s\n")

    print_summary(results, TOO_LARGE_MODELS, NOT_TESTABLE)

    # Return non-zero if any failures
    if any(r.status in ("failure", "error") for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
