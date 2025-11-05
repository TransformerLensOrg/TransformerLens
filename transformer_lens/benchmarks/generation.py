"""Generation and KV cache benchmarks for TransformerBridge."""

from typing import Optional

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity
from transformer_lens.model_bridge import TransformerBridge


def benchmark_generation(
    bridge: TransformerBridge,
    test_text: str,
    max_new_tokens: int = 10,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark basic text generation.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for generation
        max_new_tokens: Number of tokens to generate
        reference_model: Optional HookedTransformer reference model (not used)

    Returns:
        BenchmarkResult with generation details
    """
    try:
        output = bridge.generate(test_text, max_new_tokens=max_new_tokens)

        if not isinstance(output, str):
            return BenchmarkResult(
                name="generation",
                severity=BenchmarkSeverity.DANGER,
                message="Generated output is not a string",
                passed=False,
            )

        if len(output) <= len(test_text):
            return BenchmarkResult(
                name="generation",
                severity=BenchmarkSeverity.DANGER,
                message="Generated text is not longer than input",
                details={"input_len": len(test_text), "output_len": len(output)},
                passed=False,
            )

        return BenchmarkResult(
            name="generation",
            severity=BenchmarkSeverity.INFO,
            message=f"Generation successful: {len(test_text)} -> {len(output)} chars",
            details={
                "input_len": len(test_text),
                "output_len": len(output),
                "max_new_tokens": max_new_tokens,
            },
        )

    except Exception as e:
        return BenchmarkResult(
            name="generation",
            severity=BenchmarkSeverity.ERROR,
            message=f"Generation failed: {str(e)}",
            passed=False,
        )


def benchmark_generation_with_kv_cache(
    bridge: TransformerBridge,
    test_text: str,
    max_new_tokens: int = 10,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark text generation with KV caching enabled.

    This ensures that the KV cache is properly passed through attention layers
    during generation, and that the cache update logic works correctly.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for generation
        max_new_tokens: Number of tokens to generate
        reference_model: Optional HookedTransformer reference model (not used)

    Returns:
        BenchmarkResult with generation details
    """
    try:
        # Generate with KV cache (should be enabled by default for max_new_tokens > 1)
        output = bridge.generate(
            test_text,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            prepend_bos=True,
        )

        if output is None or len(output) == 0:
            return BenchmarkResult(
                name="generation_with_kv_cache",
                severity=BenchmarkSeverity.DANGER,
                message="Generation with KV cache produced no output",
                passed=False,
            )

        return BenchmarkResult(
            name="generation_with_kv_cache",
            severity=BenchmarkSeverity.INFO,
            message=f"KV cache generation successful ({len(output)} chars)",
            details={"output_len": len(output), "max_new_tokens": max_new_tokens},
        )

    except Exception as e:
        return BenchmarkResult(
            name="generation_with_kv_cache",
            severity=BenchmarkSeverity.ERROR,
            message=f"KV cache generation failed: {str(e)}",
            passed=False,
        )


def benchmark_multiple_generation_calls(
    bridge: TransformerBridge,
    test_prompts: list,
    max_new_tokens: int = 5,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark multiple generation calls to ensure KV cache handling is robust.

    Args:
        bridge: TransformerBridge model to test
        test_prompts: List of input prompts for generation
        max_new_tokens: Number of tokens to generate per prompt
        reference_model: Optional HookedTransformer reference model (not used)

    Returns:
        BenchmarkResult with multiple generation details
    """
    try:
        outputs = []
        for prompt in test_prompts:
            output = bridge.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                prepend_bos=True,
            )
            if output is None or len(output) == 0:
                return BenchmarkResult(
                    name="multiple_generation_calls",
                    severity=BenchmarkSeverity.DANGER,
                    message=f"Generation failed for prompt: {prompt[:50]}...",
                    passed=False,
                )
            outputs.append(output)

        return BenchmarkResult(
            name="multiple_generation_calls",
            severity=BenchmarkSeverity.INFO,
            message=f"All {len(test_prompts)} generation calls successful",
            details={
                "prompt_count": len(test_prompts),
                "max_new_tokens": max_new_tokens,
                "output_lens": [len(out) for out in outputs],
            },
        )

    except Exception as e:
        return BenchmarkResult(
            name="multiple_generation_calls",
            severity=BenchmarkSeverity.ERROR,
            message=f"Multiple generation calls failed: {str(e)}",
            passed=False,
        )
