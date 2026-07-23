"""Generation and KV cache benchmarks for TransformerBridge."""

from typing import Any, Optional

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    is_tiny_test_model,
)
from transformer_lens.model_bridge import TransformerBridge


def resolve_text_generator(bridge: Any):
    """Return the callable that produces text for this architecture, or None.

    Diffusion LMs sample by iterative denoising through their own sampler; that
    is still generation for verification purposes, so the benchmarks exercise it
    rather than skipping these architectures.
    """
    if getattr(bridge.adapter, "supports_generation", True):
        return bridge.generate
    if getattr(bridge.adapter, "native_sampler", None):
        return bridge.diffusion_generate
    return None


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
        if is_tiny_test_model(getattr(bridge.cfg, "model_name", "") or ""):
            return BenchmarkResult(
                name="generation",
                severity=BenchmarkSeverity.INFO,
                message="Skipped for tiny/test model (random weights produce degenerate generation)",
            )
        generator = resolve_text_generator(bridge)
        if generator is None:
            return BenchmarkResult(
                name="generation",
                severity=BenchmarkSeverity.INFO,
                message="Skipped: architecture supports no text generation",
            )
        output = generator(test_text, max_new_tokens=max_new_tokens)

        if not isinstance(output, str):
            return BenchmarkResult(
                name="generation",
                severity=BenchmarkSeverity.DANGER,
                message="Generated output is not a string",
                passed=False,
            )

        # Check token count instead of character count to handle whitespace-only generation
        input_tokens = bridge.to_tokens(test_text)
        output_tokens = bridge.to_tokens(output)

        # Strip leading BOS token if present for fair comparison
        input_len = input_tokens.shape[-1]
        output_len = output_tokens.shape[-1]

        # Compare like with like. decode->re-tokenize is lossy when the prompt
        # holds out-of-vocabulary characters (a DNA model given English drops
        # them as [UNK]), which reads as "no new tokens" though generation ran.
        prompt_roundtrip = test_text
        if bridge.tokenizer is not None:
            prompt_roundtrip = bridge.tokenizer.decode(
                input_tokens[0], skip_special_tokens=True
            )
        if output_len <= input_len and len(output) <= len(prompt_roundtrip):
            return BenchmarkResult(
                name="generation",
                severity=BenchmarkSeverity.DANGER,
                message="Generated text has no new tokens",
                details={
                    "input_tokens": input_len,
                    "output_tokens": output_len,
                    "input_chars": len(test_text),
                    "output_chars": len(output),
                },
                passed=False,
            )

        return BenchmarkResult(
            name="generation",
            severity=BenchmarkSeverity.INFO,
            message=f"Generation successful: {input_len} -> {output_len} tokens ({len(test_text)} -> {len(output)} chars)",
            details={
                "input_tokens": input_len,
                "output_tokens": output_len,
                "input_chars": len(test_text),
                "output_chars": len(output),
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
        if is_tiny_test_model(getattr(bridge.cfg, "model_name", "") or ""):
            return BenchmarkResult(
                name="generation_with_kv_cache",
                severity=BenchmarkSeverity.INFO,
                message="Skipped for tiny/test model (random weights produce degenerate generation)",
            )

        # Cache-free architectures (RWKV, HyenaDNA) would "pass" this vacuously —
        # the recompute path produces output while exercising no cache at all.
        # Diffusion samplers have no KV cache concept either.
        if not getattr(bridge.adapter, "supports_kv_cache", True) or not getattr(
            bridge.adapter, "supports_generation", True
        ):
            return BenchmarkResult(
                name="generation_with_kv_cache",
                severity=BenchmarkSeverity.INFO,
                message="Skipped: architecture has no KV cache (generation recomputes each step)",
            )

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
        if is_tiny_test_model(getattr(bridge.cfg, "model_name", "") or ""):
            return BenchmarkResult(
                name="multiple_generation_calls",
                severity=BenchmarkSeverity.INFO,
                message="Skipped for tiny/test model (random weights produce degenerate generation)",
            )

        generator = resolve_text_generator(bridge)
        if generator is None:
            return BenchmarkResult(
                name="multiple_generation_calls",
                severity=BenchmarkSeverity.INFO,
                message="Skipped: architecture supports no text generation",
            )

        outputs = []
        for prompt in test_prompts:
            output = generator(
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
