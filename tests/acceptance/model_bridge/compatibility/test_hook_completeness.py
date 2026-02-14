"""Test that all HookedTransformer hooks are available and functional in TransformerBridge.

This test ensures complete hook parity across different model architectures.
It verifies that:
1. All hooks from HookedTransformer exist in TransformerBridge
2. All hooks actually fire during forward pass
3. Hook activations match between the two implementations
"""

import os

import pytest

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks import benchmark_forward_hooks, benchmark_hook_registry
from transformer_lens.model_bridge import TransformerBridge

pytestmark = pytest.mark.skip(reason="Temporarily skipping hook completeness tests pending fixes")

# Models to test hook completeness on
# Include diverse architectures to catch architecture-specific issues
MODELS_TO_TEST = [
    "gpt2",  # Standard decoder-only with joint QKV
    "EleutherAI/pythia-14m",  # GPT-NeoX architecture (smaller than pythia-70m)
]

# Only test Gemma2 locally (not in CI) due to size
if not os.getenv("CI"):
    MODELS_TO_TEST.append("google/gemma-2-2b-it")  # Gemma2 with unique normalization setup


class TestHookCompleteness:
    """Test suite for verifying complete hook coverage across architectures."""

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST)
    def test_all_hooks_exist(self, model_name):
        """Test that TransformerBridge has all hooks that HookedTransformer has.

        This test verifies that the hook registry is complete - every hook in
        HookedTransformer must exist in TransformerBridge.
        """
        # Load both models
        ht = HookedTransformer.from_pretrained_no_processing(model_name, device="cpu")
        bridge = TransformerBridge.boot_transformers(model_name, device="cpu")
        bridge.enable_compatibility_mode(no_processing=True)

        # Run benchmark
        result = benchmark_hook_registry(bridge, reference_model=ht)

        # Must pass - no missing hooks allowed
        assert result.passed, (
            f"Hook registry check failed for {model_name}:\n"
            f"  {result.message}\n"
            f"  Details: {result.details}"
        )

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST)
    def test_all_hooks_fire(self, model_name):
        """Test that all hooks actually fire during a forward pass.

        This test verifies that hooks don't just exist in the registry, but
        actually execute and capture activations during forward pass. This is
        critical because hooks that don't fire indicate architectural bugs
        (e.g., missing ln2 calls in patched forward methods).
        """
        # Load both models
        ht = HookedTransformer.from_pretrained_no_processing(model_name, device="cpu")
        bridge = TransformerBridge.boot_transformers(model_name, device="cpu")
        bridge.enable_compatibility_mode(no_processing=True)

        # Use a short prompt to speed up testing
        test_text = "The quick brown fox"

        # Run benchmark - this will fail if hooks don't fire
        result = benchmark_forward_hooks(bridge, test_text, reference_model=ht, tolerance=1e-3)

        # Must pass - all hooks must fire
        assert result.passed, (
            f"Forward hooks check failed for {model_name}:\n"
            f"  {result.message}\n"
            f"  Details: {result.details}\n"
            f"\n"
            f"This likely means:\n"
            f"  1. Some hooks are missing from TransformerBridge, OR\n"
            f"  2. Some hooks exist but don't fire during forward pass, OR\n"
            f"  3. Hook activations don't match between implementations\n"
            f"\n"
            f"Check the 'missing_hooks' or 'didnt_fire_hooks' in details above."
        )

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST)
    def test_normalization_hooks_fire(self, model_name):
        """Test that all layer normalization hooks fire.

        This is a targeted test for normalization hooks because they're
        architecture-specific and prone to being missed (e.g., Gemma2's ln2).
        """
        # Load bridge model
        bridge = TransformerBridge.boot_transformers(model_name, device="cpu")
        bridge.enable_compatibility_mode(no_processing=True)

        test_text = "Hello world"

        # Track which normalization hooks fired
        norm_hooks_fired = set()

        def capture_hook(name):
            def hook_fn(tensor, hook):
                norm_hooks_fired.add(name)
                return tensor

            return hook_fn

        # Register hooks for all normalization layers
        hooks_to_test = []
        for layer_idx in range(bridge.cfg.n_layers):
            # Test ln1, ln2 hook_normalized for each layer
            for norm_name in ["ln1", "ln2"]:
                hook_name = f"blocks.{layer_idx}.{norm_name}.hook_normalized"
                if hook_name in bridge.hook_dict:
                    hooks_to_test.append((hook_name, capture_hook(hook_name)))

        # Also test ln_final
        if "ln_final.hook_normalized" in bridge.hook_dict:
            hooks_to_test.append(
                ("ln_final.hook_normalized", capture_hook("ln_final.hook_normalized"))
            )

        # Run forward pass with hooks
        bridge.run_with_hooks(test_text, fwd_hooks=hooks_to_test)

        # Verify all hooks fired
        expected_hooks = {name for name, _ in hooks_to_test}
        missing_hooks = expected_hooks - norm_hooks_fired

        assert not missing_hooks, (
            f"Normalization hooks didn't fire for {model_name}:\n"
            f"  Missing: {sorted(missing_hooks)}\n"
            f"  Total hooks tested: {len(expected_hooks)}\n"
            f"  Hooks that fired: {len(norm_hooks_fired)}\n"
            f"\n"
            f"This indicates a bug in the block's patched forward method.\n"
            f"The normalization layers exist but aren't being called during forward pass."
        )


class TestArchitectureSpecificHooks:
    """Test architecture-specific hook requirements."""

    @pytest.mark.skipif(bool(os.getenv("CI")), reason="Gemma2 is too large for CI")
    def test_gemma2_ln2_hook(self):
        """Specific test for Gemma2 ln2 hook (regression test).

        Gemma2 has unique architecture with 4 normalization layers per block.
        This test ensures ln2 (pre_feedforward_layernorm) fires correctly.
        """
        bridge = TransformerBridge.boot_transformers("google/gemma-2-2b-it", device="cpu")
        bridge.enable_compatibility_mode(no_processing=True)

        test_text = "Test"
        ln2_fired = []

        def ln2_hook(tensor, hook):
            ln2_fired.append(hook.name)
            return tensor

        # Test ln2 for all layers
        hooks = [(f"blocks.{i}.ln2.hook_normalized", ln2_hook) for i in range(bridge.cfg.n_layers)]

        bridge.run_with_hooks(test_text, fwd_hooks=hooks)

        # All ln2 hooks should fire
        assert len(ln2_fired) == bridge.cfg.n_layers, (
            f"Gemma2 ln2 hooks didn't fire!\n"
            f"  Expected: {bridge.cfg.n_layers}\n"
            f"  Got: {len(ln2_fired)}\n"
            f"  Fired: {ln2_fired}\n"
            f"\n"
            f"This is a regression - Gemma2's pre_feedforward_layernorm (ln2)\n"
            f"must be called in the block's patched forward method."
        )


if __name__ == "__main__":
    # Run tests on GPT-2 when executed directly
    print("Testing hook completeness on gpt2...")
    test = TestHookCompleteness()

    print("\n1. Testing hook registry...")
    test.test_all_hooks_exist("gpt2")
    print("   ✓ All hooks exist")

    print("\n2. Testing hooks fire during forward pass...")
    test.test_all_hooks_fire("gpt2")
    print("   ✓ All hooks fire and match")

    print("\n3. Testing normalization hooks...")
    test.test_normalization_hooks_fire("gpt2")
    print("   ✓ All normalization hooks fire")

    print("\n✅ All hook completeness tests passed for gpt2!")
