"""Consolidated tests for TransformerBridge cache behavior.

Tests run_with_cache output, cache contents, names filtering, and
cache equality with HookedTransformer. Consolidates overlapping tests from:
- tests/integration/model_bridge/compatibility/test_hooks.py (cache tests)
- tests/integration/model_bridge/compatibility/test_legacy_hooks.py

Uses distilgpt2 (CI-cached) for speed unless gpt2-specific behavior is tested.
"""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="module")
def bridge_compat():
    """TransformerBridge with compatibility mode."""
    b = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    b.enable_compatibility_mode()
    return b


@pytest.fixture(scope="module")
def reference_ht():
    """HookedTransformer for comparison."""
    return HookedTransformer.from_pretrained("distilgpt2", device="cpu")


EXPECTED_HOOKS = [
    "hook_embed",
    "hook_pos_embed",
    "blocks.0.hook_resid_pre",
    "blocks.0.hook_resid_mid",
    "blocks.0.hook_resid_post",
    "blocks.0.ln1.hook_scale",
    "blocks.0.ln1.hook_normalized",
    "blocks.0.ln2.hook_scale",
    "blocks.0.ln2.hook_normalized",
    "blocks.0.attn.hook_q",
    "blocks.0.attn.hook_k",
    "blocks.0.attn.hook_v",
    "blocks.0.attn.hook_z",
    "blocks.0.attn.hook_attn_scores",
    "blocks.0.attn.hook_pattern",
    "blocks.0.attn.hook_result",
    "blocks.0.mlp.hook_pre",
    "blocks.0.mlp.hook_post",
    "blocks.0.hook_attn_out",
    "blocks.0.hook_mlp_out",
    "ln_final.hook_scale",
    "ln_final.hook_normalized",
]


class TestCacheBasics:
    """Test basic cache functionality."""

    def test_run_with_cache_returns_nonempty(self, bridge_compat):
        """run_with_cache returns a non-empty cache."""
        with torch.no_grad():
            _, cache = bridge_compat.run_with_cache("Hello world")
        assert len(cache) > 0

    def test_cache_contains_residual_hooks(self, bridge_compat):
        """Cache should contain residual stream hooks."""
        with torch.no_grad():
            _, cache = bridge_compat.run_with_cache("Hello world")
        cache_keys = list(cache.keys())
        assert any("hook_resid" in k for k in cache_keys)

    def test_cache_values_are_tensors(self, bridge_compat):
        """All cached values should be tensors with correct batch dimension."""
        with torch.no_grad():
            _, cache = bridge_compat.run_with_cache("Hello")
        for key, value in cache.items():
            assert isinstance(value, torch.Tensor), f"Cache[{key}] is {type(value)}"
            assert value.shape[0] == 1, f"Cache[{key}] batch dim is {value.shape[0]}"


class TestCacheNamesFilter:
    """Test cache names filtering."""

    def test_names_filter_returns_subset(self, bridge_compat):
        """names_filter should return only matching keys."""
        with torch.no_grad():
            _, full_cache = bridge_compat.run_with_cache("Hello")
            _, filtered_cache = bridge_compat.run_with_cache(
                "Hello",
                names_filter=lambda name: "hook_resid_pre" in name,
            )

        assert len(filtered_cache) > 0
        assert len(filtered_cache) < len(full_cache)
        for key in filtered_cache:
            assert "hook_resid_pre" in key, f"Unexpected key: {key}"


class TestCacheCompleteness:
    """Test that cache contains all expected hooks."""

    def test_all_expected_hooks_in_cache(self, bridge_compat):
        """Cache should contain all expected hook names."""
        _, cache = bridge_compat.run_with_cache("Hello World!")
        actual_keys = set(cache.keys())
        missing = set(EXPECTED_HOOKS) - actual_keys
        assert len(missing) == 0, f"Missing expected hooks: {sorted(missing)}"

    def test_expected_hooks_accessible_on_model(self, bridge_compat):
        """Expected hooks should be accessible as attributes on the model."""
        from transformer_lens.hook_points import HookPoint

        missing = []
        for hook_name in EXPECTED_HOOKS:
            parts = hook_name.split(".")
            current = bridge_compat
            try:
                for part in parts:
                    current = getattr(current, part)
                if not isinstance(current, HookPoint):
                    missing.append(hook_name)
            except AttributeError:
                missing.append(hook_name)

        assert len(missing) == 0, f"Hooks not accessible on model: {sorted(missing)}"


class TestCacheEqualityWithHookedTransformer:
    """Test that cache values match between bridge and HookedTransformer."""

    def test_cache_values_match(self, bridge_compat, reference_ht):
        """Cache activations should match between bridge and HookedTransformer.

        Note: Raw attention scores use different masking sentinels:
        HookedTransformer uses -inf, Bridge uses torch.finfo(dtype).min.
        Unmasked scores and resulting patterns should still match.
        """
        prompt = "Hello World!"
        _, bridge_cache = bridge_compat.run_with_cache(prompt)
        _, ht_cache = reference_ht.run_with_cache(prompt)

        for hook in EXPECTED_HOOKS:
            if hook not in bridge_cache or hook not in ht_cache:
                continue

            ht_act = ht_cache[hook]
            bridge_act = bridge_cache[hook]

            assert (
                ht_act.shape == bridge_act.shape
            ), f"Shape mismatch for {hook}: {ht_act.shape} vs {bridge_act.shape}"

            if hook == "blocks.0.attn.hook_attn_scores":
                # Different masking sentinels — compare only unmasked positions
                masked = torch.isinf(ht_act)
                unmasked = ~masked
                assert torch.allclose(
                    ht_act[unmasked], bridge_act[unmasked], atol=1e-4, rtol=1e-4
                ), "Unmasked attention scores should match"
                continue

            mean_diff = torch.abs(ht_act - bridge_act).mean()
            assert mean_diff < 0.5, f"Hook {hook} mismatch: mean abs diff = {mean_diff:.6f}"
