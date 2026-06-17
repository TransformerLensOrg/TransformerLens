"""Tests for direct_path_patching.py

Run with:
    pytest tests/test_direct_path_patching.py -v

These tests use a tiny randomly-initialised 2-layer GPT-2 config so they run
in seconds on CPU without downloading any weights.
"""

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from transformer_lens.direct_path_patching import (
    get_act_patch_direct_path,
    get_act_patch_direct_path_all_sources,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    """A small, randomly-initialised transformer for fast tests."""
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_model=64,
        d_head=16,
        n_heads=4,
        d_mlp=128,
        d_vocab=100,
        n_ctx=16,
        act_fn="gelu",
        normalization_type="LN",
        attn_only=False,
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokens_and_caches(tiny_model):
    """Precompute clean/corrupted tokens and their caches."""
    torch.manual_seed(42)
    clean_tokens = torch.randint(0, 100, (1, 8))
    corrupted_tokens = torch.randint(0, 100, (1, 8))

    with torch.no_grad():
        _, clean_cache = tiny_model.run_with_cache(clean_tokens)
        _, corrupted_cache = tiny_model.run_with_cache(corrupted_tokens)

    return clean_tokens, corrupted_tokens, clean_cache, corrupted_cache


def simple_metric(logits):
    """Sum of last-token logits — a trivially differentiable scalar."""
    return logits[0, -1, :].sum()


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_single_source_shape(self, tiny_model, tokens_and_caches):
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        with torch.no_grad():
            results = get_act_patch_direct_path(
                model=tiny_model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=simple_metric,
                src_layer=0,
                src_head=0,
                component="q",
                verbose=False,
            )
        assert results.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.n_heads), (
            f"Expected ({tiny_model.cfg.n_layers}, {tiny_model.cfg.n_heads}), got {results.shape}"
        )

    def test_all_sources_shape(self, tiny_model, tokens_and_caches):
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        with torch.no_grad():
            results = get_act_patch_direct_path_all_sources(
                model=tiny_model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=simple_metric,
                component="q",
                verbose=False,
            )
        n = tiny_model.cfg.n_layers
        h = tiny_model.cfg.n_heads
        assert results.shape == (n, h, n, h), (
            f"Expected ({n},{h},{n},{h}), got {results.shape}"
        )

    @pytest.mark.parametrize("component", ["q", "k", "v"])
    def test_all_components(self, tiny_model, tokens_and_caches, component):
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        with torch.no_grad():
            results = get_act_patch_direct_path(
                model=tiny_model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=simple_metric,
                src_layer=0,
                src_head=1,
                component=component,
                verbose=False,
            )
        assert results.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.n_heads)


# ---------------------------------------------------------------------------
# Causal structure tests
# ---------------------------------------------------------------------------


class TestCausalStructure:
    def test_earlier_layers_are_zero(self, tiny_model, tokens_and_caches):
        """Entries for dst_layer <= src_layer must be exactly 0 (no causal path)."""
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        src_layer = 1
        with torch.no_grad():
            results = get_act_patch_direct_path(
                model=tiny_model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=simple_metric,
                src_layer=src_layer,
                src_head=0,
                component="q",
                verbose=False,
            )
        # Rows 0..src_layer (inclusive) should be exactly 0
        assert results[: src_layer + 1].eq(0).all(), (
            "Expected zero for dst_layer <= src_layer, but got non-zero entries."
        )

    def test_later_layers_are_nonzero(self, tiny_model, tokens_and_caches):
        """At least some entries for dst_layer > src_layer should be non-zero
        when clean != corrupted."""
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        src_layer = 0
        with torch.no_grad():
            results = get_act_patch_direct_path(
                model=tiny_model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=simple_metric,
                src_layer=src_layer,
                src_head=0,
                component="q",
                verbose=False,
            )
        downstream = results[src_layer + 1 :]
        assert not downstream.eq(0).all(), (
            "Expected at least some non-zero values for downstream layers, "
            "but all were zero (this is extremely unlikely with random weights)."
        )

    def test_clean_equals_corrupted_gives_zero_delta(self, tiny_model):
        """If clean == corrupted, the delta is zero and the metric should be
        identical for all pairs (patching in nothing should change nothing)."""
        torch.manual_seed(7)
        tokens = torch.randint(0, 100, (1, 6))
        with torch.no_grad():
            baseline_logits, cache = tiny_model.run_with_cache(tokens)

        baseline = simple_metric(baseline_logits).item()

        with torch.no_grad():
            results = get_act_patch_direct_path(
                model=tiny_model,
                corrupted_tokens=tokens,
                clean_cache=cache,
                corrupted_cache=cache,   # same cache → delta = 0
                patching_metric=simple_metric,
                src_layer=0,
                src_head=0,
                component="q",
                verbose=False,
            )

        # Every entry should equal the baseline (delta is zero, so hooks do nothing)
        nonzero_entries = results[results != 0]
        assert nonzero_entries.numel() == 0 or torch.allclose(
            nonzero_entries,
            torch.full_like(nonzero_entries, baseline),
            atol=1e-4,
        ), "When clean==corrupted, metric should equal baseline for all pairs."


# ---------------------------------------------------------------------------
# Correctness: manual verification for a single pair
# ---------------------------------------------------------------------------


class TestCorrectness:
    def test_manual_patch_matches_function(self, tiny_model, tokens_and_caches):
        """Manually apply the linear approximation for one (src→dst) pair and
        compare with the function's output."""
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        src_layer, src_head = 0, 0
        dst_layer, dst_head = 2, 1
        component = "q"

        # --- Manually compute the expected patched metric ---
        W_O = tiny_model.blocks[src_layer].attn.W_O  # [n_heads, d_head, d_model]
        def _head_result(cache, h):
            z = cache[f"blocks.{src_layer}.attn.hook_z"][:, :, h, :]
            return z @ W_O[h]
        delta_resid = _head_result(clean_cache, src_head) - _head_result(corrupted_cache, src_head)
        ln_scale = corrupted_cache[f"blocks.{dst_layer}.ln1.hook_scale"]
        W_Q_dst = tiny_model.blocks[dst_layer].attn.W_Q[dst_head]  # [d_model, d_head]
        delta_q = (delta_resid / ln_scale) @ W_Q_dst

        def manual_hook(value, hook):
            if value.requires_grad:
                value = value.clone()
            value[:, :, dst_head, :] = value[:, :, dst_head, :] + delta_q
            return value

        with torch.no_grad():
            patched_logits = tiny_model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(f"blocks.{dst_layer}.attn.hook_q", manual_hook)],
            )
        expected = simple_metric(patched_logits).item()

        # --- Get the function's result ---
        with torch.no_grad():
            results = get_act_patch_direct_path(
                model=tiny_model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=simple_metric,
                src_layer=src_layer,
                src_head=src_head,
                component=component,
                verbose=False,
            )

        actual = results[dst_layer, dst_head].item()
        assert abs(actual - expected) < 1e-4, (
            f"Manual patch gave {expected:.6f} but function gave {actual:.6f}"
        )

    def test_all_sources_consistent_with_single(self, tiny_model, tokens_and_caches):
        """get_act_patch_direct_path_all_sources should give the same result as
        calling get_act_patch_direct_path for each source individually."""
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches

        src_layer, src_head = 0, 2

        with torch.no_grad():
            single = get_act_patch_direct_path(
                model=tiny_model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=simple_metric,
                src_layer=src_layer,
                src_head=src_head,
                component="q",
                verbose=False,
            )
            all_sources = get_act_patch_direct_path_all_sources(
                model=tiny_model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=simple_metric,
                component="q",
                verbose=False,
            )

        assert torch.allclose(
            single, all_sources[src_layer, src_head], atol=1e-5
        ), "all_sources result doesn't match single-source call."


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_last_layer_source_all_zero(self, tiny_model, tokens_and_caches):
        """A source head in the last layer has no downstream heads → all zeros."""
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        src_layer = tiny_model.cfg.n_layers - 1
        with torch.no_grad():
            results = get_act_patch_direct_path(
                model=tiny_model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=simple_metric,
                src_layer=src_layer,
                src_head=0,
                component="q",
                verbose=False,
            )
        assert results.eq(0).all(), (
            "Source in last layer should produce all-zero results (no downstream)."
        )

    def test_returns_cpu_tensor(self, tiny_model, tokens_and_caches):
        """Return tensor should be on the same device as the model."""
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        with torch.no_grad():
            results = get_act_patch_direct_path(
                model=tiny_model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=simple_metric,
                src_layer=0,
                src_head=0,
                component="q",
                verbose=False,
            )
        assert results.device.type == tiny_model.cfg.device or results.is_cpu
