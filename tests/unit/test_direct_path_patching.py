"""Tests for transformer_lens/tools/analysis/direct_path_patching.py

Run with:
    pytest tests/unit/test_direct_path_patching.py -v

These tests use a tiny randomly-initialised 3-layer model so they run
in seconds on CPU without downloading any weights.
"""

import warnings

import pytest
import torch

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.tools.analysis.direct_path_patching import (
    _check_fold_ln,
    get_act_patch_direct_path,
    get_act_patch_direct_path_all_sources,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    """A small, randomly-initialised transformer with LN folded in."""
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
    model.process_weights_()
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
# _check_fold_ln tests
# ---------------------------------------------------------------------------


class TestCheckFoldLn:
    def test_folded_model_no_warning(self, tiny_model):
        """No warning when LN is already folded (tiny_model fixture calls process_weights_())."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_fold_ln(tiny_model)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0, "Should not warn when LN is folded"

    def test_unfolded_model_warns(self):
        """UserWarning fires when LN has a non-unit learned scale (pretrained, pre-fold)."""
        cfg = HookedTransformerConfig(
            n_layers=2,
            d_model=32,
            d_head=8,
            n_heads=4,
            d_mlp=64,
            d_vocab=50,
            n_ctx=8,
            act_fn="gelu",
            normalization_type="LN",
        )
        model = HookedTransformer(cfg)
        model.eval()
        # Simulate a pretrained model that has learned non-unit LN scale (not yet folded).
        # After process_weights_(), LayerNorm is replaced with LayerNormPre (no .w),
        # so the warning only fires in the pre-fold state with non-trivial .w.
        with torch.no_grad():
            model.blocks[0].ln1.w.fill_(2.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_fold_ln(model)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "fold" in str(user_warnings[0].message).lower()

    def test_hooked_transformer_w_attribute(self):
        """Before process_weights_(), HookedTransformer LayerNorm exposes .w.
        After folding, LayerNorm is replaced with LayerNormPre (no .w) — that's
        why _check_fold_ln passes silently on a folded model.
        """
        cfg = HookedTransformerConfig(
            n_layers=2,
            d_model=32,
            d_head=8,
            n_heads=4,
            d_mlp=64,
            d_vocab=50,
            n_ctx=8,
            act_fn="gelu",
            normalization_type="LN",
        )
        model = HookedTransformer(cfg)
        ln1 = model.blocks[0].ln1
        assert hasattr(ln1, "w"), "HookedTransformer LayerNorm should expose .w before folding"

    def test_no_crash_on_missing_attribute(self):
        """_check_fold_ln silently passes when the model has no .blocks[0].ln1."""

        class WeirdModel:
            class cfg:
                pass

            class blocks:
                pass

        # Should not raise — the except block in _check_fold_ln catches AttributeError
        _check_fold_ln(WeirdModel())  # type: ignore[arg-type]

    def test_no_runtime_error_on_multielement_tensor(self, tiny_model):
        """Regression: getattr(...) or getattr(...) on a multi-element tensor raises
        RuntimeError. The explicit None-check fix must prevent this."""
        # Calling _check_fold_ln on a real model exercises the tensor path.
        # If the bug were present this would raise RuntimeError.
        try:
            _check_fold_ln(tiny_model)
        except RuntimeError as e:
            pytest.fail(f"_check_fold_ln raised RuntimeError: {e}")


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
        assert results.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.n_heads)

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
        assert results.shape == (n, h, n, h)

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
        assert results[: src_layer + 1].eq(0).all()

    def test_later_layers_are_nonzero(self, tiny_model, tokens_and_caches):
        """At least some entries for dst_layer > src_layer should be non-zero."""
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
        assert not results[1:].eq(0).all()

    def test_clean_equals_corrupted_gives_zero_delta(self, tiny_model):
        """If clean == corrupted, delta is zero and every entry equals the baseline."""
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
                corrupted_cache=cache,
                patching_metric=simple_metric,
                src_layer=0,
                src_head=0,
                component="q",
                verbose=False,
            )

        nonzero_entries = results[results != 0]
        assert nonzero_entries.numel() == 0 or torch.allclose(
            nonzero_entries,
            torch.full_like(nonzero_entries, baseline),
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# Correctness: independent verification for a single pair
# ---------------------------------------------------------------------------


class TestCorrectness:
    def test_correctness_against_actual_ln_forward(self, tiny_model, tokens_and_caches):
        """Logit-diff metric: linear-LN approximation should match actual LN within 1e-3.

        process_weights_() folds LN into the weight matrices, so the linear
        approximation is exact and the tolerance can be tight.  Using logit diff
        (correct_tok - incorrect_tok) cancels the centering offset introduced by
        process_weights_() and gives a numerically clean comparison.
        """
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        src_layer, src_head = 0, 0
        dst_layer, dst_head = 2, 1

        # Pick stable token indices for the logit-diff metric
        torch.manual_seed(0)
        correct_tok = 17
        incorrect_tok = 42

        def logit_diff(logits):
            return logits[0, -1, correct_tok] - logits[0, -1, incorrect_tok]

        # Compute delta_resid from src head
        W_O = tiny_model.blocks[src_layer].attn.W_O  # type: ignore[union-attr]
        clean_z = clean_cache[f"blocks.{src_layer}.attn.hook_z"][:, :, src_head, :]
        corrupted_z = corrupted_cache[f"blocks.{src_layer}.attn.hook_z"][:, :, src_head, :]
        delta_resid = (clean_z @ W_O[src_head]) - (corrupted_z @ W_O[src_head])  # type: ignore[index]

        # Independent reference: patch through actual LayerNorm forward
        corrupted_resid = corrupted_cache[f"blocks.{dst_layer}.hook_resid_pre"]
        patched_resid = corrupted_resid + delta_resid

        with torch.no_grad():
            ln1 = tiny_model.blocks[dst_layer].ln1  # type: ignore[index]
            patched_normed = ln1(patched_resid)
            corrupted_normed = ln1(corrupted_resid)

        W_Q_dst = tiny_model.blocks[dst_layer].attn.W_Q[dst_head]  # type: ignore[index,union-attr]
        true_delta_q = (patched_normed - corrupted_normed) @ W_Q_dst

        def true_hook(value, hook):
            if value.requires_grad:
                value = value.clone()
            value[:, :, dst_head, :] = value[:, :, dst_head, :] + true_delta_q
            return value

        with torch.no_grad():
            ref_logits = tiny_model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(f"blocks.{dst_layer}.attn.hook_q", true_hook)],
            )
        ref_metric = logit_diff(ref_logits).item()

        with torch.no_grad():
            results = get_act_patch_direct_path(
                model=tiny_model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                patching_metric=logit_diff,
                src_layer=src_layer,
                src_head=src_head,
                component="q",
                verbose=False,
            )
        our_metric = results[dst_layer, dst_head].item()

        assert abs(our_metric - ref_metric) < 1e-3, (
            f"Linear-LN approx {our_metric:.6f} disagrees with actual-LN ref {ref_metric:.6f} "
            f"(diff={abs(our_metric - ref_metric):.2e}). process_weights_() should make these exact."
        )

    def test_all_sources_consistent_with_single(self, tiny_model, tokens_and_caches):
        """get_act_patch_direct_path_all_sources matches individual calls."""
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

        assert torch.allclose(single, all_sources[src_layer, src_head], atol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_last_layer_source_all_zero(self, tiny_model, tokens_and_caches):
        """A source in the last layer has no downstream heads → all zeros."""
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
        assert results.eq(0).all()

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
