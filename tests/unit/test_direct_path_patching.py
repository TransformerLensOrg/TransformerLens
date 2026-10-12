"""Direct path patching on a tiny random 3-layer native bridge (no downloads)."""

import warnings

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.direct_path_patching import (
    _check_fold_ln,
    get_act_patch_direct_path,
    get_act_patch_direct_path_all_sources,
)


def _tiny_cfg(**overrides):
    defaults = dict(
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
        seed=0,
    )
    defaults.update(overrides)
    return TransformerBridgeConfig(**defaults)


@pytest.fixture(scope="module")
def tiny_model():
    """Tiny native bridge with LN folded; seed pins the init because unlucky xdist-worker RNG
    draws push the linear-LN error past tolerance."""
    model = TransformerBridge.boot_native(_tiny_cfg())
    model.process_weights(
        fold_ln=True, center_writing_weights=True, center_unembed=True, fold_value_biases=True
    )
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


class TestCheckFoldLn:
    def test_folded_model_no_warning(self, tiny_model):
        """No warning when LN is already folded (tiny_model fixture calls process_weights())."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_fold_ln(tiny_model)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0, "Should not warn when LN is folded"

    def test_unfolded_model_warns(self):
        """UserWarning fires when LN has a non-unit learned scale (pretrained, pre-fold)."""
        model = TransformerBridge.boot_native(_tiny_cfg(n_layers=2, seed=1))
        model.eval()
        with torch.no_grad():
            model.blocks[0].ln1.weight.fill_(2.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_fold_ln(model)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "fold" in str(user_warnings[0].message).lower()

    def test_legacy_w_attribute_branch(self):
        """The guard also reads the legacy ``.w`` layout when ``.weight`` is absent."""

        class LegacyLN:
            w = torch.full((8,), 2.0)

        class Block:
            ln1 = LegacyLN()

        class LegacyModel:
            blocks = [Block()]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_fold_ln(LegacyModel())
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1

    def test_offset_norm_identity_weight_no_warning(self):
        """Offset-RMS (Gemma) folded identity is w = 0; reading it raw would misfire."""

        class OffsetCfg:
            rmsnorm_uses_offset = True

        class OffsetLN:
            weight = torch.zeros(8)

        class Block:
            ln1 = OffsetLN()

        class OffsetModel:
            cfg = OffsetCfg()
            blocks = [Block()]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_fold_ln(OffsetModel())
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0, "offset identity (w=0) must read as folded"

    def test_offset_norm_learned_weight_warns(self):
        """Offset-RMS with a learned scale (1 + w != 1) still warns."""

        class OffsetCfg:
            rmsnorm_uses_offset = True

        class OffsetLN:
            weight = torch.full((8,), 0.7)

        class Block:
            ln1 = OffsetLN()

        class OffsetModel:
            cfg = OffsetCfg()
            blocks = [Block()]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_fold_ln(OffsetModel())
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1

    def test_no_crash_on_missing_attribute(self):
        """_check_fold_ln silently passes when the model has no .blocks[0].ln1."""

        class WeirdModel:
            class cfg:
                pass

            class blocks:
                pass

        _check_fold_ln(WeirdModel())


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


class TestCorrectness:
    @pytest.mark.parametrize("component", ["q", "k", "v"])
    def test_matches_frozen_ln_scale_reference(self, tiny_model, tokens_and_caches, component):
        """Matches a hand-written hook dividing delta_resid by the corrupted run's ln1 scale."""
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        src_layer, src_head = 0, 0
        dst_layer, dst_head = 2, 1

        correct_tok = 17
        incorrect_tok = 42

        def logit_diff(logits):
            return logits[0, -1, correct_tok] - logits[0, -1, incorrect_tok]

        W_O = tiny_model.blocks[src_layer].attn.W_O
        clean_z = clean_cache[f"blocks.{src_layer}.attn.hook_z"][:, :, src_head, :]
        corrupted_z = corrupted_cache[f"blocks.{src_layer}.attn.hook_z"][:, :, src_head, :]
        delta_resid = (clean_z @ W_O[src_head]) - (corrupted_z @ W_O[src_head])

        # Not the true LN forward: on tiny_model its gap rivals the patch effect, hiding scale bugs.
        ln_scale = corrupted_cache[f"blocks.{dst_layer}.ln1.hook_scale"]
        W_comp = getattr(tiny_model.blocks[dst_layer].attn, f"W_{component.upper()}")[dst_head]
        ref_delta = (delta_resid / ln_scale) @ W_comp

        def ref_hook(value, hook):
            if value.requires_grad:
                value = value.clone()
            value[:, :, dst_head, :] = value[:, :, dst_head, :] + ref_delta
            return value

        with torch.no_grad():
            ref_logits = tiny_model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(f"blocks.{dst_layer}.attn.hook_{component}", ref_hook)],
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
                component=component,
                verbose=False,
            )
        our_metric = results[dst_layer, dst_head].item()

        assert abs(our_metric - ref_metric) < 1e-5, (
            f"Direct-path patch {our_metric:.6f} disagrees with frozen-scale reference "
            f"{ref_metric:.6f} (diff={abs(our_metric - ref_metric):.2e})."
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
