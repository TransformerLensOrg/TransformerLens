"""Tests for patching functions that are only covered by notebook cells.

Runs on a tiny random native bridge: the assertions are structural
(shape/finiteness/variation), so learned weights are unnecessary.
"""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.patching import get_act_patch_attn_head_all_pos_every


@pytest.fixture(scope="module")
def model():
    cfg = TransformerBridgeConfig(
        n_layers=2,
        d_model=64,
        d_head=16,
        n_heads=4,
        d_mlp=128,
        d_vocab=100,
        n_ctx=16,
        act_fn="gelu",
        seed=0,
    )
    bridge = TransformerBridge.boot_native(cfg)
    bridge.eval()
    return bridge


@pytest.fixture(scope="module")
def clean_cache(model):
    torch.manual_seed(0)
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 6))
    _, cache = model.run_with_cache(tokens)
    return cache


@pytest.fixture(scope="module")
def corrupted_tokens(model):
    torch.manual_seed(1)
    return torch.randint(0, model.cfg.d_vocab, (1, 6))


def test_get_act_patch_attn_head_all_pos_every_shape(model, corrupted_tokens, clean_cache):
    """Verify the function returns a [5, n_layers, n_heads] tensor."""

    def metric(logits):
        return logits[:, -1, :].sum()

    result = get_act_patch_attn_head_all_pos_every(model, corrupted_tokens, clean_cache, metric)

    assert result.shape == (5, model.cfg.n_layers, model.cfg.n_heads)


def test_get_act_patch_attn_head_all_pos_every_values_vary(model, corrupted_tokens, clean_cache):
    """Patching different heads should produce different metric values."""

    def metric(logits):
        return logits[:, -1, :].sum()

    result = get_act_patch_attn_head_all_pos_every(model, corrupted_tokens, clean_cache, metric)

    # Not all values should be identical — different heads have different effects
    assert not torch.all(result == result[0, 0, 0]), "All patch results are identical"
    # Values should be finite
    assert torch.isfinite(result).all()


# --- per-example results, multiple metrics, baselines and statistics -----------------


def _tiny_model(n_key_value_heads=None):
    cfg = TransformerBridgeConfig(
        n_layers=2,
        d_model=64,
        d_head=16,
        n_heads=4,
        d_mlp=128,
        d_vocab=100,
        n_ctx=16,
        act_fn="gelu",
        seed=0,
    )
    if n_key_value_heads is not None:
        cfg.n_key_value_heads = n_key_value_heads
    bridge = TransformerBridge.boot_native(cfg)
    bridge.eval()
    return bridge


@pytest.fixture(scope="module")
def batched():
    """A batch-4 clean/corrupt pair on the standard tiny model."""
    model = _tiny_model()
    torch.manual_seed(0)
    clean = torch.randint(0, model.cfg.d_vocab, (4, 6))
    torch.manual_seed(1)
    corrupt = torch.randint(0, model.cfg.d_vocab, (4, 6))
    _, cache = model.run_with_cache(clean)
    return model, clean, corrupt, cache


def _per_example(logits):
    return logits[:, -1, 3]


def _scalar(logits):
    return logits[:, -1, 3].mean()


def test_reduce_mean_equals_scalar_metric(batched):
    from transformer_lens.patching import get_act_patch_resid_pre

    model, _, corrupt, cache = batched
    legacy = get_act_patch_resid_pre(model, corrupt, cache, _scalar)
    averaged = get_act_patch_resid_pre(model, corrupt, cache, _per_example, reduce="mean")
    assert isinstance(averaged, torch.Tensor) and averaged.shape == legacy.shape
    torch.testing.assert_close(averaged, legacy, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "sweep", ["resid_pre", "attn_head_out_all_pos", "attn_head_pattern_by_pos"]
)
def test_per_example_values_match_single_example_sweeps(batched, sweep):
    import transformer_lens.patching as patching

    model, clean, corrupt, cache = batched
    fn = getattr(patching, f"get_act_patch_{sweep}")
    result = fn(model, corrupt, cache, _per_example, reduce="none")
    assert isinstance(result, patching.PatchingResult) and not result.reduced
    assert result.values.shape[-1] == 4
    for b in (0, 2):
        _, cache_b = model.run_with_cache(clean[b : b + 1])
        single = fn(model, corrupt[b : b + 1], cache_b, _per_example, reduce="none")
        torch.testing.assert_close(
            result.values[..., b], single.values[..., 0], atol=1e-5, rtol=1e-5
        )
    legacy = fn(model, corrupt, cache, _scalar)
    torch.testing.assert_close(result.mean(), legacy, atol=1e-6, rtol=1e-6)


def test_metric_shape_contract(batched):
    from transformer_lens.patching import get_act_patch_resid_pre

    model, _, corrupt, cache = batched
    with pytest.raises(ValueError, match=r"must return a \[batch\] tensor"):
        get_act_patch_resid_pre(model, corrupt, cache, _scalar, reduce="none")
    with pytest.raises(ValueError, match="must return a scalar"):
        get_act_patch_resid_pre(model, corrupt, cache, _per_example)
    with pytest.raises(ValueError, match="length 4"):
        get_act_patch_resid_pre(model, corrupt, cache, lambda l: l[:2, -1, 3], reduce="none")
    with pytest.raises(ValueError, match="reduce must be"):
        get_act_patch_resid_pre(model, corrupt, cache, _per_example, reduce="sum")
    with pytest.raises(ValueError, match="return_index_df"):
        get_act_patch_resid_pre(
            model, corrupt, cache, _per_example, reduce="none", return_index_df=True
        )


def test_multiple_metrics_share_one_sweep(batched):
    from transformer_lens.patching import get_act_patch_attn_head_out_all_pos

    model, _, corrupt, cache = batched
    other = lambda logits: logits[:, -1, 7]  # noqa: E731
    both = get_act_patch_attn_head_out_all_pos(
        model, corrupt, cache, {"first": _per_example, "second": other}, reduce="none"
    )
    assert both.has_metric_axis and both.metric_names == ["first", "second"]
    assert both.values.shape == (2, model.cfg.n_layers, model.cfg.n_heads, 4)
    alone = get_act_patch_attn_head_out_all_pos(model, corrupt, cache, other, reduce="none")
    torch.testing.assert_close(both["second"].values, alone.values)
    assert not both["second"].has_metric_axis
    with pytest.raises(KeyError, match="have"):
        both["third"]
    reduced = get_act_patch_attn_head_out_all_pos(
        model, corrupt, cache, {"first": _scalar}, reduce="scalar"
    )
    assert reduced.reduced and reduced.values.shape == (1, model.cfg.n_layers, model.cfg.n_heads)


def test_baseline_effect_and_normalization(batched):
    from transformer_lens.patching import get_act_patch_resid_pre

    model, clean, corrupt, cache = batched
    result = get_act_patch_resid_pre(
        model, corrupt, cache, _per_example, reduce="none", baseline=True
    )
    with torch.no_grad():
        corrupted_metric = _per_example(model(corrupt))
        clean_metric = _per_example(model(clean))
    torch.testing.assert_close(result.baseline, corrupted_metric, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        result.effect(), result.values - corrupted_metric, atol=1e-6, rtol=1e-6
    )
    expected = (result.values - corrupted_metric) / (clean_metric - corrupted_metric)
    torch.testing.assert_close(result.normalized(clean_metric), expected, atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="match baseline shape"):
        result.normalized(clean_metric[:2])
    without = get_act_patch_resid_pre(model, corrupt, cache, _per_example, reduce="none")
    with pytest.raises(ValueError, match="baseline=True"):
        without.effect()
    reduced = get_act_patch_resid_pre(
        model, corrupt, cache, _per_example, reduce="mean", baseline=True
    )
    assert reduced.reduced and reduced.baseline.shape == ()
    with pytest.raises(ValueError, match="reduce='none'"):
        reduced.stderr()


def test_statistics_on_patching_result(batched):
    from transformer_lens.patching import get_act_patch_resid_pre

    model, _, corrupt, cache = batched
    result = get_act_patch_resid_pre(
        model, corrupt, cache, _per_example, reduce="none", baseline=True
    )
    assert result.stderr().shape == result.mean().shape
    low, high = result.bootstrap_ci(n_resamples=100, seed=5)
    assert (low <= result.mean() + 1e-6).all() and (high >= result.mean() - 1e-6).all()
    again = result.bootstrap_ci(n_resamples=100, seed=5)
    assert torch.equal(again[0], low) and torch.equal(again[1], high)
    pvals = result.permutation_pvalue()
    assert pvals.shape == result.mean().shape
    # four examples: exact enumeration over 16 sign patterns
    assert ((pvals * 16).round() - pvals * 16).abs().max() < 1e-6
    frame = result.to_dataframe()
    assert list(frame.columns) == ["metric", "layer", "pos", "example", "value"]
    assert len(frame) == model.cfg.n_layers * 6 * 4


def test_flat_index_df_mode_keeps_examples(batched):
    from transformer_lens.patching import (
        generic_activation_patch,
        layer_pos_patch_setter,
        make_df_from_ranges,
    )

    model, _, corrupt, cache = batched
    index_df = make_df_from_ranges([model.cfg.n_layers, 2], ["layer", "pos"])
    flat = generic_activation_patch(
        model,
        corrupt,
        cache,
        _per_example,
        layer_pos_patch_setter,
        "resid_pre",
        index_df=index_df,
        reduce="none",
    )
    assert flat.axis_names == [] and flat.values.shape == (len(index_df), 4)
    legacy, df = generic_activation_patch(
        model,
        corrupt,
        cache,
        _scalar,
        layer_pos_patch_setter,
        "resid_pre",
        index_df=index_df,
        return_index_df=True,
    )
    torch.testing.assert_close(flat.mean(), legacy, atol=1e-6, rtol=1e-6)
    assert df.equals(flat.index_df)
    assert len(flat.to_dataframe()) == len(index_df) * 4


def test_every_helpers_stack_per_example_results_with_gqa_padding():
    from transformer_lens.patching import (
        get_act_patch_attn_head_all_pos_every,
        get_act_patch_attn_head_by_pos_every,
        get_act_patch_block_every,
    )

    model = _tiny_model(n_key_value_heads=2)
    torch.manual_seed(0)
    clean = torch.randint(0, model.cfg.d_vocab, (3, 5))
    torch.manual_seed(1)
    corrupt = torch.randint(0, model.cfg.d_vocab, (3, 5))
    _, cache = model.run_with_cache(clean)
    legacy = get_act_patch_attn_head_all_pos_every(model, corrupt, cache, _scalar)
    result = get_act_patch_attn_head_all_pos_every(
        model, corrupt, cache, _per_example, reduce="none"
    )
    assert result.values.shape == (5, model.cfg.n_layers, model.cfg.n_heads, 3)
    assert result.axis_names == ["patch_type", "layer", "head"]
    assert (result.values[2:4, :, 2:, :] == 0).all()  # k / v padded beyond n_key_value_heads
    torch.testing.assert_close(result.mean(), legacy, atol=1e-6, rtol=1e-6)
    legacy_pos = get_act_patch_attn_head_by_pos_every(model, corrupt, cache, _scalar)
    by_pos = get_act_patch_attn_head_by_pos_every(
        model, corrupt, cache, _per_example, reduce="none"
    )
    assert by_pos.values.shape == (5, model.cfg.n_layers, 5, model.cfg.n_heads, 3)
    torch.testing.assert_close(by_pos.mean(), legacy_pos, atol=1e-6, rtol=1e-6)
    legacy_block = get_act_patch_block_every(model, corrupt, cache, _scalar)
    block = get_act_patch_block_every(
        model, corrupt, cache, _per_example, reduce="none", baseline=True
    )
    assert block.values.shape == (3, model.cfg.n_layers, 5, 3) and block.baseline.shape == (3,)
    torch.testing.assert_close(block.mean(), legacy_block, atol=1e-6, rtol=1e-6)
    assert len(block.index_df) == 3 * model.cfg.n_layers * 5 and "patch_type" in block.index_df


def test_patching_requires_eval_mode(batched, monkeypatch):
    from transformer_lens.patching import get_act_patch_resid_pre

    model, _, corrupt, cache = batched

    def forward_ran(*args, **kwargs):
        raise AssertionError("the model was run before the eval-mode check")

    monkeypatch.setattr(model, "run_with_hooks", forward_ran)
    model.train()
    try:
        with pytest.raises(ValueError, match="evaluation mode"):
            get_act_patch_resid_pre(model, corrupt, cache, _scalar)
    finally:
        model.eval()


def test_stacked_by_pos_dataframe_labels_match_the_direct_pattern_sweep(batched):
    """The stacked frame is rebuilt from the padded grid, so pattern rows keep their cells."""
    from transformer_lens.patching import (
        get_act_patch_attn_head_by_pos_every,
        get_act_patch_attn_head_pattern_by_pos,
    )

    model, _, corrupt, cache = batched
    stacked = get_act_patch_attn_head_by_pos_every(
        model, corrupt, cache, _per_example, reduce="none"
    )
    direct = get_act_patch_attn_head_pattern_by_pos(
        model, corrupt, cache, _per_example, reduce="none"
    )
    assert stacked.patch_type_names == ["out", "q", "k", "v", "pattern"]
    frame = stacked.to_dataframe()
    assert list(frame.columns) == [
        "metric",
        "patch_type",
        "layer",
        "pos",
        "head",
        "patch_type_name",
        "example",
        "value",
    ]
    assert len(frame) == len(stacked.index_df) * 4 == 5 * model.cfg.n_layers * 6 * 4 * 4
    for layer, head, pos, example in [(1, 3, 2, 0), (0, 1, 5, 3)]:
        row = frame[
            (frame.patch_type_name == "pattern")
            & (frame.layer == layer)
            & (frame["head"] == head)
            & (frame.pos == pos)
            & (frame.example == example)
        ]
        assert len(row) == 1
        # direct sweep axes are (layer, head_index, dest_pos)
        assert row.value.item() == pytest.approx(direct.values[layer, head, pos, example].item())


def test_stacked_gqa_dataframe_has_zero_rows_for_padded_heads():
    from transformer_lens.patching import get_act_patch_attn_head_all_pos_every

    model = _tiny_model(n_key_value_heads=2)
    torch.manual_seed(0)
    clean = torch.randint(0, model.cfg.d_vocab, (3, 5))
    torch.manual_seed(1)
    corrupt = torch.randint(0, model.cfg.d_vocab, (3, 5))
    _, cache = model.run_with_cache(clean)
    stacked = get_act_patch_attn_head_all_pos_every(
        model, corrupt, cache, _per_example, reduce="none"
    )
    frame = stacked.to_dataframe()
    assert len(frame) == 5 * model.cfg.n_layers * model.cfg.n_heads * 3
    padded = frame[(frame.patch_type_name.isin(["k", "v"])) & (frame["head"] >= 2)]
    assert len(padded) == 2 * model.cfg.n_layers * 2 * 3 and (padded.value == 0).all()
    live = frame[(frame.patch_type_name == "k") & (frame["head"] < 2)]
    assert (live.value != 0).any()


def test_every_helpers_compute_the_baseline_once(batched, monkeypatch):
    from transformer_lens.patching import get_act_patch_block_every

    model, _, corrupt, cache = batched
    original = model.run_with_hooks
    unpatched_calls = []

    def counting(*args, **kwargs):
        if not kwargs.get("fwd_hooks"):
            unpatched_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "run_with_hooks", counting)
    result = get_act_patch_block_every(
        model, corrupt, cache, _per_example, reduce="none", baseline=True
    )
    assert len(unpatched_calls) == 1
    assert result.baseline is not None and result.baseline.shape == (4,)
    # a scalar metric with baseline=True also comes back typed, with one baseline forward
    unpatched_calls.clear()
    typed = get_act_patch_block_every(model, corrupt, cache, _scalar, baseline=True)
    assert len(unpatched_calls) == 1
    assert typed.reduced and typed.baseline is not None and typed.baseline.shape == ()
    assert typed.patch_type_names == ["resid_pre", "attn_out", "mlp_out"]


def test_scalar_baseline_by_pos_stack_matches_the_legacy_tensor(batched):
    """The wrapped default sweeps must carry their true axis names (the pattern sweep is
    (layer, head_index, dest_pos)) or the stack transposes or crashes."""
    from transformer_lens.patching import get_act_patch_attn_head_by_pos_every

    model, _, corrupt, cache = batched
    legacy = get_act_patch_attn_head_by_pos_every(model, corrupt, cache, _scalar)
    typed = get_act_patch_attn_head_by_pos_every(model, corrupt, cache, _scalar, baseline=True)
    assert typed.reduced and typed.axis_names == ["patch_type", "layer", "pos", "head"]
    assert typed.values.shape == legacy.shape == (5, model.cfg.n_layers, 6, model.cfg.n_heads)
    torch.testing.assert_close(typed.values, legacy)
    assert typed.baseline is not None and typed.baseline.shape == ()
