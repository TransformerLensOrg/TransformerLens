import pytest
import torch

from transformer_lens import ActivationCache
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="module", params=["LN", "RMS"])
def activation_cache(request: pytest.FixtureRequest) -> ActivationCache:
    cfg = TransformerBridgeConfig(
        n_layers=2,
        d_model=16,
        n_ctx=8,
        d_head=4,
        n_heads=4,
        d_vocab=32,
        act_fn="gelu",
        normalization_type=request.param,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = TransformerBridge.boot_native(cfg)
    tokens = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]
    )
    _, cache = model.run_with_cache(tokens)
    return cache


@pytest.mark.parametrize("layer", [1, -1], ids=["cached-scale", "recomputed-final-ln"])
@pytest.mark.parametrize(
    "pos_slice", [None, (1, 3), -1], ids=["all-positions", "position-slice", "scalar-position"]
)
@pytest.mark.parametrize("apply_ln", [False, True], ids=["raw", "normalized"])
def test_batchless_accumulated_resid_matches_batched_row(
    activation_cache: ActivationCache,
    layer: int,
    pos_slice: tuple[int, int] | int | None,
    apply_ln: bool,
) -> None:
    batch_index = 1
    batched = activation_cache.accumulated_resid(
        layer=layer,
        pos_slice=pos_slice,
        apply_ln=apply_ln,
    )

    batchless_cache = activation_cache.apply_slice_to_batch_dim(batch_index)
    assert not batchless_cache.has_batch_dim
    batchless = batchless_cache.accumulated_resid(
        layer=layer,
        pos_slice=pos_slice,
        apply_ln=apply_ln,
    )

    expected = batched[:, batch_index]
    assert batchless.shape == expected.shape
    torch.testing.assert_close(batchless, expected)


@pytest.mark.parametrize("pos_slice", [None, (1, 4), [0, 2], -1])
@pytest.mark.parametrize("batch_slice", [None, [0, 2], 1])
@pytest.mark.parametrize("logit_difference", [False, True])
def test_logit_attrs_per_example_targets_match_individual_examples(
    activation_cache: ActivationCache,
    pos_slice: tuple[int, int] | list[int] | int | None,
    batch_slice: list[int] | int | None,
    logit_difference: bool,
) -> None:
    targets = torch.tensor([10, 11, 12])
    incorrect = torch.tensor([13, 14, 15]) if logit_difference else None
    batch_indices = torch.arange(3)
    if batch_slice is not None:
        targets = targets[batch_slice]
        batch_indices = batch_indices[batch_slice]
        if incorrect is not None:
            incorrect = incorrect[batch_slice]

    residual = activation_cache.decompose_resid(pos_slice=pos_slice)
    actual = activation_cache.logit_attrs(
        residual,
        tokens=targets,
        incorrect_tokens=incorrect,
        pos_slice=pos_slice,
        batch_slice=batch_slice,
    )

    expected_rows = []
    for index in batch_indices.reshape(-1).tolist():
        single_cache = activation_cache.apply_slice_to_batch_dim(index)
        single_residual = single_cache.decompose_resid(pos_slice=pos_slice)
        expected_rows.append(
            single_cache.logit_attrs(
                single_residual,
                tokens=10 + index,
                incorrect_tokens=13 + index if logit_difference else None,
                pos_slice=pos_slice,
                has_batch_dim=False,
            )
        )
    expected = (
        expected_rows[0] if isinstance(batch_slice, int) else torch.stack(expected_rows, dim=1)
    )
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("target_layout", ["scalar", "per-position", "batchless-per-position"])
def test_logit_attrs_preserves_other_target_layouts(
    activation_cache: ActivationCache,
    target_layout: str,
) -> None:
    cache = activation_cache
    targets = torch.tensor([[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]])
    if target_layout == "scalar":
        targets = torch.tensor(10)
    elif target_layout == "batchless-per-position":
        cache = cache.apply_slice_to_batch_dim(1)
        targets = targets[1]
    residual = cache.decompose_resid()
    actual = cache.logit_attrs(residual, targets, has_batch_dim=cache.has_batch_dim)

    # A scalar target at each position is independent of target broadcasting.
    expected_positions = []
    for position in range(4):
        position_targets = targets if targets.ndim == 0 else targets[..., position]
        expected_positions.append(
            cache.logit_attrs(
                residual[..., position, :],
                position_targets,
                pos_slice=position,
                has_batch_dim=cache.has_batch_dim,
            )
        )
    expected = torch.stack(expected_positions, dim=-1)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)
