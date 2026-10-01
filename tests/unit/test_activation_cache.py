import pytest
import torch

from transformer_lens import ActivationCache
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities import Slice


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


@pytest.mark.parametrize("neuron_slice", [0, [0]], ids=["integer", "list"])
@pytest.mark.parametrize("projection_ndim", [1, 2], ids=["vector", "matrix"])
@pytest.mark.parametrize("has_batch_dim", [True, False], ids=["batched", "batchless"])
@pytest.mark.parametrize("pos_slice", [None, -1], ids=["all-positions", "last-position"])
@torch.no_grad()
def test_get_neuron_results_projection_matches_unprojected(
    activation_cache: ActivationCache,
    neuron_slice: int | list[int],
    projection_ndim: int,
    has_batch_dim: bool,
    pos_slice: int | None,
) -> None:
    cache = activation_cache if has_batch_dim else activation_cache.apply_slice_to_batch_dim(0)
    full = cache.get_neuron_results(
        layer=0,
        neuron_slice=neuron_slice,
        pos_slice=pos_slice,
    )
    projection_shape = (full.shape[-1],) if projection_ndim == 1 else (full.shape[-1], 2)
    direction = torch.randn(
        projection_shape,
        dtype=full.dtype,
        device=full.device,
    )
    expected = full @ direction

    actual = cache.get_neuron_results(
        layer=0,
        neuron_slice=neuron_slice,
        pos_slice=pos_slice,
        project_output_onto=direction,
    )

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("incl_remainder", [False, True], ids=["no-remainder", "remainder"])
@pytest.mark.parametrize(
    "projection_ndim", [None, 1, 2], ids=["unprojected", "vector-projection", "matrix-projection"]
)
@pytest.mark.parametrize("apply_ln", [False, True], ids=["raw", "normalized"])
@pytest.mark.parametrize("wrap_in_slice", [False, True], ids=["bare-int", "Slice-int"])
@torch.no_grad()
def test_stack_neuron_results_integer_neuron_slice_matches_list(
    activation_cache: ActivationCache,
    wrap_in_slice: bool,
    apply_ln: bool,
    projection_ndim: int | None,
    incl_remainder: bool,
) -> None:
    """An int ``neuron_slice`` (bare or ``Slice(n)``) keeps the neuron axis and matches ``[n]``."""
    n_layers = activation_cache.model.cfg.n_layers
    d_model = activation_cache.model.cfg.d_model
    neuron = 3
    neuron_slice = Slice(neuron) if wrap_in_slice else neuron
    direction = None
    if projection_ndim is not None:
        projection_shape = (d_model,) if projection_ndim == 1 else (d_model, 2)
        direction = torch.randn(projection_shape, generator=torch.Generator().manual_seed(0))
    kwargs = dict(
        layer=n_layers,
        apply_ln=apply_ln,
        project_output_onto=direction,
        incl_remainder=incl_remainder,
        return_labels=True,
    )

    expected, expected_labels = activation_cache.stack_neuron_results(
        neuron_slice=[neuron], **kwargs
    )
    actual, labels = activation_cache.stack_neuron_results(neuron_slice=neuron_slice, **kwargs)

    assert expected.shape[0] == n_layers + int(incl_remainder)
    assert labels == expected_labels
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("wrap_in_slice", [False, True], ids=["bare-int", "Slice-int"])
@torch.no_grad()
def test_stack_neuron_results_integer_neuron_slice_without_labels(
    activation_cache: ActivationCache,
    wrap_in_slice: bool,
) -> None:
    """With ``return_labels=False`` an int ``neuron_slice`` still returns the ``[n]`` stack."""
    layer = activation_cache.model.cfg.n_layers
    neuron = 3
    neuron_slice = Slice(neuron) if wrap_in_slice else neuron

    expected = activation_cache.stack_neuron_results(layer, neuron_slice=[neuron])
    actual = activation_cache.stack_neuron_results(layer, neuron_slice=neuron_slice)

    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)
