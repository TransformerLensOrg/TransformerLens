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


@pytest.mark.parametrize("has_batch_dim", [True, False], ids=["batched", "batchless"])
@pytest.mark.parametrize(
    "valid_layers,stale_layer",
    [
        ((), None),
        ((0,), None),
        ((1,), None),
        ((0, 1), None),
        ((0,), 1),
        ((1,), 0),
    ],
    ids=["missing-all", "valid-first", "valid-last", "valid-all", "stale-last", "stale-first"],
)
@torch.no_grad()
def test_compute_head_results_handles_per_layer_cache(
    activation_cache: ActivationCache,
    caplog: pytest.LogCaptureFixture,
    has_batch_dim: bool,
    valid_layers: tuple[int, ...],
    stale_layer: int | None,
) -> None:
    cache = ActivationCache(
        {key: value.clone() for key, value in activation_cache.cache_dict.items()},
        activation_cache.model,
    )
    if not has_batch_dim:
        cache = cache.apply_slice_to_batch_dim(0)

    expected = [
        torch.einsum(
            "...hd,hdm->...hm",
            cache[("z", layer, "attn")],
            cache.model.blocks[layer].attn.W_O,
        )
        for layer in range(cache.model.cfg.n_layers)
    ]
    for layer in range(cache.model.cfg.n_layers):
        cache.cache_dict.pop(f"blocks.{layer}.attn.hook_result", None)
    for layer in valid_layers:
        expected[layer] = expected[layer] + 7
        cache.cache_dict[f"blocks.{layer}.attn.hook_result"] = expected[layer].clone()
        del cache.cache_dict[f"blocks.{layer}.attn.hook_z"]
    if stale_layer is not None:
        cache.cache_dict[f"blocks.{stale_layer}.attn.hook_result"] = expected[stale_layer].sum(
            dim=-2
        )
    preserved = {
        layer: cache.cache_dict[f"blocks.{layer}.attn.hook_result"] for layer in valid_layers
    }

    caplog.clear()
    with caplog.at_level("WARNING"):
        cache.compute_head_results()
    already_cached = "Tried to compute head results when they were already cached"
    assert (already_cached in caplog.messages) == (len(valid_layers) == cache.model.cfg.n_layers)

    for layer, result in enumerate(expected):
        torch.testing.assert_close(cache[("result", layer, "attn")], result)
    for layer, result in preserved.items():
        assert cache[("result", layer, "attn")] is result

    computed = [cache[("result", layer, "attn")] for layer in range(cache.model.cfg.n_layers)]
    stacked = cache.stack_head_results()
    torch.testing.assert_close(stacked, torch.cat(expected, dim=-2).movedim(-2, 0))
    for layer, result in enumerate(computed):
        assert cache[("result", layer, "attn")] is result


@pytest.mark.parametrize("has_batch_dim", [True, False], ids=["batched", "batchless"])
@torch.no_grad()
def test_compute_head_results_requires_z_for_missing_results(
    activation_cache: ActivationCache,
    has_batch_dim: bool,
) -> None:
    cache = ActivationCache(
        {key: value.clone() for key, value in activation_cache.cache_dict.items()},
        activation_cache.model,
    )
    if not has_batch_dim:
        cache = cache.apply_slice_to_batch_dim(0)
    cache.cache_dict.pop("blocks.0.attn.hook_result", None)
    del cache.cache_dict["blocks.0.attn.hook_z"]

    with pytest.raises(KeyError, match=r"blocks\.0\.attn\.hook_z"):
        cache.compute_head_results()


@pytest.mark.parametrize("has_batch_dim", [True, False], ids=["batched", "batchless"])
@torch.no_grad()
def test_compute_head_results_recomputes_wrong_head_count(
    activation_cache: ActivationCache,
    has_batch_dim: bool,
) -> None:
    cache = ActivationCache(
        {key: value.clone() for key, value in activation_cache.cache_dict.items()},
        activation_cache.model,
    )
    if not has_batch_dim:
        cache = cache.apply_slice_to_batch_dim(0)
    expected = torch.einsum(
        "...hd,hdm->...hm", cache[("z", 0, "attn")], cache.model.blocks[0].attn.W_O
    )
    cache.cache_dict["blocks.0.attn.hook_result"] = expected[..., :-1, :].clone()
    preserved = (
        torch.einsum("...hd,hdm->...hm", cache[("z", 1, "attn")], cache.model.blocks[1].attn.W_O)
        + 7
    )
    cache.cache_dict["blocks.1.attn.hook_result"] = preserved
    del cache.cache_dict["blocks.1.attn.hook_z"]

    cache.compute_head_results()

    torch.testing.assert_close(cache[("result", 0, "attn")], expected)
    assert cache[("result", 1, "attn")] is preserved
