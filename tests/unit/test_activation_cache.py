from types import SimpleNamespace

import pytest
import torch

from transformer_lens import ActivationCache
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities import Slice


def _head_results_reference(z: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # Independent per-head matmuls keep the float64 reference small at wide model widths.
    return torch.stack(
        [z[..., head, :] @ weights[head] for head in range(weights.shape[0])], dim=-2
    )


@pytest.mark.parametrize("near_zero", [False, True])
@torch.enable_grad()
def test_compute_head_results_matches_wide_float32_gradients(
    monkeypatch: pytest.MonkeyPatch,
    near_zero: bool,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(20261005)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    z = torch.randn(1, 32, 32, 128, device=device, generator=generator, requires_grad=True)
    weights = torch.randn(32, 128, 4096, device=device, generator=generator, requires_grad=True)
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=1, n_heads=32),
        blocks=[SimpleNamespace(attn=SimpleNamespace(W_O=weights))],
    )
    cache = ActivationCache({"blocks.0.attn.hook_z": z}, model)
    # Freeze a head-major cotangent independently of the implementation's output layout.
    grad_output = torch.randn(32, 1, 32, 4096, device=device, generator=generator).movedim(0, -2)
    if near_zero:
        with torch.no_grad():
            weights[..., 1::2] = weights[..., 0::2]
        grad_output[..., 1::2] = -grad_output[..., 0::2] + 2**-16
    reference_z = z.detach().double().cpu().requires_grad_()
    reference_weights = weights.detach().double().cpu().requires_grad_()
    expected = _head_results_reference(reference_z, reference_weights)
    expected_grads = torch.autograd.grad(
        expected, (reference_z, reference_weights), grad_output.double().cpu()
    )

    cache.compute_head_results()

    actual = cache["blocks.0.attn.hook_result"]
    actual_grads = torch.autograd.grad(actual, (z, weights), grad_output)
    torch.testing.assert_close(actual.double().cpu(), expected, rtol=2e-5, atol=2e-4)
    # Activation gradients reduce over 4096 terms and can nearly cancel across BLAS backends.
    torch.testing.assert_close(
        actual_grads[0].double().cpu(), expected_grads[0], rtol=1e-4, atol=1e-3
    )
    torch.testing.assert_close(
        actual_grads[1].double().cpu(), expected_grads[1], rtol=2e-5, atol=2e-4
    )
    absolute_products = _head_results_reference(
        grad_output.double().cpu().abs(), reference_weights.detach().abs().transpose(-1, -2)
    )
    error = actual_grads[0].double().cpu() - expected_grads[0]
    assert error.norm() <= 8 * torch.finfo(torch.float32).eps * absolute_products.norm()


@torch.enable_grad()
def test_compute_head_results_preserves_cancelling_float16_weight_gradients() -> None:
    z = torch.full((32, 4, 32), 5000, dtype=torch.float16)
    z[16:] = -5000
    weights = torch.zeros(4, 32, 4096, dtype=torch.float16, requires_grad=True)
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=1, n_heads=4),
        blocks=[SimpleNamespace(attn=SimpleNamespace(W_O=weights))],
    )
    cache = ActivationCache({"blocks.0.attn.hook_z": z}, model, has_batch_dim=False)
    reference_weights = weights.detach().double().requires_grad_()
    expected = _head_results_reference(z.double(), reference_weights)
    expected_grad = torch.autograd.grad(expected.sum(), reference_weights)[0]
    assert torch.count_nonzero(expected_grad) == 0

    cache.compute_head_results()

    actual_grad = torch.autograd.grad(cache["blocks.0.attn.hook_result"].sum(), weights)[0]
    torch.testing.assert_close(actual_grad.double(), expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("weight_grad", [False, True])
@pytest.mark.parametrize(
    "has_batch_dim,dtype,noncontiguous",
    [
        (True, torch.float32, False),
        (False, torch.float64, True),
        (True, torch.float16, True),
        (False, torch.bfloat16, False),
    ],
)
@torch.enable_grad()
def test_compute_head_results_matches_float64_values_and_gradients(
    has_batch_dim: bool,
    dtype: torch.dtype,
    noncontiguous: bool,
    weight_grad: bool,
) -> None:
    shape = (2, 9, 7, 32) if has_batch_dim else (19, 7, 32)
    z = torch.randn(shape, dtype=dtype)
    weights = torch.randn(7, 32, 4096, dtype=dtype)
    if noncontiguous:
        z = z.transpose(-1, -2).contiguous().transpose(-1, -2)
        weights = weights.transpose(-1, -2).contiguous().transpose(-1, -2)
    z.requires_grad_()
    weights.requires_grad_(weight_grad)
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=1, n_heads=7),
        blocks=[SimpleNamespace(attn=SimpleNamespace(W_O=weights))],
    )
    cache = ActivationCache({"blocks.0.attn.hook_z": z}, model, has_batch_dim=has_batch_dim)
    reference_z = z.detach().double().requires_grad_()
    reference_weights = weights.detach().double().requires_grad_(weight_grad)
    expected = _head_results_reference(reference_z, reference_weights)
    tolerances = {
        torch.float64: (1e-12, 1e-12),
        torch.float32: (2e-5, 2e-4),
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }
    rtol, atol = tolerances[dtype]

    cache.compute_head_results()

    actual = cache["blocks.0.attn.hook_result"]
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.double(), expected, rtol=rtol, atol=atol)
    grad_output = torch.randn_like(actual)
    inputs = (z, weights) if weight_grad else (z,)
    reference_inputs = (reference_z, reference_weights) if weight_grad else (reference_z,)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output.double())
    actual_grads = torch.autograd.grad(actual, inputs, grad_output)
    for gradient_index, (actual_grad, expected_grad) in enumerate(
        zip(actual_grads, expected_grads)
    ):
        if dtype == torch.float32 and gradient_index == 0:
            torch.testing.assert_close(actual_grad.double(), expected_grad, rtol=1e-4, atol=1e-3)
        else:
            torch.testing.assert_close(actual_grad.double(), expected_grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "z_dtype,weight_dtype",
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float64),
    ],
)
@pytest.mark.parametrize("has_batch_dim", [False, True])
@torch.enable_grad()
def test_compute_head_results_promotes_mixed_dtypes(
    z_dtype: torch.dtype, weight_dtype: torch.dtype, has_batch_dim: bool
) -> None:
    shape = (2, 5, 3, 8) if has_batch_dim else (5, 3, 8)
    z = torch.randn(shape, dtype=z_dtype, requires_grad=True)
    weights = torch.randn(3, 8, 17, dtype=weight_dtype, requires_grad=True)
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=1, n_heads=3),
        blocks=[SimpleNamespace(attn=SimpleNamespace(W_O=weights))],
    )
    cache = ActivationCache({"blocks.0.attn.hook_z": z}, model, has_batch_dim=has_batch_dim)
    reference_z = z.detach().double().requires_grad_()
    reference_weights = weights.detach().double().requires_grad_()
    expected = _head_results_reference(reference_z, reference_weights)

    cache.compute_head_results()

    actual = cache["blocks.0.attn.hook_result"]
    assert actual.dtype == torch.promote_types(z_dtype, weight_dtype)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-5)
    grad_output = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (z, weights), grad_output)
    expected_grads = torch.autograd.grad(
        expected, (reference_z, reference_weights), grad_output.double()
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        # The original leaf dtype determines gradient rounding after promotion.
        torch.testing.assert_close(actual_grad, expected_grad.to(actual_grad.dtype))


@pytest.mark.parametrize(
    "n_heads,d_head,n_tokens,grad_enabled",
    [
        (4, 32, 96, False),
        (32, 128, 2, False),
        (4, 32, 64, True),
        (32, 128, 2, True),
    ],
)
def test_compute_head_results_limits_temporary_allocation(
    n_heads: int, d_head: int, n_tokens: int, grad_enabled: bool
) -> None:
    z = torch.randn(1, n_tokens, n_heads, d_head, dtype=torch.float32)
    weights = torch.randn(n_heads, d_head, 4096, dtype=torch.float32, requires_grad=True)
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=1, n_heads=n_heads),
        blocks=[SimpleNamespace(attn=SimpleNamespace(W_O=weights))],
    )
    cache = ActivationCache({"blocks.0.attn.hook_z": z}, model)

    with torch.set_grad_enabled(grad_enabled):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True, acc_events=True
        ) as profile:
            cache.compute_head_results()

    allocations = [event.cpu_memory_usage for event in profile.events()]
    actual = cache["blocks.0.attn.hook_result"]
    assert max(allocations) <= actual.numel() * actual.element_size()
    expected = _head_results_reference(z.double(), weights.detach().double())
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-4)


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
    assert isinstance(residual, torch.Tensor)
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
        assert isinstance(single_residual, torch.Tensor)
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


@pytest.mark.parametrize(
    "target_layout",
    [
        "scalar",
        "per-position",
        "batchless-per-position",
        "shared-per-position",
        "single-prompt-per-position",
    ],
)
@pytest.mark.parametrize("logit_difference", [False, True])
def test_logit_attrs_preserves_other_target_layouts(
    activation_cache: ActivationCache,
    target_layout: str,
    logit_difference: bool,
) -> None:
    cache = activation_cache
    targets = torch.tensor([[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]])
    if target_layout == "scalar":
        targets = torch.tensor(10)
    elif target_layout == "batchless-per-position":
        cache = cache.apply_slice_to_batch_dim(1)
        targets = targets[1]
    elif target_layout in ("shared-per-position", "single-prompt-per-position"):
        targets = targets[1]
        if target_layout == "single-prompt-per-position":
            cache = cache.apply_slice_to_batch_dim([1])
    incorrect = targets + 1 if logit_difference else None
    residual = cache.decompose_resid()
    assert isinstance(residual, torch.Tensor)
    actual = cache.logit_attrs(
        residual, targets, incorrect_tokens=incorrect, has_batch_dim=cache.has_batch_dim
    )

    # A scalar target at each position is independent of target broadcasting.
    expected_positions = []
    for position in range(4):
        position_targets = targets if targets.ndim == 0 else targets[..., position]
        expected_positions.append(
            cache.logit_attrs(
                residual[..., position, :],
                position_targets,
                incorrect_tokens=position_targets + 1 if logit_difference else None,
                pos_slice=position,
                has_batch_dim=cache.has_batch_dim,
            )
        )
    expected = torch.stack(expected_positions, dim=-1)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("projection_ndim", [1, 2], ids=["vector", "matrix"])
@torch.no_grad()
def test_full_resid_decomposition_projected_ln_uses_mlp_input_scale(
    activation_cache: ActivationCache,
    projection_ndim: int,
) -> None:
    """With ``mlp_input=True`` the fused LN+projection path must use the ln2 scale.

    ``project_output_onto`` only folds the projection into the decomposition, so the result
    must equal the unprojected ``apply_ln=True`` stack projected afterwards, and the stack
    must still sum to the normalized MLP input ``LN2(resid_mid)`` projected onto the same
    directions.
    """
    layer = 1
    d_model = activation_cache.model.cfg.d_model
    projection_shape = (d_model,) if projection_ndim == 1 else (d_model, 2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        direction = torch.randn(projection_shape)

    unprojected = activation_cache.get_full_resid_decomposition(
        layer=layer, mlp_input=True, apply_ln=True
    )
    expected = unprojected @ direction

    actual = activation_cache.get_full_resid_decomposition(
        layer=layer, mlp_input=True, apply_ln=True, project_output_onto=direction
    )

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)

    normalized_mlp_input = activation_cache.apply_ln_to_stack(
        activation_cache[("resid_mid", layer)][None], layer, mlp_input=True
    )[0]
    torch.testing.assert_close(actual.sum(dim=0), normalized_mlp_input @ direction)


@pytest.mark.parametrize("incl_remainder", [False, True], ids=["neurons", "with-remainder"])
@torch.no_grad()
def test_stack_neuron_results_projected_ln_honours_mlp_input(
    activation_cache: ActivationCache,
    incl_remainder: bool,
) -> None:
    """The fused LN+projection neuron stack matches the unfused one for ``mlp_input=True``."""
    layer = 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        direction = torch.randn(activation_cache.model.cfg.d_model)

    expected = (
        activation_cache.stack_neuron_results(
            layer, apply_ln=True, incl_remainder=incl_remainder, mlp_input=True
        )
        @ direction
    )
    actual = activation_cache.stack_neuron_results(
        layer,
        apply_ln=True,
        incl_remainder=incl_remainder,
        project_output_onto=direction,
        mlp_input=True,
    )

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("apply_ln", [False, True], ids=["raw", "ln"])
@pytest.mark.parametrize("project", [False, True], ids=["unprojected", "projected"])
@torch.no_grad()
def test_stack_neuron_results_mlp_input_remainder_fills_to_resid_mid(
    activation_cache: ActivationCache,
    apply_ln: bool,
    project: bool,
) -> None:
    """With ``mlp_input=True`` the stack sums to the MLP input ``resid_mid``, like ``decompose_resid``.

    With ``apply_ln=True`` that is ``LN2(resid_mid)``; with a projection, its projection.
    """
    layer = 1
    direction = None
    if project:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            direction = torch.randn(activation_cache.model.cfg.d_model)

    stack = activation_cache.stack_neuron_results(
        layer,
        apply_ln=apply_ln,
        incl_remainder=True,
        project_output_onto=direction,
        mlp_input=True,
    )

    target = activation_cache[("resid_mid", layer)]
    if apply_ln:
        target = activation_cache.apply_ln_to_stack(target[None], layer, mlp_input=True)[0]
    if direction is not None:
        target = target @ direction
    torch.testing.assert_close(stack.sum(dim=0), target)


@torch.no_grad()
def test_stack_head_results_does_not_warn_when_results_cached(
    activation_cache: ActivationCache,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Repeated stacking stays silent; direct recomputation of cached results still warns."""
    cache = ActivationCache(
        {key: value.clone() for key, value in activation_cache.cache_dict.items()},
        activation_cache.model,
    )
    for layer in range(cache.model.cfg.n_layers):
        cache.cache_dict.pop(f"blocks.{layer}.attn.hook_result", None)

    with caplog.at_level("WARNING"):
        first = cache.stack_head_results()
        second = cache.stack_head_results()
    already_cached = "Tried to compute head results when they were already cached"
    assert already_cached not in caplog.messages
    torch.testing.assert_close(second, first)

    with caplog.at_level("WARNING"):
        cache.compute_head_results()
    assert already_cached in caplog.messages


@pytest.mark.parametrize("mlp_input", [False, True], ids=["attn-input", "mlp-input"])
@torch.no_grad()
def test_stack_neuron_results_rejects_out_of_contract_remainder_layer(
    activation_cache: ActivationCache,
    mlp_input: bool,
) -> None:
    """Layers below the remainder contract raise instead of wrapping to a late layer's residual."""
    with pytest.raises(ValueError, match="incl_remainder=True"):
        activation_cache.stack_neuron_results(-2, incl_remainder=True, mlp_input=mlp_input)


@torch.no_grad()
def test_stack_neuron_results_layer_zero_remainder_is_the_whole_model(
    activation_cache: ActivationCache,
) -> None:
    """``layer=0`` decomposes the final residual stream: no neurons, remainder = resid_post[-1]."""
    stack, labels = activation_cache.stack_neuron_results(
        0, incl_remainder=True, return_labels=True
    )
    assert labels == ["remainder"]
    n_layers = activation_cache.model.cfg.n_layers
    torch.testing.assert_close(stack[0], activation_cache[("resid_post", n_layers - 1)])
    empty, empty_labels = activation_cache.stack_neuron_results(
        0, incl_remainder=False, return_labels=True
    )
    assert empty_labels == [] and empty.shape[0] == 0


@torch.no_grad()
def test_stack_neuron_results_mlp_input_layer_zero_remainder_is_resid_mid(
    activation_cache: ActivationCache,
) -> None:
    """At layer 0 with ``mlp_input=True`` the remainder is the whole MLP input ``resid_mid``."""
    stack, labels = activation_cache.stack_neuron_results(
        0, incl_remainder=True, mlp_input=True, return_labels=True
    )
    assert labels == ["remainder"]
    torch.testing.assert_close(stack[0], activation_cache[("resid_mid", 0)])


@pytest.fixture(scope="module")
def no_norm_activation_cache() -> ActivationCache:
    cfg = TransformerBridgeConfig(
        n_layers=2,
        d_model=16,
        n_ctx=8,
        d_head=4,
        n_heads=4,
        d_vocab=32,
        act_fn="gelu",
        normalization_type=None,
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


@pytest.mark.parametrize("batch_slice", [[0, 2], 1], ids=["list", "int"])
@torch.no_grad()
def test_logit_attrs_batch_slice_without_normalization(
    no_norm_activation_cache: ActivationCache,
    batch_slice: list[int] | int,
) -> None:
    """Without LN/RMS the batch slice must still be applied, not silently dropped."""
    cache = no_norm_activation_cache
    residual = cache.decompose_resid()
    assert isinstance(residual, torch.Tensor)
    all_targets = torch.tensor([10, 11, 12])

    actual = cache.logit_attrs(residual, tokens=all_targets[batch_slice], batch_slice=batch_slice)

    expected = cache.logit_attrs(residual, tokens=all_targets)[:, batch_slice]
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)
