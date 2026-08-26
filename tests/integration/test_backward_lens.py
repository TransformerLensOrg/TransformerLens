"""Integration tests for Backward Lens capture on a raw GPT-2 Bridge."""

import warnings
from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F
from beartype.roar import BeartypeCallHintParamViolation

PROMPT = "The capital of France is"
TARGET = " Paris"
LAYERS = (0, 11)

DEVICE_DTYPE_CASES = [pytest.param("cpu", torch.bfloat16, id="cpu-bfloat16")]
if torch.cuda.is_available():
    DEVICE_DTYPE_CASES.append(pytest.param("cuda", torch.float16, id="cuda-float16"))
if torch.backends.mps.is_available():
    DEVICE_DTYPE_CASES.append(pytest.param("mps", torch.float32, id="mps-float32"))


@pytest.fixture(scope="module")
def gpt2_bridge():
    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers("gpt2", device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def gradient_capture(gpt2_bridge):
    from transformer_lens.tools.analysis.backward_lens import (
        _capture_gpt2_mlp_gradient_factors,
    )

    return _capture_gpt2_mlp_gradient_factors(gpt2_bridge, PROMPT, TARGET, LAYERS)


@pytest.fixture(scope="module")
def backward_result(gpt2_bridge):
    from transformer_lens.tools.analysis import BackwardLens

    return BackwardLens(gpt2_bridge).analyze(PROMPT, TARGET, LAYERS, normalized=True)


def _projection_hook_snapshots(model, layer: int) -> list[tuple[int, ...]]:
    mlp = model.blocks[layer].mlp
    projections = (getattr(mlp, "in"), mlp.out)
    return [
        tuple(hook_point._forward_hooks)
        for projection in projections
        for hook_point in (projection.hook_in, projection.hook_out)
    ]


def test_real_gpt2_factors_reconstruct_both_mlp_weight_gradients(
    gradient_capture, gpt2_bridge
) -> None:
    assert gradient_capture.target_token_id == 6342
    assert gradient_capture.loss > 0
    assert gradient_capture.prompt_token_ids.shape == (1, 6)
    assert gradient_capture.prompt_token_ids.device.type == "cpu"
    expected_tokens = gpt2_bridge.to_tokens(PROMPT)
    assert torch.equal(gradient_capture.prompt_token_ids, expected_tokens)
    with torch.no_grad():
        logits = gpt2_bridge(expected_tokens)
        expected_loss = F.cross_entropy(
            logits[:, -1, :], torch.tensor([gradient_capture.target_token_id])
        )
    assert gradient_capture.loss == pytest.approx(float(expected_loss), abs=1e-6, rel=1e-6)
    assert [result.layer for result in gradient_capture.layers] == list(LAYERS)

    for result in gradient_capture.layers:
        first = result.input_projection
        second = result.output_projection
        assert first.forward_inputs.shape == (6, 768)
        assert first.output_gradients.shape == (6, 3072)
        assert first.weight_gradient.shape == (768, 3072)
        assert second.forward_inputs.shape == (6, 3072)
        assert second.output_gradients.shape == (6, 768)
        assert second.weight_gradient.shape == (3072, 768)
        for factors in (first, second):
            assert factors.weight_layout == "in_out"
            assert factors.reconstructed_gradient.shape == factors.weight_gradient.shape
            for tensor in (
                factors.forward_inputs,
                factors.output_gradients,
                factors.weight_gradient,
                factors.reconstructed_gradient,
            ):
                assert tensor.dtype == torch.float32
                assert tensor.device.type == "cpu"
                assert tensor.grad_fn is None
                assert torch.isfinite(tensor).all()
            torch.testing.assert_close(
                factors.reconstructed_gradient,
                factors.weight_gradient,
                atol=2e-6,
                rtol=2e-5,
            )
            assert factors.absolute_reconstruction_error <= 2e-6
            assert factors.relative_reconstruction_error <= 2e-5


@pytest.mark.parametrize("projection", ["input_projection", "output_projection"])
def test_final_layer_has_the_expected_rank_one_position_structure(
    gradient_capture, projection: str
) -> None:
    last_layer = gradient_capture.layers[-1]
    factors = getattr(last_layer, projection)
    row_norms = factors.output_gradients.norm(dim=-1)
    relative_earlier_norm = row_norms[:-1].max() / row_norms[-1].clamp_min(
        torch.finfo(row_norms.dtype).eps
    )

    assert relative_earlier_norm <= 1e-6
    assert int(torch.linalg.matrix_rank(factors.output_gradients)) == 1


def test_public_result_metadata_and_error_summaries(backward_result) -> None:
    from transformer_lens.tools.analysis import BackwardLensResult

    assert isinstance(backward_result, BackwardLensResult)
    assert backward_result.prompt == PROMPT
    assert backward_result.target_token == TARGET
    assert backward_result.target_token_id == 6342
    assert backward_result.prompt_token_ids.shape == (6,)
    assert [layer.layer for layer in backward_result.layers] == list(LAYERS)
    assert backward_result.includes_normalized_logits is True
    assert backward_result.max_absolute_reconstruction_error <= 2e-6
    assert backward_result.max_relative_reconstruction_error <= 2e-5
    assert backward_result.layer(11) is backward_result.layers[-1]
    assert not hasattr(backward_result, "model")
    with pytest.raises(KeyError, match="was not analyzed"):
        backward_result.layer(5)


def test_public_projections_match_fresh_model_readout(backward_result, gpt2_bridge) -> None:
    unembed_weight = gpt2_bridge.W_U
    for layer in backward_result.layers:
        matrices = (
            (layer.input_projection, layer.input_projection.factors.forward_inputs),
            (layer.output_projection, layer.output_projection.factors.output_gradients),
        )
        for matrix, rows in matrices:
            rows_on_model = rows.to(
                device=unembed_weight.device, dtype=unembed_weight.dtype
            ).unsqueeze(0)
            with torch.no_grad():
                direct = gpt2_bridge.unembed(gpt2_bridge.ln_final(rows_on_model)).squeeze(0)
            torch.testing.assert_close(matrix.vocabulary_logits, direct.float().cpu())

            norms = rows.norm(dim=-1)
            zero_mask = norms == 0
            normalized_rows = rows / torch.where(
                zero_mask, torch.ones_like(norms), norms
            ).unsqueeze(-1)
            with torch.no_grad():
                direct_normalized = gpt2_bridge.unembed(
                    gpt2_bridge.ln_final(
                        normalized_rows.to(
                            device=unembed_weight.device,
                            dtype=unembed_weight.dtype,
                        ).unsqueeze(0)
                    )
                ).squeeze(0)
            assert matrix.normalized_vocabulary_logits is not None
            torch.testing.assert_close(
                matrix.normalized_vocabulary_logits,
                direct_normalized.float().cpu(),
            )
            torch.testing.assert_close(matrix.factor_norms, norms)
            assert torch.equal(matrix.zero_norm_mask, zero_mask)
            for logits in (matrix.vocabulary_logits, matrix.normalized_vocabulary_logits):
                assert logits.shape == (6, gpt2_bridge.cfg.d_vocab)
                assert logits.dtype == torch.float32
                assert logits.device.type == "cpu"
                assert logits.grad_fn is None
                assert torch.isfinite(logits).all()


def test_public_rankings_decoding_and_target_ranks(backward_result, gpt2_bridge) -> None:
    output = backward_result.layer(11).output_projection
    direct_top = torch.topk(output.vocabulary_logits, k=3, dim=-1, largest=True)
    direct_bottom = torch.topk(output.vocabulary_logits, k=3, dim=-1, largest=False)

    assert torch.equal(output.top(k=3).indices, direct_top.indices)
    assert torch.equal(output.bottom(k=3).indices, direct_bottom.indices)
    assert output.top_tokens(gpt2_bridge.tokenizer, k=3)[0] == [
        gpt2_bridge.tokenizer.decode([token_id]) for token_id in direct_top.indices[0].tolist()
    ]
    assert output.bottom_tokens(gpt2_bridge.tokenizer, k=3)[0] == [
        gpt2_bridge.tokenizer.decode([token_id]) for token_id in direct_bottom.indices[0].tolist()
    ]
    target_ranks = output.gradient_descent_target_ranks(backward_result.target_token_id)
    normalized_target_ranks = output.gradient_descent_target_ranks(
        backward_result.target_token_id, normalized=True
    )
    assert target_ranks.shape == (6,)
    assert normalized_target_ranks.shape == (6,)
    assert target_ranks.dtype == torch.int64


def test_public_api_defaults_to_raw_projection_only(gpt2_bridge) -> None:
    from transformer_lens.tools.analysis import BackwardLens

    result = BackwardLens(gpt2_bridge).analyze(PROMPT, TARGET, [0])
    assert result.includes_normalized_logits is False
    for matrix in (
        result.layers[0].input_projection,
        result.layers[0].output_projection,
    ):
        assert matrix.normalized_vocabulary_logits is None
        with pytest.raises(ValueError, match="were not requested"):
            matrix.logits(normalized=True)
    with pytest.raises((TypeError, BeartypeCallHintParamViolation), match="normalized"):
        BackwardLens(gpt2_bridge).analyze(PROMPT, TARGET, [0], normalized=1)


@pytest.mark.parametrize(
    ("prompt", "target", "layers", "error", "match"),
    [
        ("", TARGET, [0], ValueError, "prompt must not be empty"),
        (PROMPT, "", [0], ValueError, "exactly one token"),
        (PROMPT, " New York", [0], ValueError, "got 2 tokens"),
        (PROMPT, TARGET, [], ValueError, "at least one"),
        (PROMPT, TARGET, [0, 0], ValueError, "duplicate"),
        (PROMPT, TARGET, [-1], ValueError, "must be in"),
        (PROMPT, TARGET, [12], ValueError, "must be in"),
        (PROMPT, TARGET, [True], TypeError, "layer"),
    ],
)
def test_capture_rejects_invalid_analysis_inputs(
    gpt2_bridge,
    prompt: str,
    target: str,
    layers: Sequence[int],
    error: type[Exception],
    match: str,
) -> None:
    from transformer_lens.tools.analysis.backward_lens import (
        _capture_gpt2_mlp_gradient_factors,
    )

    with pytest.raises(error, match=match):
        _capture_gpt2_mlp_gradient_factors(gpt2_bridge, prompt, target, layers)


def test_capture_rejects_non_bridge_and_non_raw_states(gpt2_bridge, monkeypatch) -> None:
    from transformer_lens.tools.analysis.backward_lens import (
        _capture_gpt2_mlp_gradient_factors,
    )

    with pytest.raises(TypeError, match="TransformerBridge only"):
        _capture_gpt2_mlp_gradient_factors(object(), PROMPT, TARGET, [0])
    monkeypatch.setattr(gpt2_bridge, "compatibility_mode", True)
    with pytest.raises(ValueError, match="compatibility mode"):
        _capture_gpt2_mlp_gradient_factors(gpt2_bridge, PROMPT, TARGET, [0])
    monkeypatch.setattr(gpt2_bridge, "compatibility_mode", False)
    monkeypatch.setattr(gpt2_bridge, "_weights_processed", True)
    with pytest.raises(ValueError, match="processed"):
        _capture_gpt2_mlp_gradient_factors(gpt2_bridge, PROMPT, TARGET, [0])
    monkeypatch.setattr(gpt2_bridge, "_weights_processed", False)
    adapter = gpt2_bridge.adapter
    monkeypatch.setattr(gpt2_bridge, "adapter", object())
    with pytest.raises(ValueError, match="GPT2ArchitectureAdapter"):
        _capture_gpt2_mlp_gradient_factors(gpt2_bridge, PROMPT, TARGET, [0])
    monkeypatch.setattr(gpt2_bridge, "adapter", adapter)
    monkeypatch.setattr(gpt2_bridge.cfg, "gated_mlp", True)
    with pytest.raises(ValueError, match="non-gated"):
        _capture_gpt2_mlp_gradient_factors(gpt2_bridge, PROMPT, TARGET, [0])
    monkeypatch.setattr(gpt2_bridge.cfg, "gated_mlp", False)
    monkeypatch.setattr(gpt2_bridge, "tokenizer", None)
    with pytest.raises(ValueError, match="tokenizer"):
        _capture_gpt2_mlp_gradient_factors(gpt2_bridge, PROMPT, TARGET, [0])


def test_capture_rejects_a_frozen_original_weight(gpt2_bridge) -> None:
    from transformer_lens.tools.analysis.backward_lens import (
        _capture_gpt2_mlp_gradient_factors,
    )

    weight = getattr(gpt2_bridge.blocks[0].mlp, "in").original_component.weight
    original_requires_grad = weight.requires_grad
    weight.requires_grad_(False)
    try:
        with pytest.raises(ValueError, match="trainable Parameter"):
            _capture_gpt2_mlp_gradient_factors(gpt2_bridge, PROMPT, TARGET, [0])
    finally:
        weight.requires_grad_(original_requires_grad)


def test_capture_preserves_model_state_hooks_and_uses_one_autograd_call(
    gpt2_bridge, monkeypatch
) -> None:
    from transformer_lens.model_bridge.generalized_components.normalization import (
        NATIVE_PATH_BWD_FALLBACK_WARNING,
        NATIVE_PATH_EDIT_FALLBACK_WARNING,
    )
    from transformer_lens.tools.analysis.backward_lens import (
        _capture_gpt2_mlp_gradient_factors,
    )

    mlp = gpt2_bridge.blocks[0].mlp
    projections = (getattr(mlp, "in"), mlp.out)
    weights = [projection.original_component.weight for projection in projections]
    saved_grads = [weight.grad for weight in weights]
    weight_copies = [weight.detach().clone() for weight in weights]
    training = gpt2_bridge.training
    outer_rng = torch.random.get_rng_state()
    hook_calls = 0
    autograd_calls = 0
    original_grad = torch.autograd.grad

    def existing_hook(_tensor, hook=None) -> None:
        nonlocal hook_calls
        hook_calls += 1

    def counting_grad(*args, **kwargs):
        nonlocal autograd_calls
        autograd_calls += 1
        return original_grad(*args, **kwargs)

    hook_point = mlp.out.hook_out
    hook_point.add_hook(existing_hook)
    existing_handle = hook_point.fwd_hooks[-1]
    try:
        gpt2_bridge.train(True)
        torch.manual_seed(1234)
        rng_before = torch.random.get_rng_state()
        for index, weight in enumerate(weights):
            weight.grad = torch.full_like(weight, index + 1.0)
        grad_copies = [weight.grad.clone() for weight in weights]
        requires_grad = [weight.requires_grad for weight in weights]
        hooks_before = _projection_hook_snapshots(gpt2_bridge, 0)
        monkeypatch.setattr(torch.autograd, "grad", counting_grad)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _capture_gpt2_mlp_gradient_factors(gpt2_bridge, PROMPT, TARGET, [0])

        assert len(result.layers) == 1
        assert autograd_calls == 1
        assert hook_calls == 1
        assert not any(
            str(warning.message)
            in (NATIVE_PATH_BWD_FALLBACK_WARNING, NATIVE_PATH_EDIT_FALLBACK_WARNING)
            for warning in caught
        )
        assert torch.equal(torch.random.get_rng_state(), rng_before)
        assert gpt2_bridge.training is True
        assert _projection_hook_snapshots(gpt2_bridge, 0) == hooks_before
        for weight, saved_weight, saved_grad, expected_requires_grad in zip(
            weights, weight_copies, grad_copies, requires_grad, strict=True
        ):
            assert torch.equal(weight, saved_weight)
            assert torch.equal(weight.grad, saved_grad)
            assert weight.requires_grad is expected_requires_grad
    finally:
        existing_handle.hook.remove()
        hook_point.fwd_hooks.remove(existing_handle)

        for weight, saved_grad in zip(weights, saved_grads, strict=True):
            weight.grad = saved_grad
        gpt2_bridge.train(training)
        torch.random.set_rng_state(outer_rng)


def test_capture_reconstructs_with_existing_activation_edits(gpt2_bridge) -> None:
    from transformer_lens.tools.analysis.backward_lens import (
        _capture_gpt2_mlp_gradient_factors,
    )

    mlp = gpt2_bridge.blocks[0].mlp
    input_hook_point = getattr(mlp, "in").hook_in
    output_hook_point = mlp.out.hook_out

    def scale_input(tensor, hook=None):
        return tensor * 0.5

    def shift_output(tensor, hook=None):
        return tensor + 0.01

    input_hook_point.add_hook(scale_input)
    input_handle = input_hook_point.fwd_hooks[-1]
    output_hook_point.add_hook(shift_output)
    output_handle = output_hook_point.fwd_hooks[-1]
    try:
        result = _capture_gpt2_mlp_gradient_factors(gpt2_bridge, PROMPT, TARGET, [0])
        assert input_handle in input_hook_point.fwd_hooks
        assert output_handle in output_hook_point.fwd_hooks
    finally:
        input_handle.hook.remove()
        input_hook_point.fwd_hooks.remove(input_handle)
        output_handle.hook.remove()
        output_hook_point.fwd_hooks.remove(output_handle)

    for factors in (
        result.layers[0].input_projection,
        result.layers[0].output_projection,
    ):
        torch.testing.assert_close(
            factors.reconstructed_gradient,
            factors.weight_gradient,
            atol=2e-6,
            rtol=2e-5,
        )


def test_capture_cleans_owned_hooks_when_autograd_raises(gpt2_bridge, monkeypatch) -> None:
    from transformer_lens.tools.analysis.backward_lens import (
        _capture_gpt2_mlp_gradient_factors,
    )

    hook_point = gpt2_bridge.blocks[0].mlp.out.hook_out
    hook_calls = 0

    def existing_hook(tensor, hook=None) -> None:
        nonlocal hook_calls
        hook_calls += 1

    def fail_autograd(*args, **kwargs):
        raise RuntimeError("forced autograd failure")

    hook_point.add_hook(existing_hook)
    existing_handle = hook_point.fwd_hooks[-1]
    hooks_before = _projection_hook_snapshots(gpt2_bridge, 0)
    monkeypatch.setattr(torch.autograd, "grad", fail_autograd)
    try:
        with pytest.raises(RuntimeError, match="forced autograd failure"):
            _capture_gpt2_mlp_gradient_factors(gpt2_bridge, PROMPT, TARGET, [0])
        assert hook_calls == 1
        assert existing_handle in hook_point.fwd_hooks
        assert _projection_hook_snapshots(gpt2_bridge, 0) == hooks_before
    finally:
        existing_handle.hook.remove()
        hook_point.fwd_hooks.remove(existing_handle)


def test_capture_removes_only_owned_hooks_when_forward_fails(gpt2_bridge) -> None:
    from transformer_lens.tools.analysis.backward_lens import (
        _capture_gpt2_mlp_gradient_factors,
    )

    mlp = gpt2_bridge.blocks[0].mlp
    projections = (getattr(mlp, "in"), mlp.out)
    weights = [projection.original_component.weight for projection in projections]
    saved_grads = [weight.grad for weight in weights]
    weight_copies = [weight.detach().clone() for weight in weights]
    training = gpt2_bridge.training
    outer_rng = torch.random.get_rng_state()
    hook_point = mlp.out.hook_out

    def fail(_module, _inputs, _output) -> None:
        raise RuntimeError("forced existing-hook failure")

    existing_handle = hook_point.register_forward_hook(fail)
    try:
        gpt2_bridge.train(True)
        torch.manual_seed(5678)
        rng_before = torch.random.get_rng_state()
        for index, weight in enumerate(weights):
            weight.grad = torch.full_like(weight, index + 3.0)
        grad_copies = [weight.grad.clone() for weight in weights]
        requires_grad = [weight.requires_grad for weight in weights]
        hooks_before = _projection_hook_snapshots(gpt2_bridge, 0)
        with pytest.raises(RuntimeError, match="forced existing-hook failure"):
            _capture_gpt2_mlp_gradient_factors(gpt2_bridge, PROMPT, TARGET, [0])
        assert _projection_hook_snapshots(gpt2_bridge, 0) == hooks_before
        assert torch.equal(torch.random.get_rng_state(), rng_before)
        assert gpt2_bridge.training is True
        for weight, saved_weight, saved_grad, expected_requires_grad in zip(
            weights, weight_copies, grad_copies, requires_grad, strict=True
        ):
            assert torch.equal(weight, saved_weight)
            assert torch.equal(weight.grad, saved_grad)
            assert weight.requires_grad is expected_requires_grad
    finally:
        existing_handle.remove()
        for weight, saved_grad in zip(weights, saved_grads, strict=True):
            weight.grad = saved_grad
        gpt2_bridge.train(training)
        torch.random.set_rng_state(outer_rng)


@pytest.mark.parametrize(("device", "dtype"), DEVICE_DTYPE_CASES)
def test_tiny_gpt2_capture_supports_available_devices_and_reduced_precision(
    gpt2_bridge, device: str, dtype: torch.dtype
) -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis import BackwardLens

    config = GPT2Config(
        n_layer=2,
        n_head=4,
        n_embd=32,
        n_inner=64,
        n_positions=32,
        vocab_size=gpt2_bridge.cfg.d_vocab,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    hf_model = GPT2LMHeadModel(config).to(device=device, dtype=dtype).eval()
    bridge = TransformerBridge.boot_transformers(
        "gpt2",
        hf_model=hf_model,
        tokenizer=gpt2_bridge.tokenizer,
        dtype=dtype,
    )
    bridge.train(True)
    if device == "cuda":
        rng_before = torch.cuda.get_rng_state()
    elif device == "mps":
        rng_before = torch.mps.get_rng_state()
    else:
        rng_before = torch.random.get_rng_state()

    result = BackwardLens(bridge).analyze("Small test", " token", [0, 1], normalized=True)

    if device == "cuda":
        rng_after = torch.cuda.get_rng_state()
    elif device == "mps":
        rng_after = torch.mps.get_rng_state()
    else:
        rng_after = torch.random.get_rng_state()
    assert torch.equal(rng_after, rng_before)
    for layer in result.layers:
        for matrix in (layer.input_projection, layer.output_projection):
            factors = matrix.factors
            assert torch.isfinite(factors.weight_gradient).all()
            assert factors.relative_reconstruction_error <= 5e-2
            assert torch.isfinite(matrix.vocabulary_logits).all()
            assert matrix.normalized_vocabulary_logits is not None
            assert torch.isfinite(matrix.normalized_vocabulary_logits).all()
