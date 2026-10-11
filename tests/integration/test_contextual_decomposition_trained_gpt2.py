"""Pinned trained GPT-2 characterization without circuit-recovery claims."""

import time

import pytest
import torch

from transformer_lens import TransformerBridge
from transformer_lens.tools.analysis import (
    CDBaseline,
    CDSource,
    CDTarget,
    ContextualDecomposition,
)

REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TOKENS = torch.tensor([[15496, 995, 0]])


@pytest.fixture(scope="module")
def trained():
    return TransformerBridge.boot_transformers(
        "gpt2",
        revision=REVISION,
        device="cpu",
        dtype=torch.float32,
    ).eval()


@pytest.mark.parametrize(
    "source,target",
    [
        (
            CDSource.residual(11, location="post"),
            CDTarget.logit_difference(positive_token_id=15496, negative_token_id=995, position=2),
        ),
        (
            CDSource.attention_head(10, 0, positions=(1,)),
            CDTarget.logit_difference(positive_token_id=15496, negative_token_id=995, position=2),
        ),
        (
            CDSource.residual(10, positions=(1,)),
            CDTarget.internal("blocks.11.ln1.hook_out", position=2),
        ),
    ],
)
def test_trained_native_parity_with_supplied_reference(trained, source, target, record_property):
    analysis = ContextualDecomposition(trained)
    reference = CDBaseline.from_tensor(torch.zeros(1, 3, 768), description="zero tensor reference")
    versions = {name: (id(value), value._version) for name, value in trained.named_parameters()}
    hooks = {
        id(module): (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
        for module in trained.modules()
    }
    start = time.perf_counter()
    result = analysis.decompose(TOKENS, source=source, target=target, baseline=reference)
    elapsed = time.perf_counter() - start
    for name, error in result.diagnostics.items():
        assert error.max_abs <= error.tolerance, name
    assert versions == {
        name: (id(value), value._version) for name, value in trained.named_parameters()
    }
    assert hooks == {
        id(module): (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
        for module in trained.modules()
    }
    assert not result.intermediates
    record_property("model_revision", REVISION)
    record_property("elapsed_seconds", elapsed)
    record_property(
        "returned_tensor_bytes",
        sum(
            value.numel() * value.element_size()
            for value in (result.target_beta, result.target_gamma, result.target_total)
        ),
    )
    record_property("max_absolute_error", max(e.max_abs for e in result.diagnostics.values()))
    record_property(
        "target_beta_range", [float(result.target_beta.min()), float(result.target_beta.max())]
    )
    record_property(
        "native_max_error_fraction",
        max(e.max_abs / e.tolerance for e in result.diagnostics.values()),
    )


def test_trained_failure_restores_hooks_rng_and_transient_state(trained):
    point = trained.hook_dict["blocks.10.ln2.hook_out"]

    def fail(module, args, output):
        raise RuntimeError("characterization observer failure")

    handle = point.register_forward_hook(fail)
    hooks = {
        id(module): (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
        for module in trained.modules()
    }
    rng = torch.random.get_rng_state().clone()
    pattern = trained.blocks[10].attn._pattern
    try:
        with pytest.raises(RuntimeError, match="observer failure"):
            ContextualDecomposition(trained).decompose(
                TOKENS,
                source=CDSource.residual(10),
                target=CDTarget.logit(token_id=15496, position=2),
            )
        assert hooks == {
            id(module): (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
            for module in trained.modules()
        }
        assert torch.equal(rng, torch.random.get_rng_state())
        assert trained.blocks[10].attn._pattern is pattern
    finally:
        handle.remove()


def test_native_input_projection_rounding_is_measured_separately(trained, record_property):
    with torch.no_grad():
        _, cache = trained.run_with_cache(TOKENS)
        projection = trained.blocks[10].mlp.out.original_component
        activation = cache["blocks.10.mlp.out.hook_in"]
        native = projection(activation)
        separate = activation @ projection.weight + projection.bias
    error = float((native - separate).abs().max())
    scale = float(native.abs().max())
    assert error <= 2e-6 + 2e-5 * scale
    record_property("native_input_projection_max_error", error)
    record_property("native_input_projection_scale", scale)


def test_trained_native_same_reference_removes_source(trained):
    source = CDSource.residual(11, location="post", positions=(2,))
    target = CDTarget.logit(token_id=15496, position=2)
    with torch.no_grad():
        _, cache = trained.run_with_cache(TOKENS)
    reference = CDBaseline.from_tensor(cache[source.anchor])
    result = ContextualDecomposition(trained).decompose(
        TOKENS, source=source, target=target, baseline=reference
    )
    assert torch.equal(result.target_beta, torch.zeros_like(result.target_beta))
    assert result.diagnostics["target"].max_abs <= result.diagnostics["target"].tolerance
