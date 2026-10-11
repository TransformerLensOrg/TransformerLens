"""Fail-closed coverage and numerical diagnostics on locally initialized GPT-2."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.tools.analysis.contextual_decomposition.executor import (
    _decompose_gpt2,
)


@pytest.fixture
def model():
    config = GPT2Config(vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=2)
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(2)
        bridge = build_bridge_from_module(
            GPT2LMHeadModel(config).eval(),
            "GPT2LMHeadModel",
            hf_config=config,
            dtype=torch.float32,
            device="cpu",
        )
    return bridge.eval()


TOKENS = torch.tensor([[1, 2, 3]])


def test_executor_reconstructs_unwrapped_hf_gpt2_logits():
    config = GPT2Config(vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=2)
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(12)
        hf = GPT2LMHeadModel(config).eval()
        with torch.no_grad():
            reference = hf(TOKENS, use_cache=False).logits.detach().clone()
        bridge = build_bridge_from_module(
            hf, "GPT2LMHeadModel", hf_config=config, dtype=torch.float32, device="cpu"
        ).eval()
    result = _decompose_gpt2(bridge, TOKENS)
    torch.testing.assert_close(result.native_logits, reference, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        result.logits.beta + result.logits.gamma, reference, rtol=2e-5, atol=2e-6
    )


def test_fp64_model_rejected(model):
    model.double()
    with pytest.raises(ValueError, match="fp32"):
        _decompose_gpt2(model, TOKENS)


def test_active_relevance_context_rejected(model):
    model.blocks[0].ln1._relevance_rule_active = True
    with pytest.raises(ValueError, match="relevance"):
        _decompose_gpt2(model, TOKENS)


def test_unsupported_activation_rejected(model):
    model.original_model.config.activation_function = "relu"
    with pytest.raises(ValueError, match="activation"):
        _decompose_gpt2(model, TOKENS)


def test_backward_hook_rejected_without_fallback(model):
    handle = model.hook_dict["blocks.0.ln1.hook_scale"].register_full_backward_hook(
        lambda module, grad_input, grad_output: None
    )
    try:
        with pytest.raises(ValueError, match="backward hooks"):
            _decompose_gpt2(model, TOKENS)
    finally:
        handle.remove()


def test_external_computational_hook_rejected(model):
    handle = model.blocks[0].attn.q.register_forward_hook(lambda m, args, output: None)
    try:
        with pytest.raises(ValueError, match="computational"):
            _decompose_gpt2(model, TOKENS)
    finally:
        handle.remove()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"attention_mask": torch.ones(1, 3)}, "attention_mask"),
        ({"position_ids": torch.tensor([[1, 2]])}, "position_ids"),
        ({"position_ids": torch.tensor([[1, 2, 8]])}, "position_ids"),
        ({"beta": torch.full((1, 3, 8), float("nan"))}, "finite"),
    ],
)
def test_invalid_inputs_do_not_run_model(model, kwargs, match):
    observed = []
    point = model.hook_dict["blocks.0.hook_in"]
    handle = point.register_forward_hook(lambda m, args, output: observed.append(True))
    try:
        with pytest.raises(ValueError, match=match):
            _decompose_gpt2(model, TOKENS, **kwargs)
        assert not observed
    finally:
        handle.remove()


def test_observer_modifying_weights_causes_local_reconstruction_failure(model):
    weight = model.blocks[0].attn.q.original_component.weight
    saved = weight.detach().clone()

    def modify_weight(module, args, output):
        with torch.no_grad():
            weight.mul_(2)

    handle = model.hook_dict["unembed.hook_out"].register_forward_hook(modify_weight)
    try:
        with pytest.raises(ValueError, match="changed parameters"):
            _decompose_gpt2(model, TOKENS)
    finally:
        handle.remove()
        with torch.no_grad():
            weight.copy_(saved)
    _decompose_gpt2(model, TOKENS)


@pytest.mark.parametrize("flag", ["use_split_qkv_input", "use_attn_in"])
def test_uncovered_split_attention_forks_rejected(model, flag):
    setattr(model.cfg, flag, True)
    with pytest.raises(ValueError, match="split-input"):
        _decompose_gpt2(model, TOKENS)


def test_changed_norm_implementation_is_not_hidden_by_total_correction(model):
    model.blocks[0].ln1.use_native_layernorm_autograd = True
    norm = model.blocks[0].ln1.original_component
    original = norm.forward
    norm.forward = lambda inputs: original(inputs) + 0.25
    with pytest.raises(ValueError, match="reconstruction failed at blocks.0.ln1"):
        _decompose_gpt2(model, TOKENS)


def test_repeated_capture_has_explicit_error(model):
    point = model.hook_dict["blocks.0.ln1.hook_out"]
    nested = False

    def repeat(module, args, output):
        nonlocal nested
        if not nested:
            nested = True
            point(output)

    handle = point.register_forward_hook(repeat)
    try:
        with pytest.raises(ValueError, match="more than once"):
            _decompose_gpt2(model, TOKENS)
    finally:
        handle.remove()
