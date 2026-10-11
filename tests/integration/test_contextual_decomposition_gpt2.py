"""Native GPT-2 parity and state isolation for residual-anchor CD execution."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.tools.analysis.contextual_decomposition.executor import (
    _decompose_gpt2,
)


@pytest.fixture
def tiny():
    with torch.random.fork_rng():
        torch.manual_seed(17)
        config = GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=8,
            n_layer=2,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        config._attn_implementation = "eager"
        hf = GPT2LMHeadModel(config).eval()
        bridge = build_bridge_from_module(
            hf, "GPT2LMHeadModel", hf_config=config, dtype=torch.float32, device="cpu"
        )
        bridge.eval()
        with torch.no_grad():
            for block in bridge.blocks:
                for norm in (block.ln1.original_component, block.ln2.original_component):
                    norm.weight.copy_(torch.linspace(0.8, 1.2, 8))
                    norm.bias.copy_(torch.linspace(-0.03, 0.03, 8))
                for linear in (
                    block.attn.q,
                    block.attn.k,
                    block.attn.v,
                    block.attn.o,
                    getattr(block.mlp, "in"),
                    block.mlp.out,
                ):
                    linear.original_component.bias.fill_(0.015)
            bridge.unembed.original_component.bias.copy_(torch.linspace(-0.02, 0.02, 32))
        return bridge


TOKENS = torch.tensor([[1, 2, 3, 4], [2, 5, 3, 0]])


@pytest.mark.parametrize("layer", [0, 1])
def test_residual_suffix_reconstructs_native_intermediates_and_logits(tiny, layer):
    with torch.no_grad():
        native, cache = tiny.run_with_cache(TOKENS)
    anchor = cache[f"blocks.{layer}.hook_in"]
    beta = anchor * 0.4
    result = _decompose_gpt2(tiny, TOKENS, layer=layer, beta=beta)
    torch.testing.assert_close(
        result.logits.beta + result.logits.gamma, native, rtol=2e-5, atol=2e-6
    )
    assert torch.equal(result.native_logits, native)
    torch.testing.assert_close(result.constituents[f"blocks.{layer}.hook_in"].beta, beta)
    assert result.errors and max(error.max_abs for error in result.errors.values()) < 2e-6
    for name, parts in result.constituents.items():
        torch.testing.assert_close(
            parts.beta + parts.gamma, result.native_cache[name], rtol=2e-5, atol=2e-6
        )
        assert not parts.beta.requires_grad and not parts.gamma.requires_grad
    assert "embed.hook_out" in result.native_cache and "pos_embed.hook_out" in result.native_cache
    assert f"blocks.{layer}.attn.hook_pattern" in result.constituents


def test_attention_transient_state_is_restored(tiny):
    sentinel = torch.tensor([12.0])
    tiny.blocks[0].attn._pattern = sentinel
    before = tiny.blocks[0].attn.q._fired_hook_out
    _decompose_gpt2(tiny, TOKENS)
    assert tiny.blocks[0].attn._pattern is sentinel
    assert tiny.blocks[0].attn.q._fired_hook_out == before


def test_zero_beta_preserves_zero_assigned_logit_share(tiny):
    result = _decompose_gpt2(tiny, TOKENS, beta=torch.zeros(2, 4, 8))
    assert torch.equal(result.logits.beta, torch.zeros_like(result.logits.beta))
    torch.testing.assert_close(result.logits.gamma, result.native_logits, rtol=2e-5, atol=2e-6)


def test_execution_preserves_parameters_gradients_rng_and_observer_hooks(tiny):
    observed = []
    point = tiny.hook_dict["blocks.0.ln1.hook_out"]
    handle = point.register_forward_hook(
        lambda module, args, output: observed.append(output.detach().clone())
    )
    state = {name: parameter.detach().clone() for name, parameter in tiny.named_parameters()}
    first = next(tiny.parameters())
    first.grad = torch.ones_like(first)
    gradient = first.grad.clone()
    rng = torch.random.get_rng_state().clone()
    collectors = {
        id(m): m.__dict__.get("_output_capturing_hooks_installed") for m in tiny.modules()
    }
    hooks = {id(m): (tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in tiny.modules()}
    try:
        result = _decompose_gpt2(tiny, TOKENS)
        assert observed and result.logits.beta.grad_fn is None
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert torch.equal(first.grad, gradient)
        assert collectors == {
            id(m): m.__dict__.get("_output_capturing_hooks_installed") for m in tiny.modules()
        }
        for name, parameter in tiny.named_parameters():
            assert torch.equal(parameter, state[name])
        assert hooks == {
            id(m): (tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in tiny.modules()
        }
    finally:
        handle.remove()


def test_capture_exception_removes_only_executor_hooks(tiny):
    point = tiny.hook_dict["blocks.0.ln1.hook_out"]

    def fail(module, args, output):
        raise RuntimeError("observer failure")

    handle = point.register_forward_hook(fail)
    hooks = {id(m): (tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in tiny.modules()}
    try:
        with pytest.raises(RuntimeError, match="observer failure"):
            _decompose_gpt2(tiny, TOKENS)
        assert hooks == {
            id(m): (tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in tiny.modules()
        }
    finally:
        handle.remove()
    _decompose_gpt2(tiny, TOKENS)


@pytest.mark.parametrize("inplace", [False, True])
def test_editing_hook_is_rejected_without_changing_hook_registration(tiny, inplace):
    point = tiny.hook_dict["blocks.0.ln1.hook_out"]

    def edit(module, args, output):
        if inplace:
            output.add_(1)
            return None
        return output + 1

    handle = point.register_forward_hook(edit)
    before = tuple(point._forward_hooks)
    try:
        with pytest.raises(ValueError, match="edited"):
            _decompose_gpt2(tiny, TOKENS)
        assert tuple(point._forward_hooks) == before
    finally:
        handle.remove()


@pytest.mark.parametrize("flag", ["compatibility_mode", "_weights_processed"])
def test_processed_weights_rejected(tiny, flag):
    setattr(tiny, flag, True)
    with pytest.raises(ValueError, match="raw"):
        _decompose_gpt2(tiny, TOKENS)


def test_training_submodule_rejected(tiny):
    tiny.blocks[0].mlp.train()
    with pytest.raises(ValueError, match="evaluation"):
        _decompose_gpt2(tiny, TOKENS)


@pytest.mark.parametrize("flag", ["scale_attn_by_inverse_layer_idx", "reorder_and_upcast_attn"])
def test_alternative_native_attention_modes_rejected(tiny, flag):
    setattr(tiny.blocks[0].attn.original_component, flag, True)
    with pytest.raises(ValueError, match="attention"):
        _decompose_gpt2(tiny, TOKENS)


def test_sdpa_configuration_rejected(tiny):
    tiny.original_model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        _decompose_gpt2(tiny, TOKENS)


@pytest.mark.parametrize(
    "tokens",
    [
        torch.tensor([1, 2]),
        torch.empty(0, 4, dtype=torch.long),
        torch.tensor([[32]]),
        torch.tensor([[1.0]]),
    ],
)
def test_invalid_token_inputs_rejected(tiny, tokens):
    with pytest.raises(ValueError, match="tokens"):
        _decompose_gpt2(tiny, tokens)


def test_bad_anchor_shape_or_layer_rejected(tiny):
    with pytest.raises(ValueError, match="layer"):
        _decompose_gpt2(tiny, TOKENS, layer=2)
    with pytest.raises(ValueError, match="beta.*shape"):
        _decompose_gpt2(tiny, TOKENS, beta=torch.zeros(1, 4, 8))


def test_right_padding_and_explicit_position_ids(tiny):
    mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    positions = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    result = _decompose_gpt2(tiny, TOKENS, attention_mask=mask, position_ids=positions)
    torch.testing.assert_close(
        result.logits.beta + result.logits.gamma, result.native_logits, rtol=2e-5, atol=2e-6
    )


def test_fully_masked_query_rejected_before_forward(tiny):
    mask = torch.tensor([[False, True, True, True], [True, True, True, True]])
    with pytest.raises(ValueError, match="fully masked"):
        _decompose_gpt2(tiny, TOKENS, attention_mask=mask)
