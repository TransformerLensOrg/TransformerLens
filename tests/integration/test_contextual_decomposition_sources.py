"""Single head and residual anchoring with native target reconstruction."""

import math

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.tools.analysis.contextual_decomposition.executor import (
    _decompose_gpt2,
)
from transformer_lens.tools.analysis.contextual_decomposition.selectors import (
    _CDSource,
    _CDTarget,
)


@pytest.fixture
def model():
    with torch.random.fork_rng():
        torch.manual_seed(13)
        config = GPT2Config(
            vocab_size=24,
            n_positions=12,
            n_embd=8,
            n_layer=2,
            n_head=2,
            resid_pdrop=0,
            attn_pdrop=0,
            embd_pdrop=0,
        )
        config._attn_implementation = "eager"
        bridge = build_bridge_from_module(
            GPT2LMHeadModel(config).eval(),
            "GPT2LMHeadModel",
            hf_config=config,
            dtype=torch.float32,
            device="cpu",
        ).eval()
        with torch.no_grad():
            bridge.blocks[0].attn.o.original_component.bias.fill_(0.03)
        return bridge


TOKENS = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])


@pytest.mark.parametrize("location", ["pre", "mid", "post"])
def test_residual_anchor_splits_selected_positions(model, location):
    source = _CDSource(0, location=location, positions=(1,))
    target = _CDTarget(
        "logit_difference", position=(2, 3), token_id=(3, 4), negative_token_id=(1, 2)
    )
    reference = torch.full((2, 4, 8), 0.01)
    result = _decompose_gpt2(model, TOKENS, source=source, reference=reference, target=target)
    anchor = result.constituents[source.anchor]
    expected = torch.zeros_like(reference)
    expected[:, 1] = result.native_cache[source.anchor][:, 1] - reference[:, 1]
    torch.testing.assert_close(anchor.beta, expected)
    torch.testing.assert_close(anchor.beta + anchor.gamma, result.native_cache[source.anchor])
    assert result.target is not None
    native = result.native_logits[torch.arange(2), torch.tensor([2, 3])]
    expected_target = (
        native[torch.arange(2), torch.tensor([3, 4])]
        - native[torch.arange(2), torch.tensor([1, 2])]
    )
    torch.testing.assert_close(
        result.target.beta + result.target.gamma, expected_target, atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize("head", [0, 1])
def test_head_write_projection_excludes_shared_bias(model, head):
    source = _CDSource(0, kind="head", location="mid", head=head, positions=(0,))
    result = _decompose_gpt2(model, TOKENS, source=source)
    context = result.native_cache["blocks.0.attn.o.hook_in"][:, :, head]
    weight = model.blocks[0].attn.o.original_component.weight
    expected = context @ weight[head * 4 : (head + 1) * 4]
    torch.testing.assert_close(result.source_write, expected)
    anchor = result.constituents[source.anchor]
    torch.testing.assert_close(anchor.beta[:, 0], expected[:, 0])
    assert torch.equal(anchor.beta[:, 1:], torch.zeros_like(anchor.beta[:, 1:]))
    torch.testing.assert_close(anchor.gamma, result.native_cache[source.anchor] - anchor.beta)
    assert result.constituents["blocks.1.attn.o.hook_in"].beta[:, 1:].abs().max() > 0


def test_full_head_and_position_subset_share_the_same_native_source_write(model):
    full = _decompose_gpt2(model, TOKENS, source=_CDSource(0, kind="head", location="mid", head=0))
    selected = _decompose_gpt2(
        model, TOKENS, source=_CDSource(0, kind="head", location="mid", head=0, positions=(1,))
    )
    torch.testing.assert_close(full.source_write, selected.source_write)
    torch.testing.assert_close(
        full.constituents["blocks.0.ln2.hook_in"].beta[:, 1],
        selected.constituents["blocks.0.ln2.hook_in"].beta[:, 1],
    )
    assert full.constituents["blocks.0.ln2.hook_in"].beta[:, 0].abs().max() > 0


def test_nonfinite_reference_rejected_before_capture(model):
    with pytest.raises(ValueError, match="finite"):
        _decompose_gpt2(
            model, TOKENS, source=_CDSource(0), reference=torch.full((2, 4, 8), float("nan"))
        )


def test_reference_equal_to_head_write_is_zero_source(model):
    source = _CDSource(0, kind="head", location="mid", head=0)
    initial = _decompose_gpt2(model, TOKENS, source=source)
    result = _decompose_gpt2(model, TOKENS, source=source, reference=initial.source_write)
    assert torch.equal(result.logits.beta, torch.zeros_like(result.logits.beta))


def test_internal_head_coordinate_target_and_equal_token_difference(model):
    source = _CDSource(0, location="post")
    target = _CDTarget("internal", hook_name="blocks.1.attn.q.hook_out", position=(1, 2), head=1)
    result = _decompose_gpt2(model, TOKENS, source=source, target=target)
    native = result.native_cache[target.hook_name][torch.arange(2), torch.tensor([1, 2]), 1]
    torch.testing.assert_close(
        result.target.beta + result.target.gamma, native, atol=2e-6, rtol=2e-5
    )
    zero = _decompose_gpt2(
        model,
        TOKENS,
        source=source,
        target=_CDTarget("logit_difference", position=3, token_id=2, negative_token_id=2),
    )
    assert torch.equal(zero.target.beta, torch.zeros(2)) and torch.equal(
        zero.target.gamma, torch.zeros(2)
    )


@pytest.mark.parametrize(
    "target",
    [
        _CDTarget("internal", hook_name="blocks.0.hook_in"),
        _CDTarget("internal", hook_name="blocks.0.ln2.hook_in"),
        _CDTarget("internal", hook_name="blocks.0.attn.q.hook_out"),
        _CDTarget("internal", hook_name="blocks.1.attn.hook_out", head=0),
        _CDTarget("internal", hook_name="blocks.1.unknown"),
        _CDTarget("logit", position=4, token_id=1),
        _CDTarget("logit", position=1, token_id=24),
        _CDTarget("logit", position=(1,), token_id=1),
    ],
)
def test_bad_chronology_or_target_space_rejected(model, target):
    with pytest.raises(ValueError, match="target|internal"):
        _decompose_gpt2(model, TOKENS, source=_CDSource(0, location="mid"), target=target)


def test_padding_source_reference_and_legacy_conflicts(model):
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    with pytest.raises(ValueError, match="padding"):
        _decompose_gpt2(model, TOKENS, source=_CDSource(0, positions=(3,)), attention_mask=mask)
    with pytest.raises(ValueError, match="reference"):
        _decompose_gpt2(model, TOKENS, source=_CDSource(0), reference=torch.zeros(8))
    with pytest.raises(ValueError, match="legacy"):
        _decompose_gpt2(model, TOKENS, source=_CDSource(0), beta=torch.zeros(2, 4, 8))
    with pytest.raises(ValueError, match="head"):
        _decompose_gpt2(model, TOKENS, source=_CDSource(0, kind="head", location="mid", head=2))


def test_planted_head_source_matches_scalar_normalization_and_unembed_oracle(model):
    block = model.blocks[1]
    with torch.no_grad():
        block.mlp.out.original_component.weight.zero_()
        block.mlp.out.original_component.bias.zero_()
        model.ln_final.original_component.bias.zero_()
    source = _CDSource(1, kind="head", location="mid", head=1, positions=(2,))
    result = _decompose_gpt2(
        model,
        TOKENS,
        source=source,
        target=_CDTarget("logit", position=2, token_id=3),
    )
    expected = []
    for row in range(2):
        context = result.native_cache["blocks.1.attn.o.hook_in"][row, 2, 1].tolist()
        output_weight = block.attn.o.original_component.weight[4:8].tolist()
        relevant = [sum(context[d] * output_weight[d][o] for d in range(4)) for o in range(8)]
        total = result.native_cache[source.anchor][row, 2].tolist()
        mean = sum(total) / 8
        denominator = math.sqrt(
            sum((v - mean) ** 2 for v in total) / 8 + model.ln_final.original_component.eps
        )
        relevant_mean = sum(relevant) / 8
        gain = model.ln_final.original_component.weight.tolist()
        vector = [(b - relevant_mean) * w / denominator for b, w in zip(relevant, gain)]
        unembed = model.unembed.original_component.weight[3].tolist()
        expected.append(sum(v * w for v, w in zip(vector, unembed)))
    torch.testing.assert_close(result.target.beta, torch.tensor(expected), atol=2e-7, rtol=2e-5)


def test_ill_conditioned_source_reconstruction_is_not_silently_corrected(model):
    with torch.no_grad():
        model.blocks[0].attn.o.original_component.bias.fill_(0.3)
    result = _decompose_gpt2(
        model,
        TOKENS,
        source=_CDSource(0, positions=(1,)),
        reference=torch.full((2, 4, 8), 0.01),
    )
    for name, parts in result.constituents.items():
        error = result.errors[name]
        measured = float((parts.beta + parts.gamma - result.native_cache[name]).abs().max())
        assert error.max_abs == measured
        assert measured <= error.tolerance
    assert max(error.max_abs for error in result.errors.values()) > 0


def test_late_final_residual_source_has_independent_expected_readout(model):
    source = _CDSource(1, location="post", positions=(2,))
    target = _CDTarget("logit", position=2, token_id=(2, 3))
    initial = _decompose_gpt2(model, TOKENS, source=source, target=target)
    native = initial.native_cache[source.anchor]
    reference = native.clone()
    result = _decompose_gpt2(model, TOKENS, source=source, reference=reference, target=target)
    assert torch.equal(result.target.beta, torch.zeros(2))
    torch.testing.assert_close(
        result.target.gamma,
        result.native_logits[:, 2][torch.arange(2), torch.tensor([2, 3])],
        atol=2e-6,
        rtol=2e-5,
    )
