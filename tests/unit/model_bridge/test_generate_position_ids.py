"""Explicit prompt positions survive cached and uncached generation."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.fixture(scope="module")
def bridge():
    config = GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=2, n_head=2)
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(1)
        model = GPT2LMHeadModel(config).eval()
    return build_bridge_from_module(
        model, "GPT2LMHeadModel", hf_config=config, dtype=torch.float32, device="cpu"
    ).eval()


def _positions(prompt_positions, extra):
    offsets = torch.arange(1, extra + 1, device=prompt_positions.device)
    return torch.cat([prompt_positions, prompt_positions[:, -1:] + offsets], dim=-1)


def _check_explicit_positions(bridge, cached, mask_kind, broadcast):
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    positions = torch.tensor([[7, 8, 9]]) if broadcast else torch.tensor([[7, 8, 9], [2, 4, 6]])
    original = positions.clone()
    mask = (
        None
        if mask_kind == "none"
        else torch.tensor([[0, 1, 1], [1, 1, 1]] if mask_kind == "left" else [[1, 0, 1], [1, 1, 1]])
    )
    expected = tokens.clone()
    expected_logits = []
    with torch.no_grad():
        for step in range(3):
            kwargs = {"position_ids": _positions(positions, step)}
            if mask is not None:
                kwargs["attention_mask"] = torch.cat([mask, mask.new_ones((2, step))], dim=-1)
            logits = bridge(expected, **kwargs)[:, -1]
            expected_logits.append(logits)
            expected = torch.cat([expected, logits.argmax(-1, keepdim=True)], dim=-1)
    result = bridge.generate(
        tokens,
        position_ids=positions,
        attention_mask=mask,
        max_new_tokens=3,
        do_sample=False,
        stop_at_eos=False,
        verbose=False,
        use_past_kv_cache=cached,
        output_logits=True,
    )
    assert torch.equal(result.sequences, expected)
    for actual, reference in zip(result.logits, expected_logits):
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    assert torch.equal(positions, original)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("mask_kind", ["none", "left", "gap"])
@pytest.mark.parametrize("broadcast", [False, True])
def test_explicit_positions_match_manual_loop(bridge, cached, mask_kind, broadcast):
    _check_explicit_positions(bridge, cached, mask_kind, broadcast)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("mask_kind", ["none", "gap"])
def test_rotary_positions_match_manual_loop(cached, mask_kind):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(1)
        model = LlamaForCausalLM(config).eval()
    bridge = build_bridge_from_module(
        model, "LlamaForCausalLM", hf_config=config, dtype=torch.float32, device="cpu"
    ).eval()
    _check_explicit_positions(bridge, cached, mask_kind, broadcast=False)


@pytest.mark.parametrize("cached", [False, True])
def test_issue_reproduction_changes_generated_token(bridge, cached):
    result = bridge.generate(
        torch.tensor([[1, 2, 3]]),
        position_ids=torch.tensor([[7, 8, 9]]),
        max_new_tokens=3,
        do_sample=False,
        stop_at_eos=False,
        verbose=False,
        use_past_kv_cache=cached,
    )
    assert result.tolist() == [[1, 2, 3, 3, 3, 15]]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("new_tokens", [1, 3])
def test_return_cache_replays_explicit_positions(bridge, cached, new_tokens):
    tokens = torch.tensor([[1, 2, 3]])
    positions = torch.tensor([[7, 8, 9]])
    mask = torch.tensor([[1, 0, 1]])
    output, cache = bridge.generate(
        tokens,
        position_ids=positions,
        attention_mask=mask,
        max_new_tokens=new_tokens,
        do_sample=False,
        stop_at_eos=False,
        verbose=False,
        use_past_kv_cache=cached,
        return_cache=True,
        names_filter=["hook_embed", "blocks.0.hook_resid_pre"],
    )
    _, expected = bridge.run_with_cache(
        output,
        position_ids=_positions(positions, new_tokens),
        attention_mask=torch.cat([mask, mask.new_ones((1, new_tokens))], dim=-1),
        names_filter=["hook_embed", "blocks.0.hook_resid_pre"],
    )
    assert cache.keys() == expected.keys()
    for name in expected.keys():
        torch.testing.assert_close(cache[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_default_positions_match_explicit_default(bridge, cached, masked):
    tokens = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[0, 1, 1]]) if masked else None
    positions = torch.tensor([[1, 0, 1]]) if masked else torch.tensor([[0, 1, 2]])
    kwargs = dict(
        max_new_tokens=3,
        do_sample=False,
        stop_at_eos=False,
        verbose=False,
        use_past_kv_cache=cached,
        attention_mask=mask,
        output_logits=True,
    )
    default = bridge.generate(tokens, **kwargs)
    explicit = bridge.generate(tokens, position_ids=positions, **kwargs)
    assert torch.equal(default.sequences, explicit.sequences)
    for actual, reference in zip(explicit.logits, default.logits):
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cached", [False, True])
def test_explicit_positions_do_not_depend_on_derivation_gate(bridge, monkeypatch, cached):
    monkeypatch.setattr(bridge, "_accepts_derived_position_ids", lambda: False)
    result = bridge.generate(
        torch.tensor([[1, 2, 3]]),
        position_ids=torch.tensor([[7, 8, 9]], dtype=torch.int32),
        max_new_tokens=3,
        do_sample=False,
        stop_at_eos=False,
        verbose=False,
        use_past_kv_cache=cached,
        output_attentions=False,
    )
    assert result.tolist() == [[1, 2, 3, 3, 3, 15]]


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor([7, 8, 9]),
        torch.ones(1, 1, 3, dtype=torch.long),
        torch.ones(2, 3, dtype=torch.long),
        torch.tensor([[7, 8]]),
        torch.tensor([[7.0, 8.0, 9.0]]),
        torch.tensor([[True, False, True]]),
    ],
)
def test_invalid_position_ids_raise_before_forward(bridge, positions, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("invalid positions must be rejected before forward")

    monkeypatch.setattr(bridge, "forward", fail)
    with pytest.raises(ValueError, match="position_ids"):
        bridge.generate(torch.tensor([[1, 2, 3]]), position_ids=positions, verbose=False)


@pytest.mark.parametrize("kind", ["encoder_decoder", "stateful", "embeds", "pixels"])
def test_unsupported_position_inputs_raise(bridge, kind, monkeypatch):
    tokens = torch.tensor([[1, 2, 3]])
    kwargs = {}
    if kind == "encoder_decoder":
        monkeypatch.setattr(bridge.original_model.config, "is_encoder_decoder", True)
    elif kind == "stateful":
        monkeypatch.setattr(bridge.cfg, "is_stateful", True)
    elif kind == "embeds":
        tokens = torch.zeros(1, 3, 16)
    else:
        kwargs["pixel_values"] = torch.zeros(1, 3, 2, 2)
    with pytest.raises(NotImplementedError, match="position_ids"):
        bridge.generate(tokens, position_ids=torch.tensor([[7, 8, 9]]), verbose=False, **kwargs)
