"""Execution helpers must not silently forward misplaced hook/cache arguments."""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

HOOK = "blocks.0.hook_out"


@pytest.fixture
def bridge():
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=16,
        max_position_embeddings=16,
    )
    config._attn_implementation = "eager"
    return build_bridge_from_module(
        LlamaForCausalLM(config).eval(),
        "LlamaForCausalLM",
        hf_config=config,
        tokenizer=None,
        device="cpu",
    )


def _tokens():
    return torch.tensor([[1, 2, 3, 4]])


@pytest.mark.parametrize("method", ["forward", "run_with_hooks", "run_with_cache"])
@pytest.mark.parametrize("name,correct", [("fwd_hook", "fwd_hooks"), ("bwd_hook", "bwd_hooks")])
def test_singular_hook_keywords_raise_with_the_correct_spelling(bridge, method, name, correct):
    with pytest.raises(TypeError, match=rf"{method}\(\).*{name}.*{correct}"):
        getattr(bridge, method)(_tokens(), **{name: []})


@pytest.mark.parametrize("method", ["forward", "run_with_cache"])
@pytest.mark.parametrize("name", ["fwd_hooks", "bwd_hooks"])
def test_hook_lists_require_a_hook_execution_entry_point(bridge, method, name):
    with pytest.raises(TypeError, match=rf"{method}\(\).*{name}.*run_with_hooks"):
        getattr(bridge, method)(_tokens(), **{name: []})


@pytest.mark.parametrize("method", ["forward", "run_with_hooks"])
@pytest.mark.parametrize("name", ["names_filter", "incl_bwd", "pos_slice", "return_cache_object"])
def test_cache_options_require_run_with_cache(bridge, method, name):
    with pytest.raises(TypeError, match=rf"{method}\(\).*{name}.*run_with_cache"):
        getattr(bridge, method)(_tokens(), **{name: None})


@pytest.mark.parametrize("name", ["reset_hooks_end", "clear_contexts", "remove_batch_dim"])
def test_direct_forward_rejects_helper_lifecycle_options(bridge, name):
    with pytest.raises(TypeError, match=rf"forward\(\).*{name}.*run_with"):
        bridge(_tokens(), **{name: False})


def test_bad_kwargs_are_rejected_before_installing_persistent_hooks(bridge):
    point = bridge.hook_dict[HOOK]
    bridge.add_hook(HOOK, lambda tensor, hook: tensor)
    existing = list(point.fwd_hooks)
    with pytest.raises(TypeError, match="fwd_hook"):
        bridge.run_with_hooks(
            _tokens(),
            fwd_hooks=[(HOOK, lambda tensor, hook: tensor)],
            fwd_hook=[],
            reset_hooks_end=False,
        )
    assert point.fwd_hooks == existing
    assert bridge.context_level == 0


def test_bad_kwargs_are_rejected_before_forward_state_is_armed(bridge, monkeypatch):
    def unexpected_forward(**kwargs):
        pytest.fail("Invalid execution arguments reached the driver")

    monkeypatch.setattr(bridge._driver, "forward", unexpected_forward)
    with pytest.raises(TypeError, match="fwd_hooks"):
        bridge(_tokens(), fwd_hooks=[], stop_at_layer=1, start_at_layer=0)
    assert not hasattr(bridge.blocks[0], "_stop_at_layer_idx")


@torch.no_grad()
def test_context_hooks_compose_with_cache_and_hf_forward_options(bridge):
    tokens = _tokens()
    options = {"attention_mask": torch.ones_like(tokens), "position_ids": torch.arange(4)[None]}
    baseline = bridge(tokens, **options)
    hooks = [(HOOK, lambda tensor, hook: torch.zeros_like(tensor))]
    expected = bridge.run_with_hooks(tokens, fwd_hooks=hooks, **options)
    assert not torch.allclose(expected, baseline)

    with bridge.hooks(fwd_hooks=hooks):
        actual, cache = bridge.run_with_cache(
            tokens,
            names_filter=HOOK,
            remove_batch_dim=True,
            return_cache_object=False,
            pos_slice=(0, 4),
            device="cpu",
            **options,
        )
    torch.testing.assert_close(actual, expected)
    assert cache[HOOK].shape == (4, bridge.cfg.d_model)
    assert torch.count_nonzero(cache[HOOK]) == 0
    torch.testing.assert_close(bridge(tokens, **options), baseline)


@torch.no_grad()
def test_hf_cache_arguments_still_support_incremental_decoding(bridge):
    tokens = _tokens()
    _, cache = bridge(tokens[:, :3], return_type="logits_and_cache")
    logits = bridge(tokens[:, 3:], past_key_values=cache)
    torch.testing.assert_close(logits, bridge(tokens)[:, -1:], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "name", ["custom_model_input", "decoder_input_ids", "pixel_attention_mask", "cache_position"]
)
def test_model_specific_extra_kwargs_are_not_filtered(bridge, monkeypatch, name):
    original_forward = bridge._driver.forward
    seen = []
    extra = object()

    def forward_with_extra(**kwargs):
        seen.append(kwargs.pop(name))
        return original_forward(**kwargs)

    monkeypatch.setattr(bridge._driver, "forward", forward_with_extra)
    bridge(_tokens(), **{name: extra})
    bridge.run_with_hooks(_tokens(), **{name: extra})
    bridge.run_with_cache(_tokens(), names_filter=HOOK, **{name: extra})
    assert seen == [extra, extra, extra]
