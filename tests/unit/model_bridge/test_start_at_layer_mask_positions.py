"""Masked residual re-entry preserves the original forward's position coordinates."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.fixture(scope="module", params=["llama", "gpt2"])
def bridge(request):
    if request.param == "llama":
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        model_class = LlamaForCausalLM
    else:
        config = GPT2Config(vocab_size=32, n_positions=32, n_embd=32, n_layer=3, n_head=4)
        model_class = GPT2LMHeadModel
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = model_class(config).eval()
    return build_bridge_from_module(
        model, model_class.__name__, hf_config=config, dtype=torch.float32, device="cpu"
    ).eval()


@pytest.mark.parametrize("layer", [0, 1, 2, -1])
@pytest.mark.parametrize("layout", ["gap", "left", "right", "ones", "mixed"])
@pytest.mark.parametrize("explicit", [False, True])
def test_masked_residual_round_trip(bridge, layer, layout, explicit):
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    masks = {
        "gap": [[1, 1, 0, 1, 1]],
        "left": [[0, 0, 1, 1, 1]],
        "right": [[1, 1, 1, 0, 0]],
        "ones": [[1, 1, 1, 1, 1]],
        "mixed": [[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]],
    }
    mask = torch.tensor(masks[layout])
    tokens = tokens.expand(mask.shape[0], -1)
    kwargs = {"attention_mask": mask}
    if explicit:
        kwargs["position_ids"] = torch.tensor([[3, 5, 6, 8, 9]]).expand(mask.shape[0], -1)
    with torch.no_grad():
        full, original_cache = bridge.run_with_cache(tokens, **kwargs)
        actual_layer = layer % 3
        residual = original_cache[f"blocks.{actual_layer}.hook_resid_pre"].clone()
        preserved = residual.clone()
        resumed, resumed_cache = bridge.run_with_cache(residual, start_at_layer=layer, **kwargs)
    torch.testing.assert_close(resumed, full, rtol=0, atol=1e-6)
    assert torch.equal(residual, preserved)
    for name, value in original_cache.items():
        if name.startswith("blocks.") and int(name.split(".")[1]) >= actual_layer:
            assert name in resumed_cache
            torch.testing.assert_close(resumed_cache[name], value, rtol=0, atol=1e-6, msg=name)


def test_external_inputs_embeds_still_leave_positions_to_hf(bridge):
    captured = []

    def record(module, args, kwargs):
        captured.append(kwargs.get("position_ids"))

    handle = bridge.original_model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        with torch.no_grad():
            bridge(torch.randn(1, 5, 32), attention_mask=torch.tensor([[1, 1, 0, 1, 1]]))
    finally:
        handle.remove()
    assert captured == [None]


def test_masked_reentry_honors_position_injection_gate(bridge, monkeypatch):
    monkeypatch.setattr(bridge, "_accepts_derived_position_ids", lambda: False)
    captured = []

    def record(module, args, kwargs):
        captured.append(kwargs.get("position_ids"))

    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.tensor([[1, 1, 0, 1, 1]])
    with torch.no_grad():
        full, cache = bridge.run_with_cache(tokens, attention_mask=mask)
        handle = bridge.original_model.register_forward_pre_hook(record, with_kwargs=True)
        try:
            resumed = bridge(
                cache["blocks.1.hook_resid_pre"], start_at_layer=1, attention_mask=mask
            )
        finally:
            handle.remove()
    assert captured == [None]
    torch.testing.assert_close(resumed, full, rtol=0, atol=1e-6)
