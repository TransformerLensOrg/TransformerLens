"""generate()'s stateful hf_generate fallback must forward its stop flags."""

from types import SimpleNamespace

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


def _tiny_bridge():
    config = GPT2Config(vocab_size=50, n_positions=32, n_embd=16, n_layer=1, n_head=2)
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    hf = GPT2LMHeadModel(config).eval()
    return build_bridge_from_module(
        hf, "GPT2LMHeadModel", hf_config=config, tokenizer=None, device="cpu"
    )


def _capture_hf_generate(bridge, monkeypatch):
    captured: dict = {}

    def fake_hf_generate(input, **kwargs):
        captured.update(kwargs)
        return input

    monkeypatch.setattr(bridge, "hf_generate", fake_hf_generate)
    return captured


def test_stateful_fallback_forwards_stop_at_eos(monkeypatch):
    bridge = _tiny_bridge()
    # Flagging the config stateful with use_past_kv_cache=False drives the
    # hf_generate fallback without needing a real SSM.
    bridge.cfg.is_stateful = True
    captured = _capture_hf_generate(bridge, monkeypatch)
    bridge.generate(
        torch.tensor([[1, 2, 3]]),
        stop_at_eos=False,
        use_past_kv_cache=False,
        do_sample=False,
        verbose=False,
    )
    assert captured["stop_at_eos"] is False


def test_stateful_fallback_forwards_cfg_eos_list(monkeypatch):
    bridge = _tiny_bridge()
    bridge.cfg.is_stateful = True
    # Chat adapters publish their full stop set this way (e.g. phimoe).
    bridge.cfg.eos_token_id = [5, 7]
    captured = _capture_hf_generate(bridge, monkeypatch)
    bridge.generate(
        torch.tensor([[1, 2, 3]]),
        stop_at_eos=True,
        use_past_kv_cache=False,
        do_sample=False,
        verbose=False,
    )
    assert captured["stop_at_eos"] is True
    assert captured["eos_token_id"] == [5, 7]


def test_stateful_fallback_eos_list_reaches_hf_generate(monkeypatch):
    """End to end through the real hf_generate: its annotation must accept the
    list cfg.eos_token_id carries (beartype enforces it under pytest)."""
    bridge = _tiny_bridge()
    bridge.cfg.is_stateful = True
    bridge.cfg.eos_token_id = [5, 7]
    bridge.tokenizer = SimpleNamespace(eos_token_id=0)
    captured: dict = {}

    def spy(input_ids, **kwargs):
        captured.update(kwargs)
        return input_ids

    monkeypatch.setattr(bridge.original_model, "generate", spy)
    out = bridge.generate(
        torch.tensor([[1, 2, 3]]),
        stop_at_eos=True,
        use_past_kv_cache=False,
        do_sample=False,
        verbose=False,
    )
    assert isinstance(out, torch.Tensor)
    assert captured["eos_token_id"] == [5, 7]


def test_hf_generate_normalizes_eos_sequence_to_list(monkeypatch):
    bridge = _tiny_bridge()
    bridge.tokenizer = SimpleNamespace(eos_token_id=0)
    captured: dict = {}

    def spy(input_ids, **kwargs):
        captured.update(kwargs)
        return input_ids

    monkeypatch.setattr(bridge.original_model, "generate", spy)
    bridge.hf_generate(
        torch.tensor([[1, 2]]),
        eos_token_id=(5, 7),
        max_new_tokens=1,
        do_sample=False,
        return_type="tokens",
    )
    # HF takes int or list; any other sequence must arrive normalized.
    assert captured["eos_token_id"] == [5, 7]
