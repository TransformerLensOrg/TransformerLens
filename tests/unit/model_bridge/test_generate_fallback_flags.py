"""Stateful hf_generate fallback preserves stop flags and sampling penalties; the hooked
loop rejects penalties it would otherwise ignore."""

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    MambaConfig,
    MambaForCausalLM,
    PreTrainedTokenizerFast,
)

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


@pytest.mark.parametrize("do_sample", [False, True])
def test_hf_generate_sends_temperature_only_when_sampling(monkeypatch, do_sample):
    """Whisper's generate samples whenever it is given a temperature above 0, so a
    greedy hf_generate call must not send one."""
    bridge = _tiny_bridge()
    bridge.tokenizer = SimpleNamespace(eos_token_id=0)
    captured: dict = {}

    def spy(input_ids, **kwargs):
        captured.update(kwargs)
        return input_ids

    monkeypatch.setattr(bridge.original_model, "generate", spy)
    bridge.hf_generate(
        torch.tensor([[1, 2]]),
        max_new_tokens=1,
        do_sample=do_sample,
        temperature=0.7,
        return_type="tokens",
    )
    assert ("temperature" in captured) is do_sample
    if do_sample:
        assert captured["temperature"] == 0.7


@pytest.mark.parametrize("penalty", [0.8, 1.0, 1.5])
def test_stateful_fallback_forwards_repetition_penalty(monkeypatch, penalty):
    bridge = _tiny_bridge()
    bridge.cfg.is_stateful = True
    bridge.tokenizer = SimpleNamespace(eos_token_id=0)
    captured: dict = {}

    def spy(input_ids, **kwargs):
        captured.update(kwargs)
        return input_ids

    monkeypatch.setattr(bridge.original_model, "generate", spy)
    bridge.generate(
        torch.tensor([[1, 2, 3]]),
        use_past_kv_cache=False,
        repetition_penalty=penalty,
        do_sample=False,
        verbose=False,
    )
    assert captured["repetition_penalty"] == penalty


@pytest.mark.parametrize("penalty", [-0.5, 0.5])
def test_stateful_fallback_rejects_frequency_penalty(monkeypatch, penalty):
    bridge = _tiny_bridge()
    bridge.cfg.is_stateful = True
    captured = _capture_hf_generate(bridge, monkeypatch)
    with pytest.raises(NotImplementedError, match="freq_penalty.*use_past_kv_cache=True"):
        bridge.generate(
            torch.tensor([[1, 2, 3]]),
            use_past_kv_cache=False,
            freq_penalty=penalty,
            verbose=False,
        )
    assert not captured


@pytest.mark.parametrize(
    "inputs, extra_kwargs, cause",
    [
        pytest.param(torch.tensor([[1, 2, 3]]), {"stray_kwarg": 1}, "multimodal", id="stray-kwarg"),
        pytest.param(torch.randn(1, 3, 16), {}, "inputs_embeds", id="inputs-embeds"),
    ],
)
def test_stateful_fallback_frequency_penalty_names_real_cause(
    monkeypatch, inputs, extra_kwargs, cause
):
    """With use_past_kv_cache=True already set, the error must not advise setting it."""
    bridge = _tiny_bridge()
    bridge.cfg.is_stateful = True
    captured = _capture_hf_generate(bridge, monkeypatch)
    with pytest.raises(NotImplementedError) as excinfo:
        bridge.generate(
            inputs,
            use_past_kv_cache=True,
            freq_penalty=0.5,
            verbose=False,
            **extra_kwargs,
        )
    message = str(excinfo.value)
    assert "use_past_kv_cache" not in message
    assert cause in message and "hf_generate()" in message
    assert not captured


def test_stateful_fallback_repetition_penalty_matches_hf_generation():
    bridge = _tiny_bridge()
    bridge.cfg.is_stateful = True
    bridge.tokenizer = SimpleNamespace(eos_token_id=0)
    tokens = torch.tensor([[1, 2, 3]])
    kwargs = dict(max_new_tokens=4, do_sample=False, stop_at_eos=False)
    expected = bridge.hf_generate(tokens, repetition_penalty=2.0, **kwargs)
    unpenalized = bridge.hf_generate(tokens, repetition_penalty=1.0, **kwargs)
    actual = bridge.generate(
        tokens,
        repetition_penalty=2.0,
        use_past_kv_cache=False,
        verbose=False,
        **kwargs,
    )
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, unpenalized)


@pytest.fixture(scope="module")
def mamba_bridge():
    config = MambaConfig(
        vocab_size=16,
        hidden_size=16,
        state_size=4,
        num_hidden_layers=1,
        expand=2,
        time_step_rank=2,
        conv_kernel=4,
        bos_token_id=1,
        eos_token_id=2,
    )
    tokenizer = Tokenizer(
        WordLevel({"<unk>": 0, "<bos>": 1, "<eos>": 2, "hello": 3, "world": 4}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<eos>",
    )
    return build_bridge_from_module(
        MambaForCausalLM(config).eval(),
        "MambaForCausalLM",
        hf_config=config,
        tokenizer=tokenizer,
        device="cpu",
    ).eval()


@pytest.mark.parametrize("prompt", [torch.tensor([[1, 3, 4]]), "hello world"])
@pytest.mark.parametrize("include_input", [False, True])
def test_mamba_fallback_returns_raw_logits(mamba_bridge, prompt, include_input):
    b = mamba_bridge
    expected_input = b.to_tokens(prompt, prepend_bos=True) if isinstance(prompt, str) else prompt
    kwargs = dict(max_new_tokens=2, do_sample=False, stop_at_eos=False)
    expected = b.hf_generate(expected_input, output_logits=True, return_type="tokens", **kwargs)
    result = b.generate(
        prompt,
        use_past_kv_cache=False,
        output_logits=True,
        return_type="str",
        prepend_bos=True,
        return_input_tokens=include_input,
        verbose=False,
        **kwargs,
    )
    if include_input:
        result, actual_input = result
        torch.testing.assert_close(actual_input, expected_input)
    torch.testing.assert_close(result.sequences, expected.sequences)
    assert len(result.logits) == len(expected.logits) == 2
    for actual, reference in zip(result.logits, expected.logits):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize(
    "prompt,return_type,expected_type",
    [
        (torch.tensor([[1, 3, 4]]), "str", str),
        ("hello world", "tokens", torch.Tensor),
        ("hello world", "input", str),
        (["hello world", "hello"], "input", list),
    ],
)
def test_mamba_fallback_formats_output_and_returns_actual_input(
    mamba_bridge, monkeypatch, prompt, return_type, expected_type
):
    b = mamba_bridge
    seen = {}
    generate = b.original_model.generate

    def record(tokens, **kwargs):
        seen["tokens"] = tokens.clone()
        seen["mask"] = kwargs.get("attention_mask")
        return generate(tokens, **kwargs)

    monkeypatch.setattr(b.original_model, "generate", record)
    result = b.generate(
        prompt,
        use_past_kv_cache=False,
        return_type=return_type,
        return_input_tokens=True,
        prepend_bos=True,
        max_new_tokens=2,
        do_sample=False,
        stop_at_eos=False,
        verbose=False,
    )
    assert isinstance(result, tuple)
    output, input_tokens = result
    assert isinstance(output, expected_type)
    torch.testing.assert_close(input_tokens, seen["tokens"])
    if isinstance(prompt, list):
        assert len(output) == 2
        assert seen["mask"].tolist() == [[1, 1, 1], [0, 1, 1]]


@pytest.mark.parametrize("do_sample", [True, False])
@pytest.mark.parametrize("method", ["generate", "generate_stream"])
def test_hooked_loop_rejects_negative_frequency_penalty(method, do_sample):
    bridge = _tiny_bridge()
    with pytest.raises(ValueError, match="freq_penalty must be non-negative"):
        result = getattr(bridge, method)(
            torch.tensor([[1, 2, 3]]),
            max_new_tokens=2,
            do_sample=do_sample,
            freq_penalty=-0.5,
            stop_at_eos=False,
            verbose=False,
        )
        if method == "generate_stream":
            list(result)
