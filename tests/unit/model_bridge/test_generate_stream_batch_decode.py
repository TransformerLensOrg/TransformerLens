"""``generate_stream(return_type="str")`` decodes every row of the batch.

The streaming decoder used to call ``tokenizer.decode(tokens[0])``, so a batched
stream repeated the first sequence's text for the whole batch. These tests drive
the yield/decode bookkeeping with a stub token stream — no model load — and pin
both halves of the contract: a one-row batch still yields a bare string, and a
larger batch yields one string per row, as ``generate()`` does.
"""

from types import MethodType, SimpleNamespace

import torch

from transformer_lens.model_bridge.bridge import TransformerBridge


class _Tokenizer:
    eos_token_id = None
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, str) or len(inputs) == 1:
            return {"input_ids": torch.tensor([[11, 12]])}
        return {"input_ids": torch.tensor([[11, 12], [21, 22]])}

    def decode(self, tokens, **kwargs):
        return " ".join(str(token) for token in tokens.tolist())


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)


def _make_bridge(generated_steps: list[list[int]]) -> TransformerBridge:
    bridge = object.__new__(TransformerBridge)
    torch.nn.Module.__init__(bridge)
    bridge.cfg = SimpleNamespace(device=torch.device("cpu"), eos_token_id=None)
    bridge.tokenizer = _Tokenizer()
    bridge.__dict__["original_model"] = _Model()
    bridge._ensure_generation_supported = MethodType(lambda self, api: None, bridge)
    bridge._resolve_generation_caching = MethodType(lambda self, requested, batched: False, bridge)

    def fake_generate_tokens(self, *args, **kwargs):
        for step, generated_tokens in enumerate(generated_steps):
            yield torch.tensor(generated_tokens), None, step == len(generated_steps) - 1

    bridge._generate_tokens = MethodType(fake_generate_tokens, bridge)
    return bridge


def test_batched_string_stream_decodes_every_row() -> None:
    chunks = list(
        _make_bridge([[13, 23], [14, 24]]).generate_stream(
            ["first", "second"],
            max_new_tokens=2,
            max_tokens_per_yield=1,
            stop_at_eos=False,
            do_sample=False,
            use_past_kv_cache=False,
            return_type="input",
            verbose=False,
        )
    )

    assert chunks == [
        ["11 12 13", "21 22 23"],
        ["14", "24"],
    ]


def test_batched_token_stream_decodes_every_row() -> None:
    chunks = list(
        _make_bridge([[33, 43]]).generate_stream(
            torch.tensor([[31, 32], [41, 42]]),
            max_new_tokens=1,
            max_tokens_per_yield=99,
            stop_at_eos=False,
            do_sample=False,
            use_past_kv_cache=False,
            return_type="str",
            verbose=False,
        )
    )

    assert chunks == [["31 32 33", "41 42 43"]]


def test_single_string_stream_remains_scalar() -> None:
    chunks = list(
        _make_bridge([[13]]).generate_stream(
            "first",
            max_new_tokens=1,
            max_tokens_per_yield=99,
            stop_at_eos=False,
            do_sample=False,
            use_past_kv_cache=False,
            return_type="input",
            verbose=False,
        )
    )

    assert chunks == ["11 12 13"]
