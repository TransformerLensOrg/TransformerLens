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


def _make_bridge(generated_steps: list[list[int]], tokenizer=None) -> TransformerBridge:
    bridge = object.__new__(TransformerBridge)
    torch.nn.Module.__init__(bridge)
    bridge.cfg = SimpleNamespace(device=torch.device("cpu"), eos_token_id=None)
    # First assignment only: the property validates the type on reassignment.
    bridge.tokenizer = _Tokenizer() if tokenizer is None else tokenizer
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

    # The stream is decoded cumulatively and emitted as a delta, so the stub's
    # between-token separator lands at the head of the second chunk. Real byte-level
    # tokenizers carry the separator inside the token, where old and new agree exactly.
    assert chunks == [
        ["11 12 13", "21 22 23"],
        [" 14", " 24"],
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


class _ByteTokenizer(_Tokenizer):
    """Byte-level decode, as real BPE tokenizers do — an incomplete tail becomes U+FFFD."""

    def __call__(self, inputs, **kwargs):
        return {"input_ids": torch.tensor([[]], dtype=torch.long)}

    def decode(self, tokens, **kwargs):
        return bytes(tokens.tolist()).decode("utf-8", errors="replace")


def _byte_bridge(generated_steps: list[list[int]]) -> TransformerBridge:
    return _make_bridge(generated_steps, tokenizer=_ByteTokenizer())


def _stream(bridge, steps: int):
    return list(
        bridge.generate_stream(
            "x",
            max_new_tokens=steps,
            max_tokens_per_yield=1,
            stop_at_eos=False,
            do_sample=False,
            use_past_kv_cache=False,
            return_type="str",
            verbose=False,
        )
    )


def test_character_split_across_chunks_is_not_corrupted() -> None:
    """U+65E5 is three UTF-8 bytes; decoding them in isolation yields replacement chars."""
    # 日 == e6 97 a5, arriving one byte per yield.
    chunks = _stream(_byte_bridge([[0xE6], [0x97], [0xA5]]), 3)

    assert "".join(chunks) == "日"
    assert "�" not in "".join(chunks)


def test_partial_character_is_held_back_not_emitted() -> None:
    chunks = _stream(_byte_bridge([[0xE6], [0x97], [0xA5]]), 3)

    # The first two yields carry no complete character, so they emit nothing.
    assert chunks[:2] == ["", ""]
    assert chunks[2] == "日"


def test_ascii_stream_is_unchanged_by_buffering() -> None:
    """Text that never straddles a boundary must chunk exactly as before."""
    chunks = _stream(_byte_bridge([[0x61], [0x62], [0x63]]), 3)

    assert chunks == ["a", "b", "c"]


def test_trailing_incomplete_bytes_are_flushed_at_stream_end() -> None:
    """A truncated character must still surface rather than being swallowed."""
    chunks = _stream(_byte_bridge([[0x61], [0xE6], [0x97]]), 3)

    assert "".join(chunks).startswith("a")
    assert "�" in "".join(chunks)
