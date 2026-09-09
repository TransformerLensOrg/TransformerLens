"""Batched ``hf_generate`` calls preserve each prompt's generation context."""

from types import MethodType, SimpleNamespace
from typing import Any

import torch

from transformer_lens.model_bridge.bridge import TransformerBridge


class _BatchEncoding(dict[str, torch.Tensor]):
    def to(self, device: torch.device) -> "_BatchEncoding":
        return _BatchEncoding({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    eos_token_id = 0
    padding_side = "right"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        inputs: str | list[str],
        *,
        padding_side: str | None = None,
        **kwargs: Any,
    ) -> _BatchEncoding:
        self.calls += 1
        if isinstance(inputs, str):
            return _BatchEncoding(
                {
                    "input_ids": torch.tensor([[11, 12]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }
            )
        effective_padding_side = padding_side or self.padding_side
        if effective_padding_side == "right":
            return _BatchEncoding(
                {
                    "input_ids": torch.tensor([[11, 0], [21, 22]]),
                    "attention_mask": torch.tensor([[1, 0], [1, 1]]),
                }
            )
        return _BatchEncoding(
            {
                "input_ids": torch.tensor([[0, 11], [21, 22]]),
                "attention_mask": torch.tensor([[0, 1], [1, 1]]),
            }
        )

    def decode(self, tokens: torch.Tensor, **kwargs: Any) -> str:
        return " ".join(str(token) for token in tokens.tolist())


class _Model(torch.nn.Module):
    def __init__(self, *, is_encoder_decoder: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=is_encoder_decoder)
        self.input_ids: torch.Tensor | None = None
        self.generation_kwargs: dict[str, Any] = {}

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.input_ids = input_ids
        self.generation_kwargs = kwargs
        return input_ids


def _make_bridge(
    *, is_encoder_decoder: bool = False
) -> tuple[TransformerBridge, _Tokenizer, _Model]:
    bridge = object.__new__(TransformerBridge)
    torch.nn.Module.__init__(bridge)
    bridge.cfg = SimpleNamespace(device=torch.device("cpu"))
    tokenizer = _Tokenizer()
    bridge.tokenizer = tokenizer
    model = _Model(is_encoder_decoder=is_encoder_decoder)
    bridge.original_model = model
    bridge._ensure_generation_supported = MethodType(lambda self, api: None, bridge)
    return bridge, tokenizer, model


def test_hf_generate_decoder_only_list_uses_left_padding_without_mutating_tokenizer() -> None:
    bridge, tokenizer, model = _make_bridge()

    bridge.hf_generate(["short", "a longer prompt"], return_type="tokens")

    torch.testing.assert_close(model.input_ids, torch.tensor([[0, 11], [21, 22]]))
    assert tokenizer.padding_side == "right"


def test_hf_generate_encoder_decoder_list_preserves_tokenizer_padding() -> None:
    bridge, tokenizer, model = _make_bridge(is_encoder_decoder=True)

    bridge.hf_generate(["short", "a longer prompt"], return_type="tokens")

    torch.testing.assert_close(
        model.input_ids,
        torch.tensor([[11, 0], [21, 22]]),
    )
    torch.testing.assert_close(
        model.generation_kwargs["attention_mask"], torch.tensor([[1, 0], [1, 1]])
    )
    assert tokenizer.padding_side == "right"


def test_hf_generate_list_replaces_explicit_mask_with_tokenizer_mask() -> None:
    bridge, _, model = _make_bridge()
    explicit_mask = torch.tensor([[1, 0], [1, 1]])

    bridge.hf_generate(
        ["short", "a longer prompt"],
        return_type="tokens",
        attention_mask=explicit_mask,
    )

    torch.testing.assert_close(
        model.generation_kwargs["attention_mask"], torch.tensor([[0, 1], [1, 1]])
    )


def test_hf_generate_tensor_input_does_not_synthesize_attention_mask() -> None:
    bridge, tokenizer, model = _make_bridge()

    bridge.hf_generate(torch.tensor([[11, 12]]), return_type="tokens")

    assert tokenizer.calls == 0
    assert "attention_mask" not in model.generation_kwargs
