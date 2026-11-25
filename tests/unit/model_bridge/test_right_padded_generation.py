"""Right-padded prompts sample from their last attended token."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.fixture(scope="module")
def bridge():
    cfg = GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=2, n_head=2)
    cfg._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = GPT2LMHeadModel(cfg).eval()
    return build_bridge_from_module(
        model, "GPT2LMHeadModel", hf_config=cfg, dtype=torch.float32, device="cpu"
    ).eval()


def _options(cached):
    return dict(
        max_new_tokens=3,
        do_sample=False,
        stop_at_eos=False,
        verbose=False,
        use_past_kv_cache=cached,
    )


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_padded_generation_matches_unpadded(bridge, cached, batched, explicit_positions):
    padded = torch.tensor([[2, 3, 4, 0, 0], [7, 8, 0, 0, 0], [1, 2, 3, 4, 5]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    if not batched:
        padded, mask = padded[:1], mask[:1]
    opts = _options(cached)
    positions = mask.cumsum(dim=-1) + 6 if explicit_positions else None
    result = bridge.generate(
        padded,
        attention_mask=mask,
        position_ids=positions,
        output_logits=True,
        **opts,
    )
    assert torch.equal(result.sequences[:, : padded.shape[1]], padded)
    assert len(result.logits) == 3
    for row, valid in enumerate(mask.bool()):
        row_positions = positions[row : row + 1, valid] if positions is not None else None
        reference = bridge.generate(
            padded[row : row + 1, valid],
            position_ids=row_positions,
            output_logits=True,
            **opts,
        )
        assert torch.equal(result.sequences[row, -3:], reference.sequences[0, -3:])
        for actual, expected in zip(result.logits, reference.logits):
            torch.testing.assert_close(actual[row], expected[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cached", [False, True])
def test_stream_matches_unpadded(bridge, cached):
    opts = {**_options(cached), "max_tokens_per_yield": 1}
    reference = list(bridge.generate_stream(torch.tensor([[2, 3, 4]]), **opts))
    padded = list(
        bridge.generate_stream(
            torch.tensor([[2, 3, 4, 0, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0, 0]]),
            **opts,
        )
    )
    assert len(padded) == len(reference) == 3
    for actual, expected in zip(padded, reference):
        torch.testing.assert_close(actual[:, -1:], expected[:, -1:])


@pytest.mark.parametrize("method", ["generate", "generate_stream"])
def test_all_masked_prompt_row_raises(bridge, method):
    with pytest.raises(ValueError, match="at least one attended"):
        result = getattr(bridge, method)(
            torch.tensor([[2, 3, 4], [0, 0, 0]]),
            attention_mask=torch.tensor([[1, 1, 1], [0, 0, 0]]),
            **_options(True),
        )
        if method == "generate_stream":
            list(result)


@pytest.fixture()
def shared_bos_pad_bridge(bridge, monkeypatch):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<bos-pad>": 0, "<unk>": 1}, unk_token="<unk>")),
        bos_token="<bos-pad>",
        pad_token="<bos-pad>",
        unk_token="<unk>",
    )
    monkeypatch.setattr(bridge, "tokenizer", tokenizer)
    return bridge


def _collect_generation(bridge, method, tokens, **kwargs):
    result = getattr(bridge, method)(tokens, **kwargs)
    return torch.cat(list(result), dim=1) if method == "generate_stream" else result


@pytest.mark.parametrize("method", ["generate", "generate_stream"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_right_padding_preserves_bos_only_prompt(shared_bos_pad_bridge, method, cached, batched):
    bridge = shared_bos_pad_bridge
    tokens = torch.tensor([[0, 0, 0], [0, 3, 4]]) if batched else torch.tensor([[0]])
    opts = _options(cached)
    actual = _collect_generation(
        bridge, method, tokens, padding_side="right", prepend_bos=True, **opts
    )
    assert torch.equal(actual[:, : tokens.shape[1]], tokens)
    prompts = [torch.tensor([[0]])]
    if batched:
        prompts.append(tokens[1:])
    for row, prompt in enumerate(prompts):
        expected = bridge.generate(prompt, **opts)
        torch.testing.assert_close(actual[row, -3:], expected[0, -3:])


@pytest.mark.parametrize("method", ["generate", "generate_stream"])
@pytest.mark.parametrize("explicit_mask", [False, True])
def test_right_padding_does_not_invent_bos(shared_bos_pad_bridge, method, explicit_mask):
    tokens = torch.tensor([[0, 0]])
    with pytest.raises(ValueError, match="at least one attended"):
        _collect_generation(
            shared_bos_pad_bridge,
            method,
            tokens,
            padding_side="right",
            prepend_bos=explicit_mask,
            attention_mask=torch.zeros_like(tokens) if explicit_mask else None,
            **_options(True),
        )
