"""HF generation must honor the Bridge EOS stopping switch."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
)

from tests.integration.model_bridge.helpers import make_tiny_pair


@pytest.fixture(scope="module")
def eos_bridge():
    config = GPT2Config(
        vocab_size=8,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=16,
        bos_token_id=1,
        eos_token_id=0,
        pad_token_id=0,
    )
    bridge, _ = make_tiny_pair(config, "GPT2LMHeadModel", loader=GPT2LMHeadModel)
    with torch.no_grad():
        for parameter in bridge.original_model.parameters():
            parameter.zero_()
    tokenizer = Tokenizer(WordLevel({"<eos>": 0, "a": 1, "<unk>": 2}, unk_token="<unk>"))
    bridge.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<eos>",
        pad_token="<eos>",
        unk_token="<unk>",
    )
    return bridge


@pytest.mark.parametrize("prompt", (torch.tensor([[1]]), "a", ["a", "a"]))
@pytest.mark.parametrize("explicit_eos", (None, 0, 1))
@pytest.mark.parametrize("stop_at_eos", (False, True))
def test_eos_stopping_switch(eos_bridge, prompt, explicit_eos, stop_at_eos):
    result = eos_bridge.hf_generate(
        prompt,
        max_new_tokens=3,
        do_sample=False,
        stop_at_eos=stop_at_eos,
        eos_token_id=explicit_eos,
        return_type="tokens",
    )
    batch_size = 2 if isinstance(prompt, list) else 1
    new_tokens = 1 if stop_at_eos and explicit_eos != 1 else 3
    expected = torch.tensor([[1] + [0] * new_tokens]).expand(batch_size, -1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert eos_bridge.original_model.generation_config.eos_token_id == 0


def test_eos_disabled_preserves_model_output(eos_bridge):
    result = eos_bridge.hf_generate(
        torch.tensor([[1]]),
        max_new_tokens=3,
        do_sample=False,
        stop_at_eos=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    torch.testing.assert_close(result.sequences, torch.tensor([[1, 0, 0, 0]]), rtol=0, atol=0)
    assert len(result.scores) == 3


def test_eos_disabled_preserves_custom_stopping(eos_bridge):
    class StopAfterTwoTokens(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full(
                (input_ids.shape[0],),
                input_ids.shape[1] >= 3,
                dtype=torch.bool,
                device=input_ids.device,
            )

    result = eos_bridge.hf_generate(
        torch.tensor([[1]]),
        max_new_tokens=3,
        do_sample=False,
        stop_at_eos=False,
        return_type="tokens",
        stopping_criteria=StoppingCriteriaList([StopAfterTwoTokens()]),
    )
    torch.testing.assert_close(result, torch.tensor([[1, 0, 0]]), rtol=0, atol=0)
