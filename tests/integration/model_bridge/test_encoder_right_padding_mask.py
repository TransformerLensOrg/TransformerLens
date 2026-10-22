"""Right-padded string batches on bidirectional encoders must mask the pads like HF."""

import copy
from typing import Any, Callable

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
)

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_from_module

D_MODEL = 16
N_HEADS = 4
N_LAYERS = 2
D_MLP = 32
N_CTX = 16
PARITY_ATOL = 1e-5
PROMPTS = ["w5 w6 w7 w8 w9 w10 w11", "w5 <mask> w6"]


def _tokenizer(specials: dict[str, int], bos: str, eos: str) -> PreTrainedTokenizerFast:
    """WordLevel tokenizer that wraps each sequence in ``bos ... eos`` like the hub ones."""
    vocab = dict(specials)
    vocab.update({f"w{i}": i for i in range(5, 32)})
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single=f"{bos} $A {eos}", special_tokens=[(bos, specials[bos]), (eos, specials[eos])]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
        cls_token=bos,
        sep_token=eos,
    )
    tokenizer.padding_side = "right"
    return tokenizer


def _bert() -> tuple[Any, PreTrainedTokenizerFast, str]:
    """Tiny BertForMaskedLM with BERT's pad-id-0 / [CLS] / [SEP] layout."""
    config = BertConfig(
        vocab_size=32,
        hidden_size=D_MODEL,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        intermediate_size=D_MLP,
        max_position_embeddings=N_CTX,
        pad_token_id=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    specials = {"<pad>": 0, "<unk>": 1, "<cls>": 2, "<sep>": 3, "<mask>": 4}
    return BertForMaskedLM(config).eval(), _tokenizer(specials, "<cls>", "<sep>"), "BertForMaskedLM"


def _roberta() -> tuple[Any, PreTrainedTokenizerFast, str]:
    """Tiny RobertaForMaskedLM with RoBERTa's pad-id-1 position offset."""
    config = RobertaConfig(
        vocab_size=32,
        hidden_size=D_MODEL,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        intermediate_size=D_MLP,
        max_position_embeddings=N_CTX + 2,
        type_vocab_size=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    specials = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}
    return (
        RobertaForMaskedLM(config).eval(),
        _tokenizer(specials, "<s>", "</s>"),
        "RobertaForMaskedLM",
    )


def _forward(bridge: TransformerBridge, prompts: list[str]) -> torch.Tensor:
    return bridge(prompts)


def _run_with_cache(bridge: TransformerBridge, prompts: list[str]) -> torch.Tensor:
    logits, _ = bridge.run_with_cache(prompts)
    return logits


@pytest.mark.parametrize("build", [_bert, _roberta], ids=["bert", "roberta"])
@pytest.mark.parametrize("entry", [_forward, _run_with_cache], ids=["forward", "run_with_cache"])
def test_right_padded_string_batch_matches_hf_masked(
    build: Callable[[], tuple[Any, PreTrainedTokenizerFast, str]],
    entry: Callable[[TransformerBridge, list[str]], torch.Tensor],
) -> None:
    """Real-token logits match HF fed the tokenizer's attention mask."""
    torch.manual_seed(0)
    reference, tokenizer, architecture = build()
    bridge = build_bridge_from_module(
        copy.deepcopy(reference).eval(),
        architecture,
        hf_config=copy.deepcopy(reference.config),
        tokenizer=tokenizer,
        dtype=torch.float32,
        device="cpu",
    )
    bridge.eval()
    encoded = tokenizer(PROMPTS, padding=True, return_tensors="pt")
    real = encoded.attention_mask.bool()
    assert not bool(real.all()), "batch must contain right padding"
    torch.testing.assert_close(bridge.to_tokens(PROMPTS), encoded.input_ids, rtol=0, atol=0)

    with torch.inference_mode():
        bridge_logits = entry(bridge, PROMPTS)
        hf_logits = reference(**encoded).logits
        unmasked_hf_logits = reference(input_ids=encoded.input_ids).logits

    torch.testing.assert_close(bridge_logits[real], hf_logits[real], rtol=0.0, atol=PARITY_ATOL)
    # The pads must actually move the short row, or the comparison above has no teeth.
    assert (unmasked_hf_logits[real] - hf_logits[real]).abs().max().item() > 10 * PARITY_ATOL
