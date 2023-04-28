import torch
from transformers import AutoTokenizer, BertForMaskedLM

from transformer_lens.components import Embed
from transformer_lens.HookedEncoderConfig import HookedEncoderConfig


def test_custom_embed_matches_huggingface_bert_embeddings():
    cfg = HookedEncoderConfig(
        d_vocab=28996,
        d_model=768,
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer.encode(text="Hello world!", return_tensors="pt")

    embed = Embed(cfg)

    huggingface_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
    huggingface_embed = huggingface_bert.bert.embeddings.word_embeddings

    _copy(embed.W_E, huggingface_embed.weight)

    assert torch.equal(embed(input_ids), huggingface_embed(input_ids))


def _copy(mine, theirs):
    mine.detach().copy_(theirs)
