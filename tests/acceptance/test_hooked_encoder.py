from torch.testing import assert_close
from transformers import AutoTokenizer, BertForMaskedLM

from transformer_lens.HookedEncoder import HookedEncoder


def test_hooked_encoder_full():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    sequences = [
        "Hello, world!",
        "this is another sequence of tokens",
    ]
    tokenized = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
    our_bert = HookedEncoder.from_pretrained("bert-base-cased")

    hf_bert_out = hf_bert(input_ids, attention_mask=attention_mask).logits
    our_bert_out = our_bert(input_ids, one_zero_attention_mask=attention_mask)
    assert_close(hf_bert_out, our_bert_out, rtol=1.3e-6, atol=4e-5)


def test_bert_from_pretrained_embed_one_sentence():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]

    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased").bert
    our_bert = HookedEncoder.from_pretrained("bert-base-cased")

    hf_embed = hf_bert.embeddings
    our_embed = our_bert.embed

    hf_embed_out = hf_embed(input_ids)[0]
    our_embed_out = our_embed(input_ids).squeeze(0)
    assert_close(hf_embed_out, our_embed_out)


# This test might be slightly redundant with the previous, but if anything breaks
# it will help us to track down the issue
def test_bert_from_pretrained_embed_two_sentences():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    encoding = tokenizer("First sentence.", "Second sentence.", return_tensors="pt")
    input_ids = encoding["input_ids"]
    token_type_ids = encoding["token_type_ids"]

    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased").bert
    our_bert = HookedEncoder.from_pretrained("bert-base-cased")

    hf_embed_out = hf_bert.embeddings(input_ids, token_type_ids=token_type_ids)[0]
    our_embed_out = our_bert.embed(input_ids, token_type_ids=token_type_ids).squeeze(0)
    assert_close(hf_embed_out, our_embed_out)


def test_bert_from_pretrained_attention():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]

    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased").bert
    hf_embed = hf_bert.embeddings
    hf_attn = hf_bert.encoder.layer[0].attention

    embed_out = hf_embed(input_ids)

    our_bert = HookedEncoder.from_pretrained("bert-base-cased")
    our_attn = our_bert.blocks[0].attn

    our_attn_out = our_attn(embed_out, embed_out, embed_out)
    hf_self_attn_out = hf_attn.self(embed_out)[0]
    hf_attn_out = hf_attn.output.dense(hf_self_attn_out)
    assert_close(our_attn_out, hf_attn_out)


def test_bert_block():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]

    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased").bert
    hf_embed = hf_bert.embeddings
    hf_block = hf_bert.encoder.layer[0]

    embed_out = hf_embed(input_ids)

    our_bert = HookedEncoder.from_pretrained("bert-base-cased")
    our_block = our_bert.blocks[0]

    our_block_out = our_block(embed_out)
    hf_block_out = hf_block(embed_out)[0]
    assert_close(our_block_out, hf_block_out)


def test_hooked_encoder_mlm_head():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]

    our_bert = HookedEncoder.from_pretrained("bert-base-cased")
    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
    hf_bert_core_outputs = hf_bert.bert(input_ids).last_hidden_state

    our_mlm_head_out = our_bert.mlm_head(hf_bert_core_outputs)
    our_unembed_out = our_bert.unembed(our_mlm_head_out)
    hf_predictions_out = hf_bert.cls.predictions(hf_bert_core_outputs)

    assert_close(our_unembed_out, hf_predictions_out, rtol=1.3e-6, atol=4e-5)


def test_hooked_encoder_unembed():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]

    our_bert = HookedEncoder.from_pretrained("bert-base-cased")
    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
    hf_bert_core_outputs = hf_bert.bert(input_ids).last_hidden_state

    our_mlm_head_out = our_bert.mlm_head(hf_bert_core_outputs)
    hf_predictions_out = hf_bert.cls.predictions.transform(hf_bert_core_outputs)

    print((our_mlm_head_out - hf_predictions_out).abs().max())
    assert_close(our_mlm_head_out, hf_predictions_out, rtol=1.3e-3, atol=1e-5)


def test_hooked_encoder_run_with_cache():
    model = HookedEncoder.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]
    logits, cache = model.run_with_cache(input_ids)

    # check that an arbitrary subset of the keys exist
    assert "embed.hook_embed" in cache
    assert "blocks.0.attn.hook_q" in cache
    assert "blocks.3.attn.hook_attn_scores" in cache
    assert "blocks.7.hook_resid_post" in cache
    assert "mlm_head.ln.hook_normalized" in cache


# TODO: test the masked output
# preds = F.softmax(hf_bert_out, dim=-1).argmax(dim=-1)
# pred_strings = tokenizer.batch_decode(preds)
