from torch.testing import assert_close
from transformers import AutoTokenizer, BertForMaskedLM

from transformer_lens import loading_from_pretrained as loading
from transformer_lens.components import BertMLMHead, Unembed
from transformer_lens.HookedEncoder import HookedEncoder

# preds = F.softmax(hf_bert_out, dim=-1).argmax(dim=-1)
# pred_strings = tokenizer.batch_decode(preds)


def test_hooked_encoder_padding():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    sequences = [
        "Hello, world!",
        "this is another sequence of tokens",
    ]

    tokenized = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased").bert
    our_bert = HookedEncoder.from_pretrained("bert-base-cased")

    hf_bert_out = hf_bert(input_ids, attention_mask=attention_mask)[0]
    our_bert_out = our_bert(input_ids, one_zero_attention_mask=attention_mask)

    assert_close(hf_bert_out, our_bert_out)


def test_hooked_encoder_run_with_cache():
    model = HookedEncoder.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]
    logits, cache_dict = model.run_with_cache(input_ids)
    print(f"{logits.shape=}")
    # print(f"{cache_dict.keys()=}")
    for k in cache_dict:
        if k.startswith("blocks.0") or "blocks" not in k:
            print(k, end=" ")


def test_hooked_encoder_mlm_head():
    cfg_dict = loading.convert_hf_model_config("bert-base-cased")
    mlm_head = BertMLMHead(cfg_dict)

    hf_bert_full = BertForMaskedLM.from_pretrained("bert-base-cased")
    hf_bert_core = hf_bert_full.bert

    state_dict = convert_mlm_head_state_dict(hf_bert_full.cls)

    mlm_head.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]

    hf_bert_core_outputs = hf_bert_core(input_ids).last_hidden_state
    hf_bert_mlm_head = hf_bert_full.cls.predictions.transform
    our_mlm_head_outputs = mlm_head(hf_bert_core_outputs)
    their_mlm_head_outputs = hf_bert_mlm_head(hf_bert_core_outputs)
    assert_close(our_mlm_head_outputs, their_mlm_head_outputs)
    # breakpoint()

    their_unembed = hf_bert_full.cls.predictions.decoder
    unembed_state_dict = {
        "W_U": hf_bert_core.embeddings.word_embeddings.weight.T,
        "b_U": their_unembed.bias,
    }
    our_unembed = Unembed(cfg_dict)
    our_unembed.load_state_dict(unembed_state_dict)
    their_unembed_outputs = their_unembed(our_mlm_head_outputs)
    our_unembed_outputs = our_unembed(our_mlm_head_outputs)
    assert_close(their_unembed_outputs, our_unembed_outputs)


def convert_mlm_head_state_dict(mlm_head):
    state_dict = {
        "W": mlm_head.predictions.transform.dense.weight,
        "b": mlm_head.predictions.transform.dense.bias,
        "ln.w": mlm_head.predictions.transform.LayerNorm.weight,
        "ln.b": mlm_head.predictions.transform.LayerNorm.bias,
    }
    return state_dict


def test_bert_from_pretrained_full():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]

    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased").bert
    our_bert = HookedEncoder.from_pretrained("bert-base-cased")

    hf_bert_out = hf_bert(input_ids)[0]
    our_bert_out = our_bert(input_ids)
    assert_close(hf_bert_out, our_bert_out)


# This test might be slightly redundant with the previous, but if anything breaks it will help us to track down the issue
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
