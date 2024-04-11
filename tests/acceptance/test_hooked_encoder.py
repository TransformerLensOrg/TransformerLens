from typing import List

import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch.testing import assert_close
from transformers import AutoTokenizer, BertForMaskedLM

from transformer_lens import HookedEncoder

MODEL_NAME = "bert-base-cased"


@pytest.fixture(scope="module")
def our_bert():
    return HookedEncoder.from_pretrained(MODEL_NAME, device="cpu")


@pytest.fixture(scope="module")
def huggingface_bert():
    return BertForMaskedLM.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def hello_world_tokens(tokenizer):
    return tokenizer("Hello, world!", return_tensors="pt")["input_ids"]


def test_full_model(our_bert, huggingface_bert, tokenizer):
    sequences = [
        "Hello, world!",
        "this is another sequence of tokens",
    ]
    tokenized = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    huggingface_bert_out = huggingface_bert(input_ids, attention_mask=attention_mask).logits
    our_bert_out = our_bert(input_ids, one_zero_attention_mask=attention_mask)
    assert_close(huggingface_bert_out, our_bert_out, rtol=1.3e-6, atol=4e-5)


def test_embed_one_sentence(our_bert, huggingface_bert, hello_world_tokens):
    huggingface_embed = huggingface_bert.bert.embeddings
    our_embed = our_bert.embed

    huggingface_embed_out = huggingface_embed(hello_world_tokens)[0]
    our_embed_out = our_embed(hello_world_tokens).squeeze(0)
    assert_close(huggingface_embed_out, our_embed_out)


def test_embed_two_sentences(our_bert, huggingface_bert, tokenizer):
    encoding = tokenizer("First sentence.", "Second sentence.", return_tensors="pt")
    input_ids = encoding["input_ids"]
    token_type_ids = encoding["token_type_ids"]

    huggingface_embed_out = huggingface_bert.bert.embeddings(
        input_ids, token_type_ids=token_type_ids
    )[0]
    our_embed_out = our_bert.embed(input_ids, token_type_ids=token_type_ids).squeeze(0)
    assert_close(huggingface_embed_out, our_embed_out)


def test_attention(our_bert, huggingface_bert, hello_world_tokens):
    huggingface_embed = huggingface_bert.bert.embeddings
    huggingface_attn = huggingface_bert.bert.encoder.layer[0].attention

    embed_out = huggingface_embed(hello_world_tokens)

    our_attn = our_bert.blocks[0].attn

    our_attn_out = our_attn(embed_out, embed_out, embed_out)
    huggingface_self_attn_out = huggingface_attn.self(embed_out)[0]
    huggingface_attn_out = huggingface_attn.output.dense(huggingface_self_attn_out)
    assert_close(our_attn_out, huggingface_attn_out)


def test_bert_block(our_bert, huggingface_bert, hello_world_tokens):
    huggingface_embed = huggingface_bert.bert.embeddings
    huggingface_block = huggingface_bert.bert.encoder.layer[0]

    embed_out = huggingface_embed(hello_world_tokens)

    our_block = our_bert.blocks[0]

    our_block_out = our_block(embed_out)
    huggingface_block_out = huggingface_block(embed_out)[0]
    assert_close(our_block_out, huggingface_block_out)


def test_mlm_head(our_bert, huggingface_bert, hello_world_tokens):
    huggingface_bert_core_outputs = huggingface_bert.bert(hello_world_tokens).last_hidden_state

    our_mlm_head_out = our_bert.mlm_head(huggingface_bert_core_outputs)
    our_unembed_out = our_bert.unembed(our_mlm_head_out)
    huggingface_predictions_out = huggingface_bert.cls.predictions(huggingface_bert_core_outputs)

    assert_close(our_unembed_out, huggingface_predictions_out, rtol=1.3e-6, atol=4e-5)


def test_unembed(our_bert, huggingface_bert, hello_world_tokens):
    huggingface_bert_core_outputs = huggingface_bert.bert(hello_world_tokens).last_hidden_state

    our_mlm_head_out = our_bert.mlm_head(huggingface_bert_core_outputs)
    huggingface_predictions_out = huggingface_bert.cls.predictions.transform(
        huggingface_bert_core_outputs
    )

    print((our_mlm_head_out - huggingface_predictions_out).abs().max())
    assert_close(our_mlm_head_out, huggingface_predictions_out, rtol=1.3e-3, atol=1e-5)


def test_run_with_cache(our_bert, huggingface_bert, hello_world_tokens):
    model = HookedEncoder.from_pretrained("bert-base-cased")
    logits, cache = model.run_with_cache(hello_world_tokens)

    # check that an arbitrary subset of the keys exist
    assert "embed.hook_embed" in cache
    assert "blocks.0.attn.hook_q" in cache
    assert "blocks.3.attn.hook_attn_scores" in cache
    assert "blocks.7.hook_resid_post" in cache
    assert "mlm_head.ln.hook_normalized" in cache


def test_from_pretrained_revision():
    """
    Check that the from_pretrained parameter `revision` (= git version) works
    """

    _ = HookedEncoder.from_pretrained(MODEL_NAME, revision="main")

    try:
        _ = HookedEncoder.from_pretrained(MODEL_NAME, revision="inexistent_branch_name")
    except:
        pass
    else:
        raise AssertionError("Should have raised an error")


@pytest.mark.skipif(
    torch.backends.mps.is_available() or not torch.cuda.is_available(),
    reason="bfloat16 unsupported by MPS: https://github.com/pytorch/pytorch/issues/78168 or no GPU",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision(dtype):
    """Check the 16 bits loading and inferences."""
    model = HookedEncoder.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    assert model.W_K.dtype == dtype

    _ = model(model.tokenizer("Hello, world", return_tensors="pt")["input_ids"])


def test_predictions(our_bert, huggingface_bert, tokenizer):
    input_ids = tokenizer("The [MASK] sat on the mat", return_tensors="pt")["input_ids"]

    def get_predictions(logits: Float[torch.Tensor, "batch pos d_vocab"], positions: List[int]):
        logits_at_position = logits.squeeze(0)[positions]
        predicted_tokens = F.softmax(logits_at_position, dim=-1).argmax(dim=-1)
        return tokenizer.batch_decode(predicted_tokens)

    our_bert_out = our_bert(input_ids)
    our_prediction = get_predictions(our_bert_out, [2])

    huggingface_bert_out = huggingface_bert(input_ids).logits
    huggingface_prediction = get_predictions(huggingface_bert_out, [2])

    assert our_prediction == huggingface_prediction


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
def test_cuda(hello_world_tokens):
    model = HookedEncoder.from_pretrained(MODEL_NAME)
    model(hello_world_tokens)
