import pytest
import torch
from jaxtyping import Float
from torch.testing import assert_close
from transformers import AutoTokenizer, T5ForConditionalGeneration

from transformer_lens import HookedEncoderDecoder

MODEL_NAME = "t5-small"


@pytest.fixture(scope="module")
def our_model():
    return HookedEncoderDecoder.from_pretrained(MODEL_NAME, device="cpu")


@pytest.fixture(scope="module")
def huggingface_model():
    return T5ForConditionalGeneration.from_pretrained(MODEL_NAME).eval()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def hello_world_tokens(tokenizer):
    return tokenizer("Hello, world!", return_tensors="pt")["input_ids"]


@pytest.fixture
def decoder_input_ids(tokenizer):
    return torch.LongTensor([[tokenizer.pad_token_id]])


def test_full_model(our_model, huggingface_model, tokenizer, decoder_input_ids):
    sequences = ["Hello, world!", "this is another sequence of tokens"]

    tokenized = tokenizer(sequences, return_tensors="pt", padding=True)
    decoder_ids = torch.stack([decoder_input_ids[0]] * len(sequences), dim=0)
    input_ids = tokenized["input_ids"]

    attention_mask = tokenized["attention_mask"]

    huggingface_model_out = huggingface_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_ids,
    ).logits
    our_model_out = our_model(
        input_ids,
        decoder_input=decoder_ids,
        one_zero_attention_mask=attention_mask,
    )
    assert_close(huggingface_model_out, our_model_out, rtol=1.3e-6, atol=4e-5)


def test_encoder(our_model, huggingface_model, hello_world_tokens):
    our_embeds = our_model.embed(hello_world_tokens)
    pos_bias = our_model.encoder[0].attn.compute_relative_attention_bias(
        hello_world_tokens.shape[1], hello_world_tokens.shape[1]
    )

    for our_layer in our_model.encoder:
        our_embeds = our_layer(resid_pre=our_embeds, position_bias=pos_bias)

    our_encoder_out = our_model.encoder_final_ln(our_embeds)

    huggingface_encoder_out = huggingface_model.encoder(hello_world_tokens).last_hidden_state

    assert_close(our_encoder_out, huggingface_encoder_out, rtol=1.3e-6, atol=4e-5)


def test_decoder(our_model, huggingface_model, hello_world_tokens, decoder_input_ids):
    encoder_hidden = huggingface_model.encoder(hello_world_tokens)[0]

    embeds = our_model.embed(decoder_input_ids)
    pos_bias = our_model.decoder[0].attn.compute_relative_attention_bias(
        decoder_input_ids.shape[1], decoder_input_ids.shape[1]
    )
    for layer in our_model.decoder:
        embeds = layer(embeds, encoder_hidden_states=encoder_hidden, position_bias=pos_bias)

    our_decoder_out = our_model.decoder_final_ln(embeds)
    hf_decoder_out = huggingface_model.decoder(
        decoder_input_ids, encoder_hidden_states=encoder_hidden
    )[0]

    assert_close(our_decoder_out, hf_decoder_out, rtol=1.3e-6, atol=4e-5)


def test_embed_one_sentence(our_model, huggingface_model, hello_world_tokens):
    huggingface_embed = huggingface_model.encoder.embed_tokens
    our_embed = our_model.embed

    huggingface_embed_out = huggingface_embed(hello_world_tokens)[0]
    our_embed_out = our_embed(hello_world_tokens).squeeze(0)
    assert_close(huggingface_embed_out, our_embed_out)


def test_relative_attention_bias(our_model, huggingface_model, hello_world_tokens):
    # it is used only in self attention of first layer of encoder
    huggingface_embed = huggingface_model.encoder.embed_tokens
    huggingface_attn = huggingface_model.encoder.block[0].layer[0].SelfAttention
    our_attn = our_model.encoder[0].attn

    assert huggingface_attn.has_relative_attention_bias
    assert our_attn.has_relative_attention_bias
    assert (
        our_attn.relative_attention_num_buckets == huggingface_attn.relative_attention_num_buckets
    )
    assert (
        our_attn.relative_attention_max_distance == huggingface_attn.relative_attention_max_distance
    )
    assert_close(our_attn.rel_pos_bias.weight, huggingface_attn.relative_attention_bias.weight)

    input_len = hello_world_tokens.shape[1]
    our_bias = our_attn.compute_relative_attention_bias(input_len, input_len)
    hf_bias = huggingface_attn.compute_bias(input_len, input_len)
    assert_close(our_bias, hf_bias, rtol=1e-5, atol=1e-5)

    embed_out = huggingface_embed(hello_world_tokens)

    huggingface_attn_out = huggingface_attn(embed_out)[0]
    our_attn_out = our_attn(embed_out, embed_out, embed_out, position_bias=our_bias)

    assert_close(our_attn_out, huggingface_attn_out, rtol=7.4e-4, atol=1e-5)


def test_relative_attention_layer(our_model, huggingface_model, hello_world_tokens):
    # it is used only in self attention of first layer of encoder
    hf_block = huggingface_model.encoder.block[0].layer[0]
    our_block = our_model.encoder[0]
    resid = huggingface_model.encoder.embed_tokens(hello_world_tokens)

    input_len = hello_world_tokens.shape[1]
    our_bias = our_block.attn.compute_relative_attention_bias(input_len, input_len)
    resid_norm = our_block.ln1(resid)
    our_out = resid + our_block.attn(resid_norm, resid_norm, resid_norm, position_bias=our_bias)

    hf_out = hf_block(resid)[0]
    assert_close(our_out, hf_out, rtol=1.3e-6, atol=4e-5)


def test_attention(our_model, huggingface_model, hello_world_tokens):
    huggingface_embed = huggingface_model.encoder.embed_tokens
    huggingface_attn = huggingface_model.encoder.block[1].layer[0].SelfAttention

    embed_out = huggingface_embed(hello_world_tokens)
    our_attn = our_model.encoder[1].attn

    our_attn_out = our_attn(embed_out, embed_out, embed_out)
    huggingface_attn_out = huggingface_attn(embed_out)[0]

    assert_close(our_attn_out, huggingface_attn_out, rtol=5e-4, atol=1e-5)


def test_decoder_attention(our_model, huggingface_model, hello_world_tokens):
    huggingface_embed = huggingface_model.decoder.embed_tokens
    huggingface_attn = huggingface_model.decoder.block[1].layer[0].SelfAttention

    embed_out = huggingface_embed(hello_world_tokens)
    our_attn = our_model.decoder[1].attn

    our_attn_out = our_attn(embed_out, embed_out, embed_out)
    huggingface_attn_out = huggingface_attn(embed_out)[0]
    assert_close(our_attn_out, huggingface_attn_out, rtol=3e-4, atol=1e-5)


def test_attention_layer(our_model, huggingface_model, hello_world_tokens):
    huggingface_embed = huggingface_model.encoder.embed_tokens
    huggingface_attn = huggingface_model.encoder.block[1].layer[0]

    embed_out = huggingface_embed(hello_world_tokens)
    our_attn = our_model.encoder[1].attn
    norm_embed = our_model.encoder[1].ln1(embed_out)
    our_attn_out = our_attn(norm_embed, norm_embed, norm_embed) + embed_out

    huggingface_attn_out = huggingface_attn(embed_out)[0]
    assert_close(our_attn_out, huggingface_attn_out, rtol=2e-4, atol=1e-5)


def test_decoder_attention_layer(our_model, huggingface_model, hello_world_tokens):
    huggingface_embed = huggingface_model.decoder.embed_tokens
    huggingface_attn = huggingface_model.decoder.block[1].layer[0]

    embed_out = huggingface_embed(hello_world_tokens)
    our_attn = our_model.decoder[1].attn
    norm_embed = our_model.decoder[1].ln1(embed_out)
    our_attn_out = our_attn(norm_embed, norm_embed, norm_embed) + embed_out

    huggingface_attn_out = huggingface_attn(embed_out)[0]
    assert_close(our_attn_out, huggingface_attn_out, rtol=3e-4, atol=4e-5)


def test_cross_attention(our_model, huggingface_model, hello_world_tokens, decoder_input_ids):
    encoder_hidden = huggingface_model.encoder(hello_world_tokens).last_hidden_state
    decoder_hidden = huggingface_model.decoder.embed_tokens(decoder_input_ids)

    huggingface_cross_attn = huggingface_model.decoder.block[0].layer[1].EncDecAttention
    our_cross_attn = our_model.decoder[0].cross_attn

    our_cross_attn_out = our_cross_attn(decoder_hidden, encoder_hidden, encoder_hidden)
    huggingface_cross_attn_out = huggingface_cross_attn(
        decoder_hidden, key_value_states=encoder_hidden
    )[0]
    assert_close(our_cross_attn_out, huggingface_cross_attn_out, rtol=2e-4, atol=1e-5)


def test_cross_attention_layer(our_model, huggingface_model, hello_world_tokens, decoder_input_ids):
    encoder_hidden = huggingface_model.encoder(hello_world_tokens).last_hidden_state
    decoder_hidden = huggingface_model.decoder.embed_tokens(decoder_input_ids)

    hf_layer = huggingface_model.decoder.block[0].layer[1]
    our_layer = our_model.decoder[0]
    # assert ln weights are the same
    assert_close(hf_layer.layer_norm.weight, our_layer.ln2.w)

    our_cross_attn_out = (
        our_layer.cross_attn(our_layer.ln2(decoder_hidden), encoder_hidden, encoder_hidden)
        + decoder_hidden
    )
    huggingface_cross_attn_out = hf_layer(decoder_hidden, key_value_states=encoder_hidden)[0]
    assert_close(our_cross_attn_out, huggingface_cross_attn_out, rtol=2e-4, atol=1e-5)


def test_encoder_block(our_model, huggingface_model, hello_world_tokens):
    huggingface_embed = huggingface_model.encoder.embed_tokens
    huggingface_block = huggingface_model.encoder.block[1]
    our_block = our_model.encoder[1]

    embed_out = huggingface_embed(hello_world_tokens)

    hf_out = huggingface_block(embed_out)[0]
    our_out = our_block(embed_out)

    assert_close(our_out, hf_out, rtol=2e-4, atol=2e-5)


def test_decoder_block(our_model, huggingface_model, hello_world_tokens, decoder_input_ids):
    huggingface_embed = huggingface_model.decoder.embed_tokens
    huggingface_block = huggingface_model.decoder.block[1]
    our_block = our_model.decoder[1]

    encoder_hidden = huggingface_model.encoder(hello_world_tokens)[0]
    decoder_hidden = huggingface_model.decoder.block[0](huggingface_embed(decoder_input_ids))[0]

    our_out = our_block(decoder_hidden, encoder_hidden_states=encoder_hidden)
    hf_out = huggingface_block(decoder_hidden, encoder_hidden_states=encoder_hidden)[0]

    assert_close(hf_out, our_out, rtol=2e-4, atol=2e-5)


def test_layernorm(our_model, huggingface_model, hello_world_tokens):
    huggingface_embed = huggingface_model.encoder.embed_tokens
    huggingface_layernorm = huggingface_model.encoder.block[0].layer[0].layer_norm
    our_layernorm = our_model.encoder[0].ln1

    embed_out = huggingface_embed(hello_world_tokens)

    our_layernorm_out = our_layernorm(embed_out)
    huggingface_layernorm_out = huggingface_layernorm(embed_out)
    assert_close(our_layernorm_out, huggingface_layernorm_out)


def test_unembed(our_model, huggingface_model, hello_world_tokens):
    huggingface_model_hidden = huggingface_model.decoder(hello_world_tokens).last_hidden_state

    our_model_logits = our_model.unembed(huggingface_model_hidden)
    huggingface_model_logits = huggingface_model.lm_head(huggingface_model_hidden)

    assert_close(our_model_logits, huggingface_model_logits, rtol=1.3e-3, atol=1e-5)


def test_run_with_cache(our_model, hello_world_tokens, decoder_input_ids):
    logits, cache = our_model.run_with_cache(hello_world_tokens, decoder_input=decoder_input_ids)

    # check that an arbitrary subset of the keys exist and have the right shape
    seq_len = 5
    generated_len = 1
    assert "hook_embed" in cache
    assert cache["hook_embed"].shape == (1, seq_len, 512)
    assert "encoder.1.attn.hook_v" in cache
    assert cache["encoder.1.attn.hook_v"].shape == (1, seq_len, 8, 64)
    assert "encoder.3.attn.hook_attn_scores" in cache
    assert cache["encoder.3.attn.hook_attn_scores"].shape == (1, 8, seq_len, seq_len)
    assert "decoder.0.cross_attn.hook_k" in cache
    assert cache["decoder.0.cross_attn.hook_attn_scores"].shape == (
        1,
        8,
        generated_len,
        seq_len,
    )
    assert "decoder.3.hook_resid_post" in cache
    assert cache["decoder.3.hook_resid_post"].shape == (1, generated_len, 512)


def test_from_pretrained_revision():
    """
    Check that the from_pretrained parameter `revision` (= git version) works
    """

    _ = HookedEncoderDecoder.from_pretrained(MODEL_NAME, revision="main")

    try:
        _ = HookedEncoderDecoder.from_pretrained(MODEL_NAME, revision="inexistent_branch_name")
    except:
        pass
    else:
        raise AssertionError("Should have raised an error")


def test_predictions(our_model, huggingface_model, tokenizer, decoder_input_ids):
    input_ids = tokenizer("My name is Wolfgang and I live in Berlin", return_tensors="pt")[
        "input_ids"
    ]

    def get_predictions(logits: Float[torch.Tensor, "batch pos d_vocab"]):
        predicted_tokens = logits[0].argmax(dim=-1)
        return tokenizer.batch_decode(predicted_tokens)

    our_model_logits = our_model(input_ids, decoder_input=decoder_input_ids)
    our_prediction = get_predictions(our_model_logits)

    huggingface_model_logits = huggingface_model(
        input_ids, decoder_input_ids=decoder_input_ids
    ).logits
    huggingface_prediction = get_predictions(huggingface_model_logits)

    assert our_prediction == huggingface_prediction


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
def test_cuda(hello_world_tokens, decoder_input_ids):
    model = HookedEncoderDecoder.from_pretrained(MODEL_NAME)
    model(hello_world_tokens, decoder_input=decoder_input_ids.cuda())
