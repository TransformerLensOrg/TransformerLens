import pytest
from transformers import AutoTokenizer, BertTokenizerFast

from transformer_lens import HookedEncoder, HookedTransformerConfig


@pytest.fixture
def cfg():
    return HookedTransformerConfig(d_head=4, d_model=12, n_ctx=5, n_layers=3, act_fn="gelu")


def test_pass_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = HookedEncoder(cfg, tokenizer=tokenizer)
    assert model.tokenizer == tokenizer


def test_load_tokenizer_from_config(cfg):
    cfg.tokenizer_name = "bert-base-cased"
    model = HookedEncoder(cfg)
    assert isinstance(model.tokenizer, BertTokenizerFast)


def test_load_without_tokenizer(cfg):
    cfg.d_vocab = 22
    model = HookedEncoder(cfg)
    assert model.tokenizer is None


def test_cannot_load_without_tokenizer_or_d_vocab(cfg):
    with pytest.raises(AssertionError) as e:
        HookedEncoder(cfg)
    assert "Must provide a tokenizer if d_vocab is not provided" in str(e.value)
