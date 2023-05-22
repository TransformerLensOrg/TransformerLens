from transformers import AutoTokenizer

from transformer_lens import HookedTransformer, HookedTransformerConfig


def test_d_vocab_from_tokenizer():
    cfg = HookedTransformerConfig(
        n_layers=1, d_mlp=10, d_model=10, d_head=5, n_heads=2, n_ctx=20, act_fn="relu"
    )
    model = HookedTransformer(cfg=cfg, tokenizer=AutoTokenizer.from_pretrained("gpt2"))
    assert model.cfg.d_vocab == 50257
    assert model.cfg.d_vocab_out == 50257


def test_d_vocab_from_tokenizer_name():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_mlp=10,
        d_model=10,
        d_head=5,
        n_heads=2,
        n_ctx=20,
        act_fn="relu",
        tokenizer_name="gpt2",
    )
    model = HookedTransformer(cfg=cfg)
    assert model.cfg.d_vocab == 50257
    assert model.cfg.d_vocab_out == 50257


def test_d_vocab_out_set():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_mlp=10,
        d_model=10,
        d_head=5,
        n_heads=2,
        n_ctx=20,
        act_fn="relu",
        d_vocab=100,
        d_vocab_out=90,
    )
    model = HookedTransformer(cfg=cfg)
    assert model.cfg.d_vocab == 100
    assert model.cfg.d_vocab_out == 90


def test_d_vocab_out_set_d_vocab_infer():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_mlp=10,
        d_model=10,
        d_head=5,
        n_heads=2,
        n_ctx=20,
        act_fn="relu",
        d_vocab_out=90,
        tokenizer_name="gpt2",
    )
    model = HookedTransformer(cfg=cfg)
    assert model.cfg.d_vocab == 50257
    assert model.cfg.d_vocab_out == 90
