import pytest
import torch
import torch.nn as nn
from transformers.utils import is_bitsandbytes_available

from transformer_lens.components import Attention
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

if is_bitsandbytes_available():
    from bitsandbytes.nn.modules import Params4bit


def test_attention_hooked_transformer_config():
    cfg = HookedTransformerConfig(
        n_layers=12,
        d_model=512,
        n_ctx=1024,
        d_head=64,
        n_heads=8,
        load_in_4bit=False,
        dtype=torch.float32,
        act_fn="relu",
    )
    attn = Attention(cfg)
    assert attn.cfg == cfg
    assert attn.cfg.n_layers == 12
    assert attn.cfg.d_model == 512
    assert attn.cfg.n_ctx == 1024
    assert attn.cfg.d_head == 64
    assert attn.cfg.n_heads == 8
    assert attn.cfg.load_in_4bit == False
    assert attn.cfg.dtype == torch.float32
    assert attn.cfg.act_fn == "relu"

    assert isinstance(attn.W_K, nn.Parameter)
    assert isinstance(attn.W_V, nn.Parameter)
    assert attn.W_K.shape == (cfg.n_heads, cfg.d_model, cfg.d_head)
    assert attn.W_V.shape == (cfg.n_heads, cfg.d_model, cfg.d_head)

    assert attn.b_K.shape == (cfg.n_heads, cfg.d_head)
    assert attn.b_V.shape == (cfg.n_heads, cfg.d_head)
    assert torch.all(attn.b_K == 0)
    assert torch.all(attn.b_V == 0)


@pytest.mark.skipif(not is_bitsandbytes_available(), reason="bitsandbytes is not available")
def test_attention_load_in_4bit():
    cfg = HookedTransformerConfig(
        n_layers=12,
        d_model=512,
        n_ctx=1024,
        d_head=64,
        n_heads=8,
        load_in_4bit=True,
        dtype=torch.float32,
        act_fn="relu",
    )
    attn = Attention(cfg)
    assert attn.cfg == cfg
    assert attn.cfg.n_layers == 12
    assert attn.cfg.d_model == 512
    assert attn.cfg.n_ctx == 1024
    assert attn.cfg.d_head == 64
    assert attn.cfg.n_heads == 8
    assert attn.cfg.load_in_4bit == False
    assert attn.cfg.dtype == torch.float32
    assert attn.cfg.act_fn == "relu"

    assert isinstance(attn.W_K, Params4bit)
    assert isinstance(attn.W_V, Params4bit)
    nq = int((cfg.d_model * cfg.d_model) / 2)
    assert attn.W_K.data.shape == (nq, 1)
    assert attn.W_V.data.shape == (nq, 1)

    assert attn.b_K.shape == (cfg.n_heads, cfg.d_head)
    assert attn.b_V.shape == (cfg.n_heads, cfg.d_head)
    assert torch.all(attn.b_K == 0)
    assert torch.all(attn.b_V == 0)


def test_attention_config_dict():
    cfg = {
        "n_layers": 12,
        "d_model": 512,
        "n_ctx": 1024,
        "d_head": 64,
        "n_heads": 8,
        "load_in_4bit": False,
        "dtype": torch.float32,
        "act_fn": "relu",
    }
    attn = Attention(cfg)
    assert attn.cfg.n_layers == 12
    assert attn.cfg.d_model == 512
    assert attn.cfg.n_ctx == 1024
    assert attn.cfg.d_head == 64
    assert attn.cfg.n_heads == 8
    assert attn.cfg.load_in_4bit == False
    assert attn.cfg.dtype == torch.float32
    assert attn.cfg.act_fn == "relu"
