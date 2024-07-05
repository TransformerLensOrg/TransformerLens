from unittest import mock

import torch

from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions.neo import convert_neo_weights


def get_default_config():
    return HookedTransformerConfig(
        d_model=128, d_head=8, n_heads=16, n_ctx=128, n_layers=1, d_vocab=50257, attn_only=True
    )


def test_convert_neo_weights_exposed():
    cfg = get_default_config()

    class MockNeo:
        def __init__(self):
            self.transformer = HookedTransformer(cfg)
            self.transformer.wte = torch.nn.Embedding(cfg.d_vocab, cfg.d_model)
            self.transformer.wpe = torch.nn.Embedding(cfg.n_ctx, cfg.d_model)
            self.transformer.final_norm = torch.nn.LayerNorm(cfg.d_model)
            self.transformer.h = [mock.Mock() for _ in range(cfg.n_layers)]
            self.lm_head = torch.nn.Linear(cfg.d_model, cfg.d_vocab)

            for layer in self.transformer.h:
                layer.ln_1 = torch.nn.LayerNorm(cfg.d_model)
                layer.ln_2 = torch.nn.LayerNorm(cfg.d_model)
                layer.attn = mock.Mock()
                layer.attn.attention = mock.Mock()
                layer.attn.attention.q_proj = torch.nn.Linear(cfg.d_model, cfg.d_model)
                layer.attn.attention.k_proj = torch.nn.Linear(cfg.d_model, cfg.d_model)
                layer.attn.attention.v_proj = torch.nn.Linear(cfg.d_model, cfg.d_model)
                layer.attn.attention.out_proj = torch.nn.Linear(cfg.d_model, cfg.d_model)
                layer.mlp = mock.Mock()
                layer.mlp.c_fc = torch.nn.Linear(cfg.d_model, cfg.d_model)
                layer.mlp.c_proj = torch.nn.Linear(cfg.d_model, cfg.d_model)

            self.transformer.ln_f = torch.nn.LayerNorm(cfg.d_model)

    neo = MockNeo()

    try:
        convert_neo_weights(neo, cfg)
        function_works = True
    except Exception as e:
        function_works = False
        print(f"The convert_neo_weights function raised an error: {e}")

    assert function_works
