import torch

from transformer_lens.components import MoE


def test_forward():
    cfg = {
        "d_model": 32,
        "d_mlp": 14336,
        "d_head": 4,
        "num_experts": 32,
        "n_layers": 16,
        "n_ctx": 2048,
        "experts_per_token": 4,
        "gated_mlp": True,
        "act_fn": "silu",
    }
    moe = MoE(cfg)

    x = torch.rand((1, 4, 32))
    moe(x)
