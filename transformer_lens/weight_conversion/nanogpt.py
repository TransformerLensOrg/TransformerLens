import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_nanogpt_weights(old_state_dict, cfg: HookedTransformerConfig):
    """For https://github.com/karpathy/nanoGPT
    There are two complications with converting nanogpt models:
    The first is that some state dicts have an unwanted prefix on keys that needs to be removed.
    The second is that the models can be saved with or without bias. By default, there
    is no bias. This function can handle both cases."""


    new_state_dict = {}
    # new_state_dict["pos_embed.W_pos"] = old_state_dict["transformer.wpe.weight"]
    # new_state_dict["embed.W_E"] = old_state_dict["transformer.wte.weight"]

    # new_state_dict["ln_final.w"] = old_state_dict["transformer.ln_f.weight"]
    # new_state_dict["ln_final.b"] = torch.zeros_like(old_state_dict["transformer.ln_f.weight"])
    # new_state_dict["unembed.W_U"] = old_state_dict["lm_head.weight"].T

    # bias = False
    # if "transformer.ln_f.bias" in old_state_dict:
    #     bias = True
    #     new_state_dict["ln_final.b"] = old_state_dict["transformer.ln_f.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"transformer.h.{layer}"

        # new_state_dict[f"blocks.{layer}.ln1.w"] = old_state_dict[f"{layer_key}.ln_1.weight"]
        # A bias of zeros is required for folding layer norm
        # new_state_dict[f"blocks.{layer}.ln1.b"] = torch.zeros_like(
        #     old_state_dict[f"{layer_key}.ln_1.weight"]
        # )
        # new_state_dict[f"blocks.{layer}.ln2.w"] = old_state_dict[f"{layer_key}.ln_2.weight"]
        # new_state_dict[f"blocks.{layer}.ln2.b"] = torch.zeros_like(
        #     old_state_dict[f"{layer_key}.ln_2.weight"]
        # )

        # W = old_state_dict[f"{layer_key}.attn.c_attn.weight"]
        # W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=0)
        # W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        # W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        # W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        # new_state_dict[f"blocks.{layer}.attn.W_Q"] = W_Q
        # new_state_dict[f"blocks.{layer}.attn.W_K"] = W_K
        # new_state_dict[f"blocks.{layer}.attn.W_V"] = W_V

        # W_O = old_state_dict[f"{layer_key}.attn.c_proj.weight"]
        # W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        # new_state_dict[f"blocks.{layer}.attn.W_O"] = W_O

        new_state_dict[f"blocks.{layer}.mlp.W_in"] = old_state_dict[
            f"{layer_key}.mlp.c_fc.weight"
        ].T
        new_state_dict[f"blocks.{layer}.mlp.W_out"] = old_state_dict[
            f"{layer_key}.mlp.c_proj.weight"
        ].T

        if bias:
            # new_state_dict[f"blocks.{layer}.ln1.b"] = old_state_dict[f"{layer_key}.ln_1.bias"]
            # new_state_dict[f"blocks.{layer}.ln2.b"] = old_state_dict[f"{layer_key}.ln_2.bias"]
            new_state_dict[f"blocks.{layer}.mlp.b_in"] = old_state_dict[
                f"{layer_key}.mlp.c_fc.bias"
            ]
            new_state_dict[f"blocks.{layer}.mlp.b_out"] = old_state_dict[
                f"{layer_key}.mlp.c_proj.bias"
            ]

            B = old_state_dict[f"{layer_key}.attn.c_attn.bias"]
            B_Q, B_K, B_V = torch.tensor_split(B, 3, dim=0)
            B_Q = einops.rearrange(B_Q, "(i h)->i h", i=cfg.n_heads)
            B_K = einops.rearrange(B_K, "(i h)->i h", i=cfg.n_heads)
            B_V = einops.rearrange(B_V, "(i h)->i h", i=cfg.n_heads)
            new_state_dict[f"blocks.{layer}.attn.b_Q"] = B_Q
            new_state_dict[f"blocks.{layer}.attn.b_K"] = B_K
            new_state_dict[f"blocks.{layer}.attn.b_V"] = B_V
            new_state_dict[f"blocks.{layer}.attn.b_O"] = old_state_dict[
                f"{layer_key}.attn.c_proj.bias"
            ]

    return new_state_dict

import torch
from torch import nn

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
    TernaryWeightConversion,
    ZerosLikeConversion,
)


class NanoGPTWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        number_key_value_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else 0
        super().__init__(
            {
                "pos_embed.W_pos": "transformer.wpe.weight",
                "embed.W_E": "transformer.wte.weight",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": (
                    "transformer.ln_f.bias",
                    TernaryWeightConversion(
                        fallback_conversion=("transformer.ln_f.weight", ZerosLikeConversion())
                    ),
                ),
                "unembed.W_U": "lm_head.weight.T",
                "blocks": ("transformer.h", WeightConversionSet({
                    "ln1.w": "ln_1.weight",
                    "ln1.b": (
                        "ln_1.bias",
                        TernaryWeightConversion(
                            fallback_conversion=("ln_1.weight", ZerosLikeConversion())
                        ),
                    ),
                    "ln2.w": "ln_2.weight",
                    "ln2.b": (
                        "ln_2.bias",
                        TernaryWeightConversion(
                            fallback_conversion=("ln_2.weight", ZerosLikeConversion())
                        ),
                    ),
                    "attn.W_Q": (
                        "attn.c_attn.weight",
                        RearrangeWeightConversion(
                            "(i h) m->i m h",
                            input_filter=lambda weight: torch.tensor_split(weight, 3, dim=0)[0],
                            i=cfg.n_heads,
                        )
                    ),
                    "attn.W_K": (
                        "attn.c_attn.weight",
                        RearrangeWeightConversion(
                            "(i h) m->i m h",
                            input_filter=lambda weight: torch.tensor_split(weight, 3, dim=0)[1],
                            i=cfg.n_heads,
                        )
                    ),
                    "attn.W_V": (
                        "attn.c_attn.weight",
                        RearrangeWeightConversion(
                            "(i h) m->i m h",
                            input_filter=lambda weight: torch.tensor_split(weight, 3, dim=0)[2],
                            i=cfg.n_heads,
                        )
                    ),
                    "attn.W_O": (
                        "attn.c_proj.weight",
                        RearrangeWeightConversion("m (i h)->i h m", i=cfg.n_heads)
                    )
                })),
            }
        )
    

    def convert(self, remote_module: nn.Module):
        # Nanogpt models saved after torch.compile() have this unwanted prefix
        # This is a simple way to remove it
        unwanted_prefix = "_orig_mod."
        for k, v in list(remote_module.items()):
            if k.startswith(unwanted_prefix):
                remote_module[k[len(unwanted_prefix) :]] = remote_module.pop(k)

        return super().convert(remote_module)
