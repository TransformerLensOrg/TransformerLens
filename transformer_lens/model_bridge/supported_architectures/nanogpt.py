import einops
import torch
from torch import nn

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
    UnembeddingBridge,
)


def convert_nanogpt_weights(old_state_dict, cfg: HookedTransformerConfig):
    """For https://github.com/karpathy/nanoGPT
    There are two complications with converting nanogpt models:
    The first is that some state dicts have an unwanted prefix on keys that needs to be removed.
    The second is that the models can be saved with or without bias. By default, there
    is no bias. This function can handle both cases."""

    new_state_dict = {}
    bias = False
    if "transformer.ln_f.bias" in old_state_dict:
        bias = True

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

from transformer_lens.architecture_adapter.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class NanoGPTArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for NanoGPT models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the NanoGPT architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion("(3 h d_head) d_model -> 3 h d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion("(3 h d_head) d_model -> 3 h d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion("(3 h d_head) d_model -> 3 h d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(3 h d_head) -> 3 h d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(3 h d_head) -> 3 h d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(3 h d_head) -> 3 h d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeWeightConversion("d_model (h d_head) -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": ("transformer.wte", EmbeddingBridge),  # Word token embeddings
            "blocks": (
                "transformer.h",  # Base path for blocks
                {
                    "ln1": ("ln_1", LayerNormBridge),  # Pre-attention layer norm
                    "ln2": ("ln_2", LayerNormBridge),  # Pre-MLP layer norm
                    "attn": ("attn", AttentionBridge),  # Full attention module
                    "attn.c_attn": ("attn.c_attn", AttentionBridge),  # Combined QKV projection
                    "attn.c_proj": ("attn.c_proj", AttentionBridge),  # Output projection
                    "mlp": ("mlp", MLPBridge),  # Full MLP module
                    "mlp.c_fc": ("mlp.c_fc", MLPBridge),  # First linear layer
                    "mlp.c_proj": ("mlp.c_proj", MLPBridge),  # Second linear layer
                },
            ),
            "ln_final": ("transformer.ln_f", LayerNormBridge),  # Final layer norm
            "unembed": ("lm_head", UnembeddingBridge),  # Language model head
        }

    def convert_weights(self, remote_module: nn.Module):
        # Nanogpt models saved after torch.compile() have this unwanted prefix
        # This is a simple way to remove it
        unwanted_prefix = "_orig_mod."
        for k, v in list(remote_module.items()):
            if k.startswith(unwanted_prefix):
                remote_module[k[len(unwanted_prefix) :]] = remote_module.pop(k)

        return super().convert_weights(remote_module)
