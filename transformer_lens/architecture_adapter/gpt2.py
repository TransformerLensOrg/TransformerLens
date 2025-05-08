"""GPT2 Architecture Adapter.

This module contains the GPT2 architecture adapter class.
"""

from typing import Any

import torch

from transformer_lens.architecture_adapter.conversion_utils.architecture_conversion import (
    ArchitectureConversion,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class GPT2ArchitectureAdapter(ArchitectureConversion):
    """GPT2 architecture adapter class."""

    def __init__(self, cfg: HookedTransformerConfig):
        """Initialize the GPT2 architecture adapter.

        Args:
            cfg (HookedTransformerConfig): The config to use for the adapter.
        """
        super().__init__(cfg)
        self.cfg = cfg

    def convert_weights(self, hf_model: Any) -> dict[str, torch.Tensor]:
        """Convert the weights from the HuggingFace format to the HookedTransformer format.

        Args:
            hf_model: The HuggingFace model to convert.

        Returns:
            dict[str, torch.Tensor]: The converted weights.
        """
        state_dict = {}

        # Convert the embedding weights
        state_dict["embed.W_E"] = hf_model.transformer.wte.weight
        state_dict["pos_embed.W_pos"] = hf_model.transformer.wpe.weight

        # Convert the layer weights
        for l in range(self.cfg.n_layers):
            # Convert the attention weights
            state_dict[f"blocks.{l}.attn.W_K"] = hf_model.transformer.h[l].attn.c_attn.weight[
                :, : self.cfg.d_model
            ].reshape(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model)
            state_dict[f"blocks.{l}.attn.W_Q"] = hf_model.transformer.h[l].attn.c_attn.weight[
                :, self.cfg.d_model : 2 * self.cfg.d_model
            ].reshape(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model)
            state_dict[f"blocks.{l}.attn.W_V"] = hf_model.transformer.h[l].attn.c_attn.weight[
                :, 2 * self.cfg.d_model :
            ].reshape(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model)
            state_dict[f"blocks.{l}.attn.b_K"] = hf_model.transformer.h[l].attn.c_attn.bias[
                : self.cfg.d_model
            ].reshape(self.cfg.n_heads, self.cfg.d_head)
            state_dict[f"blocks.{l}.attn.b_Q"] = hf_model.transformer.h[l].attn.c_attn.bias[
                self.cfg.d_model : 2 * self.cfg.d_model
            ].reshape(self.cfg.n_heads, self.cfg.d_head)
            state_dict[f"blocks.{l}.attn.b_V"] = hf_model.transformer.h[l].attn.c_attn.bias[
                2 * self.cfg.d_model :
            ].reshape(self.cfg.n_heads, self.cfg.d_head)
            state_dict[f"blocks.{l}.attn.W_O"] = hf_model.transformer.h[l].attn.c_proj.weight.reshape(
                self.cfg.d_model, self.cfg.n_heads, self.cfg.d_head
            )
            state_dict[f"blocks.{l}.attn.b_O"] = hf_model.transformer.h[l].attn.c_proj.bias

            # Convert the MLP weights
            state_dict[f"blocks.{l}.mlp.W_in"] = hf_model.transformer.h[l].mlp.c_fc.weight
            state_dict[f"blocks.{l}.mlp.b_in"] = hf_model.transformer.h[l].mlp.c_fc.bias
            state_dict[f"blocks.{l}.mlp.W_out"] = hf_model.transformer.h[l].mlp.c_proj.weight
            state_dict[f"blocks.{l}.mlp.b_out"] = hf_model.transformer.h[l].mlp.c_proj.bias

            # Convert the layer norm weights
            state_dict[f"blocks.{l}.ln1.w"] = hf_model.transformer.h[l].ln_1.weight
            state_dict[f"blocks.{l}.ln1.b"] = hf_model.transformer.h[l].ln_1.bias
            state_dict[f"blocks.{l}.ln2.w"] = hf_model.transformer.h[l].ln_2.weight
            state_dict[f"blocks.{l}.ln2.b"] = hf_model.transformer.h[l].ln_2.bias

        # Convert the final layer norm weights
        state_dict["ln_final.w"] = hf_model.transformer.ln_f.weight
        state_dict["ln_final.b"] = hf_model.transformer.ln_f.bias

        # Convert the unembed weights
        state_dict["unembed.W_U"] = hf_model.lm_head.weight
        if hasattr(hf_model.lm_head, "bias") and hf_model.lm_head.bias is not None:
            state_dict["unembed.b_U"] = hf_model.lm_head.bias

        return state_dict
