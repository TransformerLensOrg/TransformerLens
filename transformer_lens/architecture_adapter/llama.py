"""LLAMA Architecture Adapter.

This module contains the LLAMA architecture adapter class.
"""

from typing import Any

import torch

from transformer_lens.architecture_adapter.conversion_utils.architecture_conversion import (
    ArchitectureConversion,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class LLAMAArchitectureAdapter(ArchitectureConversion):
    """LLAMA architecture adapter class."""

    def __init__(self, cfg: HookedTransformerConfig):
        """Initialize the LLAMA architecture adapter.

        Args:
            cfg (HookedTransformerConfig): The config to use for the adapter.
        """
        super().__init__(cfg)
        self.cfg = cfg

    def _convert_weights(self, hf_model: Any) -> dict[str, torch.Tensor]:
        """Convert the weights from the HuggingFace format to the HookedTransformer format.

        Args:
            hf_model: The HuggingFace model to convert.

        Returns:
            dict[str, torch.Tensor]: The converted weights.
        """
        state_dict = {}

        # Convert the embedding weights
        state_dict["embed.W_E"] = hf_model.model.embed_tokens.weight

        # Convert the layer weights
        for l in range(self.cfg.n_layers):
            # Convert the attention weights
            if self.cfg.n_key_value_heads is not None:
                # Handle grouped query attention
                state_dict[f"blocks.{l}.attn.W_Q"] = hf_model.model.layers[l].self_attn.q_proj.weight.reshape(
                    self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model
                )
                state_dict[f"blocks.{l}.attn.W_K"] = hf_model.model.layers[l].self_attn.k_proj.weight.reshape(
                    self.cfg.n_key_value_heads, self.cfg.d_head, self.cfg.d_model
                )
                state_dict[f"blocks.{l}.attn.W_V"] = hf_model.model.layers[l].self_attn.v_proj.weight.reshape(
                    self.cfg.n_key_value_heads, self.cfg.d_head, self.cfg.d_model
                )
            else:
                state_dict[f"blocks.{l}.attn.W_Q"] = hf_model.model.layers[l].self_attn.q_proj.weight.reshape(
                    self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model
                )
                state_dict[f"blocks.{l}.attn.W_K"] = hf_model.model.layers[l].self_attn.k_proj.weight.reshape(
                    self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model
                )
                state_dict[f"blocks.{l}.attn.W_V"] = hf_model.model.layers[l].self_attn.v_proj.weight.reshape(
                    self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model
                )
            state_dict[f"blocks.{l}.attn.W_O"] = hf_model.model.layers[l].self_attn.o_proj.weight

            # Convert the MLP weights
            state_dict[f"blocks.{l}.mlp.W_gate"] = hf_model.model.layers[l].mlp.gate_proj.weight
            state_dict[f"blocks.{l}.mlp.W_in"] = hf_model.model.layers[l].mlp.up_proj.weight
            state_dict[f"blocks.{l}.mlp.W_out"] = hf_model.model.layers[l].mlp.down_proj.weight

            # Convert the layer norm weights
            state_dict[f"blocks.{l}.ln1.w"] = hf_model.model.layers[l].input_layernorm.weight
            state_dict[f"blocks.{l}.ln2.w"] = hf_model.model.layers[l].post_attention_layernorm.weight

        # Convert the final layer norm weights
        state_dict["ln_final.w"] = hf_model.model.norm.weight

        # Convert the unembed weights
        state_dict["unembed.W_U"] = hf_model.lm_head.weight

        return state_dict
