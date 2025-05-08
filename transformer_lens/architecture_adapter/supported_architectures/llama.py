"""Llama architecture adapter."""

from typing import Any

import torch

from transformer_lens.architecture_adapter.conversion_utils.architecture_conversion import (
    ArchitectureConversion,
)
from transformer_lens.architecture_adapter.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class LlamaArchitectureAdapter(ArchitectureConversion):
    """Architecture adapter for Llama models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the Llama architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        self.field_set = WeightConversionSet(
            {
                "embed.W_E": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln1.b": "model.layers.{i}.input_layernorm.bias",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.ln2.b": "model.layers.{i}.post_attention_layernorm.bias",
                "blocks.{i}.attn.W_Q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "model.layers.{i}.self_attn.q_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "model.layers.{i}.self_attn.k_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "model.layers.{i}.self_attn.v_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeWeightConversion("d_model (h d_head) -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "model.layers.{i}.self_attn.o_proj.bias",
                "blocks.{i}.mlp.W_in": "model.layers.{i}.mlp.gate_proj.weight",
                "blocks.{i}.mlp.b_in": "model.layers.{i}.mlp.gate_proj.bias",
                "blocks.{i}.mlp.W_out": "model.layers.{i}.mlp.down_proj.weight",
                "blocks.{i}.mlp.b_out": "model.layers.{i}.mlp.down_proj.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
                "ln_final.w": "model.norm.weight",
                "ln_final.b": "model.norm.bias",
            }
        )

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
