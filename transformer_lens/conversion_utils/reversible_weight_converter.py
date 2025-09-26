#!/usr/bin/env python3
"""
Reversible Weight Converter for HuggingFace ⇄ TransformerLens.

This module provides bidirectional weight conversion between HuggingFace and TransformerLens
formats with round-trip guarantees. The conversions are exact inverses up to numerical precision.

Key Features:
- HF → TLens and TLens → HF conversions that are mathematical inverses
- Handles all component types: embeddings, attention (Q/K/V/O with GQA), MLPs, norms, etc.
- Round-trip guarantee: HF → TLens → HF must match original outputs
- Comprehensive validation and debugging tools
- Integration with existing compare scripts
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import einops
import torch
from torch import nn

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig


class ConversionError(Exception):
    """Exception raised during weight conversion."""
    pass


class RoundTripError(Exception):
    """Exception raised when round-trip conversion fails validation."""
    pass


class BaseComponentConverter(ABC):
    """Base class for component-specific weight converters."""

    @abstractmethod
    def hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert HuggingFace weights to TransformerLens format."""
        pass

    @abstractmethod
    def tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert TransformerLens weights to HuggingFace format."""
        pass

    @abstractmethod
    def get_hf_keys(self, config: HookedTransformerConfig, **kwargs) -> List[str]:
        """Get expected HuggingFace parameter keys for this component."""
        pass

    @abstractmethod
    def get_tlens_keys(self, config: HookedTransformerConfig, **kwargs) -> List[str]:
        """Get expected TransformerLens parameter keys for this component."""
        pass

    def validate_round_trip(self, original_weights: Dict[str, torch.Tensor],
                          config: HookedTransformerConfig, tolerance: float = 1e-6, **kwargs) -> bool:
        """Validate that HF → TLens → HF conversion preserves weights."""
        try:
            # Forward conversion
            tlens_weights = self.hf_to_tlens(original_weights, config, **kwargs)
            # Backward conversion
            recovered_weights = self.tlens_to_hf(tlens_weights, config, **kwargs)

            # Check all keys exist
            for key in original_weights:
                if key not in recovered_weights:
                    raise RoundTripError(f"Missing key after round-trip: {key}")

            # Check values match within tolerance
            for key, original_tensor in original_weights.items():
                recovered_tensor = recovered_weights[key]
                if not torch.allclose(original_tensor, recovered_tensor, atol=tolerance, rtol=tolerance):
                    max_diff = torch.max(torch.abs(original_tensor - recovered_tensor)).item()
                    raise RoundTripError(
                        f"Round-trip mismatch for {key}: max difference {max_diff} > {tolerance}"
                    )

            return True

        except Exception as e:
            raise RoundTripError(f"Round-trip validation failed: {str(e)}")


class EmbeddingConverter(BaseComponentConverter):
    """Converter for embedding layers."""

    def hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert HF embedding weights to TLens format."""
        tlens_weights = {}

        # Word embeddings
        if "transformer.wte.weight" in hf_weights:
            tlens_weights["embed.W_E"] = hf_weights["transformer.wte.weight"]
        elif "model.embed_tokens.weight" in hf_weights:  # For LLaMA-style models
            tlens_weights["embed.W_E"] = hf_weights["model.embed_tokens.weight"]

        # Position embeddings (if exists)
        if "transformer.wpe.weight" in hf_weights:
            tlens_weights["pos_embed.W_pos"] = hf_weights["transformer.wpe.weight"]

        return tlens_weights

    def tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert TLens embedding weights to HF format."""
        hf_weights = {}

        # Determine HF format based on config or kwargs
        model_type = kwargs.get("model_type", "gpt2")

        # Word embeddings
        if "embed.W_E" in tlens_weights:
            if model_type in ["llama", "mistral", "gemma"]:
                hf_weights["model.embed_tokens.weight"] = tlens_weights["embed.W_E"]
            else:  # GPT-2 style
                hf_weights["transformer.wte.weight"] = tlens_weights["embed.W_E"]

        # Position embeddings (if exists)
        if "pos_embed.W_pos" in tlens_weights:
            hf_weights["transformer.wpe.weight"] = tlens_weights["pos_embed.W_pos"]

        return hf_weights

    def get_hf_keys(self, config: HookedTransformerConfig, **kwargs) -> List[str]:
        """Get expected HF keys for embeddings."""
        model_type = kwargs.get("model_type", "gpt2")
        keys = []

        if model_type in ["llama", "mistral", "gemma"]:
            keys.append("model.embed_tokens.weight")
        else:
            keys.extend(["transformer.wte.weight", "transformer.wpe.weight"])

        return keys

    def get_tlens_keys(self, config: HookedTransformerConfig, **kwargs) -> List[str]:
        """Get expected TLens keys for embeddings."""
        keys = ["embed.W_E"]
        if hasattr(config, 'n_ctx') and config.n_ctx > 0:
            keys.append("pos_embed.W_pos")
        return keys


class AttentionConverter(BaseComponentConverter):
    """Converter for attention layers with support for various architectures."""

    def hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert HF attention weights to TLens format."""
        tlens_weights = {}
        layer_idx = kwargs.get("layer_idx", 0)
        model_type = kwargs.get("model_type", "gpt2")

        if model_type == "gpt2":
            return self._convert_gpt2_attention_hf_to_tlens(hf_weights, config, layer_idx)
        elif model_type in ["llama", "mistral"]:
            return self._convert_llama_attention_hf_to_tlens(hf_weights, config, layer_idx)
        elif model_type == "gemma":
            return self._convert_gemma_attention_hf_to_tlens(hf_weights, config, layer_idx)
        else:
            raise ConversionError(f"Unsupported model type for attention: {model_type}")

    def tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert TLens attention weights to HF format."""
        layer_idx = kwargs.get("layer_idx", 0)
        model_type = kwargs.get("model_type", "gpt2")

        if model_type == "gpt2":
            return self._convert_gpt2_attention_tlens_to_hf(tlens_weights, config, layer_idx)
        elif model_type in ["llama", "mistral"]:
            return self._convert_llama_attention_tlens_to_hf(tlens_weights, config, layer_idx)
        elif model_type == "gemma":
            return self._convert_gemma_attention_tlens_to_hf(tlens_weights, config, layer_idx)
        else:
            raise ConversionError(f"Unsupported model type for attention: {model_type}")

    def _convert_gpt2_attention_hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                                          config: HookedTransformerConfig, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Convert GPT-2 style attention from HF to TLens."""
        tlens_weights = {}
        prefix = f"transformer.h.{layer_idx}.attn"

        # Combined QKV projection
        if f"{prefix}.c_attn.weight" in hf_weights:
            W = hf_weights[f"{prefix}.c_attn.weight"]
            W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)

            # Reshape to [n_heads, d_model, d_head] format
            W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=config.n_heads)
            W_K = einops.rearrange(W_K, "m (i h)->i m h", i=config.n_heads)
            W_V = einops.rearrange(W_V, "m (i h)->i m h", i=config.n_heads)

            tlens_weights[f"blocks.{layer_idx}.attn.W_Q"] = W_Q
            tlens_weights[f"blocks.{layer_idx}.attn.W_K"] = W_K
            tlens_weights[f"blocks.{layer_idx}.attn.W_V"] = W_V

        # Combined QKV bias
        if f"{prefix}.c_attn.bias" in hf_weights:
            qkv_bias = hf_weights[f"{prefix}.c_attn.bias"]
            qkv_bias = einops.rearrange(
                qkv_bias,
                "(qkv index head)->qkv index head",
                qkv=3,
                index=config.n_heads,
                head=config.d_head,
            )
            tlens_weights[f"blocks.{layer_idx}.attn.b_Q"] = qkv_bias[0]
            tlens_weights[f"blocks.{layer_idx}.attn.b_K"] = qkv_bias[1]
            tlens_weights[f"blocks.{layer_idx}.attn.b_V"] = qkv_bias[2]

        # Output projection
        if f"{prefix}.c_proj.weight" in hf_weights:
            W_O = hf_weights[f"{prefix}.c_proj.weight"]
            W_O = einops.rearrange(W_O, "(i h) m->i h m", i=config.n_heads)
            tlens_weights[f"blocks.{layer_idx}.attn.W_O"] = W_O

        if f"{prefix}.c_proj.bias" in hf_weights:
            tlens_weights[f"blocks.{layer_idx}.attn.b_O"] = hf_weights[f"{prefix}.c_proj.bias"]

        return tlens_weights

    def _convert_gpt2_attention_tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                                          config: HookedTransformerConfig, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Convert GPT-2 style attention from TLens to HF."""
        hf_weights = {}
        prefix = f"transformer.h.{layer_idx}.attn"

        # Combine QKV weights
        if all(f"blocks.{layer_idx}.attn.W_{x}" in tlens_weights for x in ["Q", "K", "V"]):
            W_Q = tlens_weights[f"blocks.{layer_idx}.attn.W_Q"]
            W_K = tlens_weights[f"blocks.{layer_idx}.attn.W_K"]
            W_V = tlens_weights[f"blocks.{layer_idx}.attn.W_V"]

            # Reshape from [n_heads, d_model, d_head] to [d_model, n_heads*d_head] format
            W_Q = einops.rearrange(W_Q, "i m h->m (i h)")
            W_K = einops.rearrange(W_K, "i m h->m (i h)")
            W_V = einops.rearrange(W_V, "i m h->m (i h)")

            # Combine into single weight matrix [d_model, 3*n_heads*d_head]
            combined_weight = torch.cat([W_Q, W_K, W_V], dim=1)
            hf_weights[f"{prefix}.c_attn.weight"] = combined_weight

        # Combine QKV biases
        if all(f"blocks.{layer_idx}.attn.b_{x}" in tlens_weights for x in ["Q", "K", "V"]):
            b_Q = tlens_weights[f"blocks.{layer_idx}.attn.b_Q"]
            b_K = tlens_weights[f"blocks.{layer_idx}.attn.b_K"]
            b_V = tlens_weights[f"blocks.{layer_idx}.attn.b_V"]

            # Stack and reshape
            qkv_bias = torch.stack([b_Q, b_K, b_V], dim=0)
            qkv_bias = einops.rearrange(qkv_bias, "qkv index head->(qkv index head)")
            hf_weights[f"{prefix}.c_attn.bias"] = qkv_bias

        # Output projection
        if f"blocks.{layer_idx}.attn.W_O" in tlens_weights:
            W_O = tlens_weights[f"blocks.{layer_idx}.attn.W_O"]
            W_O = einops.rearrange(W_O, "i h m->(i h) m")
            hf_weights[f"{prefix}.c_proj.weight"] = W_O

        if f"blocks.{layer_idx}.attn.b_O" in tlens_weights:
            hf_weights[f"{prefix}.c_proj.bias"] = tlens_weights[f"blocks.{layer_idx}.attn.b_O"]

        return hf_weights

    def _convert_llama_attention_hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                                           config: HookedTransformerConfig, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Convert LLaMA style attention from HF to TLens."""
        tlens_weights = {}
        prefix = f"model.layers.{layer_idx}.self_attn"

        # Separate Q, K, V projections
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            hf_key = f"{prefix}.{proj_name}.weight"
            if hf_key in hf_weights:
                weight = hf_weights[hf_key]
                # Reshape for multi-head format
                tlens_key = proj_name[0].upper()  # q_proj -> Q, etc.
                if proj_name == "k_proj" and hasattr(config, 'n_key_value_heads'):
                    # Handle GQA (Grouped Query Attention)
                    n_heads = config.n_key_value_heads
                else:
                    n_heads = config.n_heads

                weight = einops.rearrange(weight, "(i h) m->i m h", i=n_heads)
                tlens_weights[f"blocks.{layer_idx}.attn.W_{tlens_key}"] = weight

        # Output projection
        if f"{prefix}.o_proj.weight" in hf_weights:
            W_O = hf_weights[f"{prefix}.o_proj.weight"]
            W_O = einops.rearrange(W_O, "m (i h)->i h m", i=config.n_heads)
            tlens_weights[f"blocks.{layer_idx}.attn.W_O"] = W_O

        return tlens_weights

    def _convert_llama_attention_tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                                           config: HookedTransformerConfig, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Convert LLaMA style attention from TLens to HF."""
        hf_weights = {}
        prefix = f"model.layers.{layer_idx}.self_attn"

        # Separate Q, K, V projections
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            tlens_key = proj_name[0].upper()  # q_proj -> Q, etc.
            tlens_param = f"blocks.{layer_idx}.attn.W_{tlens_key}"

            if tlens_param in tlens_weights:
                weight = tlens_weights[tlens_param]
                # Reshape from multi-head format
                weight = einops.rearrange(weight, "i m h->(i h) m")
                hf_weights[f"{prefix}.{proj_name}.weight"] = weight

        # Output projection
        if f"blocks.{layer_idx}.attn.W_O" in tlens_weights:
            W_O = tlens_weights[f"blocks.{layer_idx}.attn.W_O"]
            W_O = einops.rearrange(W_O, "i h m->m (i h)")
            hf_weights[f"{prefix}.o_proj.weight"] = W_O

        return hf_weights

    def _convert_gemma_attention_hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                                           config: HookedTransformerConfig, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Convert Gemma style attention from HF to TLens."""
        # Gemma has similar structure to LLaMA but with some differences
        return self._convert_llama_attention_hf_to_tlens(hf_weights, config, layer_idx)

    def _convert_gemma_attention_tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                                           config: HookedTransformerConfig, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Convert Gemma style attention from TLens to HF."""
        return self._convert_llama_attention_tlens_to_hf(tlens_weights, config, layer_idx)

    def get_hf_keys(self, config: HookedTransformerConfig, **kwargs) -> List[str]:
        """Get expected HF keys for attention."""
        model_type = kwargs.get("model_type", "gpt2")
        layer_idx = kwargs.get("layer_idx", 0)

        if model_type == "gpt2":
            prefix = f"transformer.h.{layer_idx}.attn"
            return [f"{prefix}.c_attn.weight", f"{prefix}.c_attn.bias",
                   f"{prefix}.c_proj.weight", f"{prefix}.c_proj.bias"]
        elif model_type in ["llama", "mistral", "gemma"]:
            prefix = f"model.layers.{layer_idx}.self_attn"
            return [f"{prefix}.q_proj.weight", f"{prefix}.k_proj.weight",
                   f"{prefix}.v_proj.weight", f"{prefix}.o_proj.weight"]
        else:
            return []

    def get_tlens_keys(self, config: HookedTransformerConfig, **kwargs) -> List[str]:
        """Get expected TLens keys for attention."""
        layer_idx = kwargs.get("layer_idx", 0)
        return [f"blocks.{layer_idx}.attn.W_Q", f"blocks.{layer_idx}.attn.W_K",
               f"blocks.{layer_idx}.attn.W_V", f"blocks.{layer_idx}.attn.W_O",
               f"blocks.{layer_idx}.attn.b_Q", f"blocks.{layer_idx}.attn.b_K",
               f"blocks.{layer_idx}.attn.b_V", f"blocks.{layer_idx}.attn.b_O"]


class MLPConverter(BaseComponentConverter):
    """Converter for MLP layers."""

    def hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert HF MLP weights to TLens format."""
        tlens_weights = {}
        layer_idx = kwargs.get("layer_idx", 0)
        model_type = kwargs.get("model_type", "gpt2")

        if model_type == "gpt2":
            return self._convert_gpt2_mlp_hf_to_tlens(hf_weights, config, layer_idx)
        elif model_type in ["llama", "mistral"]:
            return self._convert_llama_mlp_hf_to_tlens(hf_weights, config, layer_idx)
        elif model_type == "gemma":
            return self._convert_gemma_mlp_hf_to_tlens(hf_weights, config, layer_idx)
        else:
            raise ConversionError(f"Unsupported model type for MLP: {model_type}")

    def tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert TLens MLP weights to HF format."""
        layer_idx = kwargs.get("layer_idx", 0)
        model_type = kwargs.get("model_type", "gpt2")

        if model_type == "gpt2":
            return self._convert_gpt2_mlp_tlens_to_hf(tlens_weights, config, layer_idx)
        elif model_type in ["llama", "mistral"]:
            return self._convert_llama_mlp_tlens_to_hf(tlens_weights, config, layer_idx)
        elif model_type == "gemma":
            return self._convert_gemma_mlp_tlens_to_hf(tlens_weights, config, layer_idx)
        else:
            raise ConversionError(f"Unsupported model type for MLP: {model_type}")

    def _convert_gpt2_mlp_hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                                    config: HookedTransformerConfig, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Convert GPT-2 style MLP from HF to TLens."""
        tlens_weights = {}
        prefix = f"transformer.h.{layer_idx}.mlp"

        # Input projection (c_fc)
        if f"{prefix}.c_fc.weight" in hf_weights:
            tlens_weights[f"blocks.{layer_idx}.mlp.W_in"] = hf_weights[f"{prefix}.c_fc.weight"]
        if f"{prefix}.c_fc.bias" in hf_weights:
            tlens_weights[f"blocks.{layer_idx}.mlp.b_in"] = hf_weights[f"{prefix}.c_fc.bias"]

        # Output projection (c_proj)
        if f"{prefix}.c_proj.weight" in hf_weights:
            tlens_weights[f"blocks.{layer_idx}.mlp.W_out"] = hf_weights[f"{prefix}.c_proj.weight"]
        if f"{prefix}.c_proj.bias" in hf_weights:
            tlens_weights[f"blocks.{layer_idx}.mlp.b_out"] = hf_weights[f"{prefix}.c_proj.bias"]

        return tlens_weights

    def _convert_gpt2_mlp_tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                                    config: HookedTransformerConfig, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Convert GPT-2 style MLP from TLens to HF."""
        hf_weights = {}
        prefix = f"transformer.h.{layer_idx}.mlp"

        # Input projection
        if f"blocks.{layer_idx}.mlp.W_in" in tlens_weights:
            hf_weights[f"{prefix}.c_fc.weight"] = tlens_weights[f"blocks.{layer_idx}.mlp.W_in"]
        if f"blocks.{layer_idx}.mlp.b_in" in tlens_weights:
            hf_weights[f"{prefix}.c_fc.bias"] = tlens_weights[f"blocks.{layer_idx}.mlp.b_in"]

        # Output projection
        if f"blocks.{layer_idx}.mlp.W_out" in tlens_weights:
            hf_weights[f"{prefix}.c_proj.weight"] = tlens_weights[f"blocks.{layer_idx}.mlp.W_out"]
        if f"blocks.{layer_idx}.mlp.b_out" in tlens_weights:
            hf_weights[f"{prefix}.c_proj.bias"] = tlens_weights[f"blocks.{layer_idx}.mlp.b_out"]

        return hf_weights

    def _convert_llama_mlp_hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                                     config: HookedTransformerConfig, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Convert LLaMA style MLP from HF to TLens (SwiGLU/GeGLU)."""
        tlens_weights = {}
        prefix = f"model.layers.{layer_idx}.mlp"

        # For SwiGLU, we have gate_proj and up_proj that get combined
        if f"{prefix}.gate_proj.weight" in hf_weights:
            tlens_weights[f"blocks.{layer_idx}.mlp.W_gate"] = hf_weights[f"{prefix}.gate_proj.weight"]
        if f"{prefix}.up_proj.weight" in hf_weights:
            tlens_weights[f"blocks.{layer_idx}.mlp.W_in"] = hf_weights[f"{prefix}.up_proj.weight"]
        if f"{prefix}.down_proj.weight" in hf_weights:
            tlens_weights[f"blocks.{layer_idx}.mlp.W_out"] = hf_weights[f"{prefix}.down_proj.weight"]

        return tlens_weights

    def _convert_llama_mlp_tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                                     config: HookedTransformerConfig, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Convert LLaMA style MLP from TLens to HF (SwiGLU/GeGLU)."""
        hf_weights = {}
        prefix = f"model.layers.{layer_idx}.mlp"

        # For SwiGLU, map back to gate_proj, up_proj, down_proj
        if f"blocks.{layer_idx}.mlp.W_gate" in tlens_weights:
            hf_weights[f"{prefix}.gate_proj.weight"] = tlens_weights[f"blocks.{layer_idx}.mlp.W_gate"]
        if f"blocks.{layer_idx}.mlp.W_in" in tlens_weights:
            hf_weights[f"{prefix}.up_proj.weight"] = tlens_weights[f"blocks.{layer_idx}.mlp.W_in"]
        if f"blocks.{layer_idx}.mlp.W_out" in tlens_weights:
            hf_weights[f"{prefix}.down_proj.weight"] = tlens_weights[f"blocks.{layer_idx}.mlp.W_out"]

        return hf_weights

    def _convert_gemma_mlp_hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                                     config: HookedTransformerConfig, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Convert Gemma style MLP from HF to TLens."""
        return self._convert_llama_mlp_hf_to_tlens(hf_weights, config, layer_idx)

    def _convert_gemma_mlp_tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                                     config: HookedTransformerConfig, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Convert Gemma style MLP from TLens to HF."""
        return self._convert_llama_mlp_tlens_to_hf(tlens_weights, config, layer_idx)

    def get_hf_keys(self, config: HookedTransformerConfig, **kwargs) -> List[str]:
        """Get expected HF keys for MLP."""
        model_type = kwargs.get("model_type", "gpt2")
        layer_idx = kwargs.get("layer_idx", 0)

        if model_type == "gpt2":
            prefix = f"transformer.h.{layer_idx}.mlp"
            return [f"{prefix}.c_fc.weight", f"{prefix}.c_fc.bias",
                   f"{prefix}.c_proj.weight", f"{prefix}.c_proj.bias"]
        elif model_type in ["llama", "mistral", "gemma"]:
            prefix = f"model.layers.{layer_idx}.mlp"
            return [f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight",
                   f"{prefix}.down_proj.weight"]
        else:
            return []

    def get_tlens_keys(self, config: HookedTransformerConfig, **kwargs) -> List[str]:
        """Get expected TLens keys for MLP."""
        layer_idx = kwargs.get("layer_idx", 0)
        model_type = kwargs.get("model_type", "gpt2")

        base_keys = [f"blocks.{layer_idx}.mlp.W_in", f"blocks.{layer_idx}.mlp.W_out"]

        if model_type == "gpt2":
            base_keys.extend([f"blocks.{layer_idx}.mlp.b_in", f"blocks.{layer_idx}.mlp.b_out"])
        elif model_type in ["llama", "mistral", "gemma"]:
            base_keys.append(f"blocks.{layer_idx}.mlp.W_gate")

        return base_keys


class NormalizationConverter(BaseComponentConverter):
    """Converter for normalization layers (LayerNorm/RMSNorm)."""

    def hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert HF normalization weights to TLens format."""
        tlens_weights = {}
        layer_idx = kwargs.get("layer_idx", None)
        norm_type = kwargs.get("norm_type", "ln1")  # ln1, ln2, or final
        model_type = kwargs.get("model_type", "gpt2")

        if model_type == "gpt2":
            return self._convert_gpt2_norm_hf_to_tlens(hf_weights, config, layer_idx, norm_type)
        elif model_type in ["llama", "mistral", "gemma"]:
            return self._convert_llama_norm_hf_to_tlens(hf_weights, config, layer_idx, norm_type)
        else:
            raise ConversionError(f"Unsupported model type for normalization: {model_type}")

    def tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert TLens normalization weights to HF format."""
        layer_idx = kwargs.get("layer_idx", None)
        norm_type = kwargs.get("norm_type", "ln1")
        model_type = kwargs.get("model_type", "gpt2")

        if model_type == "gpt2":
            return self._convert_gpt2_norm_tlens_to_hf(tlens_weights, config, layer_idx, norm_type)
        elif model_type in ["llama", "mistral", "gemma"]:
            return self._convert_llama_norm_tlens_to_hf(tlens_weights, config, layer_idx, norm_type)
        else:
            raise ConversionError(f"Unsupported model type for normalization: {model_type}")

    def _convert_gpt2_norm_hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                                     config: HookedTransformerConfig, layer_idx: Optional[int],
                                     norm_type: str) -> Dict[str, torch.Tensor]:
        """Convert GPT-2 style normalization from HF to TLens."""
        tlens_weights = {}

        if norm_type == "final":
            # Final layer norm
            if "transformer.ln_f.weight" in hf_weights:
                tlens_weights["ln_final.w"] = hf_weights["transformer.ln_f.weight"]
            if "transformer.ln_f.bias" in hf_weights:
                tlens_weights["ln_final.b"] = hf_weights["transformer.ln_f.bias"]
        elif layer_idx is not None:
            # Layer-specific norms
            hf_norm_name = "ln_1" if norm_type == "ln1" else "ln_2"
            tlens_norm_name = "ln1" if norm_type == "ln1" else "ln2"

            prefix = f"transformer.h.{layer_idx}.{hf_norm_name}"
            if f"{prefix}.weight" in hf_weights:
                tlens_weights[f"blocks.{layer_idx}.{tlens_norm_name}.w"] = hf_weights[f"{prefix}.weight"]
            if f"{prefix}.bias" in hf_weights:
                tlens_weights[f"blocks.{layer_idx}.{tlens_norm_name}.b"] = hf_weights[f"{prefix}.bias"]

        return tlens_weights

    def _convert_gpt2_norm_tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                                     config: HookedTransformerConfig, layer_idx: Optional[int],
                                     norm_type: str) -> Dict[str, torch.Tensor]:
        """Convert GPT-2 style normalization from TLens to HF."""
        hf_weights = {}

        if norm_type == "final":
            # Final layer norm
            if "ln_final.w" in tlens_weights:
                hf_weights["transformer.ln_f.weight"] = tlens_weights["ln_final.w"]
            if "ln_final.b" in tlens_weights:
                hf_weights["transformer.ln_f.bias"] = tlens_weights["ln_final.b"]
        elif layer_idx is not None:
            # Layer-specific norms
            hf_norm_name = "ln_1" if norm_type == "ln1" else "ln_2"
            tlens_norm_name = "ln1" if norm_type == "ln1" else "ln2"

            prefix = f"transformer.h.{layer_idx}.{hf_norm_name}"
            if f"blocks.{layer_idx}.{tlens_norm_name}.w" in tlens_weights:
                hf_weights[f"{prefix}.weight"] = tlens_weights[f"blocks.{layer_idx}.{tlens_norm_name}.w"]
            if f"blocks.{layer_idx}.{tlens_norm_name}.b" in tlens_weights:
                hf_weights[f"{prefix}.bias"] = tlens_weights[f"blocks.{layer_idx}.{tlens_norm_name}.b"]

        return hf_weights

    def _convert_llama_norm_hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                                      config: HookedTransformerConfig, layer_idx: Optional[int],
                                      norm_type: str) -> Dict[str, torch.Tensor]:
        """Convert LLaMA style normalization from HF to TLens."""
        tlens_weights = {}

        if norm_type == "final":
            # Final RMS norm
            if "model.norm.weight" in hf_weights:
                tlens_weights["ln_final.w"] = hf_weights["model.norm.weight"]
        elif layer_idx is not None:
            # Layer-specific RMS norms
            if norm_type == "ln1":
                prefix = f"model.layers.{layer_idx}.input_layernorm"
                tlens_key = f"blocks.{layer_idx}.ln1.w"
            else:  # ln2
                prefix = f"model.layers.{layer_idx}.post_attention_layernorm"
                tlens_key = f"blocks.{layer_idx}.ln2.w"

            if f"{prefix}.weight" in hf_weights:
                tlens_weights[tlens_key] = hf_weights[f"{prefix}.weight"]

        return tlens_weights

    def _convert_llama_norm_tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                                      config: HookedTransformerConfig, layer_idx: Optional[int],
                                      norm_type: str) -> Dict[str, torch.Tensor]:
        """Convert LLaMA style normalization from TLens to HF."""
        hf_weights = {}

        if norm_type == "final":
            # Final RMS norm
            if "ln_final.w" in tlens_weights:
                hf_weights["model.norm.weight"] = tlens_weights["ln_final.w"]
        elif layer_idx is not None:
            # Layer-specific RMS norms
            if norm_type == "ln1":
                prefix = f"model.layers.{layer_idx}.input_layernorm"
                tlens_key = f"blocks.{layer_idx}.ln1.w"
            else:  # ln2
                prefix = f"model.layers.{layer_idx}.post_attention_layernorm"
                tlens_key = f"blocks.{layer_idx}.ln2.w"

            if tlens_key in tlens_weights:
                hf_weights[f"{prefix}.weight"] = tlens_weights[tlens_key]

        return hf_weights

    def get_hf_keys(self, config: HookedTransformerConfig, **kwargs) -> List[str]:
        """Get expected HF keys for normalization."""
        model_type = kwargs.get("model_type", "gpt2")
        layer_idx = kwargs.get("layer_idx", None)
        norm_type = kwargs.get("norm_type", "ln1")

        keys = []

        if norm_type == "final":
            if model_type == "gpt2":
                keys.extend(["transformer.ln_f.weight", "transformer.ln_f.bias"])
            elif model_type in ["llama", "mistral", "gemma"]:
                keys.append("model.norm.weight")
        elif layer_idx is not None:
            if model_type == "gpt2":
                hf_norm_name = "ln_1" if norm_type == "ln1" else "ln_2"
                prefix = f"transformer.h.{layer_idx}.{hf_norm_name}"
                keys.extend([f"{prefix}.weight", f"{prefix}.bias"])
            elif model_type in ["llama", "mistral", "gemma"]:
                if norm_type == "ln1":
                    prefix = f"model.layers.{layer_idx}.input_layernorm"
                else:
                    prefix = f"model.layers.{layer_idx}.post_attention_layernorm"
                keys.append(f"{prefix}.weight")

        return keys

    def get_tlens_keys(self, config: HookedTransformerConfig, **kwargs) -> List[str]:
        """Get expected TLens keys for normalization."""
        model_type = kwargs.get("model_type", "gpt2")
        layer_idx = kwargs.get("layer_idx", None)
        norm_type = kwargs.get("norm_type", "ln1")

        keys = []

        if norm_type == "final":
            keys.append("ln_final.w")
            if model_type == "gpt2":
                keys.append("ln_final.b")
        elif layer_idx is not None:
            tlens_norm_name = "ln1" if norm_type == "ln1" else "ln2"
            keys.append(f"blocks.{layer_idx}.{tlens_norm_name}.w")
            if model_type == "gpt2":
                keys.append(f"blocks.{layer_idx}.{tlens_norm_name}.b")

        return keys


class UnembeddingConverter(BaseComponentConverter):
    """Converter for unembedding/lm_head layers."""

    def hf_to_tlens(self, hf_weights: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert HF unembedding weights to TLens format."""
        tlens_weights = {}
        model_type = kwargs.get("model_type", "gpt2")

        # Language modeling head
        if "lm_head.weight" in hf_weights:
            # Transpose for TLens format
            tlens_weights["unembed.W_U"] = hf_weights["lm_head.weight"].T
        elif "model.lm_head.weight" in hf_weights:  # Some models
            tlens_weights["unembed.W_U"] = hf_weights["model.lm_head.weight"].T

        # Bias (if exists)
        if "lm_head.bias" in hf_weights:
            tlens_weights["unembed.b_U"] = hf_weights["lm_head.bias"]
        elif "model.lm_head.bias" in hf_weights:
            tlens_weights["unembed.b_U"] = hf_weights["model.lm_head.bias"]
        else:
            # Create zero bias if none exists
            if "unembed.W_U" in tlens_weights:
                vocab_size = tlens_weights["unembed.W_U"].shape[1]
                tlens_weights["unembed.b_U"] = torch.zeros(vocab_size,
                                                         dtype=tlens_weights["unembed.W_U"].dtype,
                                                         device=tlens_weights["unembed.W_U"].device)

        return tlens_weights

    def tlens_to_hf(self, tlens_weights: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert TLens unembedding weights to HF format."""
        hf_weights = {}
        model_type = kwargs.get("model_type", "gpt2")

        # Language modeling head
        if "unembed.W_U" in tlens_weights:
            # Transpose back to HF format
            if model_type in ["llama", "mistral", "gemma"]:
                hf_weights["model.lm_head.weight"] = tlens_weights["unembed.W_U"].T
            else:  # GPT-2 style
                hf_weights["lm_head.weight"] = tlens_weights["unembed.W_U"].T

        # Bias (only for models that have it)
        if "unembed.b_U" in tlens_weights:
            bias = tlens_weights["unembed.b_U"]
            # Only add bias if it's non-zero (some models don't use bias)
            if not torch.allclose(bias, torch.zeros_like(bias)):
                if model_type in ["llama", "mistral", "gemma"]:
                    hf_weights["model.lm_head.bias"] = bias
                else:  # GPT-2 style
                    hf_weights["lm_head.bias"] = bias

        return hf_weights

    def get_hf_keys(self, config: HookedTransformerConfig, **kwargs) -> List[str]:
        """Get expected HF keys for unembedding."""
        model_type = kwargs.get("model_type", "gpt2")

        if model_type in ["llama", "mistral", "gemma"]:
            return ["model.lm_head.weight"]  # Usually no bias
        else:  # GPT-2 style
            return ["lm_head.weight", "lm_head.bias"]

    def get_tlens_keys(self, config: HookedTransformerConfig, **kwargs) -> List[str]:
        """Get expected TLens keys for unembedding."""
        return ["unembed.W_U", "unembed.b_U"]


class ReversibleWeightConverter:
    """Main class for bidirectional weight conversion between HF and TransformerLens."""

    def __init__(self):
        """Initialize the converter with component-specific converters."""
        self.embedding_converter = EmbeddingConverter()
        self.attention_converter = AttentionConverter()
        self.mlp_converter = MLPConverter()
        self.norm_converter = NormalizationConverter()
        self.unembed_converter = UnembeddingConverter()

    def hf_to_tlens(self, hf_state_dict: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig,
                   model_type: str = "gpt2") -> Dict[str, torch.Tensor]:
        """
        Convert complete HuggingFace state dict to TransformerLens format.

        Args:
            hf_state_dict: HuggingFace model state dictionary
            config: TransformerLens configuration
            model_type: Type of model ("gpt2", "llama", "mistral", "gemma")

        Returns:
            TransformerLens format state dictionary
        """
        tlens_state_dict = {}

        # Convert embeddings
        embedding_weights = self.embedding_converter.hf_to_tlens(
            hf_state_dict, config, model_type=model_type
        )
        tlens_state_dict.update(embedding_weights)

        # Convert layers
        for layer_idx in range(config.n_layers):
            # Attention
            attention_weights = self.attention_converter.hf_to_tlens(
                hf_state_dict, config, layer_idx=layer_idx, model_type=model_type
            )
            tlens_state_dict.update(attention_weights)

            # MLP
            mlp_weights = self.mlp_converter.hf_to_tlens(
                hf_state_dict, config, layer_idx=layer_idx, model_type=model_type
            )
            tlens_state_dict.update(mlp_weights)

            # Normalization layers
            for norm_type in ["ln1", "ln2"]:
                norm_weights = self.norm_converter.hf_to_tlens(
                    hf_state_dict, config, layer_idx=layer_idx,
                    norm_type=norm_type, model_type=model_type
                )
                tlens_state_dict.update(norm_weights)

        # Final normalization
        final_norm_weights = self.norm_converter.hf_to_tlens(
            hf_state_dict, config, norm_type="final", model_type=model_type
        )
        tlens_state_dict.update(final_norm_weights)

        # Unembedding
        unembed_weights = self.unembed_converter.hf_to_tlens(
            hf_state_dict, config, model_type=model_type
        )
        tlens_state_dict.update(unembed_weights)

        return tlens_state_dict

    def tlens_to_hf(self, tlens_state_dict: Dict[str, torch.Tensor],
                   config: HookedTransformerConfig,
                   model_type: str = "gpt2") -> Dict[str, torch.Tensor]:
        """
        Convert complete TransformerLens state dict to HuggingFace format.

        Args:
            tlens_state_dict: TransformerLens model state dictionary
            config: TransformerLens configuration
            model_type: Type of model ("gpt2", "llama", "mistral", "gemma")

        Returns:
            HuggingFace format state dictionary
        """
        hf_state_dict = {}

        # Convert embeddings
        embedding_weights = self.embedding_converter.tlens_to_hf(
            tlens_state_dict, config, model_type=model_type
        )
        hf_state_dict.update(embedding_weights)

        # Convert layers
        for layer_idx in range(config.n_layers):
            # Attention
            attention_weights = self.attention_converter.tlens_to_hf(
                tlens_state_dict, config, layer_idx=layer_idx, model_type=model_type
            )
            hf_state_dict.update(attention_weights)

            # MLP
            mlp_weights = self.mlp_converter.tlens_to_hf(
                tlens_state_dict, config, layer_idx=layer_idx, model_type=model_type
            )
            hf_state_dict.update(mlp_weights)

            # Normalization layers
            for norm_type in ["ln1", "ln2"]:
                norm_weights = self.norm_converter.tlens_to_hf(
                    tlens_state_dict, config, layer_idx=layer_idx,
                    norm_type=norm_type, model_type=model_type
                )
                hf_state_dict.update(norm_weights)

        # Final normalization
        final_norm_weights = self.norm_converter.tlens_to_hf(
            tlens_state_dict, config, norm_type="final", model_type=model_type
        )
        hf_state_dict.update(final_norm_weights)

        # Unembedding
        unembed_weights = self.unembed_converter.tlens_to_hf(
            tlens_state_dict, config, model_type=model_type
        )
        hf_state_dict.update(unembed_weights)

        return hf_state_dict

    def validate_round_trip_hf_to_tlens(self, hf_state_dict: Dict[str, torch.Tensor],
                                       config: HookedTransformerConfig,
                                       model_type: str = "gpt2",
                                       tolerance: float = 1e-6) -> bool:
        """
        Validate HF → TLens → HF round-trip conversion.

        Args:
            hf_state_dict: Original HuggingFace state dictionary
            config: TransformerLens configuration
            model_type: Type of model
            tolerance: Numerical tolerance for comparison

        Returns:
            True if round-trip is successful

        Raises:
            RoundTripError: If validation fails
        """
        try:
            # Forward conversion
            tlens_dict = self.hf_to_tlens(hf_state_dict, config, model_type)

            # Backward conversion
            recovered_hf_dict = self.tlens_to_hf(tlens_dict, config, model_type)

            # Validate all original keys are recovered
            missing_keys = set(hf_state_dict.keys()) - set(recovered_hf_dict.keys())
            if missing_keys:
                raise RoundTripError(f"Missing keys after round-trip: {missing_keys}")

            # Validate all values match within tolerance
            mismatched_keys = []
            for key, original_tensor in hf_state_dict.items():
                if key in recovered_hf_dict:
                    recovered_tensor = recovered_hf_dict[key]
                    if not torch.allclose(original_tensor, recovered_tensor, atol=tolerance, rtol=tolerance):
                        max_diff = torch.max(torch.abs(original_tensor - recovered_tensor)).item()
                        mismatched_keys.append((key, max_diff))

            if mismatched_keys:
                error_msg = "Round-trip mismatches:\n"
                for key, diff in mismatched_keys[:5]:  # Show first 5
                    error_msg += f"  {key}: max_diff={diff:.2e}\n"
                raise RoundTripError(error_msg)

            return True

        except Exception as e:
            raise RoundTripError(f"HF→TLens→HF round-trip failed: {str(e)}")

    def validate_round_trip_tlens_to_hf(self, tlens_state_dict: Dict[str, torch.Tensor],
                                       config: HookedTransformerConfig,
                                       model_type: str = "gpt2",
                                       tolerance: float = 1e-6) -> bool:
        """
        Validate TLens → HF → TLens round-trip conversion.

        Args:
            tlens_state_dict: Original TransformerLens state dictionary
            config: TransformerLens configuration
            model_type: Type of model
            tolerance: Numerical tolerance for comparison

        Returns:
            True if round-trip is successful

        Raises:
            RoundTripError: If validation fails
        """
        try:
            # Forward conversion
            hf_dict = self.tlens_to_hf(tlens_state_dict, config, model_type)

            # Backward conversion
            recovered_tlens_dict = self.hf_to_tlens(hf_dict, config, model_type)

            # Validate all original keys are recovered
            missing_keys = set(tlens_state_dict.keys()) - set(recovered_tlens_dict.keys())
            if missing_keys:
                raise RoundTripError(f"Missing keys after round-trip: {missing_keys}")

            # Validate all values match within tolerance
            mismatched_keys = []
            for key, original_tensor in tlens_state_dict.items():
                if key in recovered_tlens_dict:
                    recovered_tensor = recovered_tlens_dict[key]
                    if not torch.allclose(original_tensor, recovered_tensor, atol=tolerance, rtol=tolerance):
                        max_diff = torch.max(torch.abs(original_tensor - recovered_tensor)).item()
                        mismatched_keys.append((key, max_diff))

            if mismatched_keys:
                error_msg = "Round-trip mismatches:\n"
                for key, diff in mismatched_keys[:5]:  # Show first 5
                    error_msg += f"  {key}: max_diff={diff:.2e}\n"
                raise RoundTripError(error_msg)

            return True

        except Exception as e:
            raise RoundTripError(f"TLens→HF→TLens round-trip failed: {str(e)}")

    def debug_conversion_mismatch(self, original_dict: Dict[str, torch.Tensor],
                                 recovered_dict: Dict[str, torch.Tensor],
                                 tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Debug conversion mismatches by analyzing differences.

        Args:
            original_dict: Original state dictionary
            recovered_dict: Recovered state dictionary after round-trip
            tolerance: Tolerance for considering values equal

        Returns:
            Dictionary with detailed mismatch analysis
        """
        debug_info = {
            "missing_keys": [],
            "extra_keys": [],
            "shape_mismatches": [],
            "dtype_mismatches": [],
            "value_mismatches": [],
            "summary": {}
        }

        original_keys = set(original_dict.keys())
        recovered_keys = set(recovered_dict.keys())

        # Missing and extra keys
        debug_info["missing_keys"] = list(original_keys - recovered_keys)
        debug_info["extra_keys"] = list(recovered_keys - original_keys)

        # Check common keys
        common_keys = original_keys & recovered_keys

        for key in common_keys:
            orig_tensor = original_dict[key]
            rec_tensor = recovered_dict[key]

            # Shape mismatch
            if orig_tensor.shape != rec_tensor.shape:
                debug_info["shape_mismatches"].append({
                    "key": key,
                    "original_shape": orig_tensor.shape,
                    "recovered_shape": rec_tensor.shape
                })
                continue

            # Dtype mismatch
            if orig_tensor.dtype != rec_tensor.dtype:
                debug_info["dtype_mismatches"].append({
                    "key": key,
                    "original_dtype": orig_tensor.dtype,
                    "recovered_dtype": rec_tensor.dtype
                })

            # Value mismatch
            if not torch.allclose(orig_tensor, rec_tensor, atol=tolerance, rtol=tolerance):
                max_diff = torch.max(torch.abs(orig_tensor - rec_tensor)).item()
                mean_diff = torch.mean(torch.abs(orig_tensor - rec_tensor)).item()
                debug_info["value_mismatches"].append({
                    "key": key,
                    "max_difference": max_diff,
                    "mean_difference": mean_diff,
                    "relative_error": max_diff / (torch.max(torch.abs(orig_tensor)).item() + 1e-8)
                })

        # Summary
        debug_info["summary"] = {
            "total_keys_original": len(original_keys),
            "total_keys_recovered": len(recovered_keys),
            "missing_count": len(debug_info["missing_keys"]),
            "extra_count": len(debug_info["extra_keys"]),
            "shape_mismatch_count": len(debug_info["shape_mismatches"]),
            "dtype_mismatch_count": len(debug_info["dtype_mismatches"]),
            "value_mismatch_count": len(debug_info["value_mismatches"])
        }

        return debug_info