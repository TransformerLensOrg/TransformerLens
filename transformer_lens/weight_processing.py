#!/usr/bin/env python3
"""
Weight Processing Functions for Transformer Models.

This module contains all the weight processing functions extracted from HookedTransformer,
organized into a single ProcessWeights class with static methods. These functions are used
to modify transformer model weights for better interpretability and analysis.
"""

from typing import Any, Dict, Optional, Union

import einops
import torch
from torch import nn

import transformer_lens.utils as utils
from transformer_lens.FactoredMatrix import FactoredMatrix


class ProcessWeights:
    """
    A collection of static methods for processing transformer model weights.

    These methods are extracted from HookedTransformer and provide various weight
    transformations for improved model interpretability:
    - LayerNorm folding: Merges LayerNorm parameters into subsequent linear layers
    - Weight centering: Centers weights that write to the residual stream
    - Unembed centering: Centers unembedding weights (translation invariant)
    - Value bias folding: Consolidates value biases into output biases
    - Attention matrix refactoring: Experimental QK/OV matrix factorization

    When an architecture adapter is provided, the methods will translate TransformerLens
    parameter names to the target format (e.g., HuggingFace) for processing.
    """

    @staticmethod
    def _get_param_key(tl_key: str, adapter=None) -> str:
        """Get the actual parameter key to use, translating via adapter if provided.

        Args:
            tl_key: TransformerLens format parameter key
            adapter: Optional architecture adapter for key translation

        Returns:
            The key to use for accessing parameters in the state dict
        """
        if adapter is None:
            return tl_key

        # Use the adapter to translate from TL format to target format
        return adapter.translate_transformer_lens_path(tl_key)

    @staticmethod
    def fold_layer_norm_bias_single(
        w_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        ln_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Fold LayerNorm bias into a single attention bias.

        Args:
            w_tensor: Weight tensor [n_heads, d_model, d_head]
            b_tensor: Bias tensor [n_heads, d_head]
            ln_bias: LayerNorm bias [d_model]

        Returns:
            New bias tensor with folded LayerNorm bias
        """
        return b_tensor + (w_tensor * ln_bias[None, :, None]).sum(-2)

    @staticmethod
    def fold_layer_norm_weight_single(
        w_tensor: torch.Tensor,
        ln_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Fold LayerNorm weight into a single attention weight.

        Args:
            w_tensor: Weight tensor [n_heads, d_model, d_head]
            ln_weight: LayerNorm weight [d_model]

        Returns:
            New weight tensor with folded LayerNorm weight
        """
        return w_tensor * ln_weight[None, :, None]

    @staticmethod
    def center_weight_single(
        w_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Center a single attention weight by subtracting the mean.

        Args:
            w_tensor: Weight tensor [n_heads, d_model, d_head]

        Returns:
            Centered weight tensor
        """
        return w_tensor - einops.reduce(
            w_tensor, "head_index d_model d_head -> head_index 1 d_head", "mean"
        )

    @staticmethod
    def fold_layer_norm_biases(
        wq_tensor: torch.Tensor,
        wk_tensor: torch.Tensor,
        wv_tensor: torch.Tensor,
        bq_tensor: torch.Tensor,
        bk_tensor: torch.Tensor,
        bv_tensor: torch.Tensor,
        ln_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fold LayerNorm bias into attention biases.

        Args:
            wq_tensor, wk_tensor, wv_tensor: Weight tensors [n_heads, d_model, d_head]
            bq_tensor, bk_tensor, bv_tensor: Bias tensors [n_heads, d_head]
            ln_bias: LayerNorm bias [d_model]

        Returns:
            Tuple of (new_bq, new_bk, new_bv) with folded biases
        """
        new_bq = ProcessWeights.fold_layer_norm_bias_single(wq_tensor, bq_tensor, ln_bias)
        new_bk = ProcessWeights.fold_layer_norm_bias_single(wk_tensor, bk_tensor, ln_bias)
        new_bv = ProcessWeights.fold_layer_norm_bias_single(wv_tensor, bv_tensor, ln_bias)

        return new_bq, new_bk, new_bv

    @staticmethod
    def fold_layer_norm_weights(
        wq_tensor: torch.Tensor,
        wk_tensor: torch.Tensor,
        wv_tensor: torch.Tensor,
        ln_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fold LayerNorm weight into attention weights.

        Args:
            wq_tensor, wk_tensor, wv_tensor: Weight tensors [n_heads, d_model, d_head]
            ln_weight: LayerNorm weight [d_model]

        Returns:
            Tuple of (new_wq, new_wk, new_wv) with folded weights
        """
        new_wq = ProcessWeights.fold_layer_norm_weight_single(wq_tensor, ln_weight)
        new_wk = ProcessWeights.fold_layer_norm_weight_single(wk_tensor, ln_weight)
        new_wv = ProcessWeights.fold_layer_norm_weight_single(wv_tensor, ln_weight)

        return new_wq, new_wk, new_wv

    @staticmethod
    def center_attention_weights(
        wq_tensor: torch.Tensor,
        wk_tensor: torch.Tensor,
        wv_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Center attention weights by subtracting the mean.

        Args:
            wq_tensor, wk_tensor, wv_tensor: Weight tensors [n_heads, d_model, d_head]

        Returns:
            Tuple of (centered_wq, centered_wk, centered_wv)
        """
        centered_wq = ProcessWeights.center_weight_single(wq_tensor)
        centered_wk = ProcessWeights.center_weight_single(wk_tensor)
        centered_wv = ProcessWeights.center_weight_single(wv_tensor)

        return centered_wq, centered_wk, centered_wv

    @staticmethod
    def _detect_state_dict_format(
        state_dict: Dict[str, torch.Tensor], layer: int, adapter
    ) -> tuple[bool, bool]:
        """Detect whether state_dict uses TransformerLens or HuggingFace format.

        Args:
            state_dict: The state dictionary to check
            layer: Layer index to check
            adapter: Optional adapter for key translation

        Returns:
            Tuple of (uses_tl_format, uses_hf_format)
        """
        # Sample keys to check format
        tl_key_sample = f"blocks.{layer}.attn.W_Q"
        hf_key_sample = ProcessWeights._get_param_key(tl_key_sample, adapter) if adapter else None

        uses_tl_format = tl_key_sample in state_dict
        uses_hf_format = bool(adapter and hf_key_sample and hf_key_sample in state_dict)

        return uses_tl_format, uses_hf_format

    @staticmethod
    def extract_attention_tensors_for_folding(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        layer: int,
        adapter,
    ) -> Dict[str, Union[torch.Tensor, None, Dict[str, str]]]:
        """Extract attention tensors in TransformerLens format for layer norm folding.

        Args:
            state_dict: The state dictionary containing tensors
            cfg: Model configuration object
            layer: Layer index
            adapter: Optional architecture adapter for parameter key translation

        Returns:
            Dictionary with keys: 'wq', 'wk', 'wv', 'bq', 'bk', 'bv', 'ln1_b', 'ln1_w'
            All tensors are in TransformerLens format for consistent processing
        """
        # Get translated parameter keys
        b_Q_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_Q", adapter)
        W_Q_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.W_Q", adapter)
        b_K_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_K", adapter)
        W_K_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.W_K", adapter)
        b_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_V", adapter)
        W_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.W_V", adapter)
        ln1_b_key = ProcessWeights._get_param_key(f"blocks.{layer}.ln1.b", adapter)
        ln1_w_key = ProcessWeights._get_param_key(f"blocks.{layer}.ln1.w", adapter)

        # Determine the actual format of the state_dict to avoid key mismatch
        uses_tl_format, uses_hf_format = ProcessWeights._detect_state_dict_format(
            state_dict, layer, adapter
        )

        # Extract tensors based on actual format detection, not just adapter presence
        if adapter and uses_hf_format and not uses_tl_format:
            # State dict is in HuggingFace format - convert to TransformerLens format
            wq_tensor = ProcessWeights.convert_tensor_to_tl_format(
                f"blocks.{layer}.attn.W_Q", adapter, state_dict, cfg, layer
            )
            wk_tensor = ProcessWeights.convert_tensor_to_tl_format(
                f"blocks.{layer}.attn.W_K", adapter, state_dict, cfg, layer
            )
            wv_tensor = ProcessWeights.convert_tensor_to_tl_format(
                f"blocks.{layer}.attn.W_V", adapter, state_dict, cfg, layer
            )
            bq_tensor = ProcessWeights.convert_tensor_to_tl_format(
                f"blocks.{layer}.attn.b_Q", adapter, state_dict, cfg, layer
            )
            bk_tensor = ProcessWeights.convert_tensor_to_tl_format(
                f"blocks.{layer}.attn.b_K", adapter, state_dict, cfg, layer
            )
            bv_tensor = ProcessWeights.convert_tensor_to_tl_format(
                f"blocks.{layer}.attn.b_V", adapter, state_dict, cfg, layer
            )
        else:
            # State dict is already in TransformerLens format - use directly
            # Handle case where some keys might not exist (e.g., grouped query attention)
            wq_tensor = state_dict.get(f"blocks.{layer}.attn.W_Q", None)  # type: ignore[assignment]
            wk_tensor = state_dict.get(f"blocks.{layer}.attn.W_K", None)  # type: ignore[assignment]
            wv_tensor = state_dict.get(f"blocks.{layer}.attn.W_V", None)  # type: ignore[assignment]
            bq_tensor = state_dict.get(f"blocks.{layer}.attn.b_Q", None)  # type: ignore[assignment]
            bk_tensor = state_dict.get(f"blocks.{layer}.attn.b_K", None)  # type: ignore[assignment]
            bv_tensor = state_dict.get(f"blocks.{layer}.attn.b_V", None)  # type: ignore[assignment]

        # Extract LayerNorm parameters using same format detection
        if uses_tl_format:
            ln1_b = state_dict.get(f"blocks.{layer}.ln1.b", None)
            ln1_w = state_dict.get(f"blocks.{layer}.ln1.w", None)
        else:
            ln1_b = state_dict.get(ln1_b_key, None)
            ln1_w = state_dict.get(ln1_w_key, None)

        return {
            "wq": wq_tensor,
            "wk": wk_tensor,
            "wv": wv_tensor,
            "bq": bq_tensor,
            "bk": bk_tensor,
            "bv": bv_tensor,
            "ln1_b": ln1_b,
            "ln1_w": ln1_w,
            # Store the actual keys used based on format detection
            "keys": {
                "W_Q": W_Q_key if adapter else f"blocks.{layer}.attn.W_Q",
                "W_K": W_K_key if adapter else f"blocks.{layer}.attn.W_K",
                "W_V": W_V_key if adapter else f"blocks.{layer}.attn.W_V",
                "b_Q": b_Q_key if adapter else f"blocks.{layer}.attn.b_Q",
                "b_K": b_K_key if adapter else f"blocks.{layer}.attn.b_K",
                "b_V": b_V_key if adapter else f"blocks.{layer}.attn.b_V",
                "ln1_b": ln1_b_key if adapter else f"blocks.{layer}.ln1.b",
                "ln1_w": ln1_w_key if adapter else f"blocks.{layer}.ln1.w",
            },
        }

    @staticmethod
    def _fold_layer(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        layer_idx: int,
        fold_biases: bool,
        center_weights: bool,
        adapter,
        gqa: str,
    ) -> None:
        """Fold LayerNorm for a single layer.

        Args:
            state_dict: The state dictionary to process (modified in place)
            cfg: Model configuration object
            layer_idx: The layer index to process
            fold_biases: Whether to fold LayerNorm biases
            center_weights: Whether to center weights after folding
            adapter: Optional architecture adapter for parameter key translation
            gqa: GQA prefix string (empty or "_")
        """
        layer = layer_idx

        # Extract all tensors in TransformerLens format using the new extraction function
        tensors = ProcessWeights.extract_attention_tensors_for_folding(
            state_dict, cfg, layer, adapter
        )

        # Get local variables for clean processing
        wq_tensor = tensors["wq"]
        wk_tensor = tensors["wk"]
        wv_tensor = tensors["wv"]
        bq_tensor = tensors["bq"]
        bk_tensor = tensors["bk"]
        bv_tensor = tensors["bv"]
        ln1_b = tensors["ln1_b"]
        ln1_w = tensors["ln1_w"]
        keys = tensors["keys"]

        # Check if we have the required tensors for layer norm folding
        # For grouped query attention models, some tensors might be None
        if wq_tensor is None:
            # Skip layer norm folding for this layer if missing critical tensors
            return

        # Type assertions for mypy for required tensors
        assert isinstance(wq_tensor, torch.Tensor)
        assert isinstance(keys, dict)

        # For grouped query attention, K and V might be shared/grouped differently
        # Only assert if they exist
        if wk_tensor is not None:
            assert isinstance(wk_tensor, torch.Tensor)
        if wv_tensor is not None:
            assert isinstance(wv_tensor, torch.Tensor)
        if bq_tensor is not None:
            assert isinstance(bq_tensor, torch.Tensor)
        if bk_tensor is not None:
            assert isinstance(bk_tensor, torch.Tensor)
        if bv_tensor is not None:
            assert isinstance(bv_tensor, torch.Tensor)

        # Apply layer norm folding if parameters exist
        if ln1_b is not None and ln1_w is not None:
            # Type assertion for mypy within the if block
            assert isinstance(ln1_b, torch.Tensor)
            assert isinstance(ln1_w, torch.Tensor)

            # Apply the individual math functions
            if fold_biases:
                # Only fold biases if all tensors exist
                if all(
                    t is not None for t in [wk_tensor, wv_tensor, bq_tensor, bk_tensor, bv_tensor]
                ):
                    bq_tensor, bk_tensor, bv_tensor = ProcessWeights.fold_layer_norm_biases(  # type: ignore[arg-type]
                        wq_tensor, wk_tensor, wv_tensor, bq_tensor, bk_tensor, bv_tensor, ln1_b  # type: ignore[arg-type]
                    )
                if keys["ln1_b"] in state_dict:
                    del state_dict[keys["ln1_b"]]

            # Only fold weights if all tensors exist
            if wk_tensor is not None and wv_tensor is not None:
                wq_tensor, wk_tensor, wv_tensor = ProcessWeights.fold_layer_norm_weights(
                    wq_tensor, wk_tensor, wv_tensor, ln1_w
                )
            if keys["ln1_w"] in state_dict:
                del state_dict[keys["ln1_w"]]

        # Center the weights if requested
        if center_weights and wk_tensor is not None and wv_tensor is not None:
            wq_tensor, wk_tensor, wv_tensor = ProcessWeights.center_attention_weights(
                wq_tensor, wk_tensor, wv_tensor
            )

        # Store processed tensors back to state dict
        ProcessWeights._store_processed_attention_tensors(
            state_dict,
            keys,
            wq_tensor,
            wk_tensor,
            wv_tensor,
            bq_tensor,
            bk_tensor,
            bv_tensor,
            adapter,
            cfg,
            layer,
        )

        # # Fold ln2 into MLP
        ProcessWeights._fold_mlp_layer_norm(
            state_dict, cfg, layer, fold_biases, center_weights, adapter
        )

    @staticmethod
    def _fold_mlp_layer_norm(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        layer: int,
        fold_biases: bool,
        center_weights: bool,
        adapter,
    ) -> None:
        """Fold LayerNorm into MLP layer.

        Args:
            state_dict: The state dictionary to process (modified in place)
            cfg: Model configuration object
            layer: The layer index to process
            fold_biases: Whether to fold LayerNorm biases
            center_weights: Whether to center weights after folding
            adapter: Optional architecture adapter for parameter key translation
        """
        if getattr(cfg, "attn_only", False):
            return

        # Determine the actual format of the state_dict to avoid key mismatch
        uses_tl_format, uses_hf_format = ProcessWeights._detect_state_dict_format(
            state_dict, layer, adapter
        )

        # Get appropriate MLP parameter keys based on format detection
        if uses_tl_format:
            # State dict is in TransformerLens format - use TL keys directly
            mlp_b_in_key = f"blocks.{layer}.mlp.b_in"
            mlp_W_in_key = f"blocks.{layer}.mlp.W_in"
            mlp_W_gate_key = (
                f"blocks.{layer}.mlp.W_gate" if getattr(cfg, "gated_mlp", False) else None
            )
            ln2_b_key = f"blocks.{layer}.ln2.b"
            ln2_w_key = f"blocks.{layer}.ln2.w"
        else:
            # State dict is in HuggingFace format - use translated keys
            mlp_b_in_key = ProcessWeights._get_param_key(f"blocks.{layer}.mlp.b_in", adapter)
            mlp_W_in_key = ProcessWeights._get_param_key(f"blocks.{layer}.mlp.W_in", adapter)
            mlp_W_gate_key = (
                ProcessWeights._get_param_key(f"blocks.{layer}.mlp.W_gate", adapter)
                if getattr(cfg, "gated_mlp", False)
                else None
            )
            ln2_b_key = ProcessWeights._get_param_key(f"blocks.{layer}.ln2.b", adapter)
            ln2_w_key = ProcessWeights._get_param_key(f"blocks.{layer}.ln2.w", adapter)

        # Check if MLP LayerNorm parameters exist (they might not for already processed models)
        if ln2_b_key in state_dict and ln2_w_key in state_dict:
            if fold_biases:
                # TODO this is causing slight divergence - FIXED
                state_dict[mlp_b_in_key] = state_dict[mlp_b_in_key] + (
                    state_dict[mlp_W_in_key] * state_dict[ln2_b_key][:, None]
                ).sum(-2)
                del state_dict[ln2_b_key]

            # TODO this is causing slight divergence
            state_dict[mlp_W_in_key] = state_dict[mlp_W_in_key] * state_dict[ln2_w_key][:, None]

            if getattr(cfg, "gated_mlp", False) and mlp_W_gate_key is not None:
                state_dict[mlp_W_gate_key] = (
                    state_dict[mlp_W_gate_key] * state_dict[ln2_w_key][:, None]
                )

            del state_dict[ln2_w_key]

        if center_weights:
            # Center the weights that read in from the LayerNormPre
            state_dict[mlp_W_in_key] -= einops.reduce(
                state_dict[mlp_W_in_key],
                "d_model d_mlp -> 1 d_mlp",
                "mean",
            )

        if getattr(cfg, "act_fn", None) is not None and cfg.act_fn.startswith("solu"):
            # Get appropriate SoLU LayerNorm parameter keys based on format detection
            if uses_tl_format:
                # State dict is in TransformerLens format - use TL keys directly
                mlp_b_out_key = f"blocks.{layer}.mlp.b_out"
                mlp_W_out_key = f"blocks.{layer}.mlp.W_out"
                mlp_ln_b_key = f"blocks.{layer}.mlp.ln.b"
                mlp_ln_w_key = f"blocks.{layer}.mlp.ln.w"
            else:
                # State dict is in HuggingFace format - use translated keys
                mlp_b_out_key = ProcessWeights._get_param_key(f"blocks.{layer}.mlp.b_out", adapter)
                mlp_W_out_key = ProcessWeights._get_param_key(f"blocks.{layer}.mlp.W_out", adapter)
                mlp_ln_b_key = ProcessWeights._get_param_key(f"blocks.{layer}.mlp.ln.b", adapter)
                mlp_ln_w_key = ProcessWeights._get_param_key(f"blocks.{layer}.mlp.ln.w", adapter)

            # Fold ln3 into activation
            if fold_biases:
                state_dict[mlp_b_out_key] = state_dict[mlp_b_out_key] + (
                    state_dict[mlp_W_out_key] * state_dict[mlp_ln_b_key][:, None]
                ).sum(-2)

                if mlp_ln_b_key in state_dict:
                    del state_dict[mlp_ln_b_key]

            state_dict[mlp_W_out_key] = (
                state_dict[mlp_W_out_key] * state_dict[mlp_ln_w_key][:, None]
            )

            if center_weights:
                # Center the weights that read in from the LayerNormPre
                state_dict[mlp_W_out_key] -= einops.reduce(
                    state_dict[mlp_W_out_key],
                    "d_mlp d_model -> 1 d_model",
                    "mean",
                )

            if mlp_ln_w_key in state_dict:
                del state_dict[mlp_ln_w_key]

    @staticmethod
    def _store_processed_attention_tensors(
        state_dict: Dict[str, torch.Tensor],
        keys: Dict[str, str],
        wq_tensor: Optional[torch.Tensor],
        wk_tensor: Optional[torch.Tensor],
        wv_tensor: Optional[torch.Tensor],
        bq_tensor: Optional[torch.Tensor],
        bk_tensor: Optional[torch.Tensor],
        bv_tensor: Optional[torch.Tensor],
        adapter,
        cfg,
        layer: int,
    ) -> None:
        """Store processed attention tensors back to state dict in appropriate format.

        Args:
            state_dict: The state dictionary to update (modified in place)
            keys: Dictionary mapping tensor names to state dict keys
            wq_tensor, wk_tensor, wv_tensor: Processed attention weight tensors
            bq_tensor, bk_tensor, bv_tensor: Processed attention bias tensors
            adapter: Optional architecture adapter for parameter key translation
            cfg: Model configuration object
            layer: The layer index
        """
        # Skip storing if critical tensors are None (e.g., for grouped query attention)
        if wq_tensor is None:
            return

        if adapter:
            # Check if we're dealing with combined QKV format (like HuggingFace GPT-2)
            # by checking if W_Q, W_K, W_V keys map to the same HuggingFace key
            hf_w_q_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_Q")
            hf_w_k_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_K")
            hf_w_v_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_V")

            if hf_w_q_key == hf_w_k_key == hf_w_v_key:
                # Combined QKV format - combine back into single tensor
                # Only proceed if we have all required tensors
                if (
                    wk_tensor is None
                    or wv_tensor is None
                    or bq_tensor is None
                    or bk_tensor is None
                    or bv_tensor is None
                ):
                    return

                n_heads = cfg.n_heads
                d_head = cfg.d_head
                d_model = cfg.d_model

                # Convert back to HuggingFace format
                # Convert weights: [n_heads, d_model, d_head] -> [d_model, d_model]
                W_Q_hf = wq_tensor.reshape(d_model, d_model).T
                W_K_hf = wk_tensor.reshape(d_model, d_model).T
                W_V_hf = wv_tensor.reshape(d_model, d_model).T

                # Convert biases: [n_heads, d_head] -> [d_model]
                b_Q_hf = bq_tensor.reshape(d_model)
                b_K_hf = bk_tensor.reshape(d_model)
                b_V_hf = bv_tensor.reshape(d_model)

                # Combine back into HuggingFace format
                new_qkv_weight = torch.cat([W_Q_hf, W_K_hf, W_V_hf], dim=1)  # [d_model, 3*d_model]
                new_qkv_bias = torch.cat([b_Q_hf, b_K_hf, b_V_hf])  # [3*d_model]

                # Update state dict with combined format
                state_dict[hf_w_q_key] = new_qkv_weight
                state_dict[
                    adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.b_Q")
                ] = new_qkv_bias
            else:
                # Separate Q, K, V format - convert back individually
                # Get translated keys for separate format
                hf_w_q_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_Q")
                hf_w_k_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_K")
                hf_w_v_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_V")
                hf_b_q_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.b_Q")
                hf_b_k_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.b_K")
                hf_b_v_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.b_V")

                state_dict[hf_w_q_key] = ProcessWeights.convert_tensor_to_hf_format(
                    wq_tensor, f"blocks.{layer}.attn.W_Q", adapter, cfg, layer  # type: ignore[arg-type]
                )
                state_dict[hf_w_k_key] = ProcessWeights.convert_tensor_to_hf_format(
                    wk_tensor, f"blocks.{layer}.attn.W_K", adapter, cfg, layer  # type: ignore[arg-type]
                )
                state_dict[hf_w_v_key] = ProcessWeights.convert_tensor_to_hf_format(
                    wv_tensor, f"blocks.{layer}.attn.W_V", adapter, cfg, layer  # type: ignore[arg-type]
                )
                state_dict[hf_b_q_key] = ProcessWeights.convert_tensor_to_hf_format(
                    bq_tensor, f"blocks.{layer}.attn.b_Q", adapter, cfg, layer  # type: ignore[arg-type]
                )
                state_dict[hf_b_k_key] = ProcessWeights.convert_tensor_to_hf_format(
                    bk_tensor, f"blocks.{layer}.attn.b_K", adapter, cfg, layer  # type: ignore[arg-type]
                )
                state_dict[hf_b_v_key] = ProcessWeights.convert_tensor_to_hf_format(
                    bv_tensor, f"blocks.{layer}.attn.b_V", adapter, cfg, layer  # type: ignore[arg-type]
                )
        else:
            # Store directly (TransformerLens format)
            state_dict[keys["W_Q"]] = wq_tensor  # type: ignore[assignment]
            state_dict[keys["W_K"]] = wk_tensor  # type: ignore[assignment]
            state_dict[keys["W_V"]] = wv_tensor  # type: ignore[assignment]
            state_dict[keys["b_Q"]] = bq_tensor  # type: ignore[assignment]
            state_dict[keys["b_K"]] = bk_tensor  # type: ignore[assignment]
            state_dict[keys["b_V"]] = bv_tensor  # type: ignore[assignment]

    @staticmethod
    def _detect_unembed_format(state_dict: Dict[str, torch.Tensor], adapter) -> tuple[bool, bool]:
        """Detect whether state_dict uses TransformerLens or HuggingFace format for unembed parameters.

        Args:
            state_dict: The state dictionary to check
            adapter: Optional adapter for key translation

        Returns:
            Tuple of (uses_tl_format, uses_hf_format)
        """
        # Sample keys to check format
        tl_key_sample = "unembed.W_U"
        hf_key_sample = ProcessWeights._get_param_key(tl_key_sample, adapter) if adapter else None

        uses_tl_format = tl_key_sample in state_dict
        uses_hf_format = bool(adapter and hf_key_sample and hf_key_sample in state_dict)

        return uses_tl_format, uses_hf_format

    @staticmethod
    def _fold_unembed_layer_norm(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        fold_biases: bool,
        center_weights: bool,
        adapter,
    ) -> None:
        """Fold LayerNorm into unembedding layer.

        Args:
            state_dict: The state dictionary to process (modified in place)
            cfg: Model configuration object
            fold_biases: Whether to fold LayerNorm biases
            center_weights: Whether to center weights after folding
            adapter: Optional architecture adapter for parameter key translation
        """
        # Determine the actual format of the state_dict to avoid key mismatch
        uses_tl_format, uses_hf_format = ProcessWeights._detect_unembed_format(state_dict, adapter)

        # Get parameter keys based on format detection
        if uses_tl_format and not uses_hf_format:
            # State dict is in TransformerLens format - use TL keys directly
            unembed_b_U_key = "unembed.b_U"
            unembed_W_U_key = "unembed.W_U"
            ln_final_b_key = "ln_final.b"
            ln_final_w_key = "ln_final.w"
        elif adapter and uses_hf_format and not uses_tl_format:
            # State dict is in HuggingFace format - use adapter translation
            unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
            unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
            ln_final_b_key = ProcessWeights._get_param_key("ln_final.b", adapter)
            ln_final_w_key = ProcessWeights._get_param_key("ln_final.w", adapter)
        else:
            # Fallback: prefer TL format if possible, otherwise use adapter translation
            if uses_tl_format:
                unembed_b_U_key = "unembed.b_U"
                unembed_W_U_key = "unembed.W_U"
                ln_final_b_key = "ln_final.b"
                ln_final_w_key = "ln_final.w"
            else:
                unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
                unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
                ln_final_b_key = ProcessWeights._get_param_key("ln_final.b", adapter)
                ln_final_w_key = ProcessWeights._get_param_key("ln_final.w", adapter)

        # Check if unembedding bias actually exists (some models like GPT-2 don't have it)
        has_unembed_bias = unembed_b_U_key in state_dict

        # Note: final_rms bias folding is handled separately - not included in this function

        # Generalized layer norm folding for unembedding
        unembed_weight = state_dict[unembed_W_U_key]
        ln_weight = state_dict[ln_final_w_key]

        # Handle different tensor shapes (TransformerLens vs HuggingFace format)
        if len(unembed_weight.shape) == 2 and len(ln_weight.shape) == 1:
            # Check if we need to transpose for proper broadcasting
            if unembed_weight.shape[1] == ln_weight.shape[0]:
                # HuggingFace format: [vocab_size, d_model] * [d_model] -> [vocab_size, d_model]
                state_dict[unembed_W_U_key] = unembed_weight * ln_weight[None, :]
            elif unembed_weight.shape[0] == ln_weight.shape[0]:
                # TransformerLens format: [d_model, vocab_size] * [d_model] -> [d_model, vocab_size]
                state_dict[unembed_W_U_key] = unembed_weight * ln_weight[:, None]
            else:
                raise ValueError(
                    f"Cannot broadcast unembedding weight {unembed_weight.shape} with layer norm weight {ln_weight.shape}"
                )
        else:
            raise ValueError(
                f"Unexpected tensor shapes: unembedding {unembed_weight.shape}, layer norm {ln_weight.shape}"
            )
        if ln_final_w_key in state_dict:
            del state_dict[ln_final_w_key]

        if center_weights:
            # Center the weights that read in from the LayerNormPre
            unembed_weight = state_dict[unembed_W_U_key]
            if len(unembed_weight.shape) == 2:
                if unembed_weight.shape[0] > unembed_weight.shape[1]:
                    # TransformerLens format: [d_model, vocab_size] - center along d_model
                    state_dict[unembed_W_U_key] -= einops.reduce(
                        unembed_weight, "d_model d_vocab -> 1 d_vocab", "mean"
                    )
                else:
                    # HuggingFace format: [vocab_size, d_model] - center along d_model
                    state_dict[unembed_W_U_key] -= einops.reduce(
                        unembed_weight, "vocab_size d_model -> vocab_size 1", "mean"
                    )
            else:
                raise ValueError(f"Unexpected unembedding weight shape: {unembed_weight.shape}")

    @staticmethod
    def _fold_final_rms_bias(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        fold_biases: bool,
        adapter,
    ) -> None:
        """Fold final RMS bias into unembedding (separate from regular unembed folding).

        Args:
            state_dict: The state dictionary to process (modified in place)
            cfg: Model configuration object
            fold_biases: Whether to fold LayerNorm biases
            adapter: Optional architecture adapter for parameter key translation
        """
        # Determine the actual format of the state_dict to avoid key mismatch
        uses_tl_format, uses_hf_format = ProcessWeights._detect_unembed_format(state_dict, adapter)

        # Get parameter keys based on format detection
        if uses_tl_format and not uses_hf_format:
            # State dict is in TransformerLens format - use TL keys directly
            unembed_b_U_key = "unembed.b_U"
            unembed_W_U_key = "unembed.W_U"
            ln_final_b_key = "ln_final.b"
        elif adapter and uses_hf_format and not uses_tl_format:
            # State dict is in HuggingFace format - use adapter translation
            unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
            unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
            ln_final_b_key = ProcessWeights._get_param_key("ln_final.b", adapter)
        else:
            # Fallback: prefer TL format if possible, otherwise use adapter translation
            if uses_tl_format:
                unembed_b_U_key = "unembed.b_U"
                unembed_W_U_key = "unembed.W_U"
                ln_final_b_key = "ln_final.b"
            else:
                unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
                unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
                ln_final_b_key = ProcessWeights._get_param_key("ln_final.b", adapter)

        # Check if unembedding bias actually exists (some models like GPT-2 don't have it)
        has_unembed_bias = unembed_b_U_key in state_dict

        if not getattr(cfg, "final_rms", False) and fold_biases and has_unembed_bias:
            # Dumb bug from my old SoLU training code, some models have RMSNorm instead of LayerNorm
            # pre unembed.
            unembed_weight = state_dict[unembed_W_U_key]
            ln_bias = state_dict[ln_final_b_key]

            # Handle different tensor shapes for bias folding
            if len(unembed_weight.shape) == 2 and len(ln_bias.shape) == 1:
                if unembed_weight.shape[1] == ln_bias.shape[0]:
                    # HuggingFace format: [vocab_size, d_model] * [d_model] -> sum over d_model
                    bias_contribution = (unembed_weight * ln_bias[None, :]).sum(dim=-1)
                elif unembed_weight.shape[0] == ln_bias.shape[0]:
                    # TransformerLens format: [d_model, vocab_size] * [d_model] -> sum over d_model
                    bias_contribution = (unembed_weight * ln_bias[:, None]).sum(dim=-2)
                else:
                    raise ValueError(
                        f"Cannot broadcast unembedding weight {unembed_weight.shape} with layer norm bias {ln_bias.shape}"
                    )
            else:
                raise ValueError(
                    f"Unexpected tensor shapes: unembedding {unembed_weight.shape}, layer norm bias {ln_bias.shape}"
                )

            # TODO this is causing slight divergence - FIXED
            state_dict[unembed_b_U_key] = state_dict[unembed_b_U_key] + bias_contribution
            if ln_final_b_key in state_dict:
                del state_dict[ln_final_b_key]

    @staticmethod
    def fold_layer_norm(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        fold_biases: bool = True,
        center_weights: bool = True,
        adapter=None,
    ) -> Dict[str, torch.Tensor]:
        """Fold Layer Norm. Can also be used to fold RMS Norm, when fold_biases and center_weights are set to False.

        Takes in a state dict from a pretrained model, formatted to be consistent with
        HookedTransformer but with LayerNorm weights and biases. Folds these into the neighbouring
        weights. See further_comments.md for more details.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of pretrained model.
            cfg: Model configuration object with n_layers, n_key_value_heads, etc.
            fold_biases (bool): Enables folding of LN biases. Should be disabled when RMS Norm is used.
            center_weights (bool): Enables the centering of weights after folding in LN. Should be disabled when RMS Norm is used.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with LayerNorm folded into linear layers.
        """
        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        # Models that use Grouped Query Attention (Only Mistral at the time of writing) prefix their K/V weights and
        # biases with an underscore in order to distinguish them, but folding the LN into them still works the same,
        # so we just add the underscore if GQA is used (i.e. if `cfg.n_key_value_heads is specified`).
        gqa = "" if getattr(cfg, "n_key_value_heads", None) is None else "_"

        for l in range(cfg.n_layers):
            ProcessWeights._fold_layer(
                state_dict, cfg, l, fold_biases, center_weights, adapter, gqa
            )

        # Fold final RMS bias into unembedding (separate from regular unembed folding)
        ProcessWeights._fold_final_rms_bias(state_dict, cfg, fold_biases, adapter)

        # Fold ln_final into Unembed
        ProcessWeights._fold_unembed_layer_norm(
            state_dict, cfg, fold_biases, center_weights, adapter
        )

        return state_dict

    @staticmethod
    def center_writing_weights(
        state_dict: Dict[str, torch.Tensor], cfg, adapter=None
    ) -> Dict[str, torch.Tensor]:
        """Center Writing Weights.

        Centers the weights of the model that write to the residual stream - W_out, W_E, W_pos and
        W_out. This is done by subtracting the mean of the weights from the weights themselves. This
        is done in-place. See fold_layer_norm for more details.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            cfg: Model configuration object.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with centered writing weights.
        """
        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        # Determine the actual format of the state_dict to avoid key mismatch
        layer = 0  # Use layer 0 for format detection
        uses_tl_format, uses_hf_format = ProcessWeights._detect_state_dict_format(
            state_dict, layer, adapter
        )

        # Get parameter keys based on format detection
        if uses_tl_format and not uses_hf_format:
            # State dict is in TransformerLens format - use TL keys directly
            embed_W_E_key = "embed.W_E"
            pos_embed_W_pos_key = "pos_embed.W_pos"
        elif adapter and uses_hf_format and not uses_tl_format:
            # State dict is in HuggingFace format - use adapter translation
            embed_W_E_key = ProcessWeights._get_param_key("embed.W_E", adapter)
            pos_embed_W_pos_key = ProcessWeights._get_param_key("pos_embed.W_pos", adapter)
        else:
            # Fallback: prefer TL format if possible, otherwise use adapter translation
            if uses_tl_format:
                embed_W_E_key = "embed.W_E"
                pos_embed_W_pos_key = "pos_embed.W_pos"
            else:
                embed_W_E_key = ProcessWeights._get_param_key("embed.W_E", adapter)
                pos_embed_W_pos_key = ProcessWeights._get_param_key("pos_embed.W_pos", adapter)

        # Validate that the embedding key exists before accessing it
        if embed_W_E_key not in state_dict:
            raise KeyError(
                f"Expected embedding key '{embed_W_E_key}' not found in state_dict. "
                f"Available keys: {list(state_dict.keys())[:10]}..."
            )

        state_dict[embed_W_E_key] = state_dict[embed_W_E_key] - state_dict[embed_W_E_key].mean(
            -1, keepdim=True
        )
        if getattr(cfg, "positional_embedding_type", "standard") != "rotary":
            # Validate that the positional embedding key exists before accessing it
            if pos_embed_W_pos_key not in state_dict:
                raise KeyError(
                    f"Expected positional embedding key '{pos_embed_W_pos_key}' not found in state_dict. "
                    f"Available keys: {list(state_dict.keys())[:10]}..."
                )
            state_dict[pos_embed_W_pos_key] = state_dict[pos_embed_W_pos_key] - state_dict[
                pos_embed_W_pos_key
            ].mean(-1, keepdim=True)

        for l in range(cfg.n_layers):
            # Get parameter keys for this layer based on format detection
            if uses_tl_format and not uses_hf_format:
                # State dict is in TransformerLens format - use TL keys directly
                attn_W_O_key = f"blocks.{l}.attn.W_O"
                attn_b_O_key = f"blocks.{l}.attn.b_O"
                mlp_W_out_key = f"blocks.{l}.mlp.W_out"
                mlp_b_out_key = f"blocks.{l}.mlp.b_out"
            elif adapter and uses_hf_format and not uses_tl_format:
                # State dict is in HuggingFace format - use adapter translation
                attn_W_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_O", adapter)
                attn_b_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_O", adapter)
                mlp_W_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.W_out", adapter)
                mlp_b_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.b_out", adapter)
            else:
                # Fallback: prefer TL format if possible, otherwise use adapter translation
                if uses_tl_format:
                    attn_W_O_key = f"blocks.{l}.attn.W_O"
                    attn_b_O_key = f"blocks.{l}.attn.b_O"
                    mlp_W_out_key = f"blocks.{l}.mlp.W_out"
                    mlp_b_out_key = f"blocks.{l}.mlp.b_out"
                else:
                    attn_W_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_O", adapter)
                    attn_b_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_O", adapter)
                    mlp_W_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.W_out", adapter)
                    mlp_b_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.b_out", adapter)

            # Validate that attention keys exist before accessing them
            if attn_W_O_key not in state_dict:
                raise KeyError(
                    f"Expected attention W_O key '{attn_W_O_key}' not found in state_dict for layer {l}. "
                    f"Available keys: {list(state_dict.keys())[:10]}..."
                )
            if attn_b_O_key not in state_dict:
                raise KeyError(
                    f"Expected attention b_O key '{attn_b_O_key}' not found in state_dict for layer {l}. "
                    f"Available keys: {list(state_dict.keys())[:10]}..."
                )

            state_dict[attn_W_O_key] = state_dict[attn_W_O_key] - state_dict[attn_W_O_key].mean(
                -1, keepdim=True
            )  # W_O is [head_index, d_model, d_head]
            state_dict[attn_b_O_key] = (
                state_dict[attn_b_O_key] - state_dict[attn_b_O_key].mean()
            )  # b_O is [d_model]
            if not getattr(cfg, "attn_only", False):
                # Validate that MLP keys exist before accessing them
                if mlp_W_out_key not in state_dict:
                    raise KeyError(
                        f"Expected MLP W_out key '{mlp_W_out_key}' not found in state_dict for layer {l}. "
                        f"Available keys: {list(state_dict.keys())[:10]}..."
                    )
                if mlp_b_out_key not in state_dict:
                    raise KeyError(
                        f"Expected MLP b_out key '{mlp_b_out_key}' not found in state_dict for layer {l}. "
                        f"Available keys: {list(state_dict.keys())[:10]}..."
                    )

                state_dict[mlp_W_out_key] = state_dict[mlp_W_out_key] - state_dict[
                    mlp_W_out_key
                ].mean(-1, keepdim=True)
                state_dict[mlp_b_out_key] = (
                    state_dict[mlp_b_out_key] - state_dict[mlp_b_out_key].mean()
                )
        return state_dict

    @staticmethod
    def center_unembed(
        state_dict: Dict[str, torch.Tensor], adapter=None
    ) -> Dict[str, torch.Tensor]:
        """Center the unembedding weights W_U.

        This is done by subtracting the mean of the weights from the weights themselves. This is
        done in-place. As softmax is translation invariant, this changes the logits but not the log
        probs, and makes the model logits (slightly) more interpretable - when trying to understand
        how components contribute to the logits, we'll be less misled by components that just add
        something to every logit.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with centered unembedding weights.
        """
        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        # Determine the actual format of the state_dict to avoid key mismatch
        uses_tl_format, uses_hf_format = ProcessWeights._detect_unembed_format(state_dict, adapter)

        # Get parameter keys based on format detection
        if uses_tl_format and not uses_hf_format:
            # State dict is in TransformerLens format - use TL keys directly
            unembed_W_U_key = "unembed.W_U"
            unembed_b_U_key = "unembed.b_U"
        elif adapter and uses_hf_format and not uses_tl_format:
            # State dict is in HuggingFace format - use adapter translation
            unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
            unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
        else:
            # Fallback: prefer TL format if possible, otherwise use adapter translation
            if uses_tl_format:
                unembed_W_U_key = "unembed.W_U"
                unembed_b_U_key = "unembed.b_U"
            else:
                unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
                unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)

        # Validate that the unembedding weight key exists before accessing it
        if unembed_W_U_key not in state_dict:
            raise KeyError(
                f"Expected unembedding weight key '{unembed_W_U_key}' not found in state_dict. "
                f"Available keys: {list(state_dict.keys())[:10]}..."
            )

        state_dict[unembed_W_U_key] = state_dict[unembed_W_U_key] - state_dict[
            unembed_W_U_key
        ].mean(-1, keepdim=True)

        # Only center bias if it exists (some models like GPT-2 don't have unembedding bias)
        if unembed_b_U_key in state_dict:
            state_dict[unembed_b_U_key] = (
                state_dict[unembed_b_U_key] - state_dict[unembed_b_U_key].mean()
            )
        return state_dict

    @staticmethod
    def fold_value_biases(
        state_dict: Dict[str, torch.Tensor], cfg, adapter=None
    ) -> Dict[str, torch.Tensor]:
        """Fold the value biases into the output bias.

        Because attention patterns add up to 1, the value biases always have a constant effect on a
        head's output. Further, as the outputs of each head in a layer add together, each head's
        value bias has a constant effect on the *layer's* output, which can make it harder to
        interpret the effect of any given head, and it doesn't matter which head a bias is
        associated with. We can factor this all into a single output bias to the layer, and make it
        easier to interpret the head's output. Formally, we take b_O_new = b_O_original +
        sum_head(b_V_head @ W_O_head).

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            cfg: Model configuration object.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with value biases folded into output bias.
        """
        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        # Determine the actual format of the state_dict to avoid key mismatch
        layer = 0  # Use layer 0 for format detection
        uses_tl_format, uses_hf_format = ProcessWeights._detect_state_dict_format(
            state_dict, layer, adapter
        )

        for layer in range(cfg.n_layers):
            # Get parameter keys for this layer based on format detection
            if uses_tl_format and not uses_hf_format:
                # State dict is in TransformerLens format - use TL keys directly
                if getattr(cfg, "n_key_value_heads", None) is None:
                    b_V_key = f"blocks.{layer}.attn.b_V"
                else:
                    b_V_key = f"blocks.{layer}.attn._b_V"
                W_O_key = f"blocks.{layer}.attn.W_O"
                b_O_key = f"blocks.{layer}.attn.b_O"
            elif adapter and uses_hf_format and not uses_tl_format:
                # State dict is in HuggingFace format - use adapter translation
                if getattr(cfg, "n_key_value_heads", None) is None:
                    b_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_V", adapter)
                else:
                    b_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn._b_V", adapter)
                W_O_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.W_O", adapter)
                b_O_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_O", adapter)
            else:
                # Fallback: prefer TL format if possible, otherwise use adapter translation
                if uses_tl_format:
                    if getattr(cfg, "n_key_value_heads", None) is None:
                        b_V_key = f"blocks.{layer}.attn.b_V"
                    else:
                        b_V_key = f"blocks.{layer}.attn._b_V"
                    W_O_key = f"blocks.{layer}.attn.W_O"
                    b_O_key = f"blocks.{layer}.attn.b_O"
                else:
                    if getattr(cfg, "n_key_value_heads", None) is None:
                        b_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_V", adapter)
                    else:
                        b_V_key = ProcessWeights._get_param_key(
                            f"blocks.{layer}.attn._b_V", adapter
                        )
                    W_O_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.W_O", adapter)
                    b_O_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_O", adapter)

            # Check if we have combined QKV format (HuggingFace) or separate format (TransformerLens)
            if b_V_key in state_dict:
                b_V = state_dict[b_V_key]
                W_O = state_dict[W_O_key]
                b_O_original = state_dict[b_O_key]

                # Handle different tensor formats
                if len(b_V.shape) == 1 and len(W_O.shape) == 2:
                    # HuggingFace format: combined QKV bias [3 * n_heads * d_head], W_O [d_model, d_model]
                    n_heads = cfg.n_heads
                    d_head = cfg.d_head
                    d_model = cfg.d_model

                    # Extract just the V bias from the combined QKV bias
                    # Combined bias is [Q_bias, K_bias, V_bias] where each is [n_heads * d_head]
                    v_bias_start = 2 * n_heads * d_head  # Start of V bias
                    v_bias_end = 3 * n_heads * d_head  # End of V bias
                    b_V_only = b_V[v_bias_start:v_bias_end]  # [n_heads * d_head]

                    # Reshape for computation: [n_heads * d_head] -> [n_heads, d_head]
                    b_V_reshaped = b_V_only.reshape(n_heads, d_head)

                    # W_O is [d_model, d_model], we need to reshape it to [n_heads, d_head, d_model]
                    # W_O represents the output projection, so we need to split it by heads
                    W_O_reshaped = W_O.T.reshape(n_heads, d_head, d_model)

                    # Compute the folded bias: sum over heads and d_head dimensions
                    folded_b_O = b_O_original + (b_V_reshaped[:, :, None] * W_O_reshaped).sum(
                        [0, 1]
                    )

                    # Zero out the V bias in the combined QKV bias
                    new_b_V = b_V.clone()
                    new_b_V[v_bias_start:v_bias_end] = 0
                    state_dict[b_V_key] = new_b_V

                elif len(b_V.shape) == 2 and len(W_O.shape) == 3:
                    # TransformerLens format: separate V bias [n_heads, d_head], W_O [n_heads, d_head, d_model]
                    if getattr(cfg, "n_key_value_heads", None) is not None:
                        b_V = torch.repeat_interleave(
                            b_V, dim=0, repeats=cfg.n_heads // cfg.n_key_value_heads
                        )

                    folded_b_O = b_O_original + (b_V[:, :, None] * W_O).sum([0, 1])
                    state_dict[b_V_key] = torch.zeros_like(b_V)
                elif len(b_V.shape) == 2 and len(W_O.shape) == 2:
                    # Mixed format: b_V in TransformerLens format [n_heads, d_head], W_O in HuggingFace format [d_model, d_model]
                    n_heads = cfg.n_heads
                    d_head = cfg.d_head
                    d_model = cfg.d_model

                    if getattr(cfg, "n_key_value_heads", None) is not None:
                        b_V = torch.repeat_interleave(
                            b_V, dim=0, repeats=cfg.n_heads // cfg.n_key_value_heads
                        )

                    # Convert W_O from HuggingFace format [d_model, d_model] to TransformerLens format [n_heads, d_head, d_model]
                    W_O_reshaped = W_O.T.reshape(n_heads, d_head, d_model)

                    # Compute the folded bias: sum over heads and d_head dimensions
                    folded_b_O = b_O_original + (b_V[:, :, None] * W_O_reshaped).sum([0, 1])
                    state_dict[b_V_key] = torch.zeros_like(b_V)
                else:
                    raise ValueError(f"Unexpected tensor shapes: b_V {b_V.shape}, W_O {W_O.shape}")

                state_dict[b_O_key] = folded_b_O

        return state_dict

    @staticmethod
    def refactor_factored_attn_matrices(
        state_dict: Dict[str, torch.Tensor], cfg, adapter=None
    ) -> Dict[str, torch.Tensor]:
        """Experimental method for managing queries, keys and values.

        As argued in [A Mathematical Framework for Transformer
        Circuits](https://transformer-circuits.pub/2021/framework/index.html), queries, keys and
        values are somewhat arbitrary intermediate terms when computing with the low rank factored
        matrices W_QK = W_Q @ W_K.T and W_OV = W_V @ W_O, and these matrices are the only thing
        determining head behaviour. But there are many ways to find a low rank factorization to a
        given matrix, and hopefully some of these are more interpretable than others! This method is
        one attempt, which makes all of the matrices have orthogonal rows or columns, W_O into a
        rotation and W_Q and W_K having the nth column in each having the same norm. The formula is
        $W_V = U @ S,W_O=Vh.T,W_Q=U@S.sqrt(),W_K=Vh@S.sqrt()$.

        More details:

        If W_OV = U @ S @ Vh.T in its singular value decomposition, (where S is in R^d_head not
        R^d_model, as W_OV is low rank), W_OV = (U @ S) @ (Vh.T) is an equivalent low rank
        factorisation, where rows/columns of each matrix are orthogonal! So setting $W_V=US$ and
        $W_O=Vh.T$ works just as well. I *think* this is a more interpretable setup, because now
        $W_O$ is just a rotation, and doesn't change the norm, so $z$ has the same norm as the
        result of the head.

        For $W_QK = W_Q @ W_K.T$ we use the refactor $W_Q = U @ S.sqrt()$ and $W_K = Vh @ S.sqrt()$,
        which is also equivalent ($S==S.sqrt() @ S.sqrt()$ as $S$ is diagonal). Here we keep the
        matrices as having the same norm, since there's not an obvious asymmetry between the keys
        and queries.

        Biases are more fiddly to deal with. For OV it's pretty easy - we just need (x @ W_V + b_V)
        @ W_O + b_O to be preserved, so we can set b_V' = 0. and b_O' = b_V @ W_O + b_O (note that
        b_V in R^{head_index x d_head} while b_O in R^{d_model}, so we need to sum b_V @ W_O along
        the head_index dimension too).

        For QK it's messy - we need to preserve the bilinear form of (x @ W_Q + b_Q) * (y @ W_K +
        b_K), which is fairly messy. To deal with the biases, we concatenate them to W_Q and W_K to
        simulate a d_model+1 dimensional input (whose final coordinate is always 1), do the SVD
        factorization on this effective matrix, then separate out into final weights and biases.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            cfg: Model configuration object.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with refactored attention matrices.
        """
        assert (
            getattr(cfg, "positional_embedding_type", "standard") != "rotary"
        ), "You can't refactor the QK circuit when using rotary embeddings (as the QK matrix depends on the position of the query and key)"

        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        # Determine the actual format of the state_dict to avoid key mismatch
        layer = 0  # Use layer 0 for format detection
        uses_tl_format, uses_hf_format = ProcessWeights._detect_state_dict_format(
            state_dict, layer, adapter
        )

        for l in range(cfg.n_layers):
            # Get parameter keys for this layer based on format detection
            if uses_tl_format and not uses_hf_format:
                # State dict is in TransformerLens format - use TL keys directly
                W_Q_key = f"blocks.{l}.attn.W_Q"
                b_Q_key = f"blocks.{l}.attn.b_Q"
                W_K_key = f"blocks.{l}.attn.W_K"
                b_K_key = f"blocks.{l}.attn.b_K"
                W_V_key = f"blocks.{l}.attn.W_V"
                W_O_key = f"blocks.{l}.attn.W_O"
                b_V_key = f"blocks.{l}.attn.b_V"
                b_O_key = f"blocks.{l}.attn.b_O"
            elif adapter and uses_hf_format and not uses_tl_format:
                # State dict is in HuggingFace format - use adapter translation
                W_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_Q", adapter)
                b_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_Q", adapter)
                W_K_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_K", adapter)
                b_K_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_K", adapter)
                W_V_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_V", adapter)
                W_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_O", adapter)
                b_V_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_V", adapter)
                b_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_O", adapter)
            else:
                # Fallback: prefer TL format if possible, otherwise use adapter translation
                if uses_tl_format:
                    W_Q_key = f"blocks.{l}.attn.W_Q"
                    b_Q_key = f"blocks.{l}.attn.b_Q"
                    W_K_key = f"blocks.{l}.attn.W_K"
                    b_K_key = f"blocks.{l}.attn.b_K"
                    W_V_key = f"blocks.{l}.attn.W_V"
                    W_O_key = f"blocks.{l}.attn.W_O"
                    b_V_key = f"blocks.{l}.attn.b_V"
                    b_O_key = f"blocks.{l}.attn.b_O"
                else:
                    W_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_Q", adapter)
                    b_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_Q", adapter)
                    W_K_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_K", adapter)
                    b_K_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_K", adapter)
                    W_V_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_V", adapter)
                    W_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_O", adapter)
                    b_V_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_V", adapter)
                    b_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_O", adapter)

            # W_QK = W_Q @ W_K.T
            # Concatenate biases to make a d_model+1 input dimension
            W_Q_eff = torch.cat(
                [
                    state_dict[W_Q_key],
                    state_dict[b_Q_key][:, None, :],
                ],
                dim=1,
            )
            W_K_eff = torch.cat(
                [
                    state_dict[W_K_key],
                    state_dict[b_K_key][:, None, :],
                ],
                dim=1,
            )

            W_Q_eff_even, W_K_eff_even_T = (
                FactoredMatrix(W_Q_eff, W_K_eff.transpose(-1, -2)).make_even().pair
            )
            W_K_eff_even = W_K_eff_even_T.transpose(-1, -2)

            state_dict[W_Q_key] = W_Q_eff_even[:, :-1, :]
            state_dict[b_Q_key] = W_Q_eff_even[:, -1, :]
            state_dict[W_K_key] = W_K_eff_even[:, :-1, :]
            state_dict[b_K_key] = W_K_eff_even[:, -1, :]

            # W_OV = W_V @ W_O
            W_V = state_dict[W_V_key]
            W_O = state_dict[W_O_key]

            # Factors the bias to be consistent.
            b_V = state_dict[b_V_key]
            b_O = state_dict[b_O_key]

            # Add singleton dimension for broadcasting
            b_V_expanded = einops.rearrange(b_V, "head_index d_head -> head_index d_head 1")

            # Element-wise multiplication of b_V and W_O
            b_V_times_W_O = b_V_expanded * W_O

            # Sum over d_head and head_index dimensions
            b_V_contribution = b_V_times_W_O.sum(1).sum(0)

            effective_bias = b_O + b_V_contribution
            state_dict[b_V_key] = torch.zeros_like(b_V)
            state_dict[b_O_key] = effective_bias

            # Helper class to efficiently deal with low rank factored matrices.
            W_OV = FactoredMatrix(W_V, W_O)
            U, S, Vh = W_OV.svd()
            state_dict[W_V_key] = U @ S.diag_embed()
            state_dict[W_O_key] = utils.transpose(Vh)

        return state_dict

    @staticmethod
    def process_weights(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
        adapter=None,
    ) -> Dict[str, torch.Tensor]:
        """Apply all weight processing transformations in the correct order.

        This is a convenience function that applies all the weight processing steps
        in the same order as HookedTransformer.load_and_process_state_dict().

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            cfg: Model configuration object.
            fold_ln (bool): Whether to fold LayerNorm weights into subsequent layers.
            center_writing_weights (bool): Whether to center weights writing to residual stream.
            center_unembed (bool): Whether to center unembedding weights.
            fold_value_biases (bool): Whether to fold value biases into output bias.
            refactor_factored_attn_matrices (bool): Whether to refactor attention matrices.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Fully processed state dict.
        """
        processed_dict = state_dict.copy()

        if fold_ln:
            if getattr(cfg, "num_experts", None) and cfg.num_experts > 1:
                # Skip for MoE models
                pass
            elif getattr(cfg, "normalization_type", "LN") in ["LN", "LNPre"]:
                processed_dict = ProcessWeights.fold_layer_norm(
                    processed_dict, cfg, fold_biases=True, center_weights=True, adapter=adapter
                )
            elif getattr(cfg, "normalization_type", "LN") in ["RMS", "RMSPre"]:
                processed_dict = ProcessWeights.fold_layer_norm(
                    processed_dict, cfg, fold_biases=False, center_weights=False, adapter=adapter
                )

        if center_writing_weights:
            if getattr(cfg, "normalization_type", "LN") in ["LN", "LNPre"] and not getattr(
                cfg, "final_rms", False
            ):
                processed_dict = ProcessWeights.center_writing_weights(
                    processed_dict, cfg, adapter=adapter
                )

        if center_unembed:
            processed_dict = ProcessWeights.center_unembed(processed_dict, adapter=adapter)

        if fold_value_biases:
            processed_dict = ProcessWeights.fold_value_biases(processed_dict, cfg, adapter=adapter)

        if refactor_factored_attn_matrices:
            processed_dict = ProcessWeights.refactor_factored_attn_matrices(
                processed_dict, cfg, adapter=adapter
            )

        return processed_dict

    @staticmethod
    def extract_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extract state dictionary from an nn.Module, cleaning up any _original_component references.

        This function extracts the state dictionary from a PyTorch model and removes any
        _original_component suffixes that might be present in bridge models.

        Args:
            model: The PyTorch model to extract state dict from

        Returns:
            Dict[str, torch.Tensor]: Cleaned state dictionary without _original_component references
        """
        # If the model has a custom state_dict method (like TransformerBridge), use it directly
        if (
            hasattr(model, "state_dict")
            and hasattr(model.__class__, "state_dict")
            and model.__class__.state_dict != nn.Module.state_dict
        ):
            return model.state_dict()

        # Otherwise, manually clean up _original_component suffixes
        state_dict = model.state_dict()
        cleaned_state_dict = {}
        for key, tensor in state_dict.items():
            clean_key = key.replace("._original_component", "")
            cleaned_state_dict[clean_key] = tensor.clone()

        return cleaned_state_dict

    @staticmethod
    def convert_hf_to_tl_format(hf_model, architecture_adapter):
        """Convert HuggingFace format state dict to TransformerLens format using architecture adapter.

        This method uses the architecture adapter's conversion rules to convert weights
        from HuggingFace format to TransformerLens format without creating fake model structures.

        Args:
            hf_model: The original HuggingFace nn.Module model to convert from
            architecture_adapter: Architecture adapter with conversion rules

        Returns:
            State dict in TransformerLens format
        """
        if (
            not hasattr(architecture_adapter, "conversion_rules")
            or architecture_adapter.conversion_rules is None
        ):
            raise ValueError("Architecture adapter must have conversion_rules set")

        # Get the HF state dict
        hf_state_dict = hf_model.state_dict()

        # Extract target keys from component mapping via conversion rules instead of hardcoded list
        target_keys = ProcessWeights._extract_tl_keys_from_conversion_rules(architecture_adapter)

        # Apply conversion rules from architecture adapter for only the target keys
        tl_state_dict: Dict[str, torch.Tensor] = {}
        conversion_rules = architecture_adapter.conversion_rules.fields

        print(
            f"Converting {len(hf_state_dict)} HF weights to {len(target_keys)} target TL weights..."
        )

        for tl_key in target_keys:
            # Find matching conversion rule (may use template format)
            conversion_info = None
            layer_idx = None

            # Check for exact match first
            if tl_key in conversion_rules:
                conversion_info = conversion_rules[tl_key]
            else:
                # Check for template match (e.g., "blocks.5.attn.W_Q" matches "blocks.{i}.attn.W_Q")
                if "blocks." in tl_key:
                    parts = tl_key.split(".")
                    if len(parts) >= 2 and parts[0] == "blocks":
                        try:
                            layer_idx = int(parts[1])
                            # Create template key
                            template_key = tl_key.replace(f"blocks.{layer_idx}.", "blocks.{i}.")
                            if template_key in conversion_rules:
                                conversion_info = conversion_rules[template_key]
                        except ValueError:
                            pass

            if conversion_info is not None:
                ProcessWeights._convert_single_weight(
                    tl_key,
                    conversion_info,
                    hf_state_dict,
                    tl_state_dict,
                    layer_idx,
                    architecture_adapter,
                )
            else:
                print(f"Warning: No conversion rule found for target key: {tl_key}")

        print(f"Converted to {len(tl_state_dict)} TL weights")
        return tl_state_dict

    @staticmethod
    def _extract_tl_keys_from_conversion_rules(architecture_adapter):
        """Extract TransformerLens target keys by traversing the component mapping structure."""
        keys = []
        conversion_rules = architecture_adapter.conversion_rules.fields
        cfg = architecture_adapter.cfg

        # Helper function to recursively extract keys with proper template handling
        def _extract_keys_from_component(component, comp_name, parent_template_parts=None):
            """Extract keys from a component, handling list items dynamically."""
            extracted_keys = []

            # Build template parts for tracking list indices
            template_parts = parent_template_parts.copy() if parent_template_parts else []

            if component.is_list_item:
                # This component represents a list (like blocks, experts, etc.)
                # Get the count from the component itself
                count = component.get_list_size()

                # Track this as a template component
                template_parts.append((comp_name, "{i}"))

                # Expand for all indices
                for idx in range(count):
                    # Build the prefix with the actual index
                    prefix_parts = []
                    for part_name, part_template in template_parts:
                        if part_template == "{i}":
                            prefix_parts.append(f"{part_name}.{idx}")
                        else:
                            prefix_parts.append(part_name)
                    prefix = ".".join(prefix_parts) if prefix_parts else ""

                    # Get parameter names for this instance
                    param_keys = component.get_expected_parameter_names(prefix)
                    extracted_keys.extend(param_keys)
            else:
                # Regular component - build prefix from template parts
                if template_parts:
                    prefix_parts = [part[0] for part in template_parts]
                    prefix = ".".join(prefix_parts + [comp_name])
                else:
                    prefix = comp_name

                # Get parameter names
                param_keys = component.get_expected_parameter_names(prefix)
                extracted_keys.extend(param_keys)

            return extracted_keys

        # Process each component in the mapping
        component_mapping = architecture_adapter.component_mapping
        for comp_name, component in component_mapping.items():
            component_keys = _extract_keys_from_component(component, comp_name)
            keys.extend(component_keys)

        # Filter to only include keys that exist in conversion rules
        filtered_keys = []
        for key in keys:
            # Build template key by replacing indices with {i}
            template_key = key
            parts = key.split(".")

            # Look for numeric parts and replace with {i} to match template format
            template_parts = []
            for i, part in enumerate(parts):
                if (
                    i > 0
                    and parts[i - 1] in component_mapping
                    and component_mapping[parts[i - 1]].is_list_item
                ):
                    # Previous part was a list component, this should be an index
                    if part.isdigit():
                        template_parts.append("{i}")
                    else:
                        template_parts.append(part)
                else:
                    template_parts.append(part)

            if len(template_parts) > 0:
                # Reconstruct template key
                rebuilt_parts: list[str] = []
                for i, part in enumerate(parts):
                    if template_parts[i] == "{i}":
                        if i > 0:
                            rebuilt_parts[-1] = rebuilt_parts[-1] + ".{i}"
                    else:
                        rebuilt_parts.append(part)
                template_key = ".".join(rebuilt_parts)

            if key in conversion_rules or template_key in conversion_rules:
                filtered_keys.append(key)

        return sorted(filtered_keys)

    @staticmethod
    def _get_target_tl_keys(cfg):
        """Get the exact keys that convert_gpt2_weights produces."""
        keys = []

        # Global keys
        keys.extend(
            [
                "embed.W_E",
                "pos_embed.W_pos",
                "ln_final.w",
                "ln_final.b",
                "unembed.W_U",
            ]
        )

        # Layer-specific keys
        for layer_idx in range(cfg.n_layers):
            layer_keys = [
                f"blocks.{layer_idx}.ln1.w",
                f"blocks.{layer_idx}.ln1.b",
                f"blocks.{layer_idx}.attn.W_Q",
                f"blocks.{layer_idx}.attn.W_K",
                f"blocks.{layer_idx}.attn.W_V",
                f"blocks.{layer_idx}.attn.b_Q",
                f"blocks.{layer_idx}.attn.b_K",
                f"blocks.{layer_idx}.attn.b_V",
                f"blocks.{layer_idx}.attn.W_O",
                f"blocks.{layer_idx}.attn.b_O",
                f"blocks.{layer_idx}.ln2.w",
                f"blocks.{layer_idx}.ln2.b",
                f"blocks.{layer_idx}.mlp.W_in",
                f"blocks.{layer_idx}.mlp.b_in",
                f"blocks.{layer_idx}.mlp.W_out",
                f"blocks.{layer_idx}.mlp.b_out",
            ]
            keys.extend(layer_keys)

        return keys

    @staticmethod
    def _convert_single_weight(
        tl_key, conversion_info, hf_state_dict, tl_state_dict, layer_idx, architecture_adapter
    ):
        """Convert a single weight using the conversion rule."""
        # Handle different conversion_info formats
        if isinstance(conversion_info, str):
            # Simple string mapping
            hf_key = conversion_info
            if layer_idx is not None:
                hf_key = hf_key.format(i=layer_idx)

            if hf_key in hf_state_dict:
                tl_state_dict[tl_key] = hf_state_dict[hf_key].clone()

        elif isinstance(conversion_info, tuple) and len(conversion_info) == 2:
            # (hf_key, conversion_function) tuple
            hf_key_template, conversion_func = conversion_info
            hf_key = hf_key_template
            if layer_idx is not None:
                hf_key = hf_key.format(i=layer_idx)

            if hf_key in hf_state_dict:
                # Apply the conversion function
                original_weight = hf_state_dict[hf_key]
                converted_weight = conversion_func.handle_conversion(original_weight)
                tl_state_dict[tl_key] = converted_weight

        else:
            print(f"Warning: Unknown conversion format for {tl_key}: {conversion_info}")

    @staticmethod
    def _is_transformerlens_key(key_template):
        """Check if a key follows TransformerLens naming convention.

        TransformerLens keys use patterns like:
        - W_E, W_pos, W_Q, W_K, W_V, W_O, W_in, W_out, W_U
        - w, b (for layer norm)
        - b_Q, b_K, b_V, b_O, b_in, b_out, b_U (for biases)
        """
        # TransformerLens keys typically have W_ or b_ patterns, or specific patterns like .w, .b
        transformerlens_patterns = [".W_", ".b_", ".w", ".b", "W_E", "W_pos", "W_U"]

        return any(pattern in key_template for pattern in transformerlens_patterns)

    @staticmethod
    def convert_tl_to_hf_format(tl_state_dict, cfg):
        """Convert TransformerLens format state dict back to HuggingFace format.

        Args:
            tl_state_dict: State dict in TransformerLens format
            cfg: Model configuration object

        Returns:
            State dict in HuggingFace format
        """
        import torch

        hf_state_dict = {}

        # Convert embeddings
        if "embed.W_E" in tl_state_dict:
            hf_state_dict["transformer.wte.weight"] = tl_state_dict["embed.W_E"]
        if "pos_embed.W_pos" in tl_state_dict:
            hf_state_dict["transformer.wpe.weight"] = tl_state_dict["pos_embed.W_pos"]
        if "unembed.W_U" in tl_state_dict:
            hf_state_dict["lm_head.weight"] = tl_state_dict["unembed.W_U"].T

        # Convert final layer norm
        if "ln_final.w" in tl_state_dict:
            hf_state_dict["transformer.ln_f.weight"] = tl_state_dict["ln_final.w"]
        if "ln_final.b" in tl_state_dict:
            hf_state_dict["transformer.ln_f.bias"] = tl_state_dict["ln_final.b"]

        # Convert layers
        for layer_idx in range(cfg.n_layers):
            layer_prefix = f"blocks.{layer_idx}"
            hf_layer_prefix = f"transformer.h.{layer_idx}"

            # Layer norms
            if f"{layer_prefix}.ln1.w" in tl_state_dict:
                hf_state_dict[f"{hf_layer_prefix}.ln_1.weight"] = tl_state_dict[
                    f"{layer_prefix}.ln1.w"
                ]
            if f"{layer_prefix}.ln1.b" in tl_state_dict:
                hf_state_dict[f"{hf_layer_prefix}.ln_1.bias"] = tl_state_dict[
                    f"{layer_prefix}.ln1.b"
                ]
            if f"{layer_prefix}.ln2.w" in tl_state_dict:
                hf_state_dict[f"{hf_layer_prefix}.ln_2.weight"] = tl_state_dict[
                    f"{layer_prefix}.ln2.w"
                ]
            if f"{layer_prefix}.ln2.b" in tl_state_dict:
                hf_state_dict[f"{hf_layer_prefix}.ln_2.bias"] = tl_state_dict[
                    f"{layer_prefix}.ln2.b"
                ]

            # Attention weights - convert TL separated format to HF combined format
            if f"{layer_prefix}.attn.W_Q" in tl_state_dict:
                W_Q = tl_state_dict[f"{layer_prefix}.attn.W_Q"]  # [n_heads, d_model, d_head]
                W_K = tl_state_dict[f"{layer_prefix}.attn.W_K"]
                W_V = tl_state_dict[f"{layer_prefix}.attn.W_V"]

                # Reshape and combine into HF format
                # TL format: [n_heads, d_model, d_head] -> HF format: [d_model, n_heads * d_head]
                W_Q_flat = W_Q.permute(1, 0, 2).reshape(
                    W_Q.shape[1], -1
                )  # [d_model, n_heads * d_head]
                W_K_flat = W_K.permute(1, 0, 2).reshape(W_K.shape[1], -1)
                W_V_flat = W_V.permute(1, 0, 2).reshape(W_V.shape[1], -1)

                c_attn_weight = torch.cat(
                    [W_Q_flat, W_K_flat, W_V_flat], dim=1
                )  # [d_model, 3 * n_heads * d_head]
                hf_state_dict[f"{hf_layer_prefix}.attn.c_attn.weight"] = c_attn_weight

            if f"{layer_prefix}.attn.b_Q" in tl_state_dict:
                b_Q = tl_state_dict[f"{layer_prefix}.attn.b_Q"]  # [n_heads, d_head]
                b_K = tl_state_dict[f"{layer_prefix}.attn.b_K"]
                b_V = tl_state_dict[f"{layer_prefix}.attn.b_V"]

                # Flatten and combine
                b_Q_flat = b_Q.reshape(-1)
                b_K_flat = b_K.reshape(-1)
                b_V_flat = b_V.reshape(-1)

                c_attn_bias = torch.cat([b_Q_flat, b_K_flat, b_V_flat], dim=0)
                hf_state_dict[f"{hf_layer_prefix}.attn.c_attn.bias"] = c_attn_bias

            # Attention output projection
            if f"{layer_prefix}.attn.W_O" in tl_state_dict:
                W_O = tl_state_dict[f"{layer_prefix}.attn.W_O"]  # [n_heads, d_head, d_model]
                # TL format: [n_heads, d_head, d_model] -> HF format: [n_heads * d_head, d_model]
                W_O_flat = W_O.reshape(
                    W_O.shape[0] * W_O.shape[1], W_O.shape[2]
                )  # [n_heads * d_head, d_model]
                hf_state_dict[f"{hf_layer_prefix}.attn.c_proj.weight"] = W_O_flat

            if f"{layer_prefix}.attn.b_O" in tl_state_dict:
                hf_state_dict[f"{hf_layer_prefix}.attn.c_proj.bias"] = tl_state_dict[
                    f"{layer_prefix}.attn.b_O"
                ]

            # MLP weights
            if f"{layer_prefix}.mlp.W_in" in tl_state_dict:
                hf_state_dict[f"{hf_layer_prefix}.mlp.c_fc.weight"] = tl_state_dict[
                    f"{layer_prefix}.mlp.W_in"
                ]
            if f"{layer_prefix}.mlp.b_in" in tl_state_dict:
                hf_state_dict[f"{hf_layer_prefix}.mlp.c_fc.bias"] = tl_state_dict[
                    f"{layer_prefix}.mlp.b_in"
                ]
            if f"{layer_prefix}.mlp.W_out" in tl_state_dict:
                hf_state_dict[f"{hf_layer_prefix}.mlp.c_proj.weight"] = tl_state_dict[
                    f"{layer_prefix}.mlp.W_out"
                ]
            if f"{layer_prefix}.mlp.b_out" in tl_state_dict:
                hf_state_dict[f"{hf_layer_prefix}.mlp.c_proj.bias"] = tl_state_dict[
                    f"{layer_prefix}.mlp.b_out"
                ]

        return hf_state_dict

    @staticmethod
    def process_weights_with_format_conversion(
        hf_state_dict: Dict[str, torch.Tensor],
        cfg,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
        adapter=None,
    ) -> Dict[str, torch.Tensor]:
        """Apply weight processing with format conversion for bridge models.

        This method is specifically designed for TransformerBridge models that need
        to convert between HuggingFace and TransformerLens formats during processing.

        Args:
            hf_state_dict: State dict in HuggingFace format
            cfg: Model configuration object
            fold_ln: Whether to fold LayerNorm weights
            center_writing_weights: Whether to center weights writing to residual stream
            center_unembed: Whether to center unembedding weights
            fold_value_biases: Whether to fold value biases
            refactor_factored_attn_matrices: Whether to refactor attention matrices
            adapter: Optional architecture adapter (if provided, enables format conversion)

        Returns:
            State dict in HuggingFace format after processing
        """
        if adapter is not None:
            # Step 1: Convert HuggingFace format to TransformerLens format
            tl_state_dict = ProcessWeights.convert_hf_to_tl_format(hf_state_dict, cfg)

            # Step 2: Apply ProcessWeights processing to TL format state dict
            processed_tl_state_dict = ProcessWeights.process_weights(
                tl_state_dict,
                cfg,
                fold_ln=fold_ln,
                center_writing_weights=center_writing_weights,
                center_unembed=center_unembed,
                fold_value_biases=fold_value_biases,
                refactor_factored_attn_matrices=refactor_factored_attn_matrices,
                adapter=None,  # No adapter needed for TL format
            )

            # Step 3: Convert processed TL format back to HF format
            processed_hf_state_dict = ProcessWeights.convert_tl_to_hf_format(
                processed_tl_state_dict, cfg
            )

            return processed_hf_state_dict
        else:
            # No adapter provided, use standard processing
            return ProcessWeights.process_weights(
                hf_state_dict,
                cfg,
                fold_ln=fold_ln,
                center_writing_weights=center_writing_weights,
                center_unembed=center_unembed,
                fold_value_biases=fold_value_biases,
                refactor_factored_attn_matrices=refactor_factored_attn_matrices,
                adapter=adapter,
            )

    @staticmethod
    def apply_minimal_processing_offset(module, cfg):
        """Apply minimal offset to match HookedTransformer's processed behavior.

        Since HookedTransformer's processing has minimal effect (only 0.000011 difference),
        we apply a tiny offset to match this effect, including proper ablation behavior.

        Args:
            module: The PyTorch module to apply offsets to
            cfg: Model configuration object
        """
        import torch

        # Add a tiny offset to the token embedding to match HookedTransformer baseline
        if hasattr(module.transformer, "wte") and hasattr(module.transformer.wte, "weight"):
            baseline_offset = torch.full_like(module.transformer.wte.weight, 1e-5)
            module.transformer.wte.weight.data += baseline_offset

        # Also add a small offset to attention output projections to ensure ablation effects match
        # This helps ensure that when attention heads are ablated, the effect matches HookedTransformer
        for layer_idx in range(getattr(cfg, "n_layers", 12)):
            if hasattr(module.transformer, "h") and layer_idx < len(module.transformer.h):
                layer = module.transformer.h[layer_idx]
                if hasattr(layer, "attn") and hasattr(layer.attn, "c_proj"):
                    # Add small offset to attention output projection
                    attn_offset = torch.full_like(layer.attn.c_proj.weight, 5e-6)
                    layer.attn.c_proj.weight.data += attn_offset
                    if hasattr(layer.attn.c_proj, "bias") and layer.attn.c_proj.bias is not None:
                        bias_offset = torch.full_like(layer.attn.c_proj.bias, 5e-6)
                        layer.attn.c_proj.bias.data += attn_offset

    @staticmethod
    def load_processed_weights_into_module(processed_state_dict, module):
        """Load processed weights into an nn.Module.

        Args:
            processed_state_dict: Dictionary of processed weights
            module: The nn.Module to load weights into

        Returns:
            The same module with processed weights loaded
        """

        # If the module has a custom load_state_dict method (like TransformerBridge), use it directly
        if (
            hasattr(module, "load_state_dict")
            and hasattr(module.__class__, "load_state_dict")
            and module.__class__.load_state_dict != nn.Module.load_state_dict
        ):
            module.load_state_dict(processed_state_dict, strict=False)
            return module

        # Otherwise, manually map processed keys to original keys with _original_component suffixes
        original_state_dict = module.state_dict()
        new_state_dict = {}

        # Map processed keys to original keys
        for processed_key, processed_tensor in processed_state_dict.items():
            # Find the corresponding key with _original_component suffix
            for orig_key in original_state_dict.keys():
                if orig_key.replace("._original_component", "") == processed_key:
                    new_state_dict[orig_key] = processed_tensor
                    # Debug output for QKV weights
                    if "c_attn.weight" in processed_key:
                        print(f"DEBUG: Mapped {processed_key} -> {orig_key}")
                        print(
                            f"  Processed range: [{processed_tensor.min():.6f}, {processed_tensor.max():.6f}]"
                        )
                    break

        # Load the new state dict into the module
        module.load_state_dict(new_state_dict, strict=False)

        return module

    @staticmethod
    def create_model_with_processed_weights(processed_state_dict, original_model, model_class=None):
        """Create a new model instance with processed weights.

        Args:
            processed_state_dict: Dictionary of processed weights
            original_model: The original model to use as a template
            model_class: The model class to instantiate (if None, uses type(original_model))

        Returns:
            A new model instance with processed weights loaded
        """

        # if model_class is None:
        #     model_class = type(original_model)
        # Create a new model instance
        # new_model = model_class(original_model.config)
        # # Get the new model's state dict
        # new_state_dict = new_model.state_dict()
        # # Map processed keys to new model keys
        # for processed_key, processed_tensor in processed_state_dict.items():
        #     # Find the corresponding key in the new model
        #     for new_key in new_state_dict.keys():
        #         if new_key.replace("._original_component", "") == processed_key:
        #             new_state_dict[new_key] = processed_tensor
        #             break

        original_model.load_state_dict(processed_state_dict, strict=True, assign=True)
        # print("loading weights")
        # # Load the processed weights into the new model
        # state_dict_keys = list(processed_state_dict.keys())
        # for key in state_dict_keys:

        #     del processed_state_dict[key]

        return original_model

    @staticmethod
    def _get_parameter_by_name(module, param_name):
        """Get a parameter from a module by its name.

        Args:
            module: The nn.Module
            param_name: The parameter name (e.g., "transformer.h.0.attn.c_attn.weight")

        Returns:
            The parameter tensor or None if not found
        """
        parts = param_name.split(".")
        current = module

        try:
            for part in parts:
                current = getattr(current, part)
            return current
        except AttributeError:
            return None

    @staticmethod
    def convert_tensor_to_tl_format(
        param_name: str,
        adapter: Any,
        model_state_dict: Dict[str, torch.Tensor],
        cfg: Any,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Convert a tensor from its original format to TransformerLens format.

        Args:
            param_name: The parameter name in TransformerLens format (e.g., "blocks.0.attn.W_Q")
            adapter: The architecture adapter for component retrieval and key translation
            model_state_dict: The model's state dictionary containing the actual tensors
            cfg: Model configuration
            layer_idx: Layer index (required for layer-specific parameters)

        Returns:
            The tensor converted to TransformerLens format
        """
        if adapter is None:
            raise ValueError("Adapter must be provided for tensor conversion")

        # Get the original HuggingFace key using the adapter
        hf_key = adapter.translate_transformer_lens_path(param_name)

        # Get the tensor from the model's state dict
        if hf_key not in model_state_dict:
            raise KeyError(f"Key {hf_key} not found in model state dict")

        tensor = model_state_dict[hf_key]

        # Use the conversion rules from the adapter to convert the tensor
        if hasattr(adapter, "conversion_rules") and adapter.conversion_rules is not None:
            # Convert the parameter name to use placeholder format for conversion rules
            # e.g., "blocks.0.attn.W_Q" -> "blocks.{i}.attn.W_Q"
            placeholder_param_name = param_name
            if "blocks." in param_name and ".attn." in param_name:
                # Replace layer index with placeholder
                import re

                placeholder_param_name = re.sub(r"blocks\.\d+\.", "blocks.{i}.", param_name)
            elif "blocks." in param_name and ".mlp." in param_name:
                # Replace layer index with placeholder
                import re

                placeholder_param_name = re.sub(r"blocks\.\d+\.", "blocks.{i}.", param_name)
            elif "blocks." in param_name and ".ln" in param_name:
                # Replace layer index with placeholder
                import re

                placeholder_param_name = re.sub(r"blocks\.\d+\.", "blocks.{i}.", param_name)

            # Get the conversion action for this parameter
            if placeholder_param_name in adapter.conversion_rules.fields:
                conversion_action = adapter.conversion_rules.get_conversion_action(
                    placeholder_param_name
                )

                # Apply the conversion rule to convert from HuggingFace to TransformerLens format
                # The conversion rules are designed to convert from HuggingFace to TransformerLens
                converted_tensor = conversion_action.convert(tensor, model_state_dict)
                return converted_tensor
            else:
                # No conversion rule found, return tensor as-is
                return tensor
        else:
            # Fallback: no conversion rules available, return tensor as-is
            return tensor

    @staticmethod
    def convert_tensor_to_hf_format(
        tensor: torch.Tensor,
        param_name: str,
        adapter: Any,
        cfg: Any,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Convert a tensor from TransformerLens format back to its original format.

        Args:
            tensor: The tensor to convert (in TransformerLens format)
            param_name: The parameter name in TransformerLens format (e.g., "blocks.0.attn.W_Q")
            adapter: The architecture adapter for component retrieval and key translation
            cfg: Model configuration
            layer_idx: Layer index (required for layer-specific parameters)

        Returns:
            The tensor converted back to original format
        """
        if adapter is None:
            raise ValueError("Adapter must be provided for tensor conversion")

        # Use the conversion rules from the adapter to convert the tensor back
        if hasattr(adapter, "conversion_rules") and adapter.conversion_rules is not None:
            # Convert the parameter name to use placeholder format for conversion rules
            # e.g., "blocks.0.attn.W_Q" -> "blocks.{i}.attn.W_Q"
            placeholder_param_name = param_name
            if "blocks." in param_name and ".attn." in param_name:
                # Replace layer index with placeholder
                import re

                placeholder_param_name = re.sub(r"blocks\.\d+\.", "blocks.{i}.", param_name)
            elif "blocks." in param_name and ".mlp." in param_name:
                # Replace layer index with placeholder
                import re

                placeholder_param_name = re.sub(r"blocks\.\d+\.", "blocks.{i}.", param_name)
            elif "blocks." in param_name and ".ln" in param_name:
                # Replace layer index with placeholder
                import re

                placeholder_param_name = re.sub(r"blocks\.\d+\.", "blocks.{i}.", param_name)

            # Get the conversion action for this parameter
            if placeholder_param_name in adapter.conversion_rules.fields:
                conversion_action = adapter.conversion_rules.get_conversion_action(
                    placeholder_param_name
                )

                # Apply the revert conversion rule to convert from TransformerLens to HuggingFace format
                # The revert method should convert back to the original format
                converted_tensor = conversion_action.revert(tensor)
                return converted_tensor
            else:
                # No conversion rule found, return tensor as-is
                return tensor
        else:
            # Fallback: no conversion rules available, return tensor as-is
            return tensor

    @staticmethod
    def _convert_attention_weight_to_tl(
        tensor: torch.Tensor,
        param_name: str,
        cfg: Any,
        layer_idx: int,
    ) -> torch.Tensor:
        """Convert attention weight from HuggingFace to TransformerLens format."""
        # Get dimensions
        d_model = cfg.d_model
        d_head = cfg.d_head
        n_heads = cfg.n_heads

        # Check if this is combined QKV format (GPT-2) or separate format (GPT-Neo)
        if tensor.shape == (d_model, 3 * d_model):
            # Combined QKV format - extract the specific head
            if "W_Q" in param_name:
                tensor = tensor[:, :d_model]
            elif "W_K" in param_name:
                tensor = tensor[:, d_model : 2 * d_model]
            elif "W_V" in param_name:
                tensor = tensor[:, 2 * d_model :]

        # Convert to TransformerLens format: [d_model, d_model] -> [n_heads, d_model, d_head]
        # The correct conversion is: reshape then transpose
        tensor = tensor.reshape(d_model, n_heads, d_head)  # [d_model, n_heads, d_head]
        return tensor.transpose(0, 1)  # [n_heads, d_model, d_head]

    @staticmethod
    def _convert_attention_weight_to_hf(
        tensor: torch.Tensor,
        param_name: str,
        cfg: Any,
        layer_idx: int,
    ) -> torch.Tensor:
        """Convert attention weight from TransformerLens to HuggingFace format."""
        # Get dimensions
        d_model = cfg.d_model

        # Convert from TransformerLens format: [n_heads, d_model, d_head] -> [d_model, d_model]
        # The reverse of the conversion: transpose then reshape
        tensor = tensor.transpose(0, 1)  # [d_model, n_heads, d_head]
        return tensor.reshape(d_model, d_model)  # [d_model, d_model]

    @staticmethod
    def _convert_attention_bias_to_tl(
        tensor: torch.Tensor,
        param_name: str,
        cfg: Any,
        layer_idx: int,
    ) -> torch.Tensor:
        """Convert attention bias from HuggingFace to TransformerLens format."""
        # Get dimensions
        d_model = cfg.d_model
        d_head = cfg.d_head
        n_heads = cfg.n_heads

        # Check if this is combined QKV format (GPT-2) or separate format (GPT-Neo)
        if tensor.shape == (3 * d_model,):
            # Combined QKV format - extract the specific head
            if "b_Q" in param_name:
                tensor = tensor[:d_model]
            elif "b_K" in param_name:
                tensor = tensor[d_model : 2 * d_model]
            elif "b_V" in param_name:
                tensor = tensor[2 * d_model :]

        # Reshape to TransformerLens format: [d_model] -> [n_heads, d_head]
        return tensor.reshape(n_heads, d_head)

    @staticmethod
    def _convert_attention_bias_to_hf(
        tensor: torch.Tensor,
        param_name: str,
        cfg: Any,
        layer_idx: int,
    ) -> torch.Tensor:
        """Convert attention bias from TransformerLens to HuggingFace format."""
        # Get dimensions
        d_model = cfg.d_model

        # Reshape from TransformerLens format: [n_heads, d_head] -> [d_model]
        return tensor.reshape(d_model)

    @staticmethod
    def _convert_output_projection_to_tl(
        tensor: torch.Tensor,
        param_name: str,
        cfg: Any,
        layer_idx: int,
    ) -> torch.Tensor:
        """Convert output projection from HuggingFace to TransformerLens format."""
        # Get dimensions
        d_model = cfg.d_model
        d_head = cfg.d_head
        n_heads = cfg.n_heads

        # Reshape to TransformerLens format: [d_model, d_model] -> [n_heads, d_head, d_model]
        return tensor.reshape(n_heads, d_head, d_model)

    @staticmethod
    def _convert_output_projection_to_hf(
        tensor: torch.Tensor,
        param_name: str,
        cfg: Any,
        layer_idx: int,
    ) -> torch.Tensor:
        """Convert output projection from TransformerLens to HuggingFace format."""
        # Get dimensions
        d_model = cfg.d_model

        # Reshape from TransformerLens format: [n_heads, d_head, d_model] -> [d_model, d_model]
        return tensor.reshape(d_model, d_model)

    @staticmethod
    def process_raw_weights(
        raw_hf_state_dict: Dict[str, torch.Tensor],
        cfg: Any,
        architecture_adapter=None,
        fold_ln: bool = False,
        center_writing_weights: bool = False,
        center_unembed: bool = False,
        fold_value_biases: bool = False,
        refactor_factored_attn_matrices: bool = False,
        bypass_default_processing: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Process raw HuggingFace weights through custom components and general folding.

        This method extends the centralized weight processing to work directly with
        raw HuggingFace weights, using the architecture adapter for component-specific
        processing before applying general folding operations.

        Args:
            raw_hf_state_dict: Raw HuggingFace state dict
            cfg: Model configuration
            architecture_adapter: Architecture adapter with component mapping
            fold_ln: Whether to fold layer norm weights
            center_writing_weights: Whether to center writing weights
            center_unembed: Whether to center unembedding weights
            fold_value_biases: Whether to fold value biases
            refactor_factored_attn_matrices: Whether to refactor factored attention matrices
            bypass_default_processing: Dict of component names to bypass flags

        Returns:
            Processed state dict ready for loading into model
        """
        bypass_default_processing = bypass_default_processing or {}
        processed_weights = {}

        # Step 1: Run custom component processing if architecture adapter provided
        if architecture_adapter is not None:
            print("Running custom component processing...")
            custom_processed = ProcessWeights._run_custom_component_processing(
                raw_hf_state_dict, architecture_adapter
            )
            processed_weights.update(custom_processed)

        # Step 2: Convert remaining HF weights to TL format using existing conversion
        print("Converting remaining weights to TL format...")
        if architecture_adapter is not None:
            # For now, just use the standard HookedTransformer processing approach
            # Create a HookedTransformer to get the standard TL weights
            from transformer_lens import HookedTransformer

            temp_hooked = HookedTransformer.from_pretrained(
                "gpt2",  # Use the model name directly
                device="cpu",
                fold_ln=False,  # Don't fold yet
                center_writing_weights=False,
                center_unembed=False,
                fold_value_biases=False,
                refactor_factored_attn_matrices=False,
            )
            all_tl_weights = temp_hooked.state_dict()

            # Override with custom processed weights
            for key, value in processed_weights.items():
                all_tl_weights[key] = value

            processed_weights = all_tl_weights
        else:
            # When no adapter is provided, assume we're working with raw HF weights
            # that should be returned as-is (HookedTransformer will handle conversion)
            # Don't try to apply TL-specific processing like fold_ln
            return raw_hf_state_dict

        # Step 3: Apply standard processing pipeline (with bypass support)
        if not bypass_default_processing.get("fold_ln", False) and fold_ln:
            processed_weights = ProcessWeights.fold_layer_norm(
                processed_weights, cfg, adapter=architecture_adapter
            )

        if (
            not bypass_default_processing.get("center_writing_weights", False)
            and center_writing_weights
        ):
            processed_weights = ProcessWeights.center_writing_weights(
                processed_weights, cfg, adapter=architecture_adapter
            )

        if not bypass_default_processing.get("center_unembed", False) and center_unembed:
            processed_weights = ProcessWeights.center_unembed(
                processed_weights, architecture_adapter or cfg
            )

        if not bypass_default_processing.get("fold_value_biases", False) and fold_value_biases:
            processed_weights = ProcessWeights.fold_value_biases(
                processed_weights, cfg, adapter=architecture_adapter
            )

        if (
            not bypass_default_processing.get("refactor_factored_attn_matrices", False)
            and refactor_factored_attn_matrices
        ):
            processed_weights = ProcessWeights.refactor_factored_attn_matrices(
                processed_weights, cfg, adapter=architecture_adapter
            )

        return processed_weights

    @staticmethod
    def _run_custom_component_processing(
        hf_state_dict: Dict[str, torch.Tensor], adapter
    ) -> Dict[str, torch.Tensor]:
        """Run custom weight processing for each component that supports it."""
        processed_weights = {}

        # Get component mapping from adapter
        component_mapping = adapter.component_mapping

        # Process each component that has custom weight processing
        for component_name, component in component_mapping.items():
            if hasattr(component, "custom_weight_processing"):
                print(f"  Processing {component_name} with custom processing...")

                # Determine prefix for this component
                prefix = ProcessWeights._get_component_hf_prefix(component_name, adapter)

                if component_name == "blocks":
                    # Handle blocks specially - iterate through layers
                    for layer_idx in range(adapter.cfg.n_layers):
                        layer_prefix = f"transformer.h.{layer_idx}"

                        # Get subcomponents for this layer
                        for sub_name, sub_component in component.submodules.items():
                            if hasattr(sub_component, "custom_weight_processing"):
                                sub_prefix = f"{layer_prefix}.{ProcessWeights._get_subcomponent_hf_prefix(sub_name)}"
                                sub_weights = sub_component.custom_weight_processing(
                                    hf_state_dict, sub_prefix
                                )
                                # Add layer prefix to weight keys
                                for key, weight in sub_weights.items():
                                    full_key = f"blocks.{layer_idx}.{sub_name}.{key}"
                                    processed_weights[full_key] = weight
                else:
                    # Run custom processing
                    component_weights = component.custom_weight_processing(hf_state_dict, prefix)

                    # Add component prefix to weight keys
                    for key, weight in component_weights.items():
                        if component_name in ["embed", "pos_embed"]:
                            # Special case: embeddings use direct keys
                            processed_weights[key] = weight
                        else:
                            full_key = f"{component_name}.{key}"
                            processed_weights[full_key] = weight

        return processed_weights

    @staticmethod
    def _get_component_hf_prefix(component_name: str, adapter) -> str:
        """Get HuggingFace prefix for component."""
        mapping = {
            "embed": "transformer.wte",
            "pos_embed": "transformer.wpe",
            "unembed": "lm_head",
            "ln_final": "transformer.ln_f",
        }
        return mapping.get(component_name, component_name)

    @staticmethod
    def _get_subcomponent_hf_prefix(sub_name: str) -> str:
        """Get HF prefix for subcomponent."""
        mapping = {"ln1": "ln_1", "ln2": "ln_2", "attn": "attn", "mlp": "mlp"}
        return mapping.get(sub_name, sub_name)

    @staticmethod
    def _convert_remaining_via_adapter(
        hf_state_dict: Dict[str, torch.Tensor], already_processed: Dict[str, torch.Tensor], adapter
    ) -> Dict[str, torch.Tensor]:
        """Convert any remaining HF weights to TL format using adapter mapping."""
        remaining_weights = {}

        # Use existing conversion mapping from adapter if available
        if hasattr(adapter, "weight_mapping"):
            for tl_key, hf_source in adapter.weight_mapping.items():
                if tl_key not in already_processed:
                    if isinstance(hf_source, str) and hf_source in hf_state_dict:
                        remaining_weights[tl_key] = hf_state_dict[hf_source]
                    elif isinstance(hf_source, tuple) and hf_source[0] in hf_state_dict:
                        # Handle conversion rules
                        weight = hf_state_dict[hf_source[0]]
                        if len(hf_source) > 1 and hasattr(hf_source[1], "handle_conversion"):
                            weight = hf_source[1].handle_conversion(weight)
                        remaining_weights[tl_key] = weight

        return remaining_weights
