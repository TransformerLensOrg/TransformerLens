"""
Weight Processing Functions for Transformer Models.

This module contains all the weight processing functions extracted from HookedTransformer,
organized into a single ProcessWeights class with static methods. These functions are used
to modify transformer model weights for better interpretability and analysis.
"""
from typing import Any, Dict, Optional, Union, overload

import einops
import torch

import transformer_lens.utils as utils
from transformer_lens.config.TransformerLensConfig import TransformerLensConfig
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter


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
        """Convert legacy TL key format (W_Q, b_Q) to component-based format (q.weight, q.bias).

        Args:
            tl_key: TransformerLens format parameter key (e.g., "blocks.0.attn.W_Q")
            adapter: Architecture adapter for translating paths

        Returns:
            The component-based key (e.g., "blocks.0.attn.q.weight")
        """
        if adapter is None:
            return tl_key

        return ProcessWeights._prepare_component_path(tl_key)

    @staticmethod
    def _prepare_component_path(tl_key: str) -> str:
        """Map a TransformerLens key to bridge-style component path.

        Converts TransformerLens weight names (like "W_Q", "b_in") to bridge-style
        paths (like "q.weight", "in.bias"). The full path is assembled before being
        passed to the architecture adapter for translation.

        Args:
            tl_key: TransformerLens key like "blocks.0.attn.W_Q"

        Returns:
            Full path like "blocks.0.attn.q.weight"
        """
        suffix_map: Dict[str, str] = {
            "W_Q": "q.weight",
            "_W_Q": "q.weight",
            "b_Q": "q.bias",
            "_b_Q": "q.bias",
            "W_K": "k.weight",
            "_W_K": "k.weight",
            "b_K": "k.bias",
            "_b_K": "k.bias",
            "W_V": "v.weight",
            "_W_V": "v.weight",
            "b_V": "v.bias",
            "_b_V": "v.bias",
            "W_O": "o.weight",
            "b_O": "o.bias",
            "W_in": "in.weight",
            "b_in": "in.bias",
            "W_gate": "gate.weight",
            "b_gate": "gate.bias",
            "W_out": "out.weight",
            "b_out": "out.bias",
            "W_E": "weight",
            "b_E": "bias",
            "W_pos": "weight",
            "b_pos": "bias",
            "W_U": "weight",
            "b_U": "bias",
            "w": "weight",
            "b": "bias",
            "weight": "weight",
            "bias": "bias",
        }
        if "." not in tl_key:
            return tl_key
        base_path, suffix = tl_key.rsplit(".", 1)
        if suffix in suffix_map:
            replacement = suffix_map[suffix]
            return f"{base_path}.{replacement}"
        return tl_key

    @staticmethod
    def _safe_get_tensor(
        state_dict: Dict[str, torch.Tensor],
        tl_key: str,
        adapter=None,
        default: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Safely get a tensor from state_dict, handling optional parameters.

        This is the recommended way to access parameters that may not exist in all architectures
        (e.g., biases in Qwen2/LLaMA/Gemma). Returns None if the parameter doesn't exist,
        rather than raising a KeyError.

        Args:
            state_dict: Model state dictionary
            tl_key: TransformerLens format parameter key (e.g., "blocks.0.attn.b_Q")
            adapter: Optional architecture adapter for key translation
            default: Optional default value to return if key not found (defaults to None)

        Returns:
            The tensor if found, otherwise the default value (None if not specified)

        Examples:
            # Get optional bias (may be None for Qwen2/LLaMA)
            b_Q = ProcessWeights._safe_get_tensor(state_dict, "blocks.0.attn.b_Q", adapter)

            # Get required weight (will be None if missing, can check explicitly)
            W_Q = ProcessWeights._safe_get_tensor(state_dict, "blocks.0.attn.W_Q", adapter)
            if W_Q is None:
                raise ValueError("Required weight W_Q not found")
        """
        actual_key = ProcessWeights._get_param_key(tl_key, adapter)
        return state_dict.get(actual_key, default)

    @staticmethod
    def fold_layer_norm_bias_single(
        w_tensor: torch.Tensor, b_tensor: torch.Tensor, ln_bias: torch.Tensor
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
        w_tensor: torch.Tensor, ln_weight: torch.Tensor
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
    def center_weight_single(w_tensor: torch.Tensor) -> torch.Tensor:
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
        bq_tensor: Optional[torch.Tensor],
        bk_tensor: Optional[torch.Tensor],
        bv_tensor: Optional[torch.Tensor],
        ln_bias: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Fold LayerNorm bias into attention biases.

        Args:
            wq_tensor, wk_tensor, wv_tensor: Weight tensors [n_heads, d_model, d_head]
            bq_tensor, bk_tensor, bv_tensor: Bias tensors [n_heads, d_head] or None if no bias
            ln_bias: LayerNorm bias [d_model]

        Returns:
            Tuple of (new_bq, new_bk, new_bv) with folded biases (None if input bias was None)
        """
        new_bq = (
            ProcessWeights.fold_layer_norm_bias_single(wq_tensor, bq_tensor, ln_bias)
            if bq_tensor is not None
            else None
        )
        new_bk = (
            ProcessWeights.fold_layer_norm_bias_single(wk_tensor, bk_tensor, ln_bias)
            if bk_tensor is not None
            else None
        )
        new_bv = (
            ProcessWeights.fold_layer_norm_bias_single(wv_tensor, bv_tensor, ln_bias)
            if bv_tensor is not None
            else None
        )
        return (new_bq, new_bk, new_bv)

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
        return (new_wq, new_wk, new_wv)

    @staticmethod
    def center_attention_weights(
        wq_tensor: torch.Tensor, wk_tensor: torch.Tensor, wv_tensor: torch.Tensor
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
        return (centered_wq, centered_wk, centered_wv)

    @staticmethod
    def extract_attention_tensors_for_folding(
        state_dict: Dict[str, torch.Tensor], cfg, layer: int, adapter
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
        b_Q_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_Q", adapter)
        W_Q_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.W_Q", adapter)
        b_K_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_K", adapter)
        W_K_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.W_K", adapter)
        b_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_V", adapter)
        W_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.W_V", adapter)
        ln1_b_key = ProcessWeights._get_param_key(f"blocks.{layer}.ln1.b", adapter)
        ln1_w_key = ProcessWeights._get_param_key(f"blocks.{layer}.ln1.w", adapter)

        wq_tensor: Optional[torch.Tensor] = state_dict.get(W_Q_key)
        wk_tensor: Optional[torch.Tensor] = state_dict.get(W_K_key)
        wv_tensor: Optional[torch.Tensor] = state_dict.get(W_V_key)
        bq_tensor: Optional[torch.Tensor] = state_dict.get(b_Q_key)
        bk_tensor: Optional[torch.Tensor] = state_dict.get(b_K_key)
        bv_tensor: Optional[torch.Tensor] = state_dict.get(b_V_key)
        ln1_b = state_dict.get(ln1_b_key, None)
        ln1_w = state_dict.get(ln1_w_key, None)
        if adapter:
            wq_tensor = ProcessWeights.convert_tensor_to_tl_format(
                W_Q_key, state_dict, wq_tensor, cfg, adapter, layer
            )
            wk_tensor = ProcessWeights.convert_tensor_to_tl_format(
                W_K_key, state_dict, wk_tensor, cfg, adapter, layer
            )
            wv_tensor = ProcessWeights.convert_tensor_to_tl_format(
                W_V_key, state_dict, wv_tensor, cfg, adapter, layer
            )
            bq_tensor = ProcessWeights.convert_tensor_to_tl_format(
                b_Q_key, state_dict, bq_tensor, cfg, adapter, layer
            )
            bk_tensor = ProcessWeights.convert_tensor_to_tl_format(
                b_K_key, state_dict, bk_tensor, cfg, adapter, layer
            )
            bv_tensor = ProcessWeights.convert_tensor_to_tl_format(
                b_V_key, state_dict, bv_tensor, cfg, adapter, layer
            )

        return {
            "wq": wq_tensor,
            "wk": wk_tensor,
            "wv": wv_tensor,
            "bq": bq_tensor,
            "bk": bk_tensor,
            "bv": bv_tensor,
            "ln1_b": ln1_b,
            "ln1_w": ln1_w,
            "keys": {
                "W_Q": W_Q_key,
                "W_K": W_K_key,
                "W_V": W_V_key,
                "b_Q": b_Q_key,
                "b_K": b_K_key,
                "b_V": b_V_key,
                "ln1_b": ln1_b_key,
                "ln1_w": ln1_w_key,
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
    ) -> Dict[str, torch.Tensor]:
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
        tensors = ProcessWeights.extract_attention_tensors_for_folding(
            state_dict, cfg, layer, adapter
        )
        wq_tensor = tensors["wq"]
        wk_tensor = tensors["wk"]
        wv_tensor = tensors["wv"]
        bq_tensor = tensors["bq"]
        bk_tensor = tensors["bk"]
        bv_tensor = tensors["bv"]
        ln1_b = tensors["ln1_b"]
        ln1_w = tensors["ln1_w"]
        keys = tensors["keys"]
        if wq_tensor is None:
            return state_dict
        assert isinstance(wq_tensor, torch.Tensor)
        assert isinstance(keys, dict)
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
        if ln1_b is not None and ln1_w is not None:
            assert isinstance(ln1_b, torch.Tensor)
            assert isinstance(ln1_w, torch.Tensor)
            if fold_biases:
                if all(
                    (
                        t is not None
                        for t in [wq_tensor, wk_tensor, wv_tensor, bq_tensor, bk_tensor, bv_tensor]
                    )
                ):
                    # Type narrowing for mypy
                    assert wq_tensor is not None
                    assert wk_tensor is not None
                    assert wv_tensor is not None
                    bq_tensor, bk_tensor, bv_tensor = ProcessWeights.fold_layer_norm_biases(
                        wq_tensor, wk_tensor, wv_tensor, bq_tensor, bk_tensor, bv_tensor, ln1_b
                    )
                if keys["ln1_b"] in state_dict:
                    state_dict[keys["ln1_b"]] = torch.zeros_like(ln1_b)
                alternate_b_key = (
                    keys["ln1_b"].replace("ln_1", "ln1")
                    if "ln_1" in keys["ln1_b"]
                    else keys["ln1_b"].replace("ln1", "ln_1")
                )
                if alternate_b_key != keys["ln1_b"] and alternate_b_key in state_dict:
                    state_dict[alternate_b_key] = torch.zeros_like(ln1_b)
            if wk_tensor is not None and wv_tensor is not None:
                wq_tensor, wk_tensor, wv_tensor = ProcessWeights.fold_layer_norm_weights(
                    wq_tensor, wk_tensor, wv_tensor, ln1_w
                )
            if keys["ln1_w"] in state_dict:
                state_dict[keys["ln1_w"]] = torch.ones_like(ln1_w)
            alternate_w_key = (
                keys["ln1_w"].replace("ln_1", "ln1")
                if "ln_1" in keys["ln1_w"]
                else keys["ln1_w"].replace("ln1", "ln_1")
            )
            if alternate_w_key != keys["ln1_w"] and alternate_w_key in state_dict:
                state_dict[alternate_w_key] = torch.ones_like(ln1_w)
        if center_weights and wk_tensor is not None and (wv_tensor is not None):
            wq_tensor, wk_tensor, wv_tensor = ProcessWeights.center_attention_weights(
                wq_tensor, wk_tensor, wv_tensor
            )
        state_dict = ProcessWeights._store_processed_attention_tensors(
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
        state_dict = ProcessWeights._fold_mlp_layer_norm(
            state_dict, cfg, layer, fold_biases, center_weights, adapter
        )

        return state_dict

    @staticmethod
    def _fold_mlp_layer_norm(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        layer: int,
        fold_biases: bool,
        center_weights: bool,
        adapter,
    ) -> Dict[str, torch.Tensor]:
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
            return state_dict

        mlp_b_in_key = ProcessWeights._get_param_key(f"blocks.{layer}.mlp.b_in", adapter)
        mlp_W_in_key = ProcessWeights._get_param_key(f"blocks.{layer}.mlp.W_in", adapter)
        mlp_W_gate_key = (
            ProcessWeights._get_param_key(f"blocks.{layer}.mlp.W_gate", adapter)
            if getattr(cfg, "gated_mlp", False)
            else None
        )
        ln2_b_key = ProcessWeights._get_param_key(f"blocks.{layer}.ln2.b", adapter)
        ln2_w_key = ProcessWeights._get_param_key(f"blocks.{layer}.ln2.w", adapter)
        if ln2_b_key in state_dict and ln2_w_key in state_dict:
            mlp_W_in = ProcessWeights.convert_tensor_to_tl_format(
                mlp_W_in_key, state_dict, state_dict.get(mlp_W_in_key), cfg, adapter, layer
            )
            ln2_w = state_dict[ln2_w_key]
            ln2_b = state_dict[ln2_b_key]
            assert mlp_W_in is not None, f"MLP W_in not found at key {mlp_W_in_key}"
            if mlp_W_in.shape[1] == ln2_w.shape[0]:
                ln2_w_broadcast = ln2_w[None, :]
                ln2_b_broadcast = ln2_b[None, :]
                sum_dim = -1
            elif mlp_W_in.shape[0] == ln2_w.shape[0]:
                ln2_w_broadcast = ln2_w[:, None]
                ln2_b_broadcast = ln2_b[:, None]
                sum_dim = -2
            else:
                raise ValueError(
                    f"Cannot broadcast MLP weight {mlp_W_in.shape} with layer norm weight {ln2_w.shape}"
                )
            if fold_biases:
                mlp_b_in = ProcessWeights.convert_tensor_to_tl_format(
                    mlp_b_in_key, state_dict, state_dict.get(mlp_b_in_key), cfg, adapter, layer
                )
                assert mlp_b_in is not None, f"MLP b_in not found at key {mlp_b_in_key}"
                new_mlp_b_in = mlp_b_in + (mlp_W_in * ln2_b_broadcast).sum(sum_dim)
                state_dict[mlp_b_in_key] = ProcessWeights.convert_tensor_to_hf_format(
                    mlp_b_in_key, new_mlp_b_in, cfg, adapter, layer
                )
                state_dict[ln2_b_key] = torch.zeros_like(ln2_b)
                alternate_ln2_b_key = (
                    ln2_b_key.replace("ln_2", "ln2")
                    if "ln_2" in ln2_b_key
                    else ln2_b_key.replace("ln2", "ln_2")
                )
                if alternate_ln2_b_key != ln2_b_key and alternate_ln2_b_key in state_dict:
                    state_dict[alternate_ln2_b_key] = torch.zeros_like(ln2_b)
            new_mlp_W_in = mlp_W_in * ln2_w_broadcast
            state_dict[mlp_W_in_key] = ProcessWeights.convert_tensor_to_hf_format(
                mlp_W_in_key, new_mlp_W_in, cfg, adapter, layer
            )
            if getattr(cfg, "gated_mlp", False) and mlp_W_gate_key is not None:
                mlp_W_gate = ProcessWeights.convert_tensor_to_tl_format(
                    mlp_W_gate_key, state_dict, state_dict.get(mlp_W_gate_key), cfg, adapter, layer
                )
                assert mlp_W_gate is not None, f"MLP W_gate not found at key {mlp_W_gate_key}"
                new_mlp_W_gate = mlp_W_gate * ln2_w_broadcast
                state_dict[mlp_W_gate_key] = ProcessWeights.convert_tensor_to_hf_format(
                    mlp_W_gate_key, new_mlp_W_gate, cfg, adapter, layer
                )
            state_dict[ln2_w_key] = torch.ones_like(state_dict[ln2_w_key])
            alternate_ln2_w_key = (
                ln2_w_key.replace("ln_2", "ln2")
                if "ln_2" in ln2_w_key
                else ln2_w_key.replace("ln2", "ln_2")
            )
            if alternate_ln2_w_key != ln2_w_key and alternate_ln2_w_key in state_dict:
                state_dict[alternate_ln2_w_key] = torch.ones_like(state_dict[ln2_w_key])
        if center_weights and mlp_W_in_key in state_dict:
            mlp_W_in_centered = ProcessWeights.convert_tensor_to_tl_format(
                mlp_W_in_key, state_dict, state_dict.get(mlp_W_in_key), cfg, adapter, layer
            )
            assert mlp_W_in_centered is not None, f"MLP W_in not found at key {mlp_W_in_key}"
            mlp_W_in_centered = mlp_W_in_centered - einops.reduce(
                mlp_W_in_centered, "d_model d_mlp -> 1 d_mlp", "mean"
            )
            state_dict[mlp_W_in_key] = ProcessWeights.convert_tensor_to_hf_format(
                mlp_W_in_key, mlp_W_in_centered, cfg, adapter, layer
            )
        if getattr(cfg, "act_fn", None) is not None and cfg.act_fn.startswith("solu"):
            mlp_b_out_key = ProcessWeights._get_param_key(f"blocks.{layer}.mlp.b_out", adapter)
            mlp_W_out_key = ProcessWeights._get_param_key(f"blocks.{layer}.mlp.W_out", adapter)
            mlp_ln_b_key = ProcessWeights._get_param_key(f"blocks.{layer}.mlp.ln.b", adapter)
            mlp_ln_w_key = ProcessWeights._get_param_key(f"blocks.{layer}.mlp.ln.w", adapter)

            mlp_b_out = ProcessWeights.convert_tensor_to_tl_format(
                mlp_b_out_key, state_dict, state_dict.get(mlp_b_out_key), cfg, adapter, layer
            )
            mlp_W_out = ProcessWeights.convert_tensor_to_tl_format(
                mlp_W_out_key, state_dict, state_dict.get(mlp_W_out_key), cfg, adapter, layer
            )
            mlp_ln_b = state_dict.get(mlp_ln_b_key)
            mlp_ln_w = state_dict.get(mlp_ln_w_key)
            assert mlp_b_out is not None, f"MLP b_out not found at key {mlp_b_out_key}"
            assert mlp_W_out is not None, f"MLP W_out not found at key {mlp_W_out_key}"
            assert mlp_ln_b is not None, f"MLP ln.b not found at key {mlp_ln_b_key}"
            assert mlp_ln_w is not None, f"MLP ln.w not found at key {mlp_ln_w_key}"

            if fold_biases:
                new_mlp_b_out = mlp_b_out + (mlp_W_out * mlp_ln_b[:, None]).sum(-2)
                state_dict[mlp_b_out_key] = ProcessWeights.convert_tensor_to_hf_format(
                    mlp_b_out_key, new_mlp_b_out, cfg, adapter, layer
                )
                if mlp_ln_b_key in state_dict:
                    state_dict[mlp_ln_b_key] = torch.zeros_like(mlp_ln_b)

            new_mlp_W_out = mlp_W_out * mlp_ln_w[:, None]

            if center_weights:
                new_mlp_W_out = new_mlp_W_out - einops.reduce(
                    new_mlp_W_out, "d_mlp d_model -> 1 d_model", "mean"
                )

            state_dict[mlp_W_out_key] = ProcessWeights.convert_tensor_to_hf_format(
                mlp_W_out_key, new_mlp_W_out, cfg, adapter, layer
            )

            if mlp_ln_w_key in state_dict:
                state_dict[mlp_ln_w_key] = torch.ones_like(mlp_ln_w)

        return state_dict

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
    ) -> Dict[str, torch.Tensor]:
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
        if wq_tensor is None:
            return state_dict
        wq_key = keys["W_Q"]
        wk_key = keys["W_K"]
        wv_key = keys["W_V"]
        bq_key = keys["b_Q"]
        bk_key = keys["b_K"]
        bv_key = keys["b_V"]

        # Store processed tensors directly in 3D format (set_processed_weights will flatten to 2D)
        if wq_tensor is None or wk_tensor is None or wv_tensor is None:
            raise ValueError(f"Required attention weights missing for layer {layer}")
        state_dict[wq_key] = ProcessWeights.convert_tensor_to_hf_format(
            wq_key, wq_tensor, cfg, adapter, layer_idx=layer
        )
        state_dict[wk_key] = ProcessWeights.convert_tensor_to_hf_format(
            wk_key, wk_tensor, cfg, adapter, layer_idx=layer
        )
        state_dict[wv_key] = ProcessWeights.convert_tensor_to_hf_format(
            wv_key, wv_tensor, cfg, adapter, layer_idx=layer
        )
        if bq_tensor is not None:
            state_dict[bq_key] = ProcessWeights.convert_tensor_to_hf_format(
                bq_key, bq_tensor, cfg, adapter, layer_idx=layer
            )
        if bk_tensor is not None:
            state_dict[bk_key] = ProcessWeights.convert_tensor_to_hf_format(
                bk_key, bk_tensor, cfg, adapter, layer_idx=layer
            )
        if bv_tensor is not None:
            state_dict[bv_key] = ProcessWeights.convert_tensor_to_hf_format(
                bv_key, bv_tensor, cfg, adapter, layer_idx=layer
            )

        return state_dict

    @staticmethod
    def _fold_unembed_layer_norm(
        state_dict: Dict[str, torch.Tensor], cfg, fold_biases: bool, center_weights: bool, adapter
    ) -> Dict[str, torch.Tensor]:
        """Fold LayerNorm into unembedding layer.

        Args:
            state_dict: The state dictionary to process (modified in place)
            cfg: Model configuration object
            fold_biases: Whether to fold LayerNorm biases
            center_weights: Whether to center weights after folding
            adapter: Optional architecture adapter for parameter key translation
        """
        unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
        unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
        ln_final_b_key = ProcessWeights._get_param_key("ln_final.b", adapter)
        ln_final_w_key = ProcessWeights._get_param_key("ln_final.w", adapter)
        has_unembed_bias = unembed_b_U_key in state_dict
        unembed_weight = ProcessWeights.convert_tensor_to_tl_format(
            unembed_W_U_key, state_dict, state_dict.get(unembed_W_U_key), cfg, adapter, None
        )
        ln_weight = state_dict[ln_final_w_key]
        assert unembed_weight is not None, f"Unembed weight not found at key {unembed_W_U_key}"
        if len(unembed_weight.shape) == 2 and len(ln_weight.shape) == 1:
            if unembed_weight.shape[1] == ln_weight.shape[0]:
                new_unembed_weight = unembed_weight * ln_weight[None, :]
            elif unembed_weight.shape[0] == ln_weight.shape[0]:
                new_unembed_weight = unembed_weight * ln_weight[:, None]
            else:
                raise ValueError(
                    f"Cannot broadcast unembedding weight {unembed_weight.shape} with layer norm weight {ln_weight.shape}"
                )
        else:
            raise ValueError(
                f"Unexpected tensor shapes: unembedding {unembed_weight.shape}, layer norm {ln_weight.shape}"
            )
        state_dict[unembed_W_U_key] = ProcessWeights.convert_tensor_to_hf_format(
            unembed_W_U_key, new_unembed_weight, cfg, adapter, None
        )
        if ln_final_w_key in state_dict:
            state_dict[ln_final_w_key] = torch.ones_like(ln_weight)
        alternate_final_w_key = (
            ln_final_w_key.replace("ln_f", "ln_final")
            if "ln_f" in ln_final_w_key
            else ln_final_w_key.replace("ln_final", "ln_f")
        )
        if alternate_final_w_key != ln_final_w_key and alternate_final_w_key in state_dict:
            state_dict[alternate_final_w_key] = torch.ones_like(ln_weight)
        if center_weights:
            unembed_weight_centered = ProcessWeights.convert_tensor_to_tl_format(
                unembed_W_U_key, state_dict, state_dict.get(unembed_W_U_key), cfg, adapter, None
            )
            assert (
                unembed_weight_centered is not None
            ), f"Unembed weight not found at key {unembed_W_U_key}"
            if len(unembed_weight_centered.shape) == 2:
                unembed_weight_centered = unembed_weight_centered - einops.reduce(
                    unembed_weight_centered, "d_model d_vocab -> 1 d_vocab", "mean"
                )
                state_dict[unembed_W_U_key] = ProcessWeights.convert_tensor_to_hf_format(
                    unembed_W_U_key, unembed_weight_centered, cfg, adapter, None
                )
            else:
                raise ValueError(
                    f"Unexpected unembedding weight shape: {unembed_weight_centered.shape}"
                )

        return state_dict

    @staticmethod
    def _fold_final_rms_bias(
        state_dict: Dict[str, torch.Tensor], cfg, fold_biases: bool, adapter
    ) -> Dict[str, torch.Tensor]:
        """Fold final RMS bias into unembedding (separate from regular unembed folding).

        Args:
            state_dict: The state dictionary to process (modified in place)
            cfg: Model configuration object
            fold_biases: Whether to fold LayerNorm biases
            adapter: Optional architecture adapter for parameter key translation
        """
        unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
        unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
        ln_final_b_key = ProcessWeights._get_param_key("ln_final.b", adapter)
        has_unembed_bias = unembed_b_U_key in state_dict
        has_ln_final_bias = ln_final_b_key in state_dict
        if (
            not getattr(cfg, "final_rms", False)
            and fold_biases
            and has_unembed_bias
            and has_ln_final_bias
        ):
            unembed_weight = ProcessWeights.convert_tensor_to_tl_format(
                unembed_W_U_key, state_dict, state_dict.get(unembed_W_U_key), cfg, adapter, None
            )
            ln_bias = state_dict[ln_final_b_key]
            assert unembed_weight is not None, f"Unembed weight not found at key {unembed_W_U_key}"
            if len(unembed_weight.shape) == 2 and len(ln_bias.shape) == 1:
                if unembed_weight.shape[1] == ln_bias.shape[0]:
                    bias_contribution = (unembed_weight * ln_bias[None, :]).sum(dim=-1)
                elif unembed_weight.shape[0] == ln_bias.shape[0]:
                    bias_contribution = (unembed_weight * ln_bias[:, None]).sum(dim=-2)
                else:
                    raise ValueError(
                        f"Cannot broadcast unembedding weight {unembed_weight.shape} with layer norm bias {ln_bias.shape}"
                    )
            else:
                raise ValueError(
                    f"Unexpected tensor shapes: unembedding {unembed_weight.shape}, layer norm bias {ln_bias.shape}"
                )
            unembed_b_U = ProcessWeights.convert_tensor_to_tl_format(
                unembed_b_U_key, state_dict, state_dict.get(unembed_b_U_key), cfg, adapter, None
            )
            assert unembed_b_U is not None, f"Unembed bias not found at key {unembed_b_U_key}"
            new_unembed_b_U = unembed_b_U + bias_contribution
            state_dict[unembed_b_U_key] = ProcessWeights.convert_tensor_to_hf_format(
                unembed_b_U_key, new_unembed_b_U, cfg, adapter, None
            )
            if ln_final_b_key in state_dict:
                state_dict[ln_final_b_key] = torch.zeros_like(ln_bias)
            alternate_final_b_key = (
                ln_final_b_key.replace("ln_f", "ln_final")
                if "ln_f" in ln_final_b_key
                else ln_final_b_key.replace("ln_final", "ln_f")
            )
            if alternate_final_b_key != ln_final_b_key and alternate_final_b_key in state_dict:
                state_dict[alternate_final_b_key] = torch.zeros_like(ln_bias)

        return state_dict

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
        # Make a deep copy to avoid modifying the original
        state_dict = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
        }
        gqa = "" if getattr(cfg, "n_key_value_heads", None) is None else "_"
        for l in range(cfg.n_layers):
            state_dict = ProcessWeights._fold_layer(
                state_dict, cfg, l, fold_biases, center_weights, adapter, gqa
            )
        state_dict = ProcessWeights._fold_final_rms_bias(state_dict, cfg, fold_biases, adapter)
        state_dict = ProcessWeights._fold_unembed_layer_norm(
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
        # Make a deep copy to avoid modifying the original
        embed_W_E_key = ProcessWeights._get_param_key("embed.W_E", adapter)
        try:
            pos_embed_W_pos_key = (
                ProcessWeights._get_param_key("pos_embed.W_pos", adapter)
                if getattr(cfg, "positional_embedding_type", "standard") != "rotary"
                else None
            )
        except ValueError:
            pos_embed_W_pos_key = None
        if embed_W_E_key not in state_dict:
            raise KeyError(
                f"Expected embedding key '{embed_W_E_key}' not found in state_dict. Available keys: {list(state_dict.keys())[:10]}..."
            )
        embed_W_E = ProcessWeights.convert_tensor_to_tl_format(
            embed_W_E_key, state_dict, state_dict.get(embed_W_E_key), cfg, adapter, None
        )
        assert embed_W_E is not None, f"Embedding not found at key {embed_W_E_key}"
        embed_W_E = embed_W_E - embed_W_E.mean(-1, keepdim=True)
        state_dict[embed_W_E_key] = ProcessWeights.convert_tensor_to_hf_format(
            embed_W_E_key, embed_W_E, cfg, adapter, None
        )

        if (
            getattr(cfg, "positional_embedding_type", "standard") != "rotary"
            and pos_embed_W_pos_key is not None
        ):
            if pos_embed_W_pos_key not in state_dict:
                raise KeyError(
                    f"Expected positional embedding key '{pos_embed_W_pos_key}' not found in state_dict. Available keys: {list(state_dict.keys())[:10]}..."
                )
            pos_embed_W_pos = ProcessWeights.convert_tensor_to_tl_format(
                pos_embed_W_pos_key,
                state_dict,
                state_dict.get(pos_embed_W_pos_key),
                cfg,
                adapter,
                None,
            )
            assert (
                pos_embed_W_pos is not None
            ), f"Positional embedding not found at key {pos_embed_W_pos_key}"
            pos_embed_W_pos = pos_embed_W_pos - pos_embed_W_pos.mean(-1, keepdim=True)
            state_dict[pos_embed_W_pos_key] = ProcessWeights.convert_tensor_to_hf_format(
                pos_embed_W_pos_key, pos_embed_W_pos, cfg, adapter, None
            )
        for l in range(cfg.n_layers):
            attn_W_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_O", adapter)
            attn_b_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_O", adapter)
            try:
                mlp_W_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.W_out", adapter)
                mlp_b_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.b_out", adapter)
            except ValueError:
                mlp_W_out_key = None
                mlp_b_out_key = None
            if attn_W_O_key not in state_dict:
                raise KeyError(
                    f"Expected attention W_O key '{attn_W_O_key}' not found in state_dict for layer {l}. Available keys: {list(state_dict.keys())[:10]}..."
                )
            attn_W_O = ProcessWeights.convert_tensor_to_tl_format(
                attn_W_O_key, state_dict, state_dict.get(attn_W_O_key), cfg, adapter, l
            )
            assert attn_W_O is not None, f"Attention W_O not found at key {attn_W_O_key}"
            attn_W_O = attn_W_O - attn_W_O.mean(-1, keepdim=True)
            state_dict[attn_W_O_key] = ProcessWeights.convert_tensor_to_hf_format(
                attn_W_O_key, attn_W_O, cfg, adapter, l
            )
            if attn_b_O_key in state_dict:
                attn_b_O = ProcessWeights.convert_tensor_to_tl_format(
                    attn_b_O_key, state_dict, state_dict.get(attn_b_O_key), cfg, adapter, l
                )
                assert attn_b_O is not None, f"Attention b_O not found at key {attn_b_O_key}"
                attn_b_O = attn_b_O - attn_b_O.mean()
                state_dict[attn_b_O_key] = ProcessWeights.convert_tensor_to_hf_format(
                    attn_b_O_key, attn_b_O, cfg, adapter, l
                )
            if not getattr(cfg, "attn_only", False):
                is_moe = getattr(cfg, "num_experts", None) is not None and cfg.num_experts > 0
                if is_moe:
                    num_experts = cfg.num_experts
                    for e in range(num_experts):
                        expert_W_out_key = None
                        expert_b_out_key = None
                        expert_W_out_patterns = [
                            f"blocks.{l}.mlp.experts.{e}.W_out",
                            f"blocks.{l}.mlp.experts.{e}.W_out.weight",
                        ]
                        for pattern in expert_W_out_patterns:
                            if pattern in state_dict:
                                expert_W_out_key = pattern
                                break
                        if expert_W_out_key is None and adapter:
                            try:
                                expert_W_out_key = ProcessWeights._get_param_key(
                                    f"blocks.{l}.mlp.experts.{e}.W_out", adapter
                                )
                            except ValueError:
                                pass
                        if expert_W_out_key and expert_W_out_key in state_dict:
                            expert_W_out = ProcessWeights.convert_tensor_to_tl_format(
                                expert_W_out_key,
                                state_dict,
                                state_dict.get(expert_W_out_key),
                                cfg,
                                adapter,
                                l,
                            )
                            assert (
                                expert_W_out is not None
                            ), f"Expert W_out not found at key {expert_W_out_key}"
                            expert_W_out = expert_W_out - expert_W_out.mean(-1, keepdim=True)
                            state_dict[
                                expert_W_out_key
                            ] = ProcessWeights.convert_tensor_to_hf_format(
                                expert_W_out_key, expert_W_out, cfg, adapter, l
                            )
                        expert_b_out_patterns = [
                            f"blocks.{l}.mlp.experts.{e}.b_out",
                            f"blocks.{l}.mlp.experts.{e}.b_out.bias",
                        ]
                        for pattern in expert_b_out_patterns:
                            if pattern in state_dict:
                                expert_b_out_key = pattern
                                break
                        if expert_b_out_key and expert_b_out_key in state_dict:
                            expert_b_out = ProcessWeights.convert_tensor_to_tl_format(
                                expert_b_out_key,
                                state_dict,
                                state_dict.get(expert_b_out_key),
                                cfg,
                                adapter,
                                l,
                            )
                            assert (
                                expert_b_out is not None
                            ), f"Expert b_out not found at key {expert_b_out_key}"
                            expert_b_out = expert_b_out - expert_b_out.mean()
                            state_dict[
                                expert_b_out_key
                            ] = ProcessWeights.convert_tensor_to_hf_format(
                                expert_b_out_key, expert_b_out, cfg, adapter, l
                            )
                elif mlp_W_out_key is not None:
                    if mlp_W_out_key not in state_dict:
                        raise KeyError(
                            f"Expected MLP W_out key '{mlp_W_out_key}' not found in state_dict for layer {l}. Available keys: {list(state_dict.keys())[:10]}..."
                        )
                    mlp_W_out = ProcessWeights.convert_tensor_to_tl_format(
                        mlp_W_out_key, state_dict, state_dict.get(mlp_W_out_key), cfg, adapter, l
                    )
                    assert mlp_W_out is not None, f"MLP W_out not found at key {mlp_W_out_key}"
                    mlp_W_out = mlp_W_out - mlp_W_out.mean(-1, keepdim=True)
                    state_dict[mlp_W_out_key] = ProcessWeights.convert_tensor_to_hf_format(
                        mlp_W_out_key, mlp_W_out, cfg, adapter, l
                    )
                    if mlp_b_out_key is not None and mlp_b_out_key in state_dict:
                        mlp_b_out = ProcessWeights.convert_tensor_to_tl_format(
                            mlp_b_out_key,
                            state_dict,
                            state_dict.get(mlp_b_out_key),
                            cfg,
                            adapter,
                            l,
                        )
                        assert mlp_b_out is not None, f"MLP b_out not found at key {mlp_b_out_key}"
                        mlp_b_out = mlp_b_out - mlp_b_out.mean()
                        state_dict[mlp_b_out_key] = ProcessWeights.convert_tensor_to_hf_format(
                            mlp_b_out_key, mlp_b_out, cfg, adapter, l
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
        # Make a deep copy to avoid modifying the original
        state_dict = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
        }
        unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
        unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
        if unembed_W_U_key not in state_dict:
            raise KeyError(
                f"Expected unembedding weight key '{unembed_W_U_key}' not found in state_dict. Available keys: {list(state_dict.keys())[:10]}..."
            )
        W_U = ProcessWeights.convert_tensor_to_tl_format(
            unembed_W_U_key, state_dict, state_dict.get(unembed_W_U_key), None, adapter, None
        )
        assert W_U is not None, f"Unembed weight not found at key {unembed_W_U_key}"
        W_U = W_U - W_U.mean(-1, keepdim=True)
        state_dict[unembed_W_U_key] = ProcessWeights.convert_tensor_to_hf_format(
            unembed_W_U_key, W_U, None, adapter, None
        )
        if unembed_b_U_key in state_dict:
            unembed_b_U = ProcessWeights.convert_tensor_to_tl_format(
                unembed_b_U_key, state_dict, state_dict.get(unembed_b_U_key), None, adapter, None
            )
            assert unembed_b_U is not None, f"Unembed bias not found at key {unembed_b_U_key}"
            unembed_b_U = unembed_b_U - unembed_b_U.mean()
            state_dict[unembed_b_U_key] = ProcessWeights.convert_tensor_to_hf_format(
                unembed_b_U_key, unembed_b_U, None, adapter, None
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
        # Make a deep copy to avoid modifying the original
        state_dict = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
        }
        layer = 0
        for layer in range(cfg.n_layers):
            split_v_bias_key = f"blocks.{layer}.attn.v.bias"
            if split_v_bias_key in state_dict:
                b_V_key = split_v_bias_key
                W_O_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.W_O", adapter)
                b_O_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_O", adapter)
            else:
                if getattr(cfg, "n_key_value_heads", None) is None:
                    b_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_V", adapter)
                else:
                    b_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn._b_V", adapter)
                W_O_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.W_O", adapter)
                b_O_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_O", adapter)
            if b_V_key in state_dict:
                b_V = ProcessWeights.convert_tensor_to_tl_format(
                    b_V_key, state_dict, state_dict.get(b_V_key), cfg, adapter, layer
                )
                assert b_V is not None, f"Value bias not found at key {b_V_key}"
                if b_V.numel() == 0:
                    continue
                if b_O_key not in state_dict:
                    continue
                W_O = ProcessWeights.convert_tensor_to_tl_format(
                    W_O_key, state_dict, state_dict.get(W_O_key), cfg, adapter, layer
                )
                b_O_original = ProcessWeights.convert_tensor_to_tl_format(
                    b_O_key, state_dict, state_dict.get(b_O_key), cfg, adapter, layer
                )
                assert W_O is not None, f"Attention W_O not found at key {W_O_key}"
                assert b_O_original is not None, f"Attention b_O not found at key {b_O_key}"
                is_split_format = ".attn.v.bias" in b_V_key or ".attn.k.bias" in b_V_key
                if is_split_format and len(b_V.shape) == 1 and (len(W_O.shape) == 2):
                    n_heads = cfg.n_heads
                    d_head = cfg.d_head
                    d_model = cfg.d_model
                    b_V_only = b_V
                    b_V_reshaped = b_V_only.reshape(n_heads, d_head)
                    W_O_reshaped = einops.rearrange(W_O, "(i h) m -> i h m", i=n_heads)
                    folded_b_O = b_O_original + (b_V_reshaped[:, :, None] * W_O_reshaped).sum(
                        [0, 1]
                    )
                    state_dict[b_O_key] = ProcessWeights.convert_tensor_to_hf_format(
                        b_O_key, folded_b_O, cfg, adapter, layer
                    )
                    tl_b_O_key = f"blocks.{layer}.attn.b_O"
                    if tl_b_O_key in state_dict:
                        state_dict[tl_b_O_key] = ProcessWeights.convert_tensor_to_hf_format(
                            tl_b_O_key, folded_b_O, cfg, adapter, layer
                        )
                    state_dict[b_V_key] = ProcessWeights.convert_tensor_to_hf_format(
                        b_V_key, torch.zeros_like(b_V), cfg, adapter, layer
                    )
                elif len(b_V.shape) == 1 and len(W_O.shape) == 2:
                    n_heads = cfg.n_heads
                    d_head = cfg.d_head
                    d_model = cfg.d_model
                    v_bias_start = 2 * n_heads * d_head
                    v_bias_end = 3 * n_heads * d_head
                    b_V_only = b_V[v_bias_start:v_bias_end]
                    if b_V_only.numel() == 0:
                        continue
                    b_V_reshaped = b_V_only.reshape(n_heads, d_head)
                    W_O_reshaped = einops.rearrange(W_O, "(i h) m -> i h m", i=n_heads)
                    folded_b_O = b_O_original + (b_V_reshaped[:, :, None] * W_O_reshaped).sum(
                        [0, 1]
                    )
                    new_b_V = b_V.clone()
                    new_b_V[v_bias_start:v_bias_end] = 0
                    state_dict[b_V_key] = ProcessWeights.convert_tensor_to_hf_format(
                        b_V_key, new_b_V, cfg, adapter, layer
                    )
                elif len(b_V.shape) == 2 and len(W_O.shape) == 3:
                    b_V_original_shape = b_V.shape
                    if getattr(cfg, "n_key_value_heads", None) is not None:
                        b_V = torch.repeat_interleave(
                            b_V, dim=0, repeats=cfg.n_heads // cfg.n_key_value_heads
                        )
                    folded_b_O = b_O_original + (b_V[:, :, None] * W_O).sum([0, 1])
                    state_dict[b_V_key] = ProcessWeights.convert_tensor_to_hf_format(
                        b_V_key,
                        torch.zeros(b_V_original_shape, dtype=b_V.dtype, device=b_V.device),
                        cfg,
                        adapter,
                        layer,
                    )
                elif len(b_V.shape) == 2 and len(W_O.shape) == 2:
                    n_heads = cfg.n_heads
                    d_head = cfg.d_head
                    d_model = cfg.d_model
                    b_V_original_shape = b_V.shape

                    # Handle split QKV format where bias might be [1, d_model] or [n_heads, d_head]
                    is_split_format = ".attn.v.bias" in b_V_key or ".attn.k.bias" in b_V_key
                    if is_split_format and b_V.shape[0] == 1 and b_V.shape[1] == n_heads * d_head:
                        # Reshape [1, n_heads * d_head] to [n_heads, d_head]
                        b_V = b_V.reshape(n_heads, d_head)
                    elif b_V.shape != (n_heads, d_head):
                        # If not already [n_heads, d_head], try to reshape
                        if b_V.numel() == n_heads * d_head:
                            b_V = b_V.reshape(n_heads, d_head)

                    if getattr(cfg, "n_key_value_heads", None) is not None:
                        b_V = torch.repeat_interleave(
                            b_V, dim=0, repeats=cfg.n_heads // cfg.n_key_value_heads
                        )

                    W_O_reshaped = einops.rearrange(W_O, "(i h) m -> i h m", i=n_heads)
                    folded_b_O = b_O_original + (b_V[:, :, None] * W_O_reshaped).sum([0, 1])
                    state_dict[b_V_key] = ProcessWeights.convert_tensor_to_hf_format(
                        b_V_key,
                        torch.zeros(b_V_original_shape, dtype=b_V.dtype, device=b_V.device),
                        cfg,
                        adapter,
                        layer,
                    )
                else:
                    raise ValueError(f"Unexpected tensor shapes: b_V {b_V.shape}, W_O {W_O.shape}")
                state_dict[b_O_key] = ProcessWeights.convert_tensor_to_hf_format(
                    b_O_key, folded_b_O, cfg, adapter, layer
                )
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
        if fold_ln:
            if getattr(cfg, "num_experts", None) and cfg.num_experts > 1:
                pass
            elif getattr(cfg, "normalization_type", "LN") in ["LN", "LNPre"]:
                state_dict = ProcessWeights.fold_layer_norm(
                    state_dict, cfg, fold_biases=True, center_weights=True, adapter=adapter
                )
            elif getattr(cfg, "normalization_type", "LN") in ["RMS", "RMSPre"]:
                state_dict = ProcessWeights.fold_layer_norm(
                    state_dict, cfg, fold_biases=False, center_weights=False, adapter=adapter
                )
                # For RMS normalization, set all layer norm weights to 1.0 after folding
                # since RMS folding doesn't result in identity weights like LayerNorm does
                for layer_idx in range(cfg.n_layers):
                    for ln_name in ["ln1", "ln2"]:
                        ln_w_key = ProcessWeights._get_param_key(
                            f"blocks.{layer_idx}.{ln_name}.w", adapter
                        )
                        if ln_w_key in state_dict:
                            state_dict[ln_w_key] = torch.ones_like(state_dict[ln_w_key])
        if center_writing_weights:
            if getattr(cfg, "normalization_type", "LN") in ["LN", "LNPre"] and (
                not getattr(cfg, "final_rms", False)
            ):
                state_dict = ProcessWeights.center_writing_weights(state_dict, cfg, adapter=adapter)
        if center_unembed:
            state_dict = ProcessWeights.center_unembed(state_dict, adapter=adapter)
        if fold_value_biases:
            state_dict = ProcessWeights.fold_value_biases(state_dict, cfg, adapter=adapter)
            if center_writing_weights and getattr(cfg, "normalization_type", "LN") in [
                "LN",
                "LNPre",
            ]:
                for layer_idx in range(cfg.n_layers):
                    b_O_key = ProcessWeights._get_param_key(f"blocks.{layer_idx}.attn.b_O", adapter)
                    if b_O_key in state_dict:
                        b_O = ProcessWeights.convert_tensor_to_tl_format(
                            b_O_key, state_dict, state_dict.get(b_O_key), cfg, adapter, layer_idx
                        )
                        assert b_O is not None, f"Attention b_O not found at key {b_O_key}"
                        b_O = b_O - b_O.mean()
                        state_dict[b_O_key] = ProcessWeights.convert_tensor_to_hf_format(
                            b_O_key, b_O, cfg, adapter, layer_idx
                        )
        if refactor_factored_attn_matrices:
            state_dict = ProcessWeights.refactor_factored_attn_matrices(
                state_dict, cfg, adapter=adapter
            )
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
        # Make a deep copy to avoid modifying the original
        state_dict = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
        }
        assert (
            getattr(cfg, "positional_embedding_type", "standard") != "rotary"
        ), "You can't refactor the QK circuit when using rotary embeddings (as the QK matrix depends on the position of the query and key)"

        for l in range(cfg.n_layers):
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
            W_Q = ProcessWeights.convert_tensor_to_tl_format(
                W_Q_key, state_dict, state_dict.get(W_Q_key), cfg, adapter, l
            )
            b_Q = ProcessWeights.convert_tensor_to_tl_format(
                b_Q_key, state_dict, state_dict.get(b_Q_key), cfg, adapter, l
            )
            W_K = ProcessWeights.convert_tensor_to_tl_format(
                W_K_key, state_dict, state_dict.get(W_K_key), cfg, adapter, l
            )
            b_K = ProcessWeights.convert_tensor_to_tl_format(
                b_K_key, state_dict, state_dict.get(b_K_key), cfg, adapter, l
            )
            assert W_Q is not None, f"W_Q not found at key {W_Q_key}"
            assert b_Q is not None, f"b_Q not found at key {b_Q_key}"
            assert W_K is not None, f"W_K not found at key {W_K_key}"
            assert b_K is not None, f"b_K not found at key {b_K_key}"

            W_Q_eff = torch.cat([W_Q, b_Q[:, None, :]], dim=1)
            W_K_eff = torch.cat([W_K, b_K[:, None, :]], dim=1)

            W_Q_eff_even, W_K_eff_even_T = (
                FactoredMatrix(W_Q_eff, W_K_eff.transpose(-1, -2)).make_even().pair
            )
            W_K_eff_even = W_K_eff_even_T.transpose(-1, -2)

            state_dict[W_Q_key] = ProcessWeights.convert_tensor_to_hf_format(
                W_Q_key, W_Q_eff_even[:, :-1, :], cfg, adapter, l
            )
            state_dict[b_Q_key] = ProcessWeights.convert_tensor_to_hf_format(
                b_Q_key, W_Q_eff_even[:, -1, :], cfg, adapter, l
            )
            state_dict[W_K_key] = ProcessWeights.convert_tensor_to_hf_format(
                W_K_key, W_K_eff_even[:, :-1, :], cfg, adapter, l
            )
            state_dict[b_K_key] = ProcessWeights.convert_tensor_to_hf_format(
                b_K_key, W_K_eff_even[:, -1, :], cfg, adapter, l
            )

            # W_OV = W_V @ W_O
            W_V = ProcessWeights.convert_tensor_to_tl_format(
                W_V_key, state_dict, state_dict.get(W_V_key), cfg, adapter, l
            )
            W_O = ProcessWeights.convert_tensor_to_tl_format(
                W_O_key, state_dict, state_dict.get(W_O_key), cfg, adapter, l
            )

            # Factors the bias to be consistent.
            b_V = ProcessWeights.convert_tensor_to_tl_format(
                b_V_key, state_dict, state_dict.get(b_V_key), cfg, adapter, l
            )
            b_O = ProcessWeights.convert_tensor_to_tl_format(
                b_O_key, state_dict, state_dict.get(b_O_key), cfg, adapter, l
            )
            assert W_V is not None, f"W_V not found at key {W_V_key}"
            assert W_O is not None, f"W_O not found at key {W_O_key}"
            assert b_V is not None, f"b_V not found at key {b_V_key}"
            assert b_O is not None, f"b_O not found at key {b_O_key}"

            # Add singleton dimension for broadcasting
            b_V_expanded = einops.rearrange(b_V, "head_index d_head -> head_index d_head 1")

            # Element-wise multiplication of b_V and W_O
            b_V_times_W_O = b_V_expanded * W_O

            # Sum over d_head and head_index dimensions
            b_V_contribution = b_V_times_W_O.sum(1).sum(0)

            effective_bias = b_O + b_V_contribution
            state_dict[b_V_key] = ProcessWeights.convert_tensor_to_hf_format(
                b_V_key, torch.zeros_like(b_V), cfg, adapter, l
            )
            state_dict[b_O_key] = ProcessWeights.convert_tensor_to_hf_format(
                b_O_key, effective_bias, cfg, adapter, l
            )

            # Helper class to efficiently deal with low rank factored matrices.
            W_OV = FactoredMatrix(W_V, W_O)
            U, S, Vh = W_OV.svd()
            state_dict[W_V_key] = ProcessWeights.convert_tensor_to_hf_format(
                W_V_key, U @ S.diag_embed(), cfg, adapter, l
            )
            state_dict[W_O_key] = ProcessWeights.convert_tensor_to_hf_format(
                W_O_key, utils.transpose(Vh), cfg, adapter, l
            )

        return state_dict

    @overload
    @staticmethod
    def convert_tensor_to_tl_format(
        param_name: str,
        model_state_dict: Dict[str, torch.Tensor],
        tensor: torch.Tensor,
        cfg: Optional["TransformerLensConfig"],
        adapter: Optional["ArchitectureAdapter"] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        ...

    @overload
    @staticmethod
    def convert_tensor_to_tl_format(
        param_name: str,
        model_state_dict: Dict[str, torch.Tensor],
        tensor: None,
        cfg: Optional["TransformerLensConfig"],
        adapter: Optional["ArchitectureAdapter"] = None,
        layer_idx: Optional[int] = None,
    ) -> None:
        ...

    @staticmethod
    def convert_tensor_to_tl_format(
        param_name: str,
        model_state_dict: Dict[str, torch.Tensor],
        tensor: Optional[torch.Tensor],
        cfg: Optional["TransformerLensConfig"],
        adapter: Optional["ArchitectureAdapter"] = None,
        layer_idx: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Convert a tensor from its original format to TransformerLens format.

        Args:
            param_name: The parameter name in TransformerLens format (e.g., "blocks.0.attn.W_Q")
            model_state_dict: The model's state dictionary containing the actual tensors
            tensor: The tensor to convert, or None for optional parameters
            cfg: Model configuration
            adapter: Optional architecture adapter for component retrieval and key translation.
                If None, the tensor is returned unchanged.
            layer_idx: Layer index (required for layer-specific parameters)

        Returns:
            The tensor converted to TransformerLens format, or None if the parameter doesn't exist
            (which is valid for optional parameters like biases in models that don't use them).
            If adapter is None, returns the tensor unchanged.
        """
        # If no adapter provided, return tensor unchanged (handle None gracefully)
        if adapter is None:
            return tensor

        if (
            hasattr(adapter, "weight_processing_conversions")
            and adapter.weight_processing_conversions is not None
        ):
            import re

            # Create placeholder param name by replacing layer index with {i}
            placeholder_param_name = param_name
            if "blocks." in param_name:
                placeholder_param_name = re.sub(r"blocks\.(\d+)\.", "blocks.{i}.", param_name)

            # Check if we have a conversion for this parameter
            if placeholder_param_name in adapter.weight_processing_conversions:
                param_conversion = adapter.weight_processing_conversions[placeholder_param_name]

                # Handle both ParamProcessingConversion objects and legacy string mappings
                if isinstance(param_conversion, str):
                    # Legacy string mapping - just return the tensor as-is
                    # (string mappings are handled elsewhere in the architecture adapter)
                    return tensor
                else:
                    # Let ParamProcessingConversion handle the fetching and conversion
                    return param_conversion.convert(model_state_dict, param_name)
            else:
                # No conversion defined, return tensor as-is (may be None for optional params)
                return tensor
        else:
            # No conversions defined, return tensor as-is (may be None for optional params)
            return tensor

    @overload
    @staticmethod
    def convert_tensor_to_hf_format(
        param_name: str,
        tensor: torch.Tensor,
        cfg: Optional["TransformerLensConfig"],
        adapter: Optional["ArchitectureAdapter"] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        ...

    @overload
    @staticmethod
    def convert_tensor_to_hf_format(
        param_name: str,
        tensor: None,
        cfg: Optional["TransformerLensConfig"],
        adapter: Optional["ArchitectureAdapter"] = None,
        layer_idx: Optional[int] = None,
    ) -> None:
        ...

    @staticmethod
    def convert_tensor_to_hf_format(
        param_name: str,
        tensor: Optional[torch.Tensor],
        cfg: Optional["TransformerLensConfig"],
        adapter: Optional["ArchitectureAdapter"] = None,
        layer_idx: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Convert a tensor from TransformerLens format back to its original format.

        Args:
            param_name: The parameter name in TransformerLens format (e.g., "blocks.0.attn.W_Q")
            tensor: The tensor to convert (in TransformerLens format), or None if parameter is optional
            cfg: Model configuration
            adapter: Optional architecture adapter for component retrieval and key translation.
                If None, the tensor is returned unchanged.
            layer_idx: Layer index (required for layer-specific parameters)

        Returns:
            The tensor converted back to original format, or None if tensor was None.
            If adapter is None, returns the tensor unchanged.
        """
        # Handle None tensors (optional parameters)
        if tensor is None:
            return None

        # If no adapter provided, return tensor unchanged
        if adapter is None:
            return tensor

        if (
            hasattr(adapter, "weight_processing_conversions")
            and adapter.weight_processing_conversions is not None
        ):
            import re

            # Create placeholder param name by replacing layer index with {i}
            placeholder_param_name = param_name
            if "blocks." in param_name:
                placeholder_param_name = re.sub(r"blocks\.(\d+)\.", "blocks.{i}.", param_name)

            # Check if we have a conversion for this parameter
            if placeholder_param_name in adapter.weight_processing_conversions:
                param_conversion = adapter.weight_processing_conversions[placeholder_param_name]

                # Handle both ParamProcessingConversion objects and legacy string mappings
                if isinstance(param_conversion, str):
                    # Legacy string mapping - just return the tensor as-is
                    return tensor
                else:
                    # Use ParamProcessingConversion to handle reverting
                    return param_conversion.revert(tensor)
            else:
                return tensor
        else:
            return tensor

    @staticmethod
    def distribute_weights_to_components(
        state_dict: Dict[str, torch.Tensor],
        component_mapping: Dict[str, Any],
        verbose: bool = False,
    ) -> None:
        """Distribute processed weights from state_dict to generalized components.

        This function loops through the component_mapping and extracts relevant weights
        for each component using filter_dict_by_prefix, then calls set_processed_weights
        on each component. For list components (like blocks), it determines the number
        of items and distributes weights to each indexed component.

        Args:
            state_dict: Dictionary of processed weights in MODERN TransformerLens format
                (e.g., blocks.0.attn.q.weight, not transformer.h.0.attn.q.weight)
            component_mapping: Dictionary (real_components) mapping TL keys to tuples of
                (remote_path, component_instance), where component_instance can be either
                a single component or a list of components
            verbose: If True, print detailed information about weight distribution

        Example:
            For a real_components mapping like:
            {
                "embed": ("transformer.wte", <EmbeddingBridge instance>),
                "blocks": ("transformer.h", [<BlockBridge 0>, <BlockBridge 1>, ...]),
                "unembed": ("lm_head", <UnembeddingBridge instance>)
            }

            With modern TL keys in state_dict like "embed.weight", "blocks.0.attn.q.weight":
            1. Extract weights starting with "embed" and pass to embed component
            2. For blocks, extract all "blocks.*" weights, determine the number of blocks,
               then for each block index, extract weights for that specific block
            3. Extract "unembed" weights and pass to unembed component
        """
        from transformer_lens.utilities import filter_dict_by_prefix

        if verbose:
            print(f"\n{'='*80}")
            print(f"distribute_weights_to_components: Starting weight distribution")
            print(f"State dict has {len(state_dict)} keys")
            print(f"Component mapping has {len(component_mapping)} components")
            print(f"{'='*80}\n")

        for component_name, component_tuple in component_mapping.items():
            # component_mapping is real_components format: (remote_path, instance)
            # instance can be either a single component or a list of components
            if not isinstance(component_tuple, tuple):
                raise ValueError(
                    f"Expected tuple for component '{component_name}' in real_components, "
                    f"but got {type(component_tuple).__name__}: {component_tuple}"
                )
            remote_key, component = component_tuple
            is_list = isinstance(component, list)

            # Use the component_name (TL format) as prefix instead of remote_key (HF format)
            # since state_dict now has modern TL keys
            tl_prefix = component_name

            if verbose:
                print(f"\nProcessing component: {component_name}")
                print(f"  Remote key (HF): {remote_key}")
                print(f"  TL prefix: {tl_prefix}")
                print(f"  Is list: {is_list}")

            if is_list:
                # This is a list component like "blocks"
                # Extract all weights that start with this prefix
                all_list_weights = filter_dict_by_prefix(state_dict, tl_prefix)

                if verbose:
                    print(f"  Found {len(all_list_weights)} weights for list component")
                    print(f"  List has {len(component)} instances")

                # Component is a list of actual instances
                for i, instance in enumerate(component):
                    # Extract weights for this specific index
                    # This will get keys like "0.attn.q.weight" and strip the "0." to get "attn.q.weight"
                    indexed_weights = filter_dict_by_prefix(all_list_weights, str(i))

                    if verbose:
                        print(f"    Instance {i}: Found {len(indexed_weights)} weights")
                        for key in indexed_weights.keys():
                            print(f"      - {key}")

                    # Skip if no weights found for this component (e.g., Q/K/V Linear sub-components
                    # that get their weights from parent JointQKVAttentionBridge)
                    if len(indexed_weights) == 0:
                        if verbose:
                            print(f"    Skipping instance {i} - no weights found")
                        continue

                    instance.set_processed_weights(indexed_weights, verbose=verbose)
            else:
                # This is a single component (not a list)
                component_weights = filter_dict_by_prefix(state_dict, tl_prefix)

                if verbose:
                    print(f"  Found {len(component_weights)} weights for single component")
                    for key in component_weights.keys():
                        print(f"    - {key}")

                # Skip if no weights found for this component
                if len(component_weights) == 0:
                    if verbose:
                        print(f"  Skipping component - no weights found")
                    continue

                component.set_processed_weights(component_weights, verbose=verbose)
