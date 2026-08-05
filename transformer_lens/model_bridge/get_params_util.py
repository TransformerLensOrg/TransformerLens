"""Utility function for getting model parameters in TransformerLens format."""
import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _tensor_attr(obj, *names: str) -> Optional[torch.Tensor]:
    """First attribute of ``obj`` among ``names`` that is an actual tensor, else None."""
    for name in names:
        try:
            value = getattr(obj, name)
        except (AttributeError, TypeError):
            continue
        if isinstance(value, torch.Tensor):
            return value
    return None


def get_bridge_params(bridge) -> Dict[str, torch.Tensor]:
    """Model parameters in SVDInterpreter format.

    Reads the bridge components' TL-layout weight properties (``W_Q``,
    ``W_in``, ...), which already account for layout conversion and weight
    processing. For missing weights, returns zero tensors of appropriate shape
    instead of raising exceptions. Skips attn keys for non-attention layers.
    LayerNorm params (``blocks.{i}.ln1.w`` etc.) are included when the modules
    still carry them (i.e. before folding) so consumers can detect fold state.

    Returns:
        dict: Dictionary of parameter tensors with TransformerLens naming convention

    Raises:
        ValueError: If configuration is inconsistent (e.g., cfg.n_layers != len(blocks))
    """
    cfg = bridge.cfg
    params_dict: Dict[str, torch.Tensor] = {}

    def _get_device_dtype():
        """Infer device/dtype from the first available model parameter."""
        device = getattr(cfg, "device", None) or torch.device("cpu")
        dtype = torch.float32
        try:
            first_param = next(bridge.parameters())
            device = first_param.device
            dtype = first_param.dtype
        except (StopIteration, TypeError, AttributeError):
            pass
        return (device, dtype)

    def _zeros(*shape) -> torch.Tensor:
        device, dtype = _get_device_dtype()
        return torch.zeros(*shape, device=device, dtype=dtype)

    embed = _tensor_attr(getattr(bridge, "embed", None), "W_E", "weight")
    params_dict["embed.W_E"] = embed if embed is not None else _zeros(cfg.d_vocab, cfg.d_model)

    pos = _tensor_attr(getattr(bridge, "pos_embed", None), "W_pos", "weight")
    params_dict["pos_embed.W_pos"] = pos if pos is not None else _zeros(cfg.n_ctx, cfg.d_model)

    for layer_idx in range(cfg.n_layers):
        if layer_idx >= len(bridge.blocks):
            raise ValueError(
                f"Configuration mismatch: cfg.n_layers={cfg.n_layers} but only "
                f"{len(bridge.blocks)} blocks found. Layer {layer_idx} does not exist."
            )
        block = bridge.blocks[layer_idx]

        # Skip non-attention layers entirely (no zero-fill — prevents SVDInterpreter garbage)
        try:
            has_attn = "attn" in block._modules
        except (TypeError, AttributeError):
            has_attn = hasattr(block, "attn")  # Mock fallback
        if has_attn:
            attn = block.attn
            w_q = _tensor_attr(attn, "W_Q")
            w_k = _tensor_attr(attn, "W_K")
            w_v = _tensor_attr(attn, "W_V")
            w_o = _tensor_attr(attn, "W_O")
            if w_q is None or w_k is None or w_v is None or w_o is None:
                logger.debug(
                    "Block %d has 'attn' but no TL-layout W_Q/W_K/W_V/W_O properties — "
                    "skipping attention weights for this layer",
                    layer_idx,
                )
            else:
                params_dict[f"blocks.{layer_idx}.attn.W_Q"] = w_q
                params_dict[f"blocks.{layer_idx}.attn.W_K"] = w_k
                params_dict[f"blocks.{layer_idx}.attn.W_V"] = w_v
                params_dict[f"blocks.{layer_idx}.attn.W_O"] = w_o
                n_kv_heads = w_k.shape[0]
                for bias_name, n in (
                    ("b_Q", cfg.n_heads),
                    ("b_K", n_kv_heads),
                    ("b_V", n_kv_heads),
                ):
                    bias = _tensor_attr(attn, bias_name)
                    params_dict[f"blocks.{layer_idx}.attn.{bias_name}"] = (
                        bias if bias is not None else _zeros(n, cfg.d_head)
                    )
                b_O = _tensor_attr(attn, "b_O")
                params_dict[f"blocks.{layer_idx}.attn.b_O"] = (
                    b_O if b_O is not None else _zeros(cfg.d_model)
                )

        d_mlp = cfg.d_mlp if cfg.d_mlp is not None else 4 * cfg.d_model
        mlp = getattr(block, "mlp", None)
        w_in = _tensor_attr(mlp, "W_in")
        if w_in is None:
            params_dict[f"blocks.{layer_idx}.mlp.W_in"] = _zeros(cfg.d_model, d_mlp)
            params_dict[f"blocks.{layer_idx}.mlp.W_out"] = _zeros(d_mlp, cfg.d_model)
            params_dict[f"blocks.{layer_idx}.mlp.b_in"] = _zeros(d_mlp)
            params_dict[f"blocks.{layer_idx}.mlp.b_out"] = _zeros(cfg.d_model)
        else:
            params_dict[f"blocks.{layer_idx}.mlp.W_in"] = w_in
            w_out = _tensor_attr(mlp, "W_out")
            params_dict[f"blocks.{layer_idx}.mlp.W_out"] = (
                w_out if w_out is not None else _zeros(d_mlp, cfg.d_model)
            )
            b_in = _tensor_attr(mlp, "b_in")
            params_dict[f"blocks.{layer_idx}.mlp.b_in"] = (
                b_in if b_in is not None else _zeros(d_mlp)
            )
            b_out = _tensor_attr(mlp, "b_out")
            params_dict[f"blocks.{layer_idx}.mlp.b_out"] = (
                b_out if b_out is not None else _zeros(cfg.d_model)
            )
            w_gate = _tensor_attr(mlp, "W_gate")
            if w_gate is None:
                w_gate = _tensor_attr(getattr(mlp, "gate", None), "weight")
            if w_gate is not None:
                params_dict[f"blocks.{layer_idx}.mlp.W_gate"] = w_gate
                b_gate = _tensor_attr(mlp, "b_gate")
                if b_gate is None:
                    b_gate = _tensor_attr(getattr(mlp, "gate", None), "bias")
                if b_gate is not None:
                    params_dict[f"blocks.{layer_idx}.mlp.b_gate"] = b_gate

        # LN params (present pre-folding; folded models carry identities or none).
        for ln_name in ("ln1", "ln2"):
            ln = getattr(block, ln_name, None)
            ln_w = _tensor_attr(ln, "w", "weight")
            if ln_w is not None:
                params_dict[f"blocks.{layer_idx}.{ln_name}.w"] = ln_w
            ln_b = _tensor_attr(ln, "b", "bias")
            if ln_b is not None:
                params_dict[f"blocks.{layer_idx}.{ln_name}.b"] = ln_b

    ln_final_w = _tensor_attr(getattr(bridge, "ln_final", None), "w", "weight")
    if ln_final_w is not None:
        params_dict["ln_final.w"] = ln_final_w
    ln_final_b = _tensor_attr(getattr(bridge, "ln_final", None), "b", "bias")
    if ln_final_b is not None:
        params_dict["ln_final.b"] = ln_final_b

    unembed = getattr(bridge, "unembed", None)
    w_u = _tensor_attr(unembed, "W_U")
    if w_u is None:
        raw = _tensor_attr(unembed, "weight")
        w_u = raw.T if raw is not None else _zeros(cfg.d_model, cfg.d_vocab)
    params_dict["unembed.W_U"] = w_u
    b_u = _tensor_attr(unembed, "b_U")
    params_dict["unembed.b_U"] = b_u if b_u is not None else _zeros(cfg.d_vocab)

    return params_dict
