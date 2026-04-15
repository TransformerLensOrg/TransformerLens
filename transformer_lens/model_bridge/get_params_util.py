"""Utility function for getting model parameters in TransformerLens format."""
import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


def _get_n_kv_heads(cfg) -> int:
    """Resolve the number of key/value heads, falling back to n_heads."""
    if hasattr(cfg, "n_key_value_heads") and isinstance(cfg.n_key_value_heads, int):
        return cfg.n_key_value_heads
    return cfg.n_heads


def _reshape_kv_weight(weight: torch.Tensor, cfg, device, dtype) -> torch.Tensor:
    """Reshape a K or V weight matrix to (n_heads, d_model, d_head)."""
    d_head = cfg.d_model // cfg.n_heads
    if weight.shape == (cfg.d_model, cfg.d_model):
        return weight.reshape(cfg.n_heads, cfg.d_model, d_head)
    if weight.shape == (cfg.d_head, cfg.d_model) or weight.shape == (
        cfg.d_model // cfg.n_heads,
        cfg.d_model,
    ):
        return weight.transpose(0, 1).unsqueeze(0).expand(cfg.n_heads, -1, -1)
    if weight.numel() == cfg.n_heads * cfg.d_model * cfg.d_head:
        return weight.view(cfg.n_heads, cfg.d_model, cfg.d_head)
    return torch.zeros(cfg.n_heads, cfg.d_model, cfg.d_head, device=device, dtype=dtype)


def _get_or_create_bias(bias, n_heads: int, d_head: int, device, dtype) -> torch.Tensor:
    """Reshape existing bias to (n_heads, d_head), or create zeros if None."""
    if bias is not None:
        return bias.reshape(n_heads, -1)
    return torch.zeros(n_heads, d_head, device=device, dtype=dtype)


def get_bridge_params(bridge) -> Dict[str, torch.Tensor]:
    """Access to model parameters in the format expected by SVDInterpreter.

    For hybrid architectures, only layers with attention get attention keys
    (W_Q, W_K, etc.). Non-attention layers (SSM, linear-attention) are skipped
    rather than filled with zeros — this prevents downstream consumers like
    SVDInterpreter from treating synthetic zeros as real weights.

    Args:
        bridge: TransformerBridge instance

    Returns:
        dict: Dictionary of parameter tensors with TransformerLens naming convention.
            For hybrid models, attention keys only exist for layers that have attention.

    Raises:
        ValueError: If configuration is inconsistent (e.g., cfg.n_layers != len(blocks))
    """
    params_dict = {}

    def _get_device_dtype():
        """Infer device/dtype from the first available model parameter."""
        device = getattr(bridge.cfg, "device", None) or torch.device("cpu")
        dtype = torch.float32
        try:
            first_param = next(bridge.parameters())
            device = first_param.device
            dtype = first_param.dtype
        except (StopIteration, TypeError, AttributeError):
            pass
        return (device, dtype)

    try:
        params_dict["embed.W_E"] = bridge.embed.weight
    except AttributeError:
        device, dtype = _get_device_dtype()
        params_dict["embed.W_E"] = torch.zeros(
            bridge.cfg.d_vocab, bridge.cfg.d_model, device=device, dtype=dtype
        )
    try:
        params_dict["pos_embed.W_pos"] = bridge.pos_embed.weight
    except AttributeError:
        device, dtype = _get_device_dtype()
        params_dict["pos_embed.W_pos"] = torch.zeros(
            bridge.cfg.n_ctx, bridge.cfg.d_model, device=device, dtype=dtype
        )
    for layer_idx in range(bridge.cfg.n_layers):
        if layer_idx >= len(bridge.blocks):
            raise ValueError(
                f"Configuration mismatch: cfg.n_layers={bridge.cfg.n_layers} but only {len(bridge.blocks)} blocks found. Layer {layer_idx} does not exist."
            )
        block = bridge.blocks[layer_idx]

        # Only extract attention params from blocks that have attention.
        # Non-attention layers (SSM, linear-attention) are skipped entirely
        # rather than filled with zeros — this prevents consumers like
        # SVDInterpreter from treating synthetic zeros as real weights.
        try:
            has_attn = "attn" in block._modules
        except (TypeError, AttributeError):
            # Mock objects or non-nn.Module blocks: fall back to hasattr
            has_attn = hasattr(block, "attn")
        if has_attn:
            try:
                w_q = block.attn.q.weight
                w_k = block.attn.k.weight
                w_v = block.attn.v.weight
                w_o = block.attn.o.weight
                if w_q.shape == (bridge.cfg.d_model, bridge.cfg.d_model):
                    d_head = bridge.cfg.d_model // bridge.cfg.n_heads
                    w_q = w_q.reshape(bridge.cfg.n_heads, bridge.cfg.d_model, d_head)
                    w_o = w_o.reshape(bridge.cfg.n_heads, d_head, bridge.cfg.d_model)
                    device, dtype = _get_device_dtype()
                    w_k = _reshape_kv_weight(w_k, bridge.cfg, device, dtype)
                    w_v = _reshape_kv_weight(w_v, bridge.cfg, device, dtype)
                params_dict[f"blocks.{layer_idx}.attn.W_Q"] = w_q
                params_dict[f"blocks.{layer_idx}.attn.W_K"] = w_k
                params_dict[f"blocks.{layer_idx}.attn.W_V"] = w_v
                params_dict[f"blocks.{layer_idx}.attn.W_O"] = w_o
                device, dtype = _get_device_dtype()
                n_kv_heads = _get_n_kv_heads(bridge.cfg)
                params_dict[f"blocks.{layer_idx}.attn.b_Q"] = _get_or_create_bias(
                    block.attn.q.bias, bridge.cfg.n_heads, bridge.cfg.d_head, device, dtype
                )
                params_dict[f"blocks.{layer_idx}.attn.b_K"] = _get_or_create_bias(
                    block.attn.k.bias, n_kv_heads, bridge.cfg.d_head, device, dtype
                )
                params_dict[f"blocks.{layer_idx}.attn.b_V"] = _get_or_create_bias(
                    block.attn.v.bias, n_kv_heads, bridge.cfg.d_head, device, dtype
                )
                if block.attn.o.bias is not None:
                    params_dict[f"blocks.{layer_idx}.attn.b_O"] = block.attn.o.bias
                else:
                    device, dtype = _get_device_dtype()
                    params_dict[f"blocks.{layer_idx}.attn.b_O"] = torch.zeros(
                        bridge.cfg.d_model, device=device, dtype=dtype
                    )
            except AttributeError as e:
                logger.debug(
                    "Block %d has 'attn' in _modules but attention params could not "
                    "be extracted (missing q/k/v/o?): %s — skipping attention weights "
                    "for this layer",
                    layer_idx,
                    e,
                )
        try:
            mlp_in = getattr(block.mlp, "in", None) or getattr(block.mlp, "input", None)
            if mlp_in is None:
                raise AttributeError("MLP has no 'in' or 'input' attribute")
            params_dict[f"blocks.{layer_idx}.mlp.W_in"] = mlp_in.weight
            params_dict[f"blocks.{layer_idx}.mlp.W_out"] = block.mlp.out.weight
            mlp_in_bias = mlp_in.bias
            if mlp_in_bias is not None:
                params_dict[f"blocks.{layer_idx}.mlp.b_in"] = mlp_in_bias
            else:
                device, dtype = _get_device_dtype()
                d_mlp = bridge.cfg.d_mlp if bridge.cfg.d_mlp is not None else 4 * bridge.cfg.d_model
                params_dict[f"blocks.{layer_idx}.mlp.b_in"] = torch.zeros(
                    d_mlp, device=device, dtype=dtype
                )
            mlp_out_bias = block.mlp.out.bias
            if mlp_out_bias is not None:
                params_dict[f"blocks.{layer_idx}.mlp.b_out"] = mlp_out_bias
            else:
                device, dtype = _get_device_dtype()
                params_dict[f"blocks.{layer_idx}.mlp.b_out"] = torch.zeros(
                    bridge.cfg.d_model, device=device, dtype=dtype
                )
            if hasattr(block.mlp, "gate") and hasattr(block.mlp.gate, "weight"):
                params_dict[f"blocks.{layer_idx}.mlp.W_gate"] = block.mlp.gate.weight
                if hasattr(block.mlp.gate, "bias") and block.mlp.gate.bias is not None:
                    params_dict[f"blocks.{layer_idx}.mlp.b_gate"] = block.mlp.gate.bias
        except AttributeError:
            device, dtype = _get_device_dtype()
            d_mlp = bridge.cfg.d_mlp if bridge.cfg.d_mlp is not None else 4 * bridge.cfg.d_model
            params_dict[f"blocks.{layer_idx}.mlp.W_in"] = torch.zeros(
                bridge.cfg.d_model, d_mlp, device=device, dtype=dtype
            )
            params_dict[f"blocks.{layer_idx}.mlp.W_out"] = torch.zeros(
                d_mlp, bridge.cfg.d_model, device=device, dtype=dtype
            )
            params_dict[f"blocks.{layer_idx}.mlp.b_in"] = torch.zeros(
                d_mlp, device=device, dtype=dtype
            )
            params_dict[f"blocks.{layer_idx}.mlp.b_out"] = torch.zeros(
                bridge.cfg.d_model, device=device, dtype=dtype
            )
    try:
        params_dict["unembed.W_U"] = bridge.unembed.weight.T
    except AttributeError:
        device, dtype = _get_device_dtype()
        params_dict["unembed.W_U"] = torch.zeros(
            bridge.cfg.d_model, bridge.cfg.d_vocab, device=device, dtype=dtype
        )
    try:
        params_dict["unembed.b_U"] = bridge.unembed.b_U
    except AttributeError:
        device, dtype = _get_device_dtype()
        params_dict["unembed.b_U"] = torch.zeros(bridge.cfg.d_vocab, device=device, dtype=dtype)
    return params_dict
