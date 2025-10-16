"""Utility function for getting model parameters in TransformerLens format."""

from typing import Dict

import torch


def get_bridge_params(bridge) -> Dict[str, torch.Tensor]:
    """Access to model parameters in the format expected by SVDInterpreter.

    For missing weights, returns zero tensors of appropriate shape instead of raising exceptions.
    This ensures compatibility across different model architectures.

    Args:
        bridge: TransformerBridge instance

    Returns:
        dict: Dictionary of parameter tensors with TransformerLens naming convention

    Raises:
        ValueError: If configuration is inconsistent (e.g., cfg.n_layers != len(blocks))
    """
    params_dict = {}

    # Helper function to get device and dtype from existing weights
    def _get_device_dtype():
        device = bridge.cfg.device if hasattr(bridge.cfg, "device") else torch.device("cpu")
        dtype = torch.float32  # Default dtype

        # Try to get dtype from existing weights
        try:
            device = bridge.embed.weight.device
            dtype = bridge.embed.weight.dtype
        except AttributeError:
            try:
                device = bridge.pos_embed.weight.device
                dtype = bridge.pos_embed.weight.dtype
            except AttributeError:
                if len(bridge.blocks) > 0:
                    try:
                        device = bridge.blocks[0].attn.q.weight.device
                        dtype = bridge.blocks[0].attn.q.weight.dtype
                    except AttributeError:
                        pass
        return device, dtype

    # Add embedding weights
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

    # Add attention weights
    for layer_idx in range(bridge.cfg.n_layers):
        # Validate that the layer actually exists
        if layer_idx >= len(bridge.blocks):
            raise ValueError(
                f"Configuration mismatch: cfg.n_layers={bridge.cfg.n_layers} but only "
                f"{len(bridge.blocks)} blocks found. Layer {layer_idx} does not exist."
            )

        block = bridge.blocks[layer_idx]

        try:
            # Attention weights - reshape to expected format
            w_q = block.attn.q.weight
            w_k = block.attn.k.weight
            w_v = block.attn.v.weight
            w_o = block.attn.o.weight

            # Reshape from [d_model, d_model] to [n_heads, d_model, d_head] and [n_heads, d_head, d_model]
            # Handle different attention architectures (Multi-Head, Multi-Query, Grouped Query)
            if w_q.shape == (bridge.cfg.d_model, bridge.cfg.d_model):
                d_head = bridge.cfg.d_model // bridge.cfg.n_heads
                w_q = w_q.reshape(bridge.cfg.n_heads, bridge.cfg.d_model, d_head)
                w_o = w_o.reshape(bridge.cfg.n_heads, d_head, bridge.cfg.d_model)

                # Handle K and V weights - they might have different shapes in Multi-Query Attention
                if w_k.shape == (bridge.cfg.d_model, bridge.cfg.d_model):
                    w_k = w_k.reshape(bridge.cfg.n_heads, bridge.cfg.d_model, d_head)
                elif w_k.shape == (bridge.cfg.d_head, bridge.cfg.d_model) or w_k.shape == (
                    bridge.cfg.d_model // bridge.cfg.n_heads,
                    bridge.cfg.d_model,
                ):
                    # Multi-Query Attention: single K head shared across all Q heads
                    # Need to transpose to match expected [n_heads, d_model, d_head] format
                    w_k = w_k.transpose(0, 1).unsqueeze(0).expand(bridge.cfg.n_heads, -1, -1)
                else:
                    # Try to reshape based on element count
                    if w_k.numel() == bridge.cfg.n_heads * bridge.cfg.d_model * bridge.cfg.d_head:
                        w_k = w_k.view(bridge.cfg.n_heads, bridge.cfg.d_model, bridge.cfg.d_head)
                    else:
                        # Create zero tensor if can't reshape
                        device, dtype = _get_device_dtype()
                        w_k = torch.zeros(
                            bridge.cfg.n_heads,
                            bridge.cfg.d_model,
                            bridge.cfg.d_head,
                            device=device,
                            dtype=dtype,
                        )

                if w_v.shape == (bridge.cfg.d_model, bridge.cfg.d_model):
                    w_v = w_v.reshape(bridge.cfg.n_heads, bridge.cfg.d_model, d_head)
                elif w_v.shape == (bridge.cfg.d_head, bridge.cfg.d_model) or w_v.shape == (
                    bridge.cfg.d_model // bridge.cfg.n_heads,
                    bridge.cfg.d_model,
                ):
                    # Multi-Query Attention: single V head shared across all Q heads
                    # Need to transpose to match expected [n_heads, d_model, d_head] format
                    w_v = w_v.transpose(0, 1).unsqueeze(0).expand(bridge.cfg.n_heads, -1, -1)
                else:
                    # Try to reshape based on element count
                    if w_v.numel() == bridge.cfg.n_heads * bridge.cfg.d_model * bridge.cfg.d_head:
                        w_v = w_v.view(bridge.cfg.n_heads, bridge.cfg.d_model, bridge.cfg.d_head)
                    else:
                        # Create zero tensor if can't reshape
                        device, dtype = _get_device_dtype()
                        w_v = torch.zeros(
                            bridge.cfg.n_heads,
                            bridge.cfg.d_model,
                            bridge.cfg.d_head,
                            device=device,
                            dtype=dtype,
                        )

            params_dict[f"blocks.{layer_idx}.attn.W_Q"] = w_q
            params_dict[f"blocks.{layer_idx}.attn.W_K"] = w_k
            params_dict[f"blocks.{layer_idx}.attn.W_V"] = w_v
            params_dict[f"blocks.{layer_idx}.attn.W_O"] = w_o

            # Attention biases - handle None biases
            if block.attn.q.bias is not None:
                params_dict[f"blocks.{layer_idx}.attn.b_Q"] = block.attn.q.bias.reshape(
                    bridge.cfg.n_heads, -1
                )
            else:
                device, dtype = _get_device_dtype()
                params_dict[f"blocks.{layer_idx}.attn.b_Q"] = torch.zeros(
                    bridge.cfg.n_heads, bridge.cfg.d_head, device=device, dtype=dtype
                )

            if block.attn.k.bias is not None:
                params_dict[f"blocks.{layer_idx}.attn.b_K"] = block.attn.k.bias.reshape(
                    bridge.cfg.n_heads, -1
                )
            else:
                device, dtype = _get_device_dtype()
                params_dict[f"blocks.{layer_idx}.attn.b_K"] = torch.zeros(
                    bridge.cfg.n_heads, bridge.cfg.d_head, device=device, dtype=dtype
                )

            if block.attn.v.bias is not None:
                params_dict[f"blocks.{layer_idx}.attn.b_V"] = block.attn.v.bias.reshape(
                    bridge.cfg.n_heads, -1
                )
            else:
                device, dtype = _get_device_dtype()
                params_dict[f"blocks.{layer_idx}.attn.b_V"] = torch.zeros(
                    bridge.cfg.n_heads, bridge.cfg.d_head, device=device, dtype=dtype
                )

            if block.attn.o.bias is not None:
                params_dict[f"blocks.{layer_idx}.attn.b_O"] = block.attn.o.bias
            else:
                device, dtype = _get_device_dtype()
                params_dict[f"blocks.{layer_idx}.attn.b_O"] = torch.zeros(
                    bridge.cfg.d_model, device=device, dtype=dtype
                )

        except AttributeError:
            # Create zero attention weights for missing attention component
            device, dtype = _get_device_dtype()
            expected_qkv_shape = (bridge.cfg.n_heads, bridge.cfg.d_model, bridge.cfg.d_head)
            expected_o_shape = (bridge.cfg.n_heads, bridge.cfg.d_head, bridge.cfg.d_model)
            expected_qkv_bias_shape = (bridge.cfg.n_heads, bridge.cfg.d_head)
            expected_o_bias_shape = (bridge.cfg.d_model,)

            params_dict[f"blocks.{layer_idx}.attn.W_Q"] = torch.zeros(
                *expected_qkv_shape, device=device, dtype=dtype
            )
            params_dict[f"blocks.{layer_idx}.attn.W_K"] = torch.zeros(
                *expected_qkv_shape, device=device, dtype=dtype
            )
            params_dict[f"blocks.{layer_idx}.attn.W_V"] = torch.zeros(
                *expected_qkv_shape, device=device, dtype=dtype
            )
            params_dict[f"blocks.{layer_idx}.attn.W_O"] = torch.zeros(
                *expected_o_shape, device=device, dtype=dtype
            )
            params_dict[f"blocks.{layer_idx}.attn.b_Q"] = torch.zeros(
                *expected_qkv_bias_shape, device=device, dtype=dtype
            )
            params_dict[f"blocks.{layer_idx}.attn.b_K"] = torch.zeros(
                *expected_qkv_bias_shape, device=device, dtype=dtype
            )
            params_dict[f"blocks.{layer_idx}.attn.b_V"] = torch.zeros(
                *expected_qkv_bias_shape, device=device, dtype=dtype
            )
            params_dict[f"blocks.{layer_idx}.attn.b_O"] = torch.zeros(
                *expected_o_bias_shape, device=device, dtype=dtype
            )

        try:
            # MLP weights - access the actual weight tensors
            # Try "in" first (standard name), then "input" (GPT-2 naming)
            mlp_in = getattr(block.mlp, "in", None) or getattr(block.mlp, "input", None)
            if mlp_in is None:
                raise AttributeError("MLP has no 'in' or 'input' attribute")

            params_dict[f"blocks.{layer_idx}.mlp.W_in"] = mlp_in.weight
            params_dict[f"blocks.{layer_idx}.mlp.W_out"] = block.mlp.out.weight

            # MLP biases - handle None biases
            mlp_in_bias = mlp_in.bias
            if mlp_in_bias is not None:
                params_dict[f"blocks.{layer_idx}.mlp.b_in"] = mlp_in_bias
            else:
                device, dtype = _get_device_dtype()
                d_mlp = (
                    bridge.cfg.d_mlp if bridge.cfg.d_mlp is not None else (4 * bridge.cfg.d_model)
                )
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

            # Add gate weights if they exist
            if hasattr(block.mlp, "gate") and hasattr(block.mlp.gate, "weight"):
                params_dict[f"blocks.{layer_idx}.mlp.W_gate"] = block.mlp.gate.weight
                if hasattr(block.mlp.gate, "bias") and block.mlp.gate.bias is not None:
                    params_dict[f"blocks.{layer_idx}.mlp.b_gate"] = block.mlp.gate.bias

        except AttributeError:
            # Create zero MLP weights for missing MLP component
            device, dtype = _get_device_dtype()
            d_mlp = bridge.cfg.d_mlp if bridge.cfg.d_mlp is not None else (4 * bridge.cfg.d_model)
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

    # Add unembedding weights
    try:
        params_dict["unembed.W_U"] = bridge.unembed.weight.T
    except AttributeError:
        device, dtype = _get_device_dtype()
        params_dict["unembed.W_U"] = torch.zeros(
            bridge.cfg.d_model, bridge.cfg.d_vocab, device=device, dtype=dtype
        )

    # Add unembedding bias
    try:
        params_dict["unembed.b_U"] = bridge.unembed.b_U
    except AttributeError:
        device, dtype = _get_device_dtype()
        params_dict["unembed.b_U"] = torch.zeros(bridge.cfg.d_vocab, device=device, dtype=dtype)

    return params_dict
