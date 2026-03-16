import einops

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_hubert_weights(hf_model, cfg: HookedTransformerConfig):
    """
    Convert transformer encoder weights from a HuggingFace HuBERT model
    into the state_dict expected by Transformer-Lens' HookedEncoder.

    Notes:
    - This intentionally skips the convolutional frontend and feature_projection.
      Those are used directly from the HF model (hf_model.feature_extractor, hf_model.feature_projection).
    - Use model.load_state_dict(state_dict, strict=False) to load these.
    """
    state_dict = {}

    # Try to find the encoder layer list (different HF variants use .layers or .layer)
    encoder = getattr(hf_model, "encoder", None)
    if encoder is None:
        raise ValueError("hf_model has no .encoder attribute")

    encoder_layers = getattr(encoder, "layers", None) or getattr(encoder, "layer", None)
    if encoder_layers is None:
        # maybe hf_model itself is the encoder (unlikely), or a wrapped attribute
        raise ValueError("Couldn't find encoder.layers or encoder.layer on hf_model.encoder")

    # Use cfg dims for reshaping
    d_model = cfg.d_model
    n_heads = cfg.n_heads
    # d_head = d_model // n_heads  # implicit if needed

    for l, layer in enumerate(encoder_layers):
        # --- Attention module ---
        # Some HF variants might call it `attention`, others `self_attn` etc.
        att = getattr(layer, "attention", None) or getattr(layer, "self_attn", None)
        if att is None:
            raise AttributeError(f"Encoder layer {l} has no 'attention' or 'self_attn' attribute")

        # q/k/v/out proj names in HuBERT's HubertAttention: q_proj, k_proj, v_proj, out_proj
        # fall back to common alternatives if present
        q_w = getattr(att, "q_proj", None)
        k_w = getattr(att, "k_proj", None)
        v_w = getattr(att, "v_proj", None)
        o_w = getattr(att, "out_proj", None) or getattr(att, "proj", None)

        if any(x is None for x in (q_w, k_w, v_w, o_w)):
            # Try alternate nested attributes like att.q, att.k, att.v, att.o
            q_w = q_w or getattr(att, "q", None)
            k_w = k_w or getattr(att, "k", None)
            v_w = v_w or getattr(att, "v", None)
            o_w = o_w or getattr(att, "o", None)

        if any(x is None for x in (q_w, k_w, v_w, o_w)):
            raise AttributeError(f"Could not find q/k/v/out projections in layer {l}. Found: {att}")

        # weights are Linear modules: weight shape (out, in)  => same convention as Bert conversion
        # reshape to Transformer-Lens expected shapes using einops
        state_dict[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            q_w.weight, "(i h) m -> i m h", i=n_heads
        )
        if q_w.bias is not None:
            state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
                q_w.bias, "(i h) -> i h", i=n_heads
            )

        state_dict[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            k_w.weight, "(i h) m -> i m h", i=n_heads
        )
        if k_w.bias is not None:
            state_dict[f"blocks.{l}.attn.b_K"] = einops.rearrange(
                k_w.bias, "(i h) -> i h", i=n_heads
            )

        state_dict[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            v_w.weight, "(i h) m -> i m h", i=n_heads
        )
        if v_w.bias is not None:
            state_dict[f"blocks.{l}.attn.b_V"] = einops.rearrange(
                v_w.bias, "(i h) -> i h", i=n_heads
            )

        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            o_w.weight, "m (i h) -> i h m", i=n_heads
        )
        if o_w.bias is not None:
            state_dict[f"blocks.{l}.attn.b_O"] = o_w.bias

        # --- Layer norms inside the layer ---
        # HuBERT layer has `layer.layer_norm` and `layer.final_layer_norm`
        ln1 = getattr(layer, "layer_norm", None)
        ln2 = getattr(layer, "final_layer_norm", None)
        if ln1 is None or ln2 is None:
            # try alternative names
            ln1 = ln1 or getattr(layer, "attention_norm", None)
            ln2 = ln2 or getattr(layer, "output_layer_norm", None)

        if ln1 is not None:
            state_dict[f"blocks.{l}.ln1.w"] = ln1.weight
            state_dict[f"blocks.{l}.ln1.b"] = ln1.bias
        if ln2 is not None:
            state_dict[f"blocks.{l}.ln2.w"] = ln2.weight
            state_dict[f"blocks.{l}.ln2.b"] = ln2.bias

        # --- Feed-forward / MLP ---
        # HuBERT uses `feed_forward` which contains intermediate_dense and output_dense
        ff = (
            getattr(layer, "feed_forward", None)
            or getattr(layer, "feedforward", None)
            or getattr(layer, "ff", None)
        )
        if ff is None:
            raise AttributeError(f"Layer {l} has no feed_forward/ff attribute")

        # Many implementations name them intermediate_dense and output_dense
        fc1 = (
            getattr(ff, "intermediate_dense", None)
            or getattr(ff, "fc1", None)
            or getattr(ff, "linear1", None)
        )
        fc2 = (
            getattr(ff, "output_dense", None)
            or getattr(ff, "fc2", None)
            or getattr(ff, "linear2", None)
        )

        if fc1 is None or fc2 is None:
            raise AttributeError(f"Could not find FFN dense layers in layer {l}: {ff}")

        # fc1.weight shape: (d_mlp, d_model) -> Transformer-Lens expects (d_model, d_mlp)
        state_dict[f"blocks.{l}.mlp.W_in"] = einops.rearrange(fc1.weight, "mlp model -> model mlp")
        if fc1.bias is not None:
            state_dict[f"blocks.{l}.mlp.b_in"] = fc1.bias

        # fc2.weight shape: (d_model, d_mlp) -> Transformer-Lens expects (d_mlp, d_model)
        state_dict[f"blocks.{l}.mlp.W_out"] = einops.rearrange(fc2.weight, "model mlp -> mlp model")
        if fc2.bias is not None:
            state_dict[f"blocks.{l}.mlp.b_out"] = fc2.bias

    # --- Optional: encoder-level layer_norm (HubertModel.encoder.layer_norm) ---
    if hasattr(hf_model.encoder, "layer_norm"):
        ln_final = hf_model.encoder.layer_norm
        state_dict["ln_final.w"] = ln_final.weight
        state_dict["ln_final.b"] = ln_final.bias

    return state_dict
