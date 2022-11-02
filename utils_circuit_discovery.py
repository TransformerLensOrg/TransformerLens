def get_hook_tuple(layer, head_idx):
    if head_idx is None:
        return (f"blocks.{layer}.hook_mlp_out", None)
    else:
        return (f"blocks.{layer}.attn.hook_result", head_idx)
