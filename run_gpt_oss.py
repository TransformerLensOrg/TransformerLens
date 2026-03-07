"""Load GPT-OSS-20B directly from safetensors into TransformerLens.

Bypasses the HuggingFace model loading pipeline to avoid doubling memory usage.
The model is ~40GB in BF16 — loading via HF would require ~80GB peak (HF model + state dict).

Instead, we:
1. Create the TransformerLens model structure (~40GB, filled with empty tensors)
2. Load weights from safetensors one layer at a time
3. Dequantize MXFP4 expert weights on the fly using HF's convert_moe_packed_tensors
4. Copy directly into TL model parameters, freeing temp data immediately

Peak memory: ~42GB (model + one layer's temp data). Works on a 38.7GB Mac via swap.
"""

import gc
import json
from pathlib import Path

import einops
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def get_model_path():
    """Get the cached model path, downloading if necessary."""
    cache_path = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b"
    snapshots = cache_path / "snapshots"

    if snapshots.exists():
        # Use the first (usually only) snapshot
        snapshot_dirs = list(snapshots.iterdir())
        if snapshot_dirs:
            return snapshot_dirs[0]

    # Not cached — download
    print("Model not found in cache. Downloading...")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download("openai/gpt-oss-20b"))


def create_config(n_layers=24):
    """Create TransformerLens config for GPT-OSS-20B."""
    return HookedTransformerConfig(
        n_layers=n_layers,
        d_model=2880,
        d_head=64,
        n_heads=64,
        d_mlp=2880,
        n_ctx=4096,  # Reduced from 131072 to save memory
        d_vocab=201088,
        act_fn="silu",
        normalization_type="RMS",
        positional_embedding_type="rotary",
        rotary_base=150000,
        eps=1e-5,
        n_key_value_heads=8,
        gated_mlp=True,
        use_local_attn=False,
        rotary_dim=64,
        num_experts=32,
        experts_per_token=4,
        dtype=torch.bfloat16,
        device="cpu",
        original_architecture="GptOssForCausalLM",
        model_name="openai/gpt-oss-20b",
    )


def _get_tensor(hf_name, wmap, model_path, _open_files={}):
    """Load a single tensor from the correct safetensors shard."""
    st_file = wmap[hf_name]
    filepath = str(model_path / st_file)
    if filepath not in _open_files:
        _open_files[filepath] = safe_open(filepath, framework="pt", device="cpu")
    return _open_files[filepath].get_tensor(hf_name)


def load_layer_weights(l, cfg, index, model_path):
    """Load and convert weights for one transformer layer from safetensors."""
    state_dict = {}
    wmap = index["weight_map"]
    prefix = f"model.layers.{l}"

    def gt(name):
        return _get_tensor(name, wmap, model_path)

    # LayerNorms
    state_dict[f"blocks.{l}.ln1.w"] = gt(f"{prefix}.input_layernorm.weight")
    state_dict[f"blocks.{l}.ln2.w"] = gt(f"{prefix}.post_attention_layernorm.weight")

    # Attention weights
    q_w = gt(f"{prefix}.self_attn.q_proj.weight")
    k_w = gt(f"{prefix}.self_attn.k_proj.weight")
    v_w = gt(f"{prefix}.self_attn.v_proj.weight")
    o_w = gt(f"{prefix}.self_attn.o_proj.weight")

    state_dict[f"blocks.{l}.attn.W_Q"] = einops.rearrange(q_w, "(n h) m -> n m h", n=cfg.n_heads)
    state_dict[f"blocks.{l}.attn._W_K"] = einops.rearrange(k_w, "(n h) m -> n m h", n=cfg.n_key_value_heads)
    state_dict[f"blocks.{l}.attn._W_V"] = einops.rearrange(v_w, "(n h) m -> n m h", n=cfg.n_key_value_heads)
    state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(o_w, "m (n h) -> n h m", n=cfg.n_heads)
    del q_w, k_w, v_w, o_w

    # Attention biases
    q_bias_key = f"{prefix}.self_attn.q_proj.bias"
    if q_bias_key in wmap:
        state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            gt(q_bias_key), "(n h) -> n h", n=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn._b_K"] = einops.rearrange(
            gt(f"{prefix}.self_attn.k_proj.bias"), "(n h) -> n h", n=cfg.n_key_value_heads
        )
        state_dict[f"blocks.{l}.attn._b_V"] = einops.rearrange(
            gt(f"{prefix}.self_attn.v_proj.bias"), "(n h) -> n h", n=cfg.n_key_value_heads
        )
    else:
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype)

    o_bias_key = f"{prefix}.self_attn.o_proj.bias"
    if o_bias_key in wmap:
        state_dict[f"blocks.{l}.attn.b_O"] = gt(o_bias_key)
    else:
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    # Router
    state_dict[f"blocks.{l}.mlp.W_gate.weight"] = gt(f"{prefix}.mlp.router.weight")
    state_dict[f"blocks.{l}.mlp.W_gate.bias"] = gt(f"{prefix}.mlp.router.bias")

    # Expert weights — dequantize MXFP4 to BF16
    gate_up_blocks = gt(f"{prefix}.mlp.experts.gate_up_proj_blocks")
    gate_up_scales = gt(f"{prefix}.mlp.experts.gate_up_proj_scales")
    gate_up_bias = gt(f"{prefix}.mlp.experts.gate_up_proj_bias")

    # Dequantize gate_up_proj: [32, 5760, 90, 16] + [32, 5760, 90] -> [32, 2880, 5760]
    print(f"  Dequantizing layer {l} gate_up_proj...", end="", flush=True)
    gate_up_proj = convert_moe_packed_tensors(gate_up_blocks, gate_up_scales)
    del gate_up_blocks, gate_up_scales
    print(" done")

    down_blocks = gt(f"{prefix}.mlp.experts.down_proj_blocks")
    down_scales = gt(f"{prefix}.mlp.experts.down_proj_scales")
    down_bias = gt(f"{prefix}.mlp.experts.down_proj_bias")

    # Dequantize down_proj: [32, 2880, 90, 16] + [32, 2880, 90] -> [32, 2880, 2880]
    print(f"  Dequantizing layer {l} down_proj...", end="", flush=True)
    down_proj = convert_moe_packed_tensors(down_blocks, down_scales)
    del down_blocks, down_scales
    print(" done")

    # Split merged expert tensors into per-expert weights
    # gate_up_proj shape: [num_experts, hidden_size, 2*expert_dim]
    # Even columns -> gate, Odd columns -> up
    for e in range(cfg.num_experts):
        state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate.weight"] = gate_up_proj[e, :, ::2].T.contiguous()
        state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate.bias"] = gate_up_bias[e, ::2].contiguous()
        state_dict[f"blocks.{l}.mlp.experts.{e}.W_in.weight"] = gate_up_proj[e, :, 1::2].T.contiguous()
        state_dict[f"blocks.{l}.mlp.experts.{e}.W_in.bias"] = gate_up_bias[e, 1::2].contiguous()
        state_dict[f"blocks.{l}.mlp.experts.{e}.W_out.weight"] = down_proj[e].T.contiguous()
        state_dict[f"blocks.{l}.mlp.experts.{e}.W_out.bias"] = down_bias[e].contiguous()

    del gate_up_proj, gate_up_bias, down_proj, down_bias
    return state_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load GPT-OSS-20B into TransformerLens")
    parser.add_argument("--layers", type=int, default=24,
                        help="Number of layers to load (default: 24, use fewer to save memory)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt to test (default: built-in test prompts)")
    args = parser.parse_args()

    print("=" * 60)
    print("GPT-OSS-20B via TransformerLens (Direct SafeTensors)")
    print("=" * 60)

    import psutil
    ram = psutil.virtual_memory()
    print(f"\nPyTorch: {torch.__version__}")
    print(f"MPS: {torch.backends.mps.is_available()}")
    print(f"RAM: {ram.total/1e9:.1f}GB total, {ram.available/1e9:.1f}GB available")

    n_layers = args.layers
    if n_layers < 24:
        print(f"\nLoading first {n_layers} of 24 layers (reduced memory mode)")
        est_gb = 2.4 + n_layers * 1.64
        print(f"Estimated memory: ~{est_gb:.0f}GB")
    else:
        print(f"\nLoading all 24 layers (~42GB, will use swap on <40GB RAM machines)")

    model_path = get_model_path()
    print(f"Model path: {model_path}")

    with open(model_path / "model.safetensors.index.json") as f:
        index = json.load(f)

    # Create config
    cfg = create_config(n_layers=n_layers)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Create TransformerLens model (allocates parameter storage)
    print("Creating TransformerLens model structure...")
    model = HookedTransformer(cfg, tokenizer, move_to_device=False)

    # Load embeddings
    print("\nLoading embeddings...")
    embed_file = str(model_path / index["weight_map"]["model.embed_tokens.weight"])
    with safe_open(embed_file, framework="pt", device="cpu") as f:
        embed_w = f.get_tensor("model.embed_tokens.weight")
    model.load_state_dict({"embed.W_E": embed_w}, strict=False)
    del embed_w
    gc.collect()

    # Load layers one at a time
    for l in range(n_layers):
        print(f"\nLoading layer {l}/{n_layers-1}...")
        layer_dict = load_layer_weights(l, cfg, index, model_path)

        # Load into model one key at a time to minimize peak memory
        keys = list(layer_dict.keys())
        for key in keys:
            model.load_state_dict({key: layer_dict[key]}, strict=False)
            del layer_dict[key]
        del layer_dict
        gc.collect()

        ram = psutil.virtual_memory()
        print(f"  RAM: {ram.used/1e9:.1f}GB used, {ram.available/1e9:.1f}GB available")

    # Load final LayerNorm and unembed
    print("\nLoading final layers...")
    final_file = str(model_path / index["weight_map"]["model.norm.weight"])
    with safe_open(final_file, framework="pt", device="cpu") as f:
        ln_w = f.get_tensor("model.norm.weight")
        unembed_w = f.get_tensor("lm_head.weight").T

    model.load_state_dict({"ln_final.w": ln_w}, strict=False)
    del ln_w
    model.load_state_dict({"unembed.W_U": unembed_w}, strict=False)
    del unembed_w
    model.load_state_dict({"unembed.b_U": torch.zeros(cfg.d_vocab, dtype=cfg.dtype)}, strict=False)
    gc.collect()

    print("\n" + "=" * 60)
    print("Model loaded successfully!")
    print(f"Architecture: {cfg.original_architecture}")
    print(f"Layers: {cfg.n_layers}")
    print(f"Experts: {cfg.num_experts}")
    print(f"d_model: {cfg.d_model}")

    ram = psutil.virtual_memory()
    print(f"RAM: {ram.used/1e9:.1f}GB used, {ram.available/1e9:.1f}GB available")

    # Test inference
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "The capital of France is",
            "2 + 2 =",
            "The opposite of hot is",
        ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: '{prompt}'")
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)
        pred = model.to_string(logits[0, -1].argmax())
        print(f"Prediction: '{pred}'")

        # Show top 5 predictions
        probs = torch.softmax(logits[0, -1].float(), dim=-1)
        top5 = probs.topk(5)
        print("Top 5:")
        for i in range(5):
            token_str = model.to_string(top5.indices[i])
            print(f"  {token_str!r}: {top5.values[i]:.4f}")

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
