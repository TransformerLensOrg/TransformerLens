"""Tests for convert_tl_checkpoint utility."""

import torch

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils import convert_tl_checkpoint
from transformer_lens.model_bridge import TransformerBridge


def test_convert_tl_checkpoint_produces_matching_outputs():
    """Converted checkpoint produces identical outputs to original HookedTransformer."""
    torch.manual_seed(42)

    ht_cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=64,
        d_head=32,
        n_heads=2,
        n_ctx=32,
        d_vocab=64,
        d_mlp=256,
        act_fn="gelu",
    )
    ht = HookedTransformer(ht_cfg)
    tl_sd = ht.state_dict()

    bridge_cfg = TransformerBridgeConfig(
        n_layers=2,
        d_model=64,
        d_head=32,
        n_heads=2,
        n_ctx=32,
        d_vocab=64,
        d_mlp=256,
        act_fn="gelu",
    )
    bridge = TransformerBridge.boot_native(bridge_cfg)

    native_sd = convert_tl_checkpoint(tl_sd, bridge_cfg)
    bridge.load_state_dict(native_sd, strict=False)

    test_input = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        ht_out = ht(test_input)
        bridge_out = bridge(test_input)

    assert torch.allclose(
        ht_out, bridge_out, atol=1e-5
    ), f"Output mismatch: max diff = {(ht_out - bridge_out).abs().max().item()}"


def test_convert_tl_checkpoint_key_mapping():
    """Verify key mapping from TL format to native format."""
    tl_sd = {
        "embed.W_E": torch.randn(64, 32),
        "pos_embed.W_pos": torch.randn(16, 32),
        "blocks.0.ln1.w": torch.randn(32),
        "blocks.0.ln1.b": torch.randn(32),
        "blocks.0.attn.W_Q": torch.randn(2, 32, 16),
        "blocks.0.attn.b_Q": torch.randn(2, 16),
        "blocks.0.attn.W_K": torch.randn(2, 32, 16),
        "blocks.0.attn.b_K": torch.randn(2, 16),
        "blocks.0.attn.W_V": torch.randn(2, 32, 16),
        "blocks.0.attn.b_V": torch.randn(2, 16),
        "blocks.0.attn.W_O": torch.randn(2, 16, 32),
        "blocks.0.attn.b_O": torch.randn(32),
        "blocks.0.ln2.w": torch.randn(32),
        "blocks.0.ln2.b": torch.randn(32),
        "blocks.0.mlp.W_in": torch.randn(32, 128),
        "blocks.0.mlp.b_in": torch.randn(128),
        "blocks.0.mlp.W_out": torch.randn(128, 32),
        "blocks.0.mlp.b_out": torch.randn(32),
        "ln_final.w": torch.randn(32),
        "ln_final.b": torch.randn(32),
        "unembed.W_U": torch.randn(32, 64),
    }

    cfg = TransformerBridgeConfig(
        n_layers=1,
        d_model=32,
        d_head=16,
        n_heads=2,
        n_ctx=16,
        d_vocab=64,
        d_mlp=128,
    )

    native_sd = convert_tl_checkpoint(tl_sd, cfg)

    expected_keys = {
        "tok_embed.weight",
        "pos.weight",
        "layers.0.ln1.weight",
        "layers.0.ln1.bias",
        "layers.0.attn.q.weight",
        "layers.0.attn.q.bias",
        "layers.0.attn.k.weight",
        "layers.0.attn.k.bias",
        "layers.0.attn.v.weight",
        "layers.0.attn.v.bias",
        "layers.0.attn.o.weight",
        "layers.0.attn.o.bias",
        "layers.0.ln2.weight",
        "layers.0.ln2.bias",
        "layers.0.mlp.fc_in.weight",
        "layers.0.mlp.fc_in.bias",
        "layers.0.mlp.fc_out.weight",
        "layers.0.mlp.fc_out.bias",
        "ln_out.weight",
        "ln_out.bias",
        "head.weight",
    }

    assert set(native_sd.keys()) == expected_keys


def test_convert_tl_checkpoint_shape_transformations():
    """Verify shape transformations are correct."""
    n_heads, d_model, d_head, d_mlp = 4, 64, 16, 256

    tl_sd = {
        "blocks.0.attn.W_Q": torch.randn(n_heads, d_model, d_head),
        "blocks.0.attn.b_Q": torch.randn(n_heads, d_head),
        "blocks.0.attn.W_O": torch.randn(n_heads, d_head, d_model),
        "blocks.0.mlp.W_in": torch.randn(d_model, d_mlp),
        "blocks.0.mlp.W_out": torch.randn(d_mlp, d_model),
        "unembed.W_U": torch.randn(d_model, 100),
    }

    cfg = TransformerBridgeConfig(
        n_layers=1,
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        n_ctx=32,
        d_vocab=100,
        d_mlp=d_mlp,
    )

    native_sd = convert_tl_checkpoint(tl_sd, cfg)

    # Attention Q/K/V: [n_heads, d_model, d_head] -> [n_heads*d_head, d_model]
    assert native_sd["layers.0.attn.q.weight"].shape == (n_heads * d_head, d_model)
    assert native_sd["layers.0.attn.q.bias"].shape == (n_heads * d_head,)

    # Attention O: [n_heads, d_head, d_model] -> [d_model, n_heads*d_head]
    assert native_sd["layers.0.attn.o.weight"].shape == (d_model, n_heads * d_head)

    # MLP W_in: [d_model, d_mlp] -> [d_mlp, d_model] (transposed)
    assert native_sd["layers.0.mlp.fc_in.weight"].shape == (d_mlp, d_model)

    # MLP W_out: [d_mlp, d_model] -> [d_model, d_mlp] (transposed)
    assert native_sd["layers.0.mlp.fc_out.weight"].shape == (d_model, d_mlp)

    # Unembed: [d_model, d_vocab] -> [d_vocab, d_model] (transposed)
    assert native_sd["head.weight"].shape == (100, d_model)
