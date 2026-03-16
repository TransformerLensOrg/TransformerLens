import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions.openai import (
    convert_gpt_oss_weights,
)


def make_cfg():
    return HookedTransformerConfig(
        n_layers=1,
        d_model=32,
        d_head=8,
        n_heads=4,
        d_mlp=16,
        n_ctx=64,
        d_vocab=128,
        act_fn="silu",
        normalization_type="RMS",
        positional_embedding_type="rotary",
        rotary_base=150000,
        eps=1e-5,
        n_key_value_heads=2,
        gated_mlp=True,
        use_local_attn=False,
        rotary_dim=8,
        num_experts=2,
        experts_per_token=1,
        dtype=torch.float32,
        original_architecture="GptOssForCausalLM",
    )


def make_mock_model(cfg):
    """Build a minimal mock HF GPT-OSS model with the right tensor shapes."""
    from unittest import mock

    d = cfg.d_model
    d_mlp = cfg.d_mlp
    n_experts = cfg.num_experts
    n_heads = cfg.n_heads
    n_kv = cfg.n_key_value_heads
    d_head = cfg.d_head

    model = mock.Mock()

    # Embeddings
    model.model.embed_tokens.weight = torch.randn(cfg.d_vocab, d)

    # Single layer
    layer = mock.Mock()

    # LayerNorms
    layer.input_layernorm.weight = torch.randn(d)
    layer.post_attention_layernorm.weight = torch.randn(d)

    # Attention
    layer.self_attn.q_proj.weight = torch.randn(n_heads * d_head, d)
    layer.self_attn.k_proj.weight = torch.randn(n_kv * d_head, d)
    layer.self_attn.v_proj.weight = torch.randn(n_kv * d_head, d)
    layer.self_attn.o_proj.weight = torch.randn(d, n_heads * d_head)
    layer.self_attn.q_proj.bias = torch.randn(n_heads * d_head)
    layer.self_attn.k_proj.bias = torch.randn(n_kv * d_head)
    layer.self_attn.v_proj.bias = torch.randn(n_kv * d_head)
    layer.self_attn.o_proj.bias = torch.randn(d)

    # Router
    layer.mlp.router.weight = torch.randn(n_experts, d)
    layer.mlp.router.bias = torch.randn(n_experts)

    # Experts — interleaved gate_up_proj: (num_experts, d_model, 2*d_mlp)
    layer.mlp.experts.gate_up_proj = torch.randn(n_experts, d, 2 * d_mlp)
    layer.mlp.experts.gate_up_proj_bias = torch.randn(n_experts, 2 * d_mlp)
    layer.mlp.experts.down_proj = torch.randn(n_experts, d_mlp, d)
    layer.mlp.experts.down_proj_bias = torch.randn(n_experts, d)

    model.model.layers = [layer]

    # Final norm and lm_head
    model.model.norm.weight = torch.randn(d)
    model.lm_head.weight = torch.randn(cfg.d_vocab, d)

    return model


def test_interleaved_weight_split():
    """Even columns of gate_up_proj go to W_gate, odd columns go to W_in."""
    cfg = make_cfg()
    model = make_mock_model(cfg)

    state_dict = convert_gpt_oss_weights(model, cfg)

    gate_up = model.model.layers[0].mlp.experts.gate_up_proj  # (n_experts, d, 2*d_mlp)
    gate_up_bias = model.model.layers[0].mlp.experts.gate_up_proj_bias  # (n_experts, 2*d_mlp)

    for e in range(cfg.num_experts):
        # W_gate gets even columns (::2), transposed for nn.Linear format
        expected_gate_w = gate_up[e, :, ::2].T.contiguous()
        actual_gate_w = state_dict[f"blocks.0.mlp.experts.{e}.W_gate.weight"]
        assert torch.equal(actual_gate_w, expected_gate_w)

        expected_gate_b = gate_up_bias[e, ::2].contiguous()
        actual_gate_b = state_dict[f"blocks.0.mlp.experts.{e}.W_gate.bias"]
        assert torch.equal(actual_gate_b, expected_gate_b)

        # W_in gets odd columns (1::2), transposed for nn.Linear format
        expected_in_w = gate_up[e, :, 1::2].T.contiguous()
        actual_in_w = state_dict[f"blocks.0.mlp.experts.{e}.W_in.weight"]
        assert torch.equal(actual_in_w, expected_in_w)

        expected_in_b = gate_up_bias[e, 1::2].contiguous()
        actual_in_b = state_dict[f"blocks.0.mlp.experts.{e}.W_in.bias"]
        assert torch.equal(actual_in_b, expected_in_b)


def test_down_proj_transposed():
    """down_proj is transposed for nn.Linear format."""
    cfg = make_cfg()
    model = make_mock_model(cfg)

    state_dict = convert_gpt_oss_weights(model, cfg)

    down = model.model.layers[0].mlp.experts.down_proj  # (n_experts, d_mlp, d)

    for e in range(cfg.num_experts):
        expected = down[e].T.contiguous()
        actual = state_dict[f"blocks.0.mlp.experts.{e}.W_out.weight"]
        assert torch.equal(actual, expected)


def test_router_weight_and_bias():
    """Router weight and bias are correctly mapped."""
    cfg = make_cfg()
    model = make_mock_model(cfg)

    state_dict = convert_gpt_oss_weights(model, cfg)

    assert torch.equal(
        state_dict["blocks.0.mlp.W_gate.weight"],
        model.model.layers[0].mlp.router.weight,
    )
    assert torch.equal(
        state_dict["blocks.0.mlp.W_gate.bias"],
        model.model.layers[0].mlp.router.bias,
    )


def test_attention_weight_shapes():
    """Attention weights are correctly reshaped."""
    cfg = make_cfg()
    model = make_mock_model(cfg)

    state_dict = convert_gpt_oss_weights(model, cfg)

    assert state_dict["blocks.0.attn.W_Q"].shape == (cfg.n_heads, cfg.d_model, cfg.d_head)
    assert state_dict["blocks.0.attn._W_K"].shape == (
        cfg.n_key_value_heads,
        cfg.d_model,
        cfg.d_head,
    )
    assert state_dict["blocks.0.attn._W_V"].shape == (
        cfg.n_key_value_heads,
        cfg.d_model,
        cfg.d_head,
    )
    assert state_dict["blocks.0.attn.W_O"].shape == (cfg.n_heads, cfg.d_head, cfg.d_model)


def test_state_dict_completeness():
    """All expected keys are present in the converted state dict."""
    cfg = make_cfg()
    model = make_mock_model(cfg)

    state_dict = convert_gpt_oss_weights(model, cfg)

    # Check essential keys exist
    assert "embed.W_E" in state_dict
    assert "ln_final.w" in state_dict
    assert "unembed.W_U" in state_dict
    assert "unembed.b_U" in state_dict
    assert "blocks.0.ln1.w" in state_dict
    assert "blocks.0.ln2.w" in state_dict
    assert "blocks.0.mlp.W_gate.weight" in state_dict
    assert "blocks.0.mlp.W_gate.bias" in state_dict

    for e in range(cfg.num_experts):
        for name in [
            "W_gate.weight",
            "W_gate.bias",
            "W_in.weight",
            "W_in.bias",
            "W_out.weight",
            "W_out.bias",
        ]:
            assert f"blocks.0.mlp.experts.{e}.{name}" in state_dict
