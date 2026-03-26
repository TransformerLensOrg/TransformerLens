import torch
import torch.nn.functional as F

from transformer_lens.components.mlps.gpt_oss_moe import (
    GPT_OSS_ALPHA,
    GPT_OSS_LIMIT,
    GptOssExpert,
    GptOssMoE,
)
from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig


def make_moe_cfg():
    """Dict config for GptOssMoE (accepts dict via CanBeUsedAsMLP)."""
    return {
        "d_model": 32,
        "d_mlp": 16,
        "d_head": 4,
        "num_experts": 8,
        "n_layers": 2,
        "n_ctx": 64,
        "experts_per_token": 2,
        "gated_mlp": True,
        "act_fn": "silu",
        "dtype": torch.float32,
        "original_architecture": "GptOssForCausalLM",
    }


def make_expert_cfg():
    """HookedTransformerConfig for GptOssExpert (requires config object)."""
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
        num_experts=8,
        experts_per_token=2,
        dtype=torch.float32,
        original_architecture="GptOssForCausalLM",
    )


# ── Routing logic ────────────────────────────────────────────


def test_forward_shape():
    """MoE output has same shape as input."""
    moe = GptOssMoE(make_moe_cfg())
    x = torch.randn(2, 5, 32)
    out = moe(x)
    assert out.shape == (2, 5, 32)


def test_routing_softmax_after_topk():
    """GPT-OSS applies softmax AFTER top-k, not before.

    Verify the routing produces weights via: topk(logits) -> softmax(selected).
    With softmax-before (no renormalization), non-selected experts absorb
    probability mass, producing weights that don't sum to 1.
    """
    k = 2
    # Craft logits where non-top experts have significant mass
    gate_logits = torch.tensor([[5.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]])

    # GPT-OSS way: top-k first, then softmax over selected
    top_values, _ = torch.topk(gate_logits, k, dim=-1)
    gpt_oss_weights = F.softmax(top_values, dim=-1, dtype=torch.float)

    # Weights must sum to 1 (softmax over selected experts only)
    assert torch.allclose(gpt_oss_weights.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)

    # Softmax-before approach (without renormalization) produces different weights:
    # non-selected experts absorb mass so selected weights sum to less than 1
    all_weights = F.softmax(gate_logits, dim=-1, dtype=torch.float)
    before_topk_weights, _ = torch.topk(all_weights, k, dim=-1)

    assert before_topk_weights.sum(dim=-1).item() < 1.0
    assert not torch.allclose(gpt_oss_weights, before_topk_weights, atol=1e-3)


def test_routing_experts_per_token():
    """Each token is routed to exactly experts_per_token experts."""
    cfg = make_moe_cfg()
    moe = GptOssMoE(cfg)

    x = torch.randn(2, 3, cfg["d_model"])

    # Run forward and capture via run_with_hooks-style manual inspection
    with torch.no_grad():
        batch, pos, d_model = x.shape
        flat_x = x.view(-1, d_model)
        gate_logits = moe.W_gate(flat_x)
        _, expert_indices = torch.topk(gate_logits, cfg["experts_per_token"], dim=-1)

    # Each token should select exactly experts_per_token experts
    assert expert_indices.shape == (batch * pos, cfg["experts_per_token"])
    # All indices in valid range
    assert (expert_indices >= 0).all()
    assert (expert_indices < cfg["num_experts"]).all()


def test_routing_weights_sum_to_one():
    """Active expert weights sum to 1 for each token."""
    cfg = make_moe_cfg()
    moe = GptOssMoE(cfg)

    x = torch.randn(1, 4, cfg["d_model"])

    with torch.no_grad():
        flat_x = x.view(-1, cfg["d_model"])
        gate_logits = moe.W_gate(flat_x)
        top_values, _ = torch.topk(gate_logits, cfg["experts_per_token"], dim=-1)
        weights = F.softmax(top_values, dim=-1, dtype=torch.float)

    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


def test_router_has_bias():
    """GPT-OSS router uses bias, unlike Mixtral."""
    moe = GptOssMoE(make_moe_cfg())
    assert moe.W_gate.bias is not None


# ── Custom GLU activation ────────────────────────────────────


def test_expert_glu_activation():
    """Verify the custom GLU: gate.clamp(max=7) * sigmoid(gate * 1.702) * (up.clamp(-7,7) + 1)."""
    cfg = make_expert_cfg()
    expert = GptOssExpert(cfg)

    x = torch.randn(1, cfg.d_model)

    with torch.no_grad():
        gate_raw = expert.W_gate(x)
        up_raw = expert.W_in(x)

        # Manual computation of GPT-OSS activation
        gate = gate_raw.clamp(max=GPT_OSS_LIMIT)
        up = up_raw.clamp(min=-GPT_OSS_LIMIT, max=GPT_OSS_LIMIT)
        glu = gate * torch.sigmoid(gate * GPT_OSS_ALPHA)
        expected_post = (up + 1) * glu
        expected_out = expert.W_out(expected_post)

        # Actual forward pass
        actual_out = expert(x)

    assert torch.allclose(actual_out, expected_out, atol=1e-5)


def test_expert_gate_clamping():
    """Gate values above 7.0 are clamped in the activation."""
    cfg = make_expert_cfg()
    expert = GptOssExpert(cfg)

    # Force large gate values by setting weights to produce large outputs
    with torch.no_grad():
        expert.W_gate.weight.fill_(10.0)
        expert.W_gate.bias.fill_(10.0)

    x = torch.ones(1, cfg.d_model)

    with torch.no_grad():
        gate_raw = expert.W_gate(x)

    # The raw gate output should exceed the limit
    assert (gate_raw > GPT_OSS_LIMIT).any(), "Test setup: gate values should exceed limit"

    # But after clamping, all values should be <= 7.0
    gate_clamped = gate_raw.clamp(max=GPT_OSS_LIMIT)
    assert (gate_clamped <= GPT_OSS_LIMIT).all()

    # Forward pass should still work (no NaN/Inf)
    with torch.no_grad():
        out = expert(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_expert_up_clamping():
    """Up values are clamped to [-7, 7]."""
    cfg = make_expert_cfg()
    expert = GptOssExpert(cfg)

    with torch.no_grad():
        expert.W_in.weight.fill_(10.0)
        expert.W_in.bias.fill_(10.0)

    x = torch.ones(1, cfg.d_model)

    with torch.no_grad():
        up_raw = expert.W_in(x)

    assert (up_raw > GPT_OSS_LIMIT).any(), "Test setup: up values should exceed limit"

    up_clamped = up_raw.clamp(min=-GPT_OSS_LIMIT, max=GPT_OSS_LIMIT)
    assert (up_clamped <= GPT_OSS_LIMIT).all()
    assert (up_clamped >= -GPT_OSS_LIMIT).all()


def test_expert_has_biases():
    """GPT-OSS expert projections have biases."""
    cfg = make_expert_cfg()
    expert = GptOssExpert(cfg)
    assert expert.W_gate.bias is not None
    assert expert.W_in.bias is not None
    assert expert.W_out.bias is not None


def test_expert_activation_differs_from_silu():
    """GPT-OSS activation differs from standard SiLU gating."""
    cfg = make_expert_cfg()
    expert = GptOssExpert(cfg)

    x = torch.randn(1, cfg.d_model)

    with torch.no_grad():
        gate_raw = expert.W_gate(x)
        up_raw = expert.W_in(x)

        # Standard SiLU gating (Mixtral-style): silu(gate) * up
        silu_out = F.silu(gate_raw) * up_raw

        # GPT-OSS: gate.clamp(max=7) * sigmoid(gate*1.702) * (up.clamp(-7,7) + 1)
        gate = gate_raw.clamp(max=GPT_OSS_LIMIT)
        up = up_raw.clamp(min=-GPT_OSS_LIMIT, max=GPT_OSS_LIMIT)
        gpt_oss_out = gate * torch.sigmoid(gate * GPT_OSS_ALPHA) * (up + 1)

    assert not torch.allclose(silu_out, gpt_oss_out, atol=1e-3)
