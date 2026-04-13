"""Unit tests for ALiBiJointQKVAttentionBridge.

Exercises the reimplemented ALiBi attention with mock weights — no model download needed.
Covers MHA, MQA, and GQA head configurations to catch shape mismatches.
"""

import torch

from transformer_lens.model_bridge.generalized_components.alibi_joint_qkv_attention import (
    ALiBiJointQKVAttentionBridge,
)


class _MockConfig:
    """Minimal config for ALiBiJointQKVAttentionBridge."""

    def __init__(self, n_heads: int, d_model: int, n_key_value_heads: int | None = None):
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_key_value_heads = n_key_value_heads


class _MockAttention(torch.nn.Module):
    """Stub original component so the bridge's forward doesn't raise."""

    def __init__(self):
        super().__init__()
        self.attn_dropout = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _build_bridge(
    n_heads: int, d_model: int, n_key_value_heads: int | None = None
) -> ALiBiJointQKVAttentionBridge:
    """Build a wired-up ALiBiJointQKVAttentionBridge with random Q/K/V weights."""
    cfg = _MockConfig(n_heads, d_model, n_key_value_heads)
    head_dim = d_model // n_heads
    n_kv = n_key_value_heads or n_heads

    q_linear = torch.nn.Linear(d_model, n_heads * head_dim)
    k_linear = torch.nn.Linear(d_model, n_kv * head_dim)
    v_linear = torch.nn.Linear(d_model, n_kv * head_dim)
    o_linear = torch.nn.Linear(d_model, d_model)

    def split_qkv(_component):
        return q_linear, k_linear, v_linear

    bridge = ALiBiJointQKVAttentionBridge(
        name="self_attention",
        config=cfg,
        split_qkv_matrix=split_qkv,
    )
    mock_attn = _MockAttention()
    mock_attn.dense = o_linear
    bridge.set_original_component(mock_attn)
    return bridge


def _random_inputs(bridge: ALiBiJointQKVAttentionBridge, batch: int = 2, seq: int = 6):
    """Generate random inputs via the bridge's own method."""
    return bridge.get_random_inputs(batch_size=batch, seq_len=seq)


class TestALiBiJointQKVForward:
    """Forward pass runs and produces valid output for all head configs."""

    def test_mha_forward(self):
        """Standard MHA: n_heads == n_kv_heads."""
        bridge = _build_bridge(n_heads=4, d_model=32)
        inputs = _random_inputs(bridge)
        with torch.no_grad():
            output, weights = bridge(
                inputs["hidden_states"], **{k: v for k, v in inputs.items() if k != "hidden_states"}
            )
        assert output.shape == (2, 6, 32)
        assert not torch.isnan(output).any()

    def test_mqa_forward(self):
        """Multi-query: K/V have 1 head, Q has n_heads."""
        bridge = _build_bridge(n_heads=8, d_model=64, n_key_value_heads=1)
        inputs = _random_inputs(bridge)
        with torch.no_grad():
            output, weights = bridge(
                inputs["hidden_states"], **{k: v for k, v in inputs.items() if k != "hidden_states"}
            )
        assert output.shape == (2, 6, 64)
        assert not torch.isnan(output).any()
        # Attention weights should have full n_heads after expansion
        assert weights.shape[1] == 8

    def test_gqa_forward(self):
        """Grouped-query: K/V have fewer heads than Q (but more than 1)."""
        bridge = _build_bridge(n_heads=8, d_model=64, n_key_value_heads=2)
        inputs = _random_inputs(bridge)
        with torch.no_grad():
            output, weights = bridge(
                inputs["hidden_states"], **{k: v for k, v in inputs.items() if k != "hidden_states"}
            )
        assert output.shape == (2, 6, 64)
        assert not torch.isnan(output).any()
        assert weights.shape[1] == 8


class TestALiBiEffect:
    """ALiBi bias actually affects attention scores."""

    def test_alibi_changes_output(self):
        """Output with ALiBi should differ from output without."""
        bridge = _build_bridge(n_heads=4, d_model=32)
        inputs = _random_inputs(bridge)
        hidden = inputs["hidden_states"]
        mask = inputs["attention_mask"]

        with torch.no_grad():
            out_with, _ = bridge(hidden, alibi=inputs["alibi"], attention_mask=mask)
            out_without, _ = bridge(hidden, attention_mask=mask)

        assert not torch.allclose(out_with, out_without), "ALiBi should change the output"

    def test_pattern_is_causal(self):
        """Upper triangle of attention pattern should be zero (causal masking)."""
        bridge = _build_bridge(n_heads=4, d_model=32)
        inputs = _random_inputs(bridge, batch=1, seq=4)

        with torch.no_grad():
            _, weights = bridge(
                inputs["hidden_states"], **{k: v for k, v in inputs.items() if k != "hidden_states"}
            )
        # weights: [batch, heads, seq, seq] — upper triangle (above diagonal) should be 0
        upper = torch.triu(weights[0, 0], diagonal=1)
        assert (upper == 0).all()


class TestHooksFireInForward:
    """Hooks fire correctly during the reimplemented attention forward."""

    def test_attn_scores_hook(self):
        bridge = _build_bridge(n_heads=4, d_model=32)
        inputs = _random_inputs(bridge, batch=1, seq=4)
        captured = {}

        def hook_fn(tensor, hook):
            captured["attn_scores"] = tensor.clone()
            return tensor

        bridge.hook_attn_scores.add_hook(hook_fn)
        with torch.no_grad():
            bridge(
                inputs["hidden_states"], **{k: v for k, v in inputs.items() if k != "hidden_states"}
            )
        assert "attn_scores" in captured
        assert captured["attn_scores"].shape == (1, 4, 4, 4)

    def test_pattern_hook(self):
        bridge = _build_bridge(n_heads=4, d_model=32)
        inputs = _random_inputs(bridge, batch=1, seq=4)
        captured = {}

        def hook_fn(tensor, hook):
            captured["pattern"] = tensor.clone()
            return tensor

        bridge.hook_pattern.add_hook(hook_fn)
        with torch.no_grad():
            bridge(
                inputs["hidden_states"], **{k: v for k, v in inputs.items() if k != "hidden_states"}
            )
        assert "pattern" in captured
        # Pattern rows should sum to 1 (softmax output)
        row_sums = captured["pattern"].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
