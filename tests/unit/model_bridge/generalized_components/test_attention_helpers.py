"""Tests for AttentionBridge helper methods (_get_n_heads, _reshape_weight_to_3d)."""
import einops
import torch
from transformers.pytorch_utils import Conv1D

from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class _FakeCfg:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_bridge(config):
    """Create a minimal AttentionBridge with a config for testing helpers."""
    bridge = AttentionBridge.__new__(AttentionBridge)
    bridge.config = config
    return bridge


class TestGetNHeads:
    def test_prefers_n_heads_attr(self):
        bridge = _make_bridge(_FakeCfg(n_heads=12, n_head=8))
        # n_heads takes priority over n_head
        assert bridge._get_n_heads() == 12

    def test_falls_back_to_n_head(self):
        bridge = _make_bridge(_FakeCfg(n_head=8))
        assert bridge._get_n_heads() == 8
        assert not hasattr(bridge.config, "n_heads")

    def test_kv_returns_different_value(self):
        bridge = _make_bridge(_FakeCfg(n_heads=12, n_key_value_heads=4))
        assert bridge._get_n_heads(use_kv=False) == 12
        assert bridge._get_n_heads(use_kv=True) == 4
        assert bridge._get_n_heads(use_kv=False) != bridge._get_n_heads(use_kv=True)

    def test_kv_fallback_when_no_kv_heads(self):
        bridge = _make_bridge(_FakeCfg(n_heads=12))
        # Without n_key_value_heads, use_kv should still return n_heads
        assert bridge._get_n_heads(use_kv=True) == 12


class TestReshapeWeightTo3D:
    def test_linear_format_preserves_elements(self):
        bridge = _make_bridge(_FakeCfg(n_heads=4))
        weight = torch.randn(64, 32)  # (n_heads*d_head, d_model) Linear format
        result = bridge._reshape_weight_to_3d(weight, n_heads=4)
        assert result.shape == (4, 32, 16)
        assert result.numel() == weight.numel()
        # Verify specific element mapping: Linear rearranges "(n d) m -> n m d"
        # So result[0] should contain the first d_head=16 rows of weight, transposed
        # result[0, :, 0] should be weight[0, :]
        assert torch.equal(result[0, :, 0], weight[0, :])

    def test_conv1d_format_detected_correctly(self):
        bridge = _make_bridge(_FakeCfg(n_heads=3))
        # (d_model, n_heads*d_head) — first dim (32) not divisible by 3 → Conv1D branch
        weight = torch.randn(32, 48)
        result = bridge._reshape_weight_to_3d(weight, n_heads=3)
        assert result.shape == (3, 32, 16)
        # Conv1D rearranges "m (n d) -> n m d"
        # So result[0, :, 0] should be weight[:, 0]
        assert torch.equal(result[0, :, 0], weight[:, 0])

    def test_o_pattern_linear_format(self):
        bridge = _make_bridge(_FakeCfg(n_heads=4))
        # Linear O weight: (d_model, n_heads*d_head) = (32, 64)
        # shape[0]=32, n_heads*(shape[1]//n_heads)=4*16=64 != 32, so transpose path
        # weight.T = (64, 32), rearranged → (4, 16, 32)
        weight = torch.randn(32, 64)
        o_result = bridge._reshape_weight_to_3d(weight, n_heads=4, pattern="o")
        assert o_result.shape == (4, 16, 32)

    def test_qkv_and_o_produce_different_shapes(self):
        bridge = _make_bridge(_FakeCfg(n_heads=4))
        # Use a shape where both patterns work: (64, 32)
        weight = torch.randn(64, 32)
        qkv_result = bridge._reshape_weight_to_3d(weight, n_heads=4, pattern="qkv")
        o_result = bridge._reshape_weight_to_3d(weight, n_heads=4, pattern="o")
        # QKV: "(n d) m -> n m d" → (4, 32, 16)
        # O: shape[0]=64, n_heads*(shape[1]//4)=4*8=32 != 64 → transpose path
        #    weight.T=(32,64), "(n d) m -> n d m" → (4, 8, 64)
        assert qkv_result.shape == (4, 32, 16)
        assert o_result.shape == (4, 8, 64)
        assert qkv_result.shape != o_result.shape


class TestWeightLayoutInOut:
    def test_linear_detected_as_out_in(self):
        bridge = _make_bridge(_FakeCfg(n_heads=4))
        proj = LinearBridge(name="o_proj")
        proj.set_original_component(torch.nn.Linear(8, 8))
        assert bridge._weight_layout_in_out(proj) is False

    def test_conv1d_detected_as_in_out(self):
        bridge = _make_bridge(_FakeCfg(n_heads=4))
        proj = LinearBridge(name="c_proj")
        proj.set_original_component(Conv1D(8, 8))
        assert bridge._weight_layout_in_out(proj) is True

    def test_unknown_component_returns_none(self):
        bridge = _make_bridge(_FakeCfg(n_heads=4))
        proj = LinearBridge(name="o_proj")
        assert bridge._weight_layout_in_out(proj) is None


class TestReshapeWeightSquareDisambiguation:
    """Square weights are shape-ambiguous: nn.Linear stores [out, in], Conv1D [in, out].

    Each test reproduces the wrapped module's projection from the reshaped 3D
    weight, so a wrong-axis split fails numerically, not just on shape.
    """

    def test_square_linear_o_weight_reproduces_projection(self):
        bridge = _make_bridge(_FakeCfg(n_heads=4))
        linear = torch.nn.Linear(32, 32, bias=False)
        w3d = bridge._reshape_weight_to_3d(
            linear.weight, n_heads=4, pattern="o", in_out_layout=False
        )
        assert w3d.shape == (4, 8, 32)
        z = torch.randn(2, 4, 8)
        expected = linear(z.reshape(2, 32))
        actual = torch.einsum("bhd,hdm->bm", z, w3d)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_square_conv1d_o_weight_reproduces_projection(self):
        bridge = _make_bridge(_FakeCfg(n_heads=4))
        conv = Conv1D(32, 32)
        w3d = bridge._reshape_weight_to_3d(conv.weight, n_heads=4, pattern="o", in_out_layout=True)
        assert w3d.shape == (4, 8, 32)
        z = torch.randn(2, 4, 8)
        expected = conv(z.reshape(2, 32))
        actual = torch.einsum("bhd,hdm->bm", z, w3d) + conv.bias
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_square_linear_qkv_weight_reproduces_projection(self):
        bridge = _make_bridge(_FakeCfg(n_heads=4))
        linear = torch.nn.Linear(32, 32, bias=False)
        w3d = bridge._reshape_weight_to_3d(
            linear.weight, n_heads=4, pattern="qkv", in_out_layout=False
        )
        assert w3d.shape == (4, 32, 8)
        x = torch.randn(2, 32)
        expected = linear(x).reshape(2, 4, 8)
        actual = torch.einsum("bm,hmd->bhd", x, w3d)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_square_conv1d_qkv_weight_reproduces_projection(self):
        bridge = _make_bridge(_FakeCfg(n_heads=4))
        conv = Conv1D(32, 32)
        w3d = bridge._reshape_weight_to_3d(
            conv.weight, n_heads=4, pattern="qkv", in_out_layout=True
        )
        assert w3d.shape == (4, 32, 8)
        x = torch.randn(2, 32)
        expected = conv(x).reshape(2, 4, 8)
        actual = torch.einsum("bm,hmd->bhd", x, w3d) + conv.bias.reshape(4, 8)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_square_heuristic_fallback_unchanged(self):
        # Without layout info the legacy heuristic assumes [in, out] on square weights.
        bridge = _make_bridge(_FakeCfg(n_heads=4))
        weight = torch.randn(32, 32)
        legacy = einops.rearrange(weight, "(n d) m -> n d m", n=4)
        assert torch.equal(bridge._reshape_weight_to_3d(weight, n_heads=4, pattern="o"), legacy)
