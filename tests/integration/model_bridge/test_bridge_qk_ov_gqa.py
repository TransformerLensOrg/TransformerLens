"""Regression tests for bridge QK/OV circuits on GQA models.

https://github.com/TransformerLensOrg/TransformerLens/issues/1553: bridge.QK/OV
built FactoredMatrices from a per-query-head W_Q/W_O and a grouped (n_kv_heads)
W_K/W_V, so the leading head axes did not broadcast. The grouped K/V weights must
be expanded to n_heads first, matching HookedTransformer.
"""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge


@pytest.fixture(scope="module")
def gqa_bridge():
    """TransformerBridge wrapping Qwen2-0.5B, a small GQA model (n_kv_heads < n_heads)."""
    return TransformerBridge.boot_transformers("Qwen/Qwen2-0.5B", device="cpu")


def _expanded_kv(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """repeat_interleave the grouped K/V head axis up to n_heads."""
    return torch.repeat_interleave(w, dim=1, repeats=n_heads // w.shape[1])


class TestGQABridgeQKOV:
    def test_raw_kv_storage_stays_grouped(self, gqa_bridge):
        cfg = gqa_bridge.cfg
        assert cfg.n_key_value_heads is not None
        assert cfg.n_key_value_heads < cfg.n_heads
        assert gqa_bridge.W_K.shape == (
            cfg.n_layers,
            cfg.n_key_value_heads,
            cfg.d_model,
            cfg.d_head,
        )
        assert gqa_bridge.W_V.shape == (
            cfg.n_layers,
            cfg.n_key_value_heads,
            cfg.d_model,
            cfg.d_head,
        )

    def test_qk_shapes_aligned(self, gqa_bridge):
        cfg = gqa_bridge.cfg
        QK = gqa_bridge.QK
        assert QK.A.shape == (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head)
        assert QK.B.shape == (cfg.n_layers, cfg.n_heads, cfg.d_head, cfg.d_model)
        assert QK.shape == (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_model)

    def test_ov_shapes_aligned(self, gqa_bridge):
        cfg = gqa_bridge.cfg
        OV = gqa_bridge.OV
        assert OV.A.shape == (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head)
        assert OV.B.shape == (cfg.n_layers, cfg.n_heads, cfg.d_head, cfg.d_model)
        assert OV.shape == (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_model)

    def test_qk_components_match_expanded_raw_weights(self, gqa_bridge):
        cfg = gqa_bridge.cfg
        QK = gqa_bridge.QK
        assert torch.equal(QK.A, gqa_bridge.W_Q)
        expected_B = _expanded_kv(gqa_bridge.W_K, cfg.n_heads).transpose(-2, -1)
        assert torch.equal(QK.B, expected_B)

    def test_ov_components_match_expanded_raw_weights(self, gqa_bridge):
        cfg = gqa_bridge.cfg
        OV = gqa_bridge.OV
        assert torch.equal(OV.A, _expanded_kv(gqa_bridge.W_V, cfg.n_heads))
        assert torch.equal(OV.B, gqa_bridge.W_O)

    def test_qk_ov_for_attn_layers_aligned(self, gqa_bridge):
        cfg = gqa_bridge.cfg
        _, QK = gqa_bridge.QK_for_attn_layers()
        assert QK.A.shape == (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head)
        assert QK.B.shape == (cfg.n_layers, cfg.n_heads, cfg.d_head, cfg.d_model)
        _, OV = gqa_bridge.OV_for_attn_layers()
        assert OV.A.shape == (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head)
        assert OV.B.shape == (cfg.n_layers, cfg.n_heads, cfg.d_head, cfg.d_model)


class TestMHABridgeQKOVUnchanged:
    def test_mha_qk_ov_unchanged(self, distilgpt2_bridge):
        cfg = distilgpt2_bridge.cfg
        QK = distilgpt2_bridge.QK
        OV = distilgpt2_bridge.OV
        assert QK.A.shape == (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head)
        assert QK.B.shape == (cfg.n_layers, cfg.n_heads, cfg.d_head, cfg.d_model)
        assert torch.equal(QK.A, distilgpt2_bridge.W_Q)
        assert torch.equal(QK.B, distilgpt2_bridge.W_K.transpose(-2, -1))
        assert torch.equal(OV.A, distilgpt2_bridge.W_V)
        assert torch.equal(OV.B, distilgpt2_bridge.W_O)
