"""Unit tests for TransformerBridge._ungroup_kv_weights().

Regression for https://github.com/TransformerLensOrg/TransformerLens/issues/1553:
bridge QK/OV circuits must expand grouped K/V weights to n_heads before building
FactoredMatrices, mirroring HookedTransformer's unconditional repeat_interleave.
"""

import torch

from transformer_lens.model_bridge.transformer_bridge import TransformerBridge


class TestUngroupKvWeights:
    def _make_bridge(self, n_heads):
        bridge = TransformerBridge.__new__(TransformerBridge)

        class Cfg:
            pass

        cfg = Cfg()
        cfg.n_heads = n_heads
        bridge.cfg = cfg
        return bridge

    def test_expands_grouped_kv_to_n_heads(self):
        bridge = self._make_bridge(n_heads=8)
        grouped = torch.arange(2 * 2 * 4 * 2, dtype=torch.float32).reshape(2, 2, 4, 2)
        result = bridge._ungroup_kv_weights(grouped)
        assert result.shape == (2, 8, 4, 2)
        repeats = 8 // 2
        for h in range(8):
            assert torch.equal(result[:, h], grouped[:, h // repeats])

    def test_mha_weights_returned_unchanged(self):
        bridge = self._make_bridge(n_heads=8)
        mha = torch.randn(2, 8, 4, 2)
        assert bridge._ungroup_kv_weights(mha) is mha

    def test_does_not_require_cfg_n_key_value_heads(self):
        bridge = self._make_bridge(n_heads=8)
        assert not hasattr(bridge.cfg, "n_key_value_heads")
        grouped = torch.randn(2, 2, 4, 2)
        assert bridge._ungroup_kv_weights(grouped).shape == (2, 8, 4, 2)

    def test_non_4d_weights_returned_unchanged(self):
        bridge = self._make_bridge(n_heads=8)
        flat = torch.randn(2, 8, 16)
        assert bridge._ungroup_kv_weights(flat) is flat
