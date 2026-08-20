"""Unit tests for TransformerBridge._expand_kv_heads.

Regression for https://github.com/TransformerLensOrg/TransformerLens/issues/1553:
weight circuits on GQA models must expand the grouped K/V head axis to n_heads
(repeat_interleave, matching HookedTransformer's GroupedQueryAttention layout)
before factoring, while MHA weights pass through untouched.
"""

import pytest
import torch

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge


def _bridge_stub(n_heads: int) -> TransformerBridge:
    """Uninitialized bridge carrying only the cfg that _expand_kv_heads reads."""
    bridge = TransformerBridge.__new__(TransformerBridge)
    bridge.cfg = TransformerBridgeConfig(
        d_model=8,
        d_head=2,
        n_heads=n_heads,
        n_layers=2,
        n_ctx=8,
        d_vocab=16,
    )
    return bridge


class TestExpandKvHeads:
    def test_grouped_kv_expands_by_repeat_interleave(self):
        bridge = _bridge_stub(n_heads=4)
        grouped = torch.arange(2 * 2 * 3 * 2, dtype=torch.float32).reshape(2, 2, 3, 2)

        expanded = bridge._expand_kv_heads(grouped)

        assert expanded.shape == (2, 4, 3, 2)
        # Query head h must read kv head h // (n_heads // n_kv_heads).
        for h in range(4):
            assert torch.equal(expanded[:, h], grouped[:, h // 2])

    def test_mha_weights_pass_through_untouched(self):
        bridge = _bridge_stub(n_heads=4)
        mha = torch.randn(2, 4, 3, 2)
        assert bridge._expand_kv_heads(mha) is mha

    def test_non_4d_input_passes_through_untouched(self):
        bridge = _bridge_stub(n_heads=4)
        bias_stack = torch.randn(2, 2, 2)
        assert bridge._expand_kv_heads(bias_stack) is bias_stack

    def test_indivisible_head_counts_raise(self):
        bridge = _bridge_stub(n_heads=4)
        grouped = torch.randn(2, 3, 3, 2)
        with pytest.raises(ValueError, match="multiple of n_kv_heads"):
            bridge._expand_kv_heads(grouped)
