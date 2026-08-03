"""GQA weight-circuit parity between TransformerBridge and HookedTransformer.

Acceptance for https://github.com/TransformerLensOrg/TransformerLens/issues/1553:
bridge.QK/OV on a GQA model must match HookedTransformer.QK/OV (whose
GroupedQueryAttention repeat_interleaves grouped K/V) within fp tolerance.
Qwen2-0.5B (14 query heads, 2 kv heads) is a small ungated GQA model both
systems support; it is not CI-cached, hence @pytest.mark.slow.
"""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "Qwen/Qwen2-0.5B"


@pytest.mark.slow
class TestGQAWeightCircuitParity:
    @pytest.fixture(scope="class")
    def bridge_and_hooked(self):
        bridge = TransformerBridge.boot_transformers(MODEL, device="cpu")
        hooked = HookedTransformer.from_pretrained_no_processing(MODEL, device="cpu")
        return bridge, hooked

    def test_model_is_gqa(self, bridge_and_hooked):
        bridge, _ = bridge_and_hooked
        assert bridge.cfg.n_key_value_heads is not None
        assert bridge.cfg.n_key_value_heads < bridge.cfg.n_heads

    def test_qk_factors_match_hooked(self, bridge_and_hooked):
        bridge, hooked = bridge_and_hooked
        torch.testing.assert_close(bridge.QK.A, hooked.QK.A)
        torch.testing.assert_close(bridge.QK.B, hooked.QK.B)

    def test_ov_factors_match_hooked(self, bridge_and_hooked):
        bridge, hooked = bridge_and_hooked
        torch.testing.assert_close(bridge.OV.A, hooked.OV.A)
        torch.testing.assert_close(bridge.OV.B, hooked.OV.B)

    def test_qk_product_matches_hooked(self, bridge_and_hooked):
        bridge, hooked = bridge_and_hooked
        torch.testing.assert_close(bridge.QK.A[0] @ bridge.QK.B[0], hooked.QK.A[0] @ hooked.QK.B[0])

    def test_for_attn_layers_match_hooked(self, bridge_and_hooked):
        bridge, hooked = bridge_and_hooked
        indices, QK = bridge.QK_for_attn_layers()
        assert indices == list(range(bridge.cfg.n_layers))
        torch.testing.assert_close(QK.A, hooked.QK.A)
        torch.testing.assert_close(QK.B, hooked.QK.B)
        _, OV = bridge.OV_for_attn_layers()
        torch.testing.assert_close(OV.A, hooked.OV.A)
        torch.testing.assert_close(OV.B, hooked.OV.B)
