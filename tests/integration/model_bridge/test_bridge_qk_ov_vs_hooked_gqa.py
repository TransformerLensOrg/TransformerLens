"""GQA weight-circuit parity between TransformerBridge and the frozen HT goldens.

Acceptance for https://github.com/TransformerLensOrg/TransformerLens/issues/1553:
bridge.QK/OV on a GQA model must match the legacy HookedTransformer factors
(whose GroupedQueryAttention repeat_interleaves grouped K/V) within fp
tolerance. The frozen factors live in the Qwen2-0.5B golden views
(tests/goldens.py). Qwen2-0.5B (14 query heads, 2 kv heads) is a small ungated
GQA model; it is not CI-cached, hence @pytest.mark.slow.
"""

import pytest
import torch

from tests import goldens
from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "Qwen/Qwen2-0.5B"


@pytest.mark.slow
class TestGQAWeightCircuitParity:
    @pytest.fixture(scope="class")
    def bridge(self):
        return TransformerBridge.boot_transformers(MODEL, device="cpu")

    @pytest.fixture(scope="class")
    def golden_views(self):
        if not goldens.goldens_available(MODEL, "no_processing"):
            pytest.skip("TL goldens dataset unavailable (set TL_GOLDENS_DIR or enable network)")
        return goldens.GoldenCell(MODEL, "no_processing").tensors("views")

    def test_model_is_gqa(self, bridge):
        assert bridge.cfg.n_key_value_heads is not None
        assert bridge.cfg.n_key_value_heads < bridge.cfg.n_heads

    def test_qk_factors_match_hooked(self, bridge, golden_views):
        torch.testing.assert_close(bridge.QK.A, golden_views["QK.A"])
        torch.testing.assert_close(bridge.QK.B, golden_views["QK.B"])

    def test_ov_factors_match_hooked(self, bridge, golden_views):
        torch.testing.assert_close(bridge.OV.A, golden_views["OV.A"])
        torch.testing.assert_close(bridge.OV.B, golden_views["OV.B"])

    def test_qk_product_matches_hooked(self, bridge, golden_views):
        torch.testing.assert_close(
            bridge.QK.A[0] @ bridge.QK.B[0], golden_views["QK.A"][0] @ golden_views["QK.B"][0]
        )

    def test_kv_expansion_matches_raw_hf(self, bridge):
        """The grouped K expansion must be repeat_interleave over the raw HF k_proj.

        Independent of the goldens: recomputes the expanded K factor directly from
        the underlying HF weights, pinning the expansion semantics themselves.
        """
        cfg = bridge.cfg
        repeats = cfg.n_heads // cfg.n_key_value_heads
        hf_k = bridge.original_model.model.layers[0].self_attn.k_proj.weight
        # HF k_proj: [n_kv_heads * d_head, d_model] -> TL [n_kv_heads, d_model, d_head]
        k_tl = hf_k.reshape(cfg.n_key_value_heads, cfg.d_head, cfg.d_model).permute(0, 2, 1)
        expanded = torch.repeat_interleave(k_tl, repeats, dim=0)
        # QK.B is W_K^T stacked over layers: [n_layers, n_heads, d_head, d_model]
        torch.testing.assert_close(bridge.QK.B[0], expanded.transpose(-2, -1))

    def test_for_attn_layers_match_hooked(self, bridge, golden_views):
        indices, QK = bridge.QK_for_attn_layers()
        assert indices == list(range(bridge.cfg.n_layers))
        torch.testing.assert_close(QK.A, golden_views["QK.A"])
        torch.testing.assert_close(QK.B, golden_views["QK.B"])
        _, OV = bridge.OV_for_attn_layers()
        torch.testing.assert_close(OV.A, golden_views["OV.A"])
        torch.testing.assert_close(OV.B, golden_views["OV.B"])
