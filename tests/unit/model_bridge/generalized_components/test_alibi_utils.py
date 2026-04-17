"""Tests for shared ALiBi utility functions."""

import torch

from transformer_lens.model_bridge.generalized_components.alibi_utils import (
    build_alibi_slopes,
    build_alibi_tensor,
)


class TestBuildAlibiSlopes:
    """Test ALiBi slope computation against hand-derived values."""

    def test_power_of_2_heads_4(self):
        # base = 2^(-(2^-(log2(4)-3))) = 2^(-(2^(-1))) = 2^(-2) = 0.25
        slopes = build_alibi_slopes(4, torch.device("cpu"))
        expected = torch.tensor([0.25, 0.0625, 0.015625, 0.00390625])
        assert torch.allclose(slopes, expected, atol=1e-7)

    def test_power_of_2_heads_8(self):
        # base = 2^(-(2^-(3-3))) = 2^(-1) = 0.5
        slopes = build_alibi_slopes(8, torch.device("cpu"))
        expected = torch.tensor([0.5**i for i in range(1, 9)])
        assert torch.allclose(slopes, expected, atol=1e-7)

    def test_non_power_of_2_heads_6(self):
        # closest_pow2=4, base=0.25 → first 4
        # extra_base=0.5, extra_powers=[1,3] → 0.5^1, 0.5^3
        slopes = build_alibi_slopes(6, torch.device("cpu"))
        expected = torch.tensor([0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125])
        assert slopes.shape == (6,)
        assert torch.allclose(slopes, expected, atol=1e-7)

    def test_slopes_length_matches_num_heads(self):
        for n in [1, 2, 3, 5, 7, 16, 32]:
            slopes = build_alibi_slopes(n, torch.device("cpu"))
            assert slopes.shape == (n,), f"Failed for n_heads={n}"

    def test_all_slopes_positive(self):
        slopes = build_alibi_slopes(32, torch.device("cpu"))
        assert (slopes > 0).all()


class TestBuildAlibiTensor:
    """Test full ALiBi tensor generation."""

    def test_output_shape(self):
        mask = torch.ones(2, 8, dtype=torch.long)
        alibi = build_alibi_tensor(mask, 4, torch.float32)
        assert alibi.shape == (2, 4, 1, 8)

    def test_first_position_is_zero(self):
        """Position 0 should always have zero bias regardless of slope."""
        mask = torch.ones(1, 4, dtype=torch.long)
        alibi = build_alibi_tensor(mask, 8, torch.float32)
        assert (alibi[:, :, :, 0] == 0).all()

    def test_values_against_hand_computation(self):
        """Verify against manually computed values for 2 heads, seq_len=4."""
        # base = 0.0625, slopes = [0.0625, 0.00390625]
        # positions = [0, 1, 2, 3]
        mask = torch.ones(1, 4, dtype=torch.long)
        alibi = build_alibi_tensor(mask, 2, torch.float32)
        # shape: [1, 2, 1, 4]
        head0 = alibi[0, 0, 0]  # [4]
        head1 = alibi[0, 1, 0]  # [4]
        expected_h0 = torch.tensor([0.0, 0.0625, 0.125, 0.1875])
        expected_h1 = torch.tensor([0.0, 0.00390625, 0.0078125, 0.01171875])
        assert torch.allclose(head0, expected_h0, atol=1e-6)
        assert torch.allclose(head1, expected_h1, atol=1e-6)

    def test_masked_positions_are_zero(self):
        """Positions where attention_mask=0 should produce zero bias."""
        mask = torch.tensor([[1, 1, 0, 0]])  # last 2 positions masked
        alibi = build_alibi_tensor(mask, 4, torch.float32)
        assert (alibi[:, :, :, 2:] == 0).all()

    def test_batch_independence(self):
        """Each batch element should be computed independently."""
        mask = torch.ones(3, 6, dtype=torch.long)
        alibi = build_alibi_tensor(mask, 4, torch.float32)
        # All batch elements have same mask → same alibi
        assert torch.allclose(alibi[0], alibi[1])
        assert torch.allclose(alibi[1], alibi[2])

    def test_matches_hf_falcon_slopes(self):
        """Verify slopes match HF Falcon (the bfloat16-free part of their implementation).

        HF Falcon applies a bfloat16 cast to slopes before multiplying with positions,
        so full tensor values diverge slightly. We verify that the underlying slope
        values (which determine the relative bias per head) are identical.
        """
        from transformers.models.falcon.modeling_falcon import (
            build_alibi_tensor as hf_falcon_alibi,
        )

        mask = torch.ones(1, 4, dtype=torch.long)
        for n_heads in [8, 16, 32]:
            ours = build_alibi_tensor(mask, n_heads, torch.float32)
            hf = hf_falcon_alibi(mask, n_heads, torch.float32)
            # Extract slope per head from position 1 (avoids bfloat16 compounding)
            ours_slopes = ours.reshape(n_heads, 1, 4)[:, 0, 1]
            hf_slopes = hf[:, 0, 1]
            assert torch.allclose(
                ours_slopes, hf_slopes, rtol=0.01
            ), f"Slope mismatch for {n_heads} heads: max_diff={(ours_slopes - hf_slopes).abs().max()}"

    def test_matches_hf_bloom(self):
        """Verify against HuggingFace Bloom's ALiBi implementation."""
        from transformers.models.bloom.modeling_bloom import (
            build_alibi_tensor as hf_bloom_alibi,
        )

        mask = torch.ones(1, 16, dtype=torch.long)
        for n_heads in [8, 16, 32]:
            ours = build_alibi_tensor(mask, n_heads, torch.float32)
            hf = hf_bloom_alibi(mask, n_heads, torch.float32)
            ours_flat = ours.reshape(n_heads, 1, 16)
            assert torch.allclose(
                ours_flat, hf, atol=1e-5
            ), f"Mismatch for {n_heads} heads: max_diff={(ours_flat - hf).abs().max()}"

    def test_dtype_preserved(self):
        mask = torch.ones(1, 4, dtype=torch.long)
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            alibi = build_alibi_tensor(mask, 4, dtype)
            assert alibi.dtype == dtype
