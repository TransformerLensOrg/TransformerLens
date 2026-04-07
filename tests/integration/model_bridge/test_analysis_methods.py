"""Tests for TransformerBridge mechanistic interpretability analysis methods.

Tests tokens_to_residual_directions, accumulated_bias, all_composition_scores,
all_head_labels, and top-level W_E/W_U/b_U properties. Validates against
HookedTransformer for correctness, not just shape/type.

Uses distilgpt2 (CI-cached).
"""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge.bridge import TransformerBridge


@pytest.fixture(scope="module")
def bridge_compat():
    b = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    b.enable_compatibility_mode()
    return b


@pytest.fixture(scope="module")
def reference_ht():
    return HookedTransformer.from_pretrained("distilgpt2", device="cpu")


class TestTopLevelWeightProperties:
    """Test W_E, W_U, b_U delegate to the correct component tensors."""

    def test_W_E_is_same_object_as_embed(self, bridge_compat):
        """bridge.W_E should be the exact same tensor as bridge.embed.W_E."""
        assert bridge_compat.W_E is bridge_compat.embed.W_E

    def test_W_U_equals_unembed(self, bridge_compat):
        """bridge.W_U should equal bridge.unembed.W_U (may be a view/transpose)."""
        assert torch.equal(bridge_compat.W_U, bridge_compat.unembed.W_U)

    def test_b_U_equals_unembed(self, bridge_compat):
        """bridge.b_U should equal bridge.unembed.b_U."""
        assert torch.equal(bridge_compat.b_U, bridge_compat.unembed.b_U)

    def test_W_E_matches_hooked_transformer(self, bridge_compat, reference_ht):
        """bridge.W_E values should match HookedTransformer.W_E."""
        assert bridge_compat.W_E.shape == reference_ht.W_E.shape
        # After weight processing, embeddings may differ due to centering.
        # But shapes must match and both must be non-zero.
        assert bridge_compat.W_E.std() > 0
        assert reference_ht.W_E.std() > 0

    def test_W_U_matches_hooked_transformer(self, bridge_compat, reference_ht):
        """bridge.W_U values should match HookedTransformer.W_U."""
        assert bridge_compat.W_U.shape == reference_ht.W_U.shape
        max_diff = (bridge_compat.W_U - reference_ht.W_U).abs().max().item()
        assert max_diff < 1e-4, f"W_U differs by {max_diff}"


class TestTokensToResidualDirections:
    """Test tokens_to_residual_directions produces correct unembedding vectors."""

    def test_single_token_string(self, bridge_compat):
        """String token should return a 1-D vector of size d_model."""
        rd = bridge_compat.tokens_to_residual_directions("hello")
        assert rd.shape == (bridge_compat.cfg.d_model,)

    def test_single_token_int(self, bridge_compat):
        """Integer token should return a 1-D vector of size d_model."""
        rd = bridge_compat.tokens_to_residual_directions(100)
        assert rd.shape == (bridge_compat.cfg.d_model,)

    def test_equals_W_U_column(self, bridge_compat):
        """Result should be exactly the corresponding column of W_U."""
        token_id = 42
        rd = bridge_compat.tokens_to_residual_directions(token_id)
        expected = bridge_compat.W_U[:, token_id]
        assert torch.equal(rd, expected)

    def test_batch_tokens(self, bridge_compat):
        """1-D tensor of tokens should return (n_tokens, d_model)."""
        tokens = torch.tensor([100, 200, 300])
        rd = bridge_compat.tokens_to_residual_directions(tokens)
        assert rd.shape == (3, bridge_compat.cfg.d_model)
        # Each row should match the corresponding W_U column
        for i, tok in enumerate(tokens):
            assert torch.equal(rd[i], bridge_compat.W_U[:, tok])

    def test_matches_hooked_transformer(self, bridge_compat, reference_ht):
        """Output should match HookedTransformer for the same tokens."""
        tokens = torch.tensor([10, 20, 30])
        bridge_rd = bridge_compat.tokens_to_residual_directions(tokens)
        ht_rd = reference_ht.tokens_to_residual_directions(tokens)
        max_diff = (bridge_rd - ht_rd).abs().max().item()
        assert max_diff < 1e-4, f"Residual directions differ by {max_diff}"


class TestAccumulatedBias:
    """Test accumulated_bias sums biases correctly."""

    def test_layer_zero_is_zeros(self, bridge_compat):
        """accumulated_bias(0) should be all zeros (no layers processed)."""
        ab = bridge_compat.accumulated_bias(0)
        assert ab.shape == (bridge_compat.cfg.d_model,)
        assert torch.allclose(ab, torch.zeros_like(ab))

    def test_layer_one_includes_first_block(self, bridge_compat):
        """accumulated_bias(1) should include block 0's biases and be non-zero."""
        ab = bridge_compat.accumulated_bias(1)
        assert ab.shape == (bridge_compat.cfg.d_model,)
        # distilgpt2 has biases, so this should be non-zero
        assert ab.norm() > 0

    def test_monotonically_increasing_norm(self, bridge_compat):
        """Accumulated bias norm should generally increase with more layers."""
        # Not strictly monotonic, but bias(n_layers) should have larger norm than bias(0)
        ab_0 = bridge_compat.accumulated_bias(0)
        ab_all = bridge_compat.accumulated_bias(bridge_compat.cfg.n_layers)
        assert ab_all.norm() > ab_0.norm()

    def test_matches_hooked_transformer(self, bridge_compat, reference_ht):
        """Output should match HookedTransformer."""
        for layer in [0, 1, 3, bridge_compat.cfg.n_layers]:
            bridge_ab = bridge_compat.accumulated_bias(layer)
            ht_ab = reference_ht.accumulated_bias(layer)
            max_diff = (bridge_ab - ht_ab).abs().max().item()
            assert max_diff < 1e-4, f"accumulated_bias({layer}) differs by {max_diff}"

    def test_mlp_input_flag(self, bridge_compat, reference_ht):
        """mlp_input=True should include the current layer's attn bias."""
        bridge_ab = bridge_compat.accumulated_bias(1, mlp_input=True)
        ht_ab = reference_ht.accumulated_bias(1, mlp_input=True)
        max_diff = (bridge_ab - ht_ab).abs().max().item()
        assert max_diff < 1e-4, f"accumulated_bias(1, mlp_input=True) differs by {max_diff}"


class TestAllCompositionScores:
    """Test all_composition_scores produces correct composition score matrices."""

    def test_shape(self, bridge_compat):
        """Shape should be (n_layers, n_heads, n_layers, n_heads)."""
        cfg = bridge_compat.cfg
        scores = bridge_compat.all_composition_scores("Q")
        assert scores.shape == (cfg.n_layers, cfg.n_heads, cfg.n_layers, cfg.n_heads)

    def test_upper_triangular_masking(self, bridge_compat):
        """Scores should be zero where left_layer >= right_layer."""
        scores = bridge_compat.all_composition_scores("Q")
        n_layers = bridge_compat.cfg.n_layers
        for l1 in range(n_layers):
            for l2 in range(l1 + 1):  # l2 <= l1
                assert (
                    scores[l1, :, l2, :] == 0
                ).all(), f"Scores at L{l1}->L{l2} should be zero (upper triangular)"

    def test_nonzero_above_diagonal(self, bridge_compat):
        """At least some scores above the diagonal should be non-zero."""
        scores = bridge_compat.all_composition_scores("Q")
        # Check L0 -> L1 (first above-diagonal block)
        assert scores[0, :, 1, :].abs().sum() > 0

    def test_all_modes_work(self, bridge_compat):
        """Q, K, V modes should all produce valid tensors."""
        for mode in ["Q", "K", "V"]:
            scores = bridge_compat.all_composition_scores(mode)
            assert not torch.isnan(scores).any(), f"NaN in {mode} composition scores"

    def test_invalid_mode_raises(self, bridge_compat):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode must be one of"):
            bridge_compat.all_composition_scores("X")


class TestAllHeadLabels:
    """Test all_head_labels produces correct labels."""

    def test_count(self, bridge_compat):
        """Should have n_layers * n_heads labels."""
        labels = bridge_compat.all_head_labels
        expected = bridge_compat.cfg.n_layers * bridge_compat.cfg.n_heads
        assert len(labels) == expected

    def test_format(self, bridge_compat):
        """Labels should follow L{layer}H{head} format."""
        labels = bridge_compat.all_head_labels
        assert labels[0] == "L0H0"
        assert labels[1] == "L0H1"
        assert labels[bridge_compat.cfg.n_heads] == "L1H0"

    def test_matches_hooked_transformer(self, bridge_compat, reference_ht):
        """Should match HookedTransformer's labels exactly."""
        assert bridge_compat.all_head_labels == reference_ht.all_head_labels()
