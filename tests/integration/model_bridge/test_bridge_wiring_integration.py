"""Integration tests for bridge internal wiring with real models.

Verifies that:
- Reshaped attention biases produce correct computation (not just correct shapes)
- Adapter path translations resolve to actual weight tensors with expected properties

Uses distilgpt2 (CI-cached).
"""

import pytest
import torch


@pytest.fixture()
def bridge_compat(distilgpt2_bridge_compat):
    """Alias session fixture for backward compatibility with test signatures."""
    return distilgpt2_bridge_compat


@pytest.fixture()
def reference_ht(distilgpt2_hooked_processed):
    """Alias session fixture for backward compatibility with test signatures."""
    return distilgpt2_hooked_processed


class TestReshapeBiasIntegration:
    """Verify reshaped biases produce correct attention computation on a real model."""

    def test_reshaped_b_Q_produces_matching_hook_q(self, bridge_compat, reference_ht):
        """b_Q reshaped via _reshape_bias should produce hook_q values matching HookedTransformer."""
        text = "The quick brown fox"

        with torch.no_grad():
            _, ht_cache = reference_ht.run_with_cache(text)
            _, br_cache = bridge_compat.run_with_cache(text)

        ht_q = ht_cache["blocks.0.attn.hook_q"]
        br_q = br_cache["blocks.0.attn.hook_q"]

        assert ht_q.shape == br_q.shape, f"hook_q shapes differ: {ht_q.shape} vs {br_q.shape}"
        max_diff = (ht_q - br_q).abs().max().item()
        assert max_diff < 1e-4, (
            f"hook_q values differ by {max_diff:.6f} — " f"bias reshaping may be incorrect"
        )

    def test_reshaped_b_V_produces_matching_hook_v(self, bridge_compat, reference_ht):
        """b_V reshaped via _reshape_bias should produce hook_v values matching HookedTransformer."""
        text = "The quick brown fox"

        with torch.no_grad():
            _, ht_cache = reference_ht.run_with_cache(text)
            _, br_cache = bridge_compat.run_with_cache(text)

        ht_v = ht_cache["blocks.0.attn.hook_v"]
        br_v = br_cache["blocks.0.attn.hook_v"]

        assert ht_v.shape == br_v.shape, f"hook_v shapes differ: {ht_v.shape} vs {br_v.shape}"
        max_diff = (ht_v - br_v).abs().max().item()
        assert max_diff < 1e-4, (
            f"hook_v values differ by {max_diff:.6f} — " f"bias reshaping may be incorrect"
        )


class TestAdapterPathResolution:
    """Verify adapter path translations resolve to real weight tensors."""

    def test_embed_path_resolves_to_weight(self, bridge_compat):
        """embed.W_E should resolve to a real embedding weight tensor."""
        W_E = bridge_compat.embed.W_E
        assert W_E is not None
        assert W_E.ndim == 2
        assert W_E.shape == (bridge_compat.cfg.d_vocab, bridge_compat.cfg.d_model)
        assert not torch.isnan(W_E).any()
        assert W_E.std() > 0, "Embedding weights should not be all zeros"

    def test_unembed_path_resolves_to_weight(self, bridge_compat):
        """unembed.W_U should resolve to a real unembedding weight tensor."""
        W_U = bridge_compat.unembed.W_U
        assert W_U is not None
        assert W_U.ndim == 2
        assert W_U.shape == (bridge_compat.cfg.d_model, bridge_compat.cfg.d_vocab)
        assert not torch.isnan(W_U).any()
        assert W_U.std() > 0

    def test_attention_weight_paths_resolve(self, bridge_compat):
        """W_Q, W_K, W_V, W_O per-block should resolve to real weight tensors."""
        cfg = bridge_compat.cfg
        block = bridge_compat.blocks[0]

        W_Q = block.attn.W_Q
        assert W_Q is not None
        assert W_Q.shape == (cfg.n_heads, cfg.d_model, cfg.d_head)
        assert not torch.isnan(W_Q).any()

        W_K = block.attn.W_K
        assert W_K.shape == (cfg.n_heads, cfg.d_model, cfg.d_head)

        W_V = block.attn.W_V
        assert W_V.shape == (cfg.n_heads, cfg.d_model, cfg.d_head)

        W_O = block.attn.W_O
        assert W_O.shape == (cfg.n_heads, cfg.d_head, cfg.d_model)

    def test_mlp_weight_paths_resolve(self, bridge_compat):
        """MLP weight paths should resolve to real weight tensors."""
        block = bridge_compat.blocks[0]

        W_in = block.mlp.W_in
        assert W_in is not None
        assert W_in.ndim == 2
        assert not torch.isnan(W_in).any()
        assert W_in.std() > 0

        W_out = block.mlp.W_out
        assert W_out is not None
        assert W_out.ndim == 2
        assert not torch.isnan(W_out).any()

    def test_stacked_weight_properties_match_per_block(self, bridge_compat):
        """Stacked W_Q property should match per-block W_Q values."""
        stacked_W_Q = bridge_compat.W_Q  # [n_layers, n_heads, d_model, d_head]
        block0_W_Q = bridge_compat.blocks[0].attn.W_Q  # [n_heads, d_model, d_head]

        assert torch.allclose(
            stacked_W_Q[0], block0_W_Q, atol=1e-6
        ), "Stacked W_Q[0] should match blocks[0].attn.W_Q"

    def test_translated_paths_match_hf_weights(self, bridge_compat):
        """Bridge weight properties should contain the same data as the underlying HF model."""
        hf_model = bridge_compat.original_model

        # distilgpt2 embedding: transformer.wte.weight
        hf_embed = hf_model.transformer.wte.weight
        bridge_embed = bridge_compat.embed.W_E

        assert torch.equal(
            hf_embed, bridge_embed
        ), "Bridge embed.W_E should be the same tensor as HF transformer.wte.weight"
