"""Integration tests for ProcessWeights._fold_layer on real GPT-2 weights.

Inputs come from the frozen unprocessed gpt2 golden (tests/goldens.py), whose
LayerNorm parameters are still present — so these tests exercise the real
folding path deterministically. _fold_layer folds ln1 into the attention
weights and ln2 into the MLP for one layer, setting the folded LN params to
identity (w=1, b=0) rather than deleting them.
"""

import pytest
import torch

from tests import goldens
from transformer_lens.weight_processing import ProcessWeights


@pytest.fixture(scope="module")
def gpt2_cfg():
    """gpt2 bridge boot supplying the config for ProcessWeights calls."""
    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers("gpt2", device="cpu").cfg


@pytest.fixture(scope="module")
def _golden_unprocessed():
    if not goldens.goldens_available("gpt2", "no_processing"):
        pytest.skip("TL goldens dataset unavailable (set TL_GOLDENS_DIR or enable network)")
    return goldens.GoldenCell("gpt2", "no_processing").tensors("state_dict")


@pytest.fixture()
def state_dict(_golden_unprocessed):
    """Fresh mutable copy per test — _fold_layer mutates in place."""
    return {k: v.clone() for k, v in _golden_unprocessed.items()}


def _assert_identity_ln(state_dict, key_w, key_b):
    torch.testing.assert_close(state_dict[key_w], torch.ones_like(state_dict[key_w]))
    torch.testing.assert_close(state_dict[key_b], torch.zeros_like(state_dict[key_b]))


class TestFoldLayerIntegration:
    """Integration tests for _fold_layer with real golden weights."""

    def test_fold_layer_transformer_lens_format(self, state_dict, gpt2_cfg):
        """_fold_layer folds ln1/ln2 into attn/MLP and leaves identity LN params."""
        layer_idx = 0
        original = {k: v.clone() for k, v in state_dict.items()}

        assert f"blocks.{layer_idx}.ln1.w" in state_dict, "golden must be unprocessed"

        ProcessWeights._fold_layer(
            state_dict,
            gpt2_cfg,
            layer_idx=layer_idx,
            fold_biases=True,
            center_weights=True,
            adapter=None,
            gqa="",
        )

        # Folded LN params become identities
        _assert_identity_ln(state_dict, f"blocks.{layer_idx}.ln1.w", f"blocks.{layer_idx}.ln1.b")
        _assert_identity_ln(state_dict, f"blocks.{layer_idx}.ln2.w", f"blocks.{layer_idx}.ln2.b")

        # Attention weights are centered over d_model
        for name in ("W_Q", "W_K", "W_V"):
            w = state_dict[f"blocks.{layer_idx}.attn.{name}"]
            mean = torch.mean(w, dim=1, keepdim=True)
            assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)

        # Biases absorbed the projected ln1 bias (they change from the original)
        for name in ("b_Q", "b_K", "b_V"):
            key = f"blocks.{layer_idx}.attn.{name}"
            assert not torch.equal(state_dict[key], original[key]), f"{key} unchanged by fold"

        # MLP input weights absorbed ln2 (change) and other layers are untouched
        assert not torch.equal(
            state_dict[f"blocks.{layer_idx}.mlp.W_in"], original[f"blocks.{layer_idx}.mlp.W_in"]
        )
        untouched_layer = layer_idx + 1
        for key in (f"blocks.{untouched_layer}.ln1.w", f"blocks.{untouched_layer}.attn.W_Q"):
            assert torch.equal(state_dict[key], original[key]), f"{key} modified unexpectedly"

    def test_fold_layer_with_different_layers(self, _golden_unprocessed, gpt2_cfg):
        """_fold_layer works identically on first, middle, and last layers."""
        for layer_idx in [0, 1, gpt2_cfg.n_layers - 1]:
            state_dict = {k: v.clone() for k, v in _golden_unprocessed.items()}

            ProcessWeights._fold_layer(
                state_dict,
                gpt2_cfg,
                layer_idx=layer_idx,
                fold_biases=True,
                center_weights=True,
                adapter=None,
                gqa="",
            )

            _assert_identity_ln(
                state_dict, f"blocks.{layer_idx}.ln1.w", f"blocks.{layer_idx}.ln1.b"
            )
            w_q = state_dict[f"blocks.{layer_idx}.attn.W_Q"]
            w_q_mean = torch.mean(w_q, dim=1, keepdim=True)
            assert torch.allclose(w_q_mean, torch.zeros_like(w_q_mean), atol=1e-6)

    def test_fold_layer_with_different_options(self, _golden_unprocessed, gpt2_cfg):
        """fold_biases and center_weights act independently."""
        layer_idx = 0

        # No bias folding, with centering: LN bias survives, weights centered.
        state_dict = {k: v.clone() for k, v in _golden_unprocessed.items()}
        original_ln1_b = state_dict[f"blocks.{layer_idx}.ln1.b"].clone()
        ProcessWeights._fold_layer(
            state_dict,
            gpt2_cfg,
            layer_idx=layer_idx,
            fold_biases=False,
            center_weights=True,
            adapter=None,
            gqa="",
        )
        assert torch.equal(state_dict[f"blocks.{layer_idx}.ln1.b"], original_ln1_b)
        w_q = state_dict[f"blocks.{layer_idx}.attn.W_Q"]
        w_q_mean = torch.mean(w_q, dim=1, keepdim=True)
        assert torch.allclose(w_q_mean, torch.zeros_like(w_q_mean), atol=1e-6)

        # With bias folding, no centering: LN folded to identity, weights NOT centered.
        state_dict = {k: v.clone() for k, v in _golden_unprocessed.items()}
        ProcessWeights._fold_layer(
            state_dict,
            gpt2_cfg,
            layer_idx=layer_idx,
            fold_biases=True,
            center_weights=False,
            adapter=None,
            gqa="",
        )
        _assert_identity_ln(state_dict, f"blocks.{layer_idx}.ln1.w", f"blocks.{layer_idx}.ln1.b")
        w_q = state_dict[f"blocks.{layer_idx}.attn.W_Q"]
        w_q_mean = torch.mean(w_q, dim=1, keepdim=True)
        assert not torch.allclose(w_q_mean, torch.zeros_like(w_q_mean), atol=1e-6)
