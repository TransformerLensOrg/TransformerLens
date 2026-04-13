"""Test mathematical correctness of weight processing operations.

Verifies that weight processing transformations produce the expected
mathematical properties, not just that they run without error.
Uses distilgpt2 (CI-cached).
"""

import copy

import pytest
import torch


@pytest.fixture(scope="module")
def bridge_fold_ln(distilgpt2_bridge):
    """Bridge with only fold_ln applied."""
    bridge = copy.deepcopy(distilgpt2_bridge)
    bridge.enable_compatibility_mode(
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return bridge


@pytest.fixture(scope="module")
def bridge_center_writing(distilgpt2_bridge):
    """Bridge with fold_ln + center_writing_weights applied."""
    bridge = copy.deepcopy(distilgpt2_bridge)
    bridge.enable_compatibility_mode(
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return bridge


@pytest.fixture(scope="module")
def bridge_center_unembed(distilgpt2_bridge):
    """Bridge with fold_ln + center_writing + center_unembed applied."""
    bridge = copy.deepcopy(distilgpt2_bridge)
    bridge.enable_compatibility_mode(
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return bridge


@pytest.fixture()
def bridge_fold_value_biases(distilgpt2_bridge_compat):
    """Bridge with all processing applied."""
    return distilgpt2_bridge_compat


class TestFoldLayerNorm:
    """After fold_ln, LayerNorm weights should be identity (w=1, b=0)."""

    def test_ln1_weights_are_ones(self, bridge_fold_ln):
        """After folding, ln1.w should be all ones."""
        checked = 0
        for i in range(bridge_fold_ln.cfg.n_layers):
            block = bridge_fold_ln.blocks[i]
            ln = block.ln1.original_component
            assert torch.allclose(
                ln.weight, torch.ones_like(ln.weight), atol=1e-6
            ), f"Layer {i} ln1.weight should be ones after fold_ln"
            checked += 1
        assert checked > 0, "No ln1 weights were checked — test is vacuous"

    def test_ln1_biases_are_zeros(self, bridge_fold_ln):
        """After folding, ln1.b should be all zeros."""
        checked = 0
        for i in range(bridge_fold_ln.cfg.n_layers):
            block = bridge_fold_ln.blocks[i]
            ln = block.ln1.original_component
            if ln.bias is not None:
                assert torch.allclose(
                    ln.bias, torch.zeros_like(ln.bias), atol=1e-6
                ), f"Layer {i} ln1.bias should be zeros after fold_ln"
                checked += 1
        assert checked > 0, "No ln1 biases were checked — test is vacuous"

    def test_ln2_weights_are_ones(self, bridge_fold_ln):
        """After folding, ln2.w should be all ones."""
        checked = 0
        for i in range(bridge_fold_ln.cfg.n_layers):
            block = bridge_fold_ln.blocks[i]
            ln = block.ln2.original_component
            assert torch.allclose(
                ln.weight, torch.ones_like(ln.weight), atol=1e-6
            ), f"Layer {i} ln2.weight should be ones after fold_ln"
            checked += 1
        assert checked > 0, "No ln2 weights were checked — test is vacuous"

    def test_ln_final_weights_are_ones(self, bridge_fold_ln):
        """After folding, ln_final.w should be all ones."""
        ln = bridge_fold_ln.ln_final.original_component
        assert torch.allclose(
            ln.weight, torch.ones_like(ln.weight), atol=1e-6
        ), "ln_final.weight should be ones after fold_ln"

    def test_fold_preserves_output(self, bridge_fold_ln, distilgpt2_bridge):
        """Folding should not change model output (mathematically equivalent)."""
        # Compare against an unprocessed bridge (deepcopy + no-processing compat mode)
        bridge_unproc = copy.deepcopy(distilgpt2_bridge)
        bridge_unproc.enable_compatibility_mode(no_processing=True)

        text = "The quick brown fox"
        with torch.no_grad():
            folded_loss = bridge_fold_ln(text, return_type="loss").item()
            unfolded_loss = bridge_unproc(text, return_type="loss").item()

        # Folding is mathematically equivalent — output should be very close
        assert abs(folded_loss - unfolded_loss) < 0.01, (
            f"fold_ln should not change output: folded={folded_loss:.6f}, "
            f"unfolded={unfolded_loss:.6f}"
        )


class TestCenterWritingWeights:
    """After center_writing_weights, writing weights should have zero column mean."""

    def test_W_O_centered(self, bridge_center_writing):
        """W_O columns should have zero mean after centering."""
        W_O = bridge_center_writing.W_O  # [n_layers, n_heads, d_head, d_model]
        # Mean along the output dimension (d_model) should be ~0
        # W_O writes to the residual stream along d_model
        col_mean = W_O.mean(dim=-1)  # [n_layers, n_heads, d_head]
        assert torch.allclose(
            col_mean, torch.zeros_like(col_mean), atol=1e-5
        ), f"W_O column mean should be ~0 after centering, max: {col_mean.abs().max():.6f}"


class TestCenterUnembed:
    """After center_unembed, unembed weights should have zero row mean."""

    def test_unembed_rows_centered(self, bridge_center_unembed):
        """W_U rows should have zero mean after centering."""
        # W_U shape: [d_model, d_vocab] — rows are indexed by d_model
        # center_unembed subtracts the mean along d_vocab (columns)
        W_U = bridge_center_unembed.unembed.W_U  # [d_model, d_vocab]
        col_mean = W_U.mean(dim=-1)  # [d_model]
        assert torch.allclose(
            col_mean, torch.zeros_like(col_mean), atol=1e-5
        ), f"W_U column mean should be ~0 after centering, max: {col_mean.abs().max():.6f}"
