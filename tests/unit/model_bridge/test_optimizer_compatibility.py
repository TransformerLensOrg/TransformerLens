"""Tests for TransformerBridge optimizer compatibility.

Ensures that TransformerBridge.parameters() returns only leaf tensors
that are compatible with PyTorch optimizers.
"""

import pytest
import torch
from torch import nn

from transformer_lens.model_bridge.bridge import TransformerBridge


@pytest.fixture
def small_bridge_model():
    """Create a small TransformerBridge model for testing."""
    model_name = "distilgpt2"  # Use smaller model for faster tests
    bridge = TransformerBridge.boot_transformers(model_name, device="cpu")

    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token

    return bridge


class TestParametersAreLeafTensors:
    """Test that parameters() returns only leaf tensors."""

    def test_all_parameters_are_leaf_tensors(self, small_bridge_model):
        """Verify all parameters returned by parameters() are leaf tensors."""
        for i, param in enumerate(small_bridge_model.parameters()):
            assert param.is_leaf, (
                f"Parameter {i} is non-leaf (has grad_fn={param.grad_fn}). "
                "Non-leaf tensors cannot be optimized."
            )
            assert isinstance(param, nn.Parameter), f"Parameter {i} is not an nn.Parameter"

    def test_tl_parameters_provides_tl_style_names(self, small_bridge_model):
        """Verify tl_parameters() provides TransformerLens-style parameter dictionary.

        tl_parameters() returns processed weights for analysis (via SVDInterpreter, etc.).
        These may include non-leaf tensors created by einops.rearrange(), which is
        expected and correct for TransformerLens compatibility.

        For optimization, use parameters() which returns only leaf tensors.
        """
        # Get TL-style parameters
        tl_params = small_bridge_model.tl_parameters()

        # Check that we have TL-style names (blocks.X.attn.W_Y format)
        assert any(
            "blocks." in name and ".attn." in name for name in tl_params.keys()
        ), "Expected TransformerLens-style parameter names like 'blocks.0.attn.W_Q'"

        # Check that some common TL parameter names exist
        assert any(
            name.endswith(".W_E") for name in tl_params.keys()
        ), "Expected embedding parameter 'W_E'"

    def test_tl_named_parameters_provides_iterator(self, small_bridge_model):
        """Verify tl_named_parameters() provides iterator with TL-style names.

        This method provides the same content as tl_parameters() but as an iterator,
        maintaining consistency with PyTorch's named_parameters() API pattern.
        """
        # Get TL-style parameters as iterator
        tl_named_params = list(small_bridge_model.tl_named_parameters())
        tl_params_dict = small_bridge_model.tl_parameters()

        # Verify iterator returns same content as dictionary
        assert len(tl_named_params) == len(
            tl_params_dict
        ), "Iterator should yield same number of parameters as dict"

        # Verify names and tensors match
        iterator_dict = dict(tl_named_params)
        for name, tensor in tl_params_dict.items():
            assert name in iterator_dict, f"Name {name} should be in iterator output"
            assert torch.equal(iterator_dict[name], tensor), f"Tensor for {name} should match"

        # Check that we have TL-style names (blocks.X.attn.W_Y format)
        param_names = [name for name, _ in tl_named_params]
        assert any(
            "blocks." in name and ".attn." in name for name in param_names
        ), "Expected TransformerLens-style parameter names like 'blocks.0.attn.W_Q'"

    def test_no_processed_weights_in_parameters(self, small_bridge_model):
        """Verify processed weight attributes are not included in parameters().

        Note: This test verifies that parameters() (e.g. for optimizers) doesn't include
        internal processed weight attributes. These weights should only appear in
        tl_parameters().
        """
        # Enable compatibility mode to create processed weights
        small_bridge_model.enable_compatibility_mode(no_processing=True)

        # Get all parameter names from PyTorch-style named_parameters()
        hf_param_names = {name for name, _ in small_bridge_model.named_parameters()}

        # Check that processed weight attribute names are NOT in parameters
        # (They should exist as attributes but not be trainable parameters)
        for block_idx in range(small_bridge_model.cfg.n_layers):
            # These are the processed weight attributes created by _set_processed_weight_attributes
            processed_weight_attrs = [
                f"blocks.{block_idx}.attn._processed_W_Q",
                f"blocks.{block_idx}.attn._processed_W_K",
                f"blocks.{block_idx}.attn._processed_W_V",
                f"blocks.{block_idx}.attn._processed_W_O",
            ]

            for attr_name in processed_weight_attrs:
                # The attribute might exist on the object but should NOT be in parameters()
                assert attr_name not in hf_param_names, (
                    f"Processed weight attribute '{attr_name}' should not be in parameters(). "
                    "Processed weights are views for analysis, not trainable parameters."
                )


class TestOptimizerCompatibility:
    """Test that TransformerBridge works with standard PyTorch optimizers."""

    def test_adamw_accepts_parameters(self, small_bridge_model):
        """Test that AdamW optimizer accepts TransformerBridge parameters."""
        # This should not raise "can't optimize a non-leaf Tensor"
        try:
            optimizer = torch.optim.AdamW(small_bridge_model.parameters(), lr=1e-4)
            assert optimizer is not None
        except ValueError as e:
            if "can't optimize a non-leaf Tensor" in str(e):
                pytest.fail(
                    "AdamW rejected TransformerBridge parameters. "
                    "This indicates non-leaf tensors are being returned by parameters()."
                )
            raise

    def test_gradient_flow_after_backward(self, small_bridge_model):
        """Test that gradients flow correctly after backward pass."""
        small_bridge_model.train()
        input_ids = torch.randint(0, small_bridge_model.cfg.d_vocab, (1, 10))
        logits = small_bridge_model(input_ids, return_type="logits")
        loss = logits.sum()
        loss.backward()

        # Verify that parameters have gradients
        params_with_grad = 0
        total_params = 0

        for param in small_bridge_model.parameters():
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
                # Verify gradient is on a leaf tensor
                assert param.is_leaf, "Gradient was computed for a non-leaf tensor"

        # At least some parameters should have gradients
        assert params_with_grad > 0, "No parameters received gradients after backward pass"

    def test_optimizer_step_updates_parameters(self, small_bridge_model):
        """Test that optimizer.step() actually updates model parameters."""
        small_bridge_model.train()
        optimizer = torch.optim.SGD(small_bridge_model.parameters(), lr=0.1)

        # Get initial parameter values (first few params for efficiency)
        initial_params = {}
        for i, (name, param) in enumerate(small_bridge_model.named_parameters()):
            if i >= 5:  # Just check first 5 parameters
                break
            initial_params[name] = param.data.clone()

        # Create dummy input and compute loss
        input_ids = torch.randint(0, small_bridge_model.cfg.d_vocab, (1, 10))
        logits = small_bridge_model(input_ids, return_type="logits")
        loss = logits.sum()

        # Backward and step
        loss.backward()
        optimizer.step()

        # Verify parameters were updated
        params_updated = 0
        for name, initial_value in initial_params.items():
            current_value = dict(small_bridge_model.named_parameters())[name].data

            # Check if parameter changed
            if not torch.allclose(initial_value, current_value, atol=1e-8):
                params_updated += 1

        assert params_updated > 0, (
            "No parameters were updated after optimizer.step(). "
            "This suggests the optimizer is not correctly connected to the model parameters."
        )


class TestParametersAfterCompatibilityMode:
    """Test parameters() behavior after enabling compatibility mode."""

    def test_parameters_still_leaf_after_compatibility_mode(self, small_bridge_model):
        """Verify parameters() returns leaf tensors even after enabling compatibility mode."""
        # Enable compatibility mode (which creates processed weights)
        small_bridge_model.enable_compatibility_mode(no_processing=True)

        # All parameters from parameters() should still be leaf tensors
        # (named_parameters() may include non-leaf processed weights for TL compatibility)
        for i, param in enumerate(small_bridge_model.parameters()):
            assert param.is_leaf, (
                f"Parameter {i} from parameters() is non-leaf after compatibility mode. "
                "Compatibility mode should not affect trainable parameters from parameters()."
            )

    def test_optimizer_works_after_compatibility_mode(self, small_bridge_model):
        """Test that optimizers still work after enabling compatibility mode."""
        # Enable compatibility mode
        small_bridge_model.enable_compatibility_mode(no_processing=True)

        # Should still be able to create optimizer
        try:
            optimizer = torch.optim.AdamW(small_bridge_model.parameters(), lr=1e-4)
            assert optimizer is not None
        except ValueError as e:
            if "can't optimize a non-leaf Tensor" in str(e):
                pytest.fail(
                    "AdamW rejected parameters after compatibility mode. "
                    "This indicates non-leaf tensors are being returned."
                )
            raise


class TestParametersMatchOriginalModel:
    """Test that parameters() returns the same parameters as the original HF model."""

    def test_parameter_count_matches(self, small_bridge_model):
        """Verify parameter count matches original model."""
        bridge_param_count = sum(1 for _ in small_bridge_model.parameters())
        original_param_count = sum(1 for _ in small_bridge_model.original_model.parameters())

        assert bridge_param_count == original_param_count, (
            f"Parameter count mismatch: Bridge has {bridge_param_count}, "
            f"original model has {original_param_count}"
        )

    def test_parameters_are_same_objects(self, small_bridge_model):
        """Verify that parameters() returns the actual original model parameters."""
        bridge_params = list(small_bridge_model.parameters())
        original_params = list(small_bridge_model.original_model.parameters())

        # Should have same number of parameters
        assert len(bridge_params) == len(original_params)

        # Parameters should be the same objects (same id)
        # This ensures gradients flow to the original model
        for bridge_param, original_param in zip(bridge_params, original_params):
            assert bridge_param is original_param, (
                "Bridge parameters should be the exact same objects as original model parameters. "
                "This ensures gradient flow and memory efficiency."
            )
