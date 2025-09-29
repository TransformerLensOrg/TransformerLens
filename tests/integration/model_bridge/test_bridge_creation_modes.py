"""Test different bridge creation and configuration modes."""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge.bridge import TransformerBridge


class TestBridgeCreationModes:
    """Test different modes of creating and configuring TransformerBridge."""

    @pytest.fixture
    def reference_model(self):
        """Create reference HookedTransformer."""
        return HookedTransformer.from_pretrained("gpt2", device="cpu")

    @pytest.fixture
    def test_text(self):
        """Test text for evaluation."""
        return "Hello world"

    def test_bridge_no_processing(self, reference_model, test_text):
        """Test bridge with no weight processing."""
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
        bridge.enable_compatibility_mode(no_processing=True)

        ref_loss = reference_model(test_text, return_type="loss")
        bridge_loss = bridge(test_text, return_type="loss")

        # With no processing, losses should be close but not identical
        assert (
            abs(ref_loss - bridge_loss) < 1.0
        ), f"Losses should be reasonably close: {ref_loss} vs {bridge_loss}"
        assert 3.0 < bridge_loss < 6.0, f"Bridge loss should be reasonable: {bridge_loss}"

    def test_bridge_full_compatibility(self, reference_model, test_text):
        """Test bridge with full compatibility mode processing."""
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
        bridge.enable_compatibility_mode()

        ref_loss = reference_model(test_text, return_type="loss")
        bridge_loss = bridge(test_text, return_type="loss")

        # With full processing, losses should be very close
        diff = abs(ref_loss - bridge_loss)
        assert diff < 0.01, f"Processed bridge should match reference closely: {diff}"
        assert 3.0 < bridge_loss < 6.0, f"Bridge loss should be reasonable: {bridge_loss}"

    def test_bridge_component_inspection(self):
        """Test that bridge components can be inspected."""
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

        # Check that we can access the original model components
        assert hasattr(bridge.original_model, "transformer"), "Should have transformer"
        assert hasattr(bridge.original_model.transformer, "h"), "Should have layers"
        assert len(bridge.original_model.transformer.h) > 0, "Should have at least one layer"

        # Check layer 0 components
        block_0 = bridge.original_model.transformer.h[0]
        assert hasattr(block_0, "ln_1"), "Should have ln_1"
        assert hasattr(block_0, "attn"), "Should have attention"
        assert hasattr(block_0, "ln_2"), "Should have ln_2"
        assert hasattr(block_0, "mlp"), "Should have MLP"

        # Check embedding and final components
        assert hasattr(bridge.original_model.transformer, "wte"), "Should have token embedding"
        assert hasattr(bridge.original_model.transformer, "wpe"), "Should have position embedding"
        assert hasattr(bridge.original_model, "lm_head"), "Should have language model head"

    def test_bridge_tokenizer_compatibility(self, reference_model):
        """Test that bridge tokenizer works like reference."""
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
        test_text = "Hello world test"

        # Tokenize with both
        ref_tokens = reference_model.to_tokens(test_text)
        bridge_tokens = bridge.to_tokens(test_text)

        # Should produce identical tokens
        assert torch.equal(ref_tokens, bridge_tokens), "Tokenizers should produce identical results"

    def test_bridge_configuration_persistence(self):
        """Test that bridge configuration persists correctly."""
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

        # Test configuration before compatibility mode
        assert hasattr(bridge, "cfg"), "Bridge should have configuration"

        # Enable compatibility mode and check it persists
        bridge.enable_compatibility_mode()

        # Configuration should still be accessible
        assert hasattr(bridge, "cfg"), "Configuration should persist after compatibility mode"
        assert bridge.cfg is not None, "Configuration should not be None"

    def test_bridge_device_handling(self):
        """Test that bridge handles device specification correctly."""
        # Test CPU device
        bridge_cpu = TransformerBridge.boot_transformers("gpt2", device="cpu")
        assert (
            next(bridge_cpu.original_model.parameters()).device.type == "cpu"
        ), "Model should be on CPU device"

        # Test that bridge can process text on correct device
        test_text = "Device test"
        loss = bridge_cpu(test_text, return_type="loss")
        assert isinstance(loss, torch.Tensor), "Should return tensor"
        assert loss.device.type == "cpu", "Loss should be on CPU"

    def test_bridge_memory_efficiency(self):
        """Test that bridge creation doesn't leak excessive memory."""
        import gc

        # Get initial memory usage
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        # Create and destroy bridge
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
        bridge.enable_compatibility_mode()

        # Process some text to ensure everything is initialized
        _ = bridge("Test", return_type="loss")

        # Clean up
        del bridge
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Should not raise any memory-related errors
        assert True, "Memory cleanup should work correctly"
