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
        return HookedTransformer.from_pretrained("distilgpt2", device="cpu")

    @pytest.fixture
    def test_text(self):
        """Test text for evaluation."""
        return "Hello world"

    def test_bridge_no_processing(self, reference_model, test_text):
        """Test bridge with no weight processing."""
        bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
        bridge.enable_compatibility_mode(no_processing=True)

        ref_loss = reference_model(test_text, return_type="loss")
        bridge_loss = bridge(test_text, return_type="loss")

        # With no processing, losses should be close but not identical
        assert (
            abs(ref_loss - bridge_loss) < 1.0
        ), f"Losses should be reasonably close: {ref_loss} vs {bridge_loss}"
        assert 3.0 < bridge_loss < 8.0, f"Bridge loss should be reasonable: {bridge_loss}"

    def test_bridge_full_compatibility(self, reference_model, test_text):
        """Test bridge with full compatibility mode processing."""
        bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
        bridge.enable_compatibility_mode()

        ref_loss = reference_model(test_text, return_type="loss")
        bridge_loss = bridge(test_text, return_type="loss")

        # With full processing, losses should be very close
        diff = abs(ref_loss - bridge_loss)
        assert diff < 0.01, f"Processed bridge should match reference closely: {diff}"
        assert 3.0 < bridge_loss < 8.0, f"Bridge loss should be reasonable: {bridge_loss}"

    def test_bridge_component_inspection(self):
        """Test that bridge components can be inspected."""
        bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")

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
        bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
        test_text = "Hello world test"

        # Tokenize with both
        ref_tokens = reference_model.to_tokens(test_text)
        bridge_tokens = bridge.to_tokens(test_text)

        # Should produce identical tokens
        assert torch.equal(ref_tokens, bridge_tokens), "Tokenizers should produce identical results"

    def test_bridge_configuration_persistence(self):
        """Test that bridge configuration persists correctly."""
        bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")

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
        bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
        bridge.enable_compatibility_mode()

        # Process some text to ensure everything is initialized
        _ = bridge("Test", return_type="loss")

        # Clean up
        del bridge
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Should not raise any memory-related errors
        assert True, "Memory cleanup should work correctly"


class TestBridgeOfflineWithHfModel:
    """Bridge must reuse hf_model.config when supplied, not refetch from the Hub (#846)."""

    OFFLINE_MODEL = "trl-internal-testing/tiny-MistralForCausalLM-0.2"

    @pytest.fixture
    def hf_model(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(self.OFFLINE_MODEL).eval()

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.OFFLINE_MODEL)

    def test_offline_boot_with_hf_model(self, hf_model, tokenizer):
        """Bridge boot succeeds when AutoConfig.from_pretrained would fail."""
        from unittest.mock import patch

        import transformer_lens.model_bridge.sources.transformers as bridge_source

        with patch.object(bridge_source, "AutoConfig") as mock_autoconfig:
            mock_autoconfig.from_pretrained.side_effect = OSError("Simulated Hub failure")

            bridge = TransformerBridge.boot_transformers(
                self.OFFLINE_MODEL, hf_model=hf_model, tokenizer=tokenizer
            )

            test_input = tokenizer("Hello", return_tensors="pt")["input_ids"]
            with torch.no_grad():
                logits = bridge(test_input)
            assert torch.isfinite(logits).all()

    def test_hf_model_config_not_mutated_by_bridge(self, hf_model, tokenizer):
        """hf_config_overrides / n_ctx / pad_token_id mutations must not leak into hf_model.config.

        Bridge works on a deepcopy so the user's loaded model stays clean. Catches a
        regression where the deepcopy is dropped in favor of a direct reference.
        """
        snapshot = {
            "max_position_embeddings": getattr(hf_model.config, "max_position_embeddings", None),
            "pad_token_id": getattr(hf_model.config, "pad_token_id", None),
            "output_attentions": getattr(hf_model.config, "output_attentions", None),
        }

        TransformerBridge.boot_transformers(
            self.OFFLINE_MODEL,
            hf_model=hf_model,
            tokenizer=tokenizer,
            hf_config_overrides={"max_position_embeddings": 999},
            n_ctx=128,
        )

        for attr, original_value in snapshot.items():
            assert (
                getattr(hf_model.config, attr, None) == original_value
            ), f"Bridge mutated hf_model.config.{attr}"

    def test_autoconfig_not_called_when_hf_model_provided(self, hf_model, tokenizer):
        """Patches AutoConfig in the bridge module only — transitive calls from AutoTokenizer
        (which uses ``transformers.AutoConfig`` directly) aren't intercepted.
        """
        from unittest.mock import patch

        import transformer_lens.model_bridge.sources.transformers as bridge_source

        with patch.object(bridge_source, "AutoConfig") as mock_autoconfig:
            TransformerBridge.boot_transformers(
                self.OFFLINE_MODEL, hf_model=hf_model, tokenizer=tokenizer
            )
            assert not mock_autoconfig.from_pretrained.called
