"""Unit tests for bridge component inspection functionality."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge


class TestBridgeComponentInspection:
    """Test inspection of bridge components and their properties."""

    @pytest.fixture
    def bridge(self):
        """Create a TransformerBridge for testing."""
        return TransformerBridge.boot_transformers("gpt2", device="cpu")

    def test_bridge_has_required_components(self, bridge):
        """Test that bridge has all required transformer components."""
        # Check main transformer structure
        assert hasattr(bridge.original_model, "transformer"), "Should have transformer module"
        transformer = bridge.original_model.transformer

        # Check core components
        assert hasattr(transformer, "wte"), "Should have token embedding (wte)"
        assert hasattr(transformer, "wpe"), "Should have position embedding (wpe)"
        assert hasattr(transformer, "h"), "Should have transformer layers (h)"
        assert hasattr(transformer, "ln_f"), "Should have final layer norm (ln_f)"
        assert hasattr(bridge.original_model, "lm_head"), "Should have language model head"

    def test_transformer_layers_structure(self, bridge):
        """Test the structure of transformer layers."""
        layers = bridge.original_model.transformer.h
        assert len(layers) > 0, "Should have at least one transformer layer"

        # Check first layer structure
        layer_0 = layers[0]
        assert hasattr(layer_0, "ln_1"), "Layer should have first layer norm"
        assert hasattr(layer_0, "attn"), "Layer should have attention"
        assert hasattr(layer_0, "ln_2"), "Layer should have second layer norm"
        assert hasattr(layer_0, "mlp"), "Layer should have MLP"

        # Check that all layers have consistent structure
        for i, layer in enumerate(layers):
            assert hasattr(layer, "ln_1"), f"Layer {i} should have ln_1"
            assert hasattr(layer, "attn"), f"Layer {i} should have attn"
            assert hasattr(layer, "ln_2"), f"Layer {i} should have ln_2"
            assert hasattr(layer, "mlp"), f"Layer {i} should have mlp"

    def test_attention_component_structure(self, bridge):
        """Test the structure of attention components."""
        attn = bridge.original_model.transformer.h[0].attn

        # GPT-2 style attention should have these components
        expected_attrs = ["c_attn", "c_proj"]  # GPT-2 specific naming
        for attr in expected_attrs:
            assert hasattr(attn, attr), f"Attention should have {attr}"

        # Check weight shapes are reasonable
        c_attn = attn.c_attn
        if hasattr(c_attn, "weight"):
            weight_shape = c_attn.weight.shape
            assert len(weight_shape) == 2, f"c_attn weight should be 2D: {weight_shape}"
            assert (
                weight_shape[0] > 0 and weight_shape[1] > 0
            ), f"Weight should have positive dimensions: {weight_shape}"

    def test_mlp_component_structure(self, bridge):
        """Test the structure of MLP components."""
        mlp = bridge.original_model.transformer.h[0].mlp

        # GPT-2 style MLP should have these components
        expected_attrs = ["c_fc", "c_proj"]  # GPT-2 specific naming
        for attr in expected_attrs:
            assert hasattr(mlp, attr), f"MLP should have {attr}"

        # Check weight shapes
        c_fc = mlp.c_fc
        if hasattr(c_fc, "weight"):
            weight_shape = c_fc.weight.shape
            assert len(weight_shape) == 2, f"c_fc weight should be 2D: {weight_shape}"

    def test_embedding_components(self, bridge):
        """Test embedding component properties."""
        transformer = bridge.original_model.transformer

        # Token embedding
        wte = transformer.wte
        assert hasattr(wte, "weight"), "Token embedding should have weight"
        wte_shape = wte.weight.shape
        assert len(wte_shape) == 2, f"Token embedding should be 2D: {wte_shape}"
        assert (
            wte_shape[0] > 0 and wte_shape[1] > 0
        ), "Token embedding should have positive dimensions"

        # Position embedding
        wpe = transformer.wpe
        assert hasattr(wpe, "weight"), "Position embedding should have weight"
        wpe_shape = wpe.weight.shape
        assert len(wpe_shape) == 2, f"Position embedding should be 2D: {wpe_shape}"
        assert (
            wpe_shape[1] == wte_shape[1]
        ), "Position and token embeddings should have same hidden dimension"

    def test_lm_head_structure(self, bridge):
        """Test language model head structure."""
        lm_head = bridge.original_model.lm_head
        assert hasattr(lm_head, "weight"), "LM head should have weight"

        lm_head_shape = lm_head.weight.shape
        assert len(lm_head_shape) == 2, f"LM head should be 2D: {lm_head_shape}"

        # LM head vocab size should match token embedding
        wte_shape = bridge.original_model.transformer.wte.weight.shape
        assert (
            lm_head_shape[0] == wte_shape[0]
        ), "LM head and token embedding should have same vocab size"

    def test_component_types(self, bridge):
        """Test that components are of expected PyTorch types."""
        transformer = bridge.original_model.transformer

        # All components should be nn.Module subclasses
        assert isinstance(transformer.wte, torch.nn.Module), "Token embedding should be nn.Module"
        assert isinstance(
            transformer.wpe, torch.nn.Module
        ), "Position embedding should be nn.Module"
        assert isinstance(transformer.ln_f, torch.nn.Module), "Final layer norm should be nn.Module"

        # Layer components
        layer_0 = transformer.h[0]
        assert isinstance(layer_0.ln_1, torch.nn.Module), "Layer norm 1 should be nn.Module"
        assert isinstance(layer_0.attn, torch.nn.Module), "Attention should be nn.Module"
        assert isinstance(layer_0.ln_2, torch.nn.Module), "Layer norm 2 should be nn.Module"
        assert isinstance(layer_0.mlp, torch.nn.Module), "MLP should be nn.Module"

    def test_parameter_devices(self, bridge):
        """Test that all parameters are on the expected device."""
        expected_device = torch.device("cpu")

        # Check embedding parameters
        transformer = bridge.original_model.transformer
        assert transformer.wte.weight.device == expected_device, "Token embedding should be on CPU"
        assert (
            transformer.wpe.weight.device == expected_device
        ), "Position embedding should be on CPU"

        # Check layer parameters
        layer_0 = transformer.h[0]
        for name, param in layer_0.named_parameters():
            assert (
                param.device == expected_device
            ), f"Parameter {name} should be on CPU, got {param.device}"

        # Check LM head
        assert (
            bridge.original_model.lm_head.weight.device == expected_device
        ), "LM head should be on CPU"

    def test_parameter_dtypes(self, bridge):
        """Test that parameters have expected data types."""
        # Most parameters should be float32 or float16
        valid_dtypes = {torch.float32, torch.float16, torch.bfloat16}

        transformer = bridge.original_model.transformer

        # Check key parameters
        assert (
            transformer.wte.weight.dtype in valid_dtypes
        ), f"Token embedding dtype: {transformer.wte.weight.dtype}"
        assert (
            transformer.wpe.weight.dtype in valid_dtypes
        ), f"Position embedding dtype: {transformer.wpe.weight.dtype}"

        # Check layer 0 parameters
        for name, param in transformer.h[0].named_parameters():
            assert (
                param.dtype in valid_dtypes
            ), f"Parameter {name} has unexpected dtype: {param.dtype}"

    def test_model_configuration_accessible(self, bridge):
        """Test that model configuration is accessible."""
        # Should have access to the original model's config
        assert hasattr(bridge.original_model, "config"), "Model should have configuration"

        config = bridge.original_model.config
        assert hasattr(config, "n_layer"), "Config should specify number of layers"
        assert hasattr(config, "n_head"), "Config should specify number of heads"
        assert hasattr(config, "n_embd"), "Config should specify embedding dimension"

        # Verify config matches actual model structure
        actual_layers = len(bridge.original_model.transformer.h)
        assert (
            config.n_layer == actual_layers
        ), f"Config layers ({config.n_layer}) should match actual ({actual_layers})"
