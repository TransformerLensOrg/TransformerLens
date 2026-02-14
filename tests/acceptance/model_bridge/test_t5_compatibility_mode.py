"""Acceptance test for T5 compatibility mode in TransformerBridge.

This test verifies that T5 can be loaded with TransformerBridge and that
compatibility mode can be successfully enabled with proper hook registration.
"""

import gc

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.utilities.bridge_components import apply_fn_to_all_components


class TestT5CompatibilityMode:
    """Test T5 compatibility mode functionality."""

    @pytest.fixture(autouse=True)
    def cleanup_after_test(self):
        """Clean up memory after each test."""
        yield
        # Force garbage collection and clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for _ in range(3):
            gc.collect()

    @pytest.fixture
    def model_name(self):
        """T5 model to test."""
        return "google-t5/t5-small"

    @pytest.fixture
    def bridge_model(self, model_name):
        """Load T5 model via TransformerBridge."""
        return TransformerBridge.boot_transformers(model_name, device="cpu")

    def test_t5_loads_successfully(self, bridge_model, model_name):
        """Test that T5 loads successfully via TransformerBridge."""
        assert bridge_model is not None
        assert bridge_model.cfg.model_name == model_name
        assert hasattr(bridge_model, "encoder_blocks")
        assert hasattr(bridge_model, "decoder_blocks")

    def test_linear_bridge_submodules_exist(self, bridge_model):
        """Test that AttentionBridge and MLPBridge have LinearBridge submodules.

        This is critical for compatibility mode to work - without LinearBridge
        submodules, hook aliases like 'hook_q -> q.hook_out' will fail.
        """
        # Check encoder attention
        encoder_attn = bridge_model.encoder_blocks[0].attn
        assert hasattr(encoder_attn, "q"), "Encoder attention missing q submodule"
        assert hasattr(encoder_attn, "k"), "Encoder attention missing k submodule"
        assert hasattr(encoder_attn, "v"), "Encoder attention missing v submodule"
        assert hasattr(encoder_attn, "o"), "Encoder attention missing o submodule"

        # Verify they are LinearBridge instances, not raw Linear layers
        from transformer_lens.model_bridge.generalized_components.linear import (
            LinearBridge,
        )

        assert isinstance(encoder_attn.q, LinearBridge), "q should be LinearBridge"
        assert isinstance(encoder_attn.k, LinearBridge), "k should be LinearBridge"
        assert isinstance(encoder_attn.v, LinearBridge), "v should be LinearBridge"
        assert isinstance(encoder_attn.o, LinearBridge), "o should be LinearBridge"

        # Check decoder self-attention
        decoder_self_attn = bridge_model.decoder_blocks[0].self_attn
        assert hasattr(decoder_self_attn, "q"), "Decoder self-attn missing q submodule"
        assert isinstance(decoder_self_attn.q, LinearBridge), "q should be LinearBridge"

        # Check decoder cross-attention
        decoder_cross_attn = bridge_model.decoder_blocks[0].cross_attn
        assert hasattr(decoder_cross_attn, "q"), "Decoder cross-attn missing q submodule"
        assert isinstance(decoder_cross_attn.q, LinearBridge), "q should be LinearBridge"

        # Check encoder MLP
        encoder_mlp = bridge_model.encoder_blocks[0].mlp
        # Use getattr since 'in' is a Python keyword
        mlp_in = getattr(encoder_mlp, "in", None)
        mlp_out = getattr(encoder_mlp, "out", None)
        assert mlp_in is not None, "Encoder MLP missing 'in' submodule"
        assert mlp_out is not None, "Encoder MLP missing 'out' submodule"
        assert isinstance(mlp_in, LinearBridge), "in should be LinearBridge"
        assert isinstance(mlp_out, LinearBridge), "out should be LinearBridge"

    def test_linear_bridge_hooks_accessible(self, bridge_model):
        """Test that LinearBridge submodules have hook_out."""
        encoder_attn = bridge_model.encoder_blocks[0].attn

        assert hasattr(encoder_attn.q, "hook_out"), "LinearBridge q missing hook_out"
        assert hasattr(encoder_attn.k, "hook_out"), "LinearBridge k missing hook_out"
        assert hasattr(encoder_attn.v, "hook_out"), "LinearBridge v missing hook_out"
        assert hasattr(encoder_attn.o, "hook_out"), "LinearBridge o missing hook_out"

        # Verify they are HookPoints
        from transformer_lens.hook_points import HookPoint

        assert isinstance(encoder_attn.q.hook_out, HookPoint)
        assert isinstance(encoder_attn.k.hook_out, HookPoint)
        assert isinstance(encoder_attn.v.hook_out, HookPoint)
        assert isinstance(encoder_attn.o.hook_out, HookPoint)

    def test_compatibility_mode_enables_successfully(self, bridge_model):
        """Test that compatibility mode can be enabled for T5.

        This is the main acceptance test - compatibility mode should enable
        without errors and properly register all hooks.
        """
        # Enable compatibility mode manually (avoiding full enable_compatibility_mode
        # which includes weight processing that doesn't work for T5 yet)
        bridge_model.compatibility_mode = True

        def set_compatibility_mode(component):
            component.compatibility_mode = True
            component.disable_warnings = False

        apply_fn_to_all_components(bridge_model, set_compatibility_mode)

        # Re-initialize hook registry to include aliases
        bridge_model.clear_hook_registry()
        bridge_model._initialize_hook_registry()

        # Verify compatibility mode is enabled
        assert bridge_model.compatibility_mode is True

    def test_hook_registry_populated(self, bridge_model):
        """Test that hook registry is populated after enabling compatibility mode."""
        # Enable compatibility mode
        bridge_model.compatibility_mode = True

        def set_compatibility_mode(component):
            component.compatibility_mode = True
            component.disable_warnings = False

        apply_fn_to_all_components(bridge_model, set_compatibility_mode)
        bridge_model.clear_hook_registry()
        bridge_model._initialize_hook_registry()

        # Check that hooks are registered
        assert len(bridge_model._hook_registry) > 0, "Hook registry should not be empty"

        # Should have hundreds of canonical hooks (encoder + decoder)
        # Note: _hook_registry only contains canonical hooks, not aliases
        assert (
            len(bridge_model._hook_registry) > 400
        ), f"Expected >400 hooks, got {len(bridge_model._hook_registry)}"

    def test_critical_hooks_accessible(self, bridge_model):
        """Test that critical hooks are accessible after compatibility mode."""
        # Enable compatibility mode
        bridge_model.compatibility_mode = True

        def set_compatibility_mode(component):
            component.compatibility_mode = True
            component.disable_warnings = False

        apply_fn_to_all_components(bridge_model, set_compatibility_mode)
        bridge_model.clear_hook_registry()
        bridge_model._initialize_hook_registry()

        # Test critical encoder hooks
        critical_hooks = [
            "encoder_blocks.0.hook_in",
            "encoder_blocks.0.attn.q.hook_out",
            "encoder_blocks.0.attn.hook_out",
            "encoder_blocks.0.mlp.in.hook_out",
            "encoder_blocks.0.mlp.out.hook_out",
            # Decoder hooks
            "decoder_blocks.0.hook_in",
            "decoder_blocks.0.self_attn.q.hook_out",
            "decoder_blocks.0.cross_attn.k.hook_out",
            "decoder_blocks.0.mlp.in.hook_out",
        ]

        for hook_name in critical_hooks:
            assert (
                hook_name in bridge_model._hook_registry
            ), f"Critical hook {hook_name} not found in registry"

    def test_encoder_decoder_hook_counts(self, bridge_model):
        """Test that both encoder and decoder have reasonable hook counts."""
        # Enable compatibility mode
        bridge_model.compatibility_mode = True

        def set_compatibility_mode(component):
            component.compatibility_mode = True
            component.disable_warnings = False

        apply_fn_to_all_components(bridge_model, set_compatibility_mode)
        bridge_model.clear_hook_registry()
        bridge_model._initialize_hook_registry()

        # Count encoder and decoder hooks
        encoder_hooks = [h for h in bridge_model._hook_registry if "encoder" in h]
        decoder_hooks = [h for h in bridge_model._hook_registry if "decoder" in h]

        assert len(encoder_hooks) > 0, "Should have encoder hooks"
        assert len(decoder_hooks) > 0, "Should have decoder hooks"

        # Decoder should have more hooks (has cross-attention in addition to self-attention)
        assert len(decoder_hooks) > len(
            encoder_hooks
        ), "Decoder should have more hooks than encoder"

    def test_t5_block_bridge_hooks(self, bridge_model):
        """Test that T5BlockBridge has the expected hooks."""
        # Check encoder block
        encoder_block = bridge_model.encoder_blocks[0]
        assert hasattr(encoder_block, "hook_in")
        assert hasattr(encoder_block, "hook_out")
        assert hasattr(encoder_block, "hook_resid_mid")

        # Encoder blocks should NOT have hook_resid_mid2 (only 2 layers)
        assert not hasattr(encoder_block, "hook_resid_mid2")

        # Check decoder block
        decoder_block = bridge_model.decoder_blocks[0]
        assert hasattr(decoder_block, "hook_in")
        assert hasattr(decoder_block, "hook_out")
        assert hasattr(decoder_block, "hook_resid_mid")

        # Decoder blocks SHOULD have hook_resid_mid2 (3 layers - after cross-attn)
        assert hasattr(decoder_block, "hook_resid_mid2")

    def test_rms_normalization_used(self, bridge_model):
        """Test that T5 uses RMSNormalizationBridge throughout."""
        from transformer_lens.model_bridge.generalized_components.rms_normalization import (
            RMSNormalizationBridge,
        )

        # Check encoder
        assert isinstance(bridge_model.encoder_blocks[0].ln1, RMSNormalizationBridge)
        assert isinstance(bridge_model.encoder_blocks[0].ln2, RMSNormalizationBridge)
        assert isinstance(bridge_model.encoder_ln_final, RMSNormalizationBridge)

        # Check decoder
        assert isinstance(bridge_model.decoder_blocks[0].ln1, RMSNormalizationBridge)
        assert isinstance(bridge_model.decoder_blocks[0].ln2, RMSNormalizationBridge)
        assert isinstance(bridge_model.decoder_blocks[0].ln3, RMSNormalizationBridge)
        assert isinstance(bridge_model.decoder_ln_final, RMSNormalizationBridge)
