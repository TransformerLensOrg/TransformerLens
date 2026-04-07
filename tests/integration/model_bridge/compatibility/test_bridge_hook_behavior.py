"""Consolidated tests for TransformerBridge hook behavior.

Tests hook firing, modification, ablation, shapes, context managers, error handling,
and registry completeness. Consolidates overlapping tests from:
- tests/acceptance/model_bridge/compatibility/test_bridge_hooks.py
- tests/integration/model_bridge/compatibility/test_hooks.py
- tests/integration/model_bridge/test_attention_hook_compatibility.py

Uses distilgpt2 (CI-cached) for speed unless gpt2-specific behavior is being tested.
"""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="module")
def bridge():
    """TransformerBridge without compatibility mode."""
    return TransformerBridge.boot_transformers("distilgpt2", device="cpu")


@pytest.fixture(scope="module")
def bridge_compat():
    """TransformerBridge with compatibility mode."""
    b = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    b.enable_compatibility_mode()
    return b


@pytest.fixture(scope="module")
def reference_ht():
    """HookedTransformer for comparison."""
    return HookedTransformer.from_pretrained("distilgpt2", device="cpu")


class TestHookFiring:
    """Test that hooks fire correctly during forward passes."""

    def test_hook_fires_once_per_forward(self, bridge):
        """A registered forward hook fires exactly once per forward pass."""
        count = 0

        def hook_fn(tensor, hook):
            nonlocal count
            count += 1
            return tensor

        bridge.run_with_hooks(
            "Hello world",
            fwd_hooks=[("blocks.0.hook_resid_pre", hook_fn)],
        )
        assert count == 1

    def test_hook_receives_tensor_with_batch_and_seq(self, bridge):
        """Hook receives a tensor with at least batch and sequence dimensions."""
        captured = {}

        def hook_fn(tensor, hook):
            captured["shape"] = tensor.shape
            return tensor

        bridge.run_with_hooks(
            "Hello",
            fwd_hooks=[("blocks.0.hook_resid_pre", hook_fn)],
        )
        assert len(captured["shape"]) >= 2
        assert captured["shape"][0] >= 1  # batch >= 1

    def test_multiple_hooks_fire_independently(self, bridge):
        """Multiple hooks on different points each fire independently."""
        fired = set()

        def make_hook(name):
            def hook_fn(tensor, hook):
                fired.add(name)
                return tensor

            return hook_fn

        bridge.run_with_hooks(
            "Hello",
            fwd_hooks=[
                ("blocks.0.hook_resid_pre", make_hook("resid_pre_0")),
                ("blocks.0.hook_resid_post", make_hook("resid_post_0")),
            ],
        )
        assert fired == {"resid_pre_0", "resid_post_0"}

    @pytest.mark.xfail(reason="add_perma_hook not yet implemented on TransformerBridge")
    def test_perma_hook_persists_across_calls(self, bridge):
        """A permanent hook fires on every forward pass until removed."""
        count = 0

        def hook_fn(tensor, hook):
            nonlocal count
            count += 1
            return tensor

        bridge.add_perma_hook("blocks.0.hook_resid_pre", hook_fn)
        try:
            with torch.no_grad():
                bridge("Hello")
                assert count == 1
                bridge("World")
                assert count == 2
        finally:
            bridge.reset_hooks()


class TestHookModification:
    """Test that hooks can modify activations and affect output."""

    def test_zeroing_residual_changes_output(self, bridge):
        """Zeroing a residual stream hook changes the final output."""
        with torch.no_grad():
            normal_output = bridge("Hello world")

            def zero_hook(tensor, hook):
                return torch.zeros_like(tensor)

            modified_output = bridge.run_with_hooks(
                "Hello world",
                fwd_hooks=[("blocks.0.hook_resid_pre", zero_hook)],
            )

        assert not torch.allclose(normal_output, modified_output)

    def test_ablation_has_nonzero_effect(self, bridge_compat):
        """Ablating an attention head changes the loss."""
        test_text = "Natural language processing"
        baseline_loss = bridge_compat(test_text, return_type="loss")

        def ablation_hook(activation, hook):
            activation[:, :, 0, :] = 0
            return activation

        ablated_loss = bridge_compat.run_with_hooks(
            test_text,
            return_type="loss",
            fwd_hooks=[("blocks.0.attn.hook_v", ablation_hook)],
        )

        effect = abs(ablated_loss - baseline_loss)
        assert effect > 1e-6, f"Ablation should have meaningful effect (got {effect:.6f})"


class TestHookAblationEquivalence:
    """Test that ablation effects match between bridge and HookedTransformer."""

    def test_ablation_effect_matches_reference(self, bridge_compat, reference_ht):
        """Ablation effects should match between bridge and HookedTransformer."""
        test_text = "Natural language processing"

        def ablation_hook(activation, hook):
            activation[:, :, 5, :] = 0
            return activation

        ht_baseline = reference_ht(test_text, return_type="loss")
        ht_ablated = reference_ht.run_with_hooks(
            test_text,
            return_type="loss",
            fwd_hooks=[("blocks.0.attn.hook_v", ablation_hook)],
        )

        bridge_baseline = bridge_compat(test_text, return_type="loss")
        bridge_ablated = bridge_compat.run_with_hooks(
            test_text,
            return_type="loss",
            fwd_hooks=[("blocks.0.attn.hook_v", ablation_hook)],
        )

        ht_effect = ht_ablated - ht_baseline
        bridge_effect = bridge_ablated - bridge_baseline
        effect_diff = abs(ht_effect - bridge_effect)

        assert (
            effect_diff < 2e-4
        ), f"Hook effects should match between models (diff: {effect_diff:.6f})"


class TestHookActivationShapes:
    """Test that hook activations have expected shapes."""

    def test_embedding_shape_3d(self, bridge_compat):
        """Embedding hook should produce 3D tensor [batch, seq, d_model]."""
        shapes = {}

        def capture(name):
            def hook_fn(activation, hook):
                shapes[name] = activation.shape
                return activation

            return hook_fn

        bridge_compat.run_with_hooks(
            "The quick brown fox",
            return_type="logits",
            fwd_hooks=[("hook_embed", capture("embed"))],
        )
        assert len(shapes["embed"]) == 3
        assert shapes["embed"][-1] == bridge_compat.cfg.d_model

    def test_attention_v_shape_4d(self, bridge_compat):
        """Attention V hook should produce 4D tensor [batch, seq, n_heads, d_head]."""
        shapes = {}

        def capture(name):
            def hook_fn(activation, hook):
                shapes[name] = activation.shape
                return activation

            return hook_fn

        bridge_compat.run_with_hooks(
            "The quick brown fox",
            return_type="logits",
            fwd_hooks=[("blocks.0.attn.hook_v", capture("v"))],
        )
        assert len(shapes["v"]) == 4
        assert shapes["v"][2] == bridge_compat.cfg.n_heads

    def test_shapes_match_reference(self, bridge_compat, reference_ht):
        """Activation shapes should match between bridge and HookedTransformer."""
        hook_name = "blocks.0.attn.hook_v"
        tokens = reference_ht.to_tokens("The cat sat on")

        ref_act: list[torch.Tensor] = []
        bridge_act: list[torch.Tensor] = []

        def collect_ref(a: torch.Tensor, hook: object) -> torch.Tensor:
            ref_act.append(a)
            return a

        def collect_bridge(a: torch.Tensor, hook: object) -> torch.Tensor:
            bridge_act.append(a)
            return a

        reference_ht.add_hook(hook_name, collect_ref)
        bridge_compat.add_hook(hook_name, collect_bridge)

        with torch.no_grad():
            reference_ht(tokens)
            bridge_compat(tokens)

        reference_ht.reset_hooks()
        bridge_compat.reset_hooks()

        assert ref_act[0].shape == bridge_act[0].shape


class TestHookContextManager:
    """Test hook cleanup and context management."""

    def test_run_with_hooks_cleans_up(self, bridge):
        """Hooks from run_with_hooks don't persist after the call."""
        count = 0

        def hook_fn(tensor, hook):
            nonlocal count
            count += 1
            return tensor

        with torch.no_grad():
            bridge.run_with_hooks(
                "Hello",
                fwd_hooks=[("blocks.0.hook_resid_pre", hook_fn)],
            )
        assert count == 1

        count = 0
        with torch.no_grad():
            bridge("Hello")
        assert count == 0, "Hook persisted after run_with_hooks returned"

    def test_hooks_context_manager(self, bridge_compat):
        """hooks() context manager adds and removes hooks correctly."""
        hook_fired = False

        def test_hook(activation, hook):
            nonlocal hook_fired
            hook_fired = True
            return activation

        with bridge_compat.hooks(fwd_hooks=[("hook_embed", test_hook)]):
            bridge_compat("Natural language", return_type="logits")

        assert hook_fired, "Hook should have fired in context"

        hook_fired = False
        bridge_compat("Natural language", return_type="logits")
        assert not hook_fired, "Hook should be removed after context"


class TestHookRegistry:
    """Test hook registry completeness."""

    def test_key_hooks_present(self, bridge_compat, reference_ht):
        """Key hooks should be present in both bridge and reference."""
        key_hooks = [
            "hook_embed",
            "hook_pos_embed",
            "blocks.0.attn.hook_q",
            "blocks.0.attn.hook_k",
            "blocks.0.attn.hook_v",
            "blocks.0.attn.hook_z",
        ]
        for hook_name in key_hooks:
            assert hook_name in reference_ht.hook_dict, f"Reference missing {hook_name}"
            assert hook_name in bridge_compat.hook_dict, f"Bridge missing {hook_name}"

    def test_bridge_has_substantial_hooks(self, bridge_compat):
        """Bridge should have a substantial number of hooks.

        distilgpt2 has ~301 hooks, gpt2 has ~589. Threshold of 200 catches
        regressions where large portions of the hook registry are lost.
        """
        assert len(bridge_compat.hook_dict) > 200

    def test_expected_attention_hooks_available(self, bridge_compat):
        """Expected attention hook names should be available."""
        expected = [
            "blocks.0.attn.hook_v",
            "blocks.0.attn.hook_q",
            "blocks.0.attn.hook_k",
        ]
        hook_names = set(bridge_compat.hook_dict.keys())
        for hook_name in expected:
            assert hook_name in hook_names, f"Bridge missing hook: {hook_name}"


class TestHookErrorHandling:
    """Test error handling in hooks."""

    def test_hook_error_propagates(self, bridge_compat):
        """Errors in hooks should propagate to the caller."""
        tokens = bridge_compat.to_tokens("test")

        def error_hook(activation, hook):
            raise ValueError("Test error in hook")

        bridge_compat.add_hook("blocks.0.attn.hook_v", error_hook)
        with pytest.raises(ValueError, match="Test error in hook"):
            with torch.no_grad():
                bridge_compat(tokens)
        bridge_compat.reset_hooks()
