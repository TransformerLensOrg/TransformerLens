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


@pytest.fixture()
def bridge(distilgpt2_bridge):
    """Alias session fixture for backward compatibility with test signatures."""
    return distilgpt2_bridge


@pytest.fixture()
def bridge_compat(distilgpt2_bridge_compat):
    """Alias session fixture for backward compatibility with test signatures."""
    return distilgpt2_bridge_compat


@pytest.fixture(scope="module")
def golden(distilgpt2_goldens_processed):
    """Golden cell replacing the live-HT reference."""
    return distilgpt2_goldens_processed


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
            # Session-scoped fixture: plain reset_hooks() keeps permanent hooks,
            # leaking this one onto blocks.0.hook_in for every later test.
            bridge.reset_hooks(including_permanent=True)


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
    """Test that ablation effects match the frozen HookedTransformer golden anchors."""

    def test_ablation_effect_matches_reference(self, bridge_compat, golden):
        """Replaying the golden ablation must reproduce the frozen HT loss delta."""
        anchor = golden.scalars["ablation"]
        head = anchor["head"]

        def ablation_hook(activation, hook):
            activation[:, :, head, :] = 0
            return activation

        bridge_baseline = bridge_compat(anchor["text"], return_type="loss")
        bridge_ablated = bridge_compat.run_with_hooks(
            anchor["text"],
            return_type="loss",
            fwd_hooks=[(anchor["hook"], ablation_hook)],
        )

        golden_effect = anchor["ablated_loss"] - anchor["orig_loss"]
        bridge_effect = (bridge_ablated - bridge_baseline).item()
        effect_diff = abs(golden_effect - bridge_effect)

        assert (
            abs(bridge_baseline.item() - anchor["orig_loss"]) < 0.01
        ), f"Baseline loss drifted from golden: {bridge_baseline.item()} vs {anchor['orig_loss']}"
        assert (
            effect_diff < 2e-4
        ), f"Hook effect should match the golden anchor (diff: {effect_diff:.6f})"


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

    def test_shapes_match_reference(self, bridge_compat, golden):
        """Activation shapes should match the golden hook manifest."""
        hook_name = "blocks.0.attn.hook_v"
        golden_shape = golden.hook_manifest[hook_name]
        assert golden_shape is not None, f"{hook_name} unfired in the golden capture"

        bridge_act: list[torch.Tensor] = []

        def collect_bridge(a: torch.Tensor, hook: object) -> torch.Tensor:
            bridge_act.append(a)
            return a

        bridge_compat.add_hook(hook_name, collect_bridge)
        try:
            with torch.no_grad():
                bridge_compat(golden.scalars["short_prompt"])
        finally:
            bridge_compat.reset_hooks()

        assert list(bridge_act[0].shape) == golden_shape


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

    def test_key_hooks_present(self, bridge_compat, golden):
        """Key hooks should be present in both the golden manifest and the bridge."""
        key_hooks = [
            "hook_embed",
            "hook_pos_embed",
            "blocks.0.attn.hook_q",
            "blocks.0.attn.hook_k",
            "blocks.0.attn.hook_v",
            "blocks.0.attn.hook_z",
        ]
        manifest = golden.hook_manifest
        for hook_name in key_hooks:
            assert hook_name in manifest, f"Golden manifest missing {hook_name}"
            assert hook_name in bridge_compat.hook_dict, f"Bridge missing {hook_name}"

    def test_bridge_has_substantial_hooks(self, bridge_compat):
        """Bridge should have a substantial number of hooks.

        distilgpt2 has ~301 hooks, gpt2 has ~589. Threshold of 200 catches
        regressions where large portions of the hook registry are lost.
        """
        assert len(bridge_compat.hook_dict) > 200


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
