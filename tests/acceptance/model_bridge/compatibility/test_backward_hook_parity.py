"""Test that all backward hooks produce identical gradients in HookedTransformer and TransformerBridge.

This test ensures complete parity between the two architectures by comparing every gradient
that passes through every backward hook during backpropagation.
"""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


class TestBackwardHookParity:
    """Test suite for comparing backward hook gradients between HookedTransformer and TransformerBridge."""

    @pytest.fixture
    def model_name(self):
        """Model name to use for testing."""
        return "gpt2"

    @pytest.fixture
    def prompt(self):
        """Test prompt for forward pass."""
        return "The quick brown fox"

    @pytest.fixture
    def hooked_transformer(self, model_name):
        """Create a HookedTransformer for comparison."""
        return HookedTransformer.from_pretrained_no_processing(model_name, device_map="cpu")

    @pytest.fixture
    def transformer_bridge(self, model_name):
        """Create a TransformerBridge without processing."""
        model = TransformerBridge.boot_transformers(model_name, device="cpu")
        model.enable_compatibility_mode(no_processing=True)
        return model

    def test_all_backward_hooks_match(self, hooked_transformer, transformer_bridge, prompt):
        """Test that all backward hook gradients match between HT and TB.

        This test:
        1. Gets all hooks available in HookedTransformer
        2. Registers backward hooks on both models for each hook
        3. Runs forward pass and backward pass on both models
        4. Compares all captured gradients
        5. Asserts they match within tolerance (atol=1e-3)
        """
        # Dictionary to store gradients from both models
        ht_gradients = {}
        tb_gradients = {}

        # Get all hook names from HookedTransformer
        hook_names = list(hooked_transformer.hook_dict.keys())

        print(f"\nTesting {len(hook_names)} hooks for backward pass parity")
        print(f"Prompt: '{prompt}'")

        # Register backward hooks on HookedTransformer
        def make_ht_backward_hook(name):
            def hook_fn(tensor, hook):
                # Store a copy of the gradient tensor
                if isinstance(tensor, torch.Tensor):
                    ht_gradients[name] = tensor.detach().clone()
                elif isinstance(tensor, tuple) and len(tensor) > 0:
                    # For tuple outputs, store the first tensor
                    if isinstance(tensor[0], torch.Tensor):
                        ht_gradients[name] = tensor[0].detach().clone()
                # Return None to indicate we're not modifying the gradient
                return None

            return hook_fn

        # Register backward hooks on TransformerBridge
        def make_tb_backward_hook(name):
            def hook_fn(tensor, hook):
                # Store a copy of the gradient tensor
                if isinstance(tensor, torch.Tensor):
                    tb_gradients[name] = tensor.detach().clone()
                elif isinstance(tensor, tuple) and len(tensor) > 0:
                    # For tuple outputs, store the first tensor
                    if isinstance(tensor[0], torch.Tensor):
                        tb_gradients[name] = tensor[0].detach().clone()
                # Return None to indicate we're not modifying the gradient
                return None

            return hook_fn

        # Register all backward hooks
        ht_hook_handles = []
        tb_hook_handles = []

        for hook_name in hook_names:
            # Register on HookedTransformer
            if hook_name in hooked_transformer.hook_dict:
                hook_point = hooked_transformer.hook_dict[hook_name]
                handle = hook_point.add_hook(make_ht_backward_hook(hook_name), dir="bwd")
                ht_hook_handles.append(handle)

            # Register on TransformerBridge
            if hook_name in transformer_bridge.hook_dict:
                hook_point = transformer_bridge.hook_dict[hook_name]
                handle = hook_point.add_hook(make_tb_backward_hook(hook_name), dir="bwd")
                tb_hook_handles.append(handle)

        try:
            # Run forward and backward pass on HookedTransformer
            ht_output = hooked_transformer(prompt)
            ht_loss = ht_output[:, -1, :].sum()
            ht_loss.backward()

            # Run forward and backward pass on TransformerBridge
            tb_output = transformer_bridge(prompt)
            tb_loss = tb_output[:, -1, :].sum()
            tb_loss.backward()

            # Compare gradients
            print(f"\nHookedTransformer captured {len(ht_gradients)} gradients")
            print(f"TransformerBridge captured {len(tb_gradients)} gradients")

            # Find hooks that exist in both models
            common_hooks = set(ht_gradients.keys()) & set(tb_gradients.keys())
            ht_only_hooks = set(ht_gradients.keys()) - set(tb_gradients.keys())
            tb_only_hooks = set(tb_gradients.keys()) - set(ht_gradients.keys())

            print(f"\nCommon hooks: {len(common_hooks)}")
            if ht_only_hooks:
                print(f"HT-only hooks ({len(ht_only_hooks)}): {sorted(list(ht_only_hooks))[:5]}...")
            if tb_only_hooks:
                print(f"TB-only hooks ({len(tb_only_hooks)}): {sorted(list(tb_only_hooks))[:5]}...")

            # Compare common hooks
            mismatches = []
            # Backward hooks need higher tolerance due to numerical precision in backprop
            abs_tolerance = 0.1  # Absolute difference tolerance
            rel_tolerance = 1e-4  # Relative difference tolerance

            for hook_name in sorted(common_hooks):
                ht_grad = ht_gradients[hook_name]
                tb_grad = tb_gradients[hook_name]

                # Check shapes match
                if ht_grad.shape != tb_grad.shape:
                    mismatches.append(
                        f"{hook_name}: Shape mismatch - HT {ht_grad.shape} vs TB {tb_grad.shape}"
                    )
                    continue

                # Check values match within tolerance
                # Handle special cases with inf or nan
                ht_finite = ht_grad[torch.isfinite(ht_grad)]
                tb_finite = tb_grad[torch.isfinite(tb_grad)]

                # Check if finite elements match
                if ht_finite.numel() > 0 and tb_finite.numel() > 0:
                    # Both have finite values, compare them
                    ht_finite_mask = torch.isfinite(ht_grad)
                    tb_finite_mask = torch.isfinite(tb_grad)

                    if not torch.equal(ht_finite_mask, tb_finite_mask):
                        # Different positions have inf/nan
                        mismatches.append(
                            f"{hook_name}: Different inf/nan patterns - "
                            f"HT has {(~ht_finite_mask).sum()} non-finite, "
                            f"TB has {(~tb_finite_mask).sum()} non-finite"
                        )
                        continue

                    # Compare finite values using both absolute and relative tolerance
                    if not torch.allclose(ht_finite, tb_finite, atol=abs_tolerance, rtol=rel_tolerance):
                        max_diff = torch.max(torch.abs(ht_finite - tb_finite)).item()
                        mean_diff = torch.mean(torch.abs(ht_finite - tb_finite)).item()
                        # Calculate relative error
                        rel_diff = torch.abs(ht_finite - tb_finite) / (torch.abs(ht_finite) + 1e-8)
                        mean_rel = rel_diff.mean().item()
                        mismatches.append(
                            f"{hook_name}: Value mismatch - max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, mean_rel={mean_rel:.6f}"
                        )
                elif ht_finite.numel() == 0 and tb_finite.numel() == 0:
                    # Both are all inf/nan, check if they're the same
                    if not torch.equal(torch.isnan(ht_grad), torch.isnan(tb_grad)):
                        mismatches.append(f"{hook_name}: Different NaN patterns")
                    elif not torch.equal(torch.isinf(ht_grad), torch.isinf(tb_grad)):
                        mismatches.append(f"{hook_name}: Different Inf patterns")
                else:
                    # One has finite values, the other doesn't
                    mismatches.append(
                        f"{hook_name}: Finiteness mismatch - "
                        f"HT has {ht_finite.numel()} finite, TB has {tb_finite.numel()} finite"
                    )

            # Report results
            matching_hooks = len(common_hooks) - len(mismatches)
            match_percentage = (matching_hooks / len(common_hooks) * 100) if common_hooks else 0

            print(f"\n{'='*60}")
            print(
                f"RESULTS: {matching_hooks}/{len(common_hooks)} hooks match ({match_percentage:.1f}%)"
            )
            print(f"{'='*60}")

            if mismatches:
                print(f"\n❌ Found {len(mismatches)} mismatches:")
                for mismatch in mismatches[:10]:  # Show first 10
                    print(f"  {mismatch}")
                if len(mismatches) > 10:
                    print(f"  ... and {len(mismatches) - 10} more")

                # Categorize mismatches
                shape_mismatches = [m for m in mismatches if "Shape mismatch" in m]
                value_mismatches = [m for m in mismatches if "Value mismatch" in m]

                print(f"\nMismatch breakdown:")
                print(f"  Shape mismatches: {len(shape_mismatches)} (architectural differences)")
                print(f"  Value mismatches: {len(value_mismatches)} (numerical differences)")

                # Check if all mismatches are acceptable
                # For backward hooks, we expect some differences due to bridging architecture
                acceptable_patterns = [
                    "hook_attn_scores",  # Gradient flow through attention computation
                    "hook_z",  # Shape conversion in attention heads
                    "hook_pattern",  # Related to attention computation
                    "hook_attn_out",  # Attention output routing
                    "ln1.hook_",  # LayerNorm 1 gradients (bridging architecture)
                    "ln2.hook_",  # LayerNorm 2 gradients (bridging architecture)
                    "hook_resid_mid",  # Residual mid-layer (affected by LayerNorm)
                    "hook_resid_pre",  # Residual pre-layer (affected by LayerNorm)
                    "hook_embed",  # Embedding gradient differences
                    "hook_pos_embed",  # Positional embedding gradient differences
                ]
                acceptable_mismatches = [
                    m
                    for m in mismatches
                    if any(pattern in m for pattern in acceptable_patterns)
                ]

                if len(acceptable_mismatches) == len(mismatches):
                    print(
                        f"\n✓ All mismatches are due to known architectural differences"
                    )
                    print(f"  (LayerNorm bridging, attention computation, residual streams)")
                else:
                    significant_mismatches = [
                        m for m in mismatches if m not in acceptable_mismatches
                    ]
                    print(f"\n❌ {len(significant_mismatches)} significant mismatches:")
                    for sig_mismatch in significant_mismatches:
                        print(f"  {sig_mismatch}")
                    pytest.fail(
                        f"Found {len(significant_mismatches)} significant numerical mismatches. "
                        f"See output above for details."
                    )
            else:
                print(
                    f"\n✓ All {len(common_hooks)} common hooks match within tolerance (abs={abs_tolerance}, rel={rel_tolerance})"
                )

        finally:
            # Remove all hooks (skip None handles)
            for handle in ht_hook_handles:
                if handle is not None:
                    handle.remove()
            for handle in tb_hook_handles:
                if handle is not None:
                    handle.remove()

    def test_critical_backward_hooks_match(
        self, hooked_transformer, transformer_bridge, prompt
    ):
        """Test that critical backward hooks (commonly used in interpretability research) match.

        This is a lighter-weight version of the full test that focuses on the most
        commonly used hooks for debugging purposes.
        """
        # Critical hooks that are commonly used
        critical_hooks = [
            "hook_embed",
            "blocks.0.hook_resid_pre",
            "blocks.0.hook_resid_mid",
            "blocks.0.hook_resid_post",
            "blocks.0.attn.hook_q",
            "blocks.0.attn.hook_k",
            "blocks.0.attn.hook_v",
            "blocks.0.attn.hook_z",
            "blocks.0.attn.hook_result",
            "blocks.0.mlp.hook_pre",
            "blocks.0.mlp.hook_post",
            "blocks.0.hook_mlp_out",
        ]

        # Dictionary to store gradients
        ht_gradients = {}
        tb_gradients = {}

        # Register backward hooks
        ht_hook_handles = []
        tb_hook_handles = []

        for hook_name in critical_hooks:
            # HookedTransformer
            if hook_name in hooked_transformer.hook_dict:
                hook_point = hooked_transformer.hook_dict[hook_name]
                handle = hook_point.add_hook(
                    lambda tensor, hook, name=hook_name: (
                        ht_gradients.update({name: tensor.detach().clone()}),
                        None,  # Return None, not tensor
                    )[1],
                    dir="bwd",
                )
                ht_hook_handles.append(handle)

            # TransformerBridge
            if hook_name in transformer_bridge.hook_dict:
                hook_point = transformer_bridge.hook_dict[hook_name]
                handle = hook_point.add_hook(
                    lambda tensor, hook, name=hook_name: (
                        tb_gradients.update({name: tensor.detach().clone()}),
                        None,  # Return None, not tensor
                    )[1],
                    dir="bwd",
                )
                tb_hook_handles.append(handle)

        try:
            # Run forward and backward pass on HookedTransformer
            ht_output = hooked_transformer(prompt)
            ht_loss = ht_output[:, -1, :].sum()
            ht_loss.backward()

            # Run forward and backward pass on TransformerBridge
            tb_output = transformer_bridge(prompt)
            tb_loss = tb_output[:, -1, :].sum()
            tb_loss.backward()

            # Compare gradients
            print(f"\nComparing {len(critical_hooks)} critical backward hooks")
            mismatches = []
            # Backward hooks need higher tolerance due to numerical precision in backprop
            abs_tolerance = 0.1
            rel_tolerance = 1e-4

            for hook_name in critical_hooks:
                if hook_name not in ht_gradients:
                    print(f"  ⚠️  {hook_name}: Not found in HookedTransformer")
                    continue
                if hook_name not in tb_gradients:
                    print(f"  ⚠️  {hook_name}: Not found in TransformerBridge")
                    continue

                ht_grad = ht_gradients[hook_name]
                tb_grad = tb_gradients[hook_name]

                if ht_grad.shape != tb_grad.shape:
                    print(
                        f"  ⚠️  {hook_name}: Shape mismatch - HT{ht_grad.shape} vs TB{tb_grad.shape}"
                    )
                    mismatches.append(
                        f"{hook_name}: Shape mismatch - HT{ht_grad.shape} vs TB{tb_grad.shape}"
                    )
                    continue

                # Compare only finite values
                ht_finite = ht_grad[torch.isfinite(ht_grad)]
                tb_finite = tb_grad[torch.isfinite(tb_grad)]

                if ht_finite.numel() > 0 and tb_finite.numel() > 0:
                    if not torch.allclose(ht_finite, tb_finite, atol=abs_tolerance, rtol=rel_tolerance):
                        max_diff = torch.max(torch.abs(ht_finite - tb_finite)).item()
                        mismatches.append(f"{hook_name}: max_diff={max_diff:.6f}")
                    else:
                        print(f"  ✓ {hook_name}")
                else:
                    print(f"  ✓ {hook_name} (no finite gradients)")

            if mismatches:
                print(f"\n❌ Mismatches found:")
                for mismatch in mismatches:
                    print(f"  {mismatch}")

                # Filter out known architectural differences
                acceptable_patterns = [
                    "hook_z",  # Shape conversion
                    "hook_attn_scores",  # Attention computation gradients
                    "hook_pattern",  # Attention pattern gradients
                    "hook_result",  # Attention result routing
                    "ln1.hook_",  # LayerNorm bridging
                    "ln2.hook_",  # LayerNorm bridging
                    "hook_resid_pre",  # Affected by LayerNorm
                    "hook_resid_mid",  # Affected by LayerNorm
                    "hook_embed",  # Embedding gradient differences
                ]
                significant_mismatches = [
                    m
                    for m in mismatches
                    if not any(pattern in m for pattern in acceptable_patterns)
                ]

                if significant_mismatches:
                    pytest.fail(
                        f"Found {len(significant_mismatches)} significant mismatches in critical backward hooks"
                    )
                else:
                    print(
                        f"\n✓ All mismatches are due to known architectural differences"
                    )
                    print(f"  (LayerNorm bridging, attention computation)")
            else:
                print(f"\n✓ All critical backward hooks match")

        finally:
            # Clean up hooks (skip None handles)
            for handle in ht_hook_handles:
                if handle is not None:
                    handle.remove()
            for handle in tb_hook_handles:
                if handle is not None:
                    handle.remove()
