"""Integration test for TransformerBridge optimizer compatibility.

Tests that TransformerBridge works correctly with PyTorch optimizers,
including parameter access, gradient flow, and parameter updates.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch

from transformer_lens.model_bridge.bridge import TransformerBridge


class StageThresholds(NamedTuple):
    """Thresholds for a specific stage of validation."""

    logits_max: float = 0.0
    logits_mean: float = 0.0
    loss_relative: float = 0.0
    params_max: float = 0.0  # Only used for parameter update stages
    params_mean: float = 0.0  # Only used for parameter update stages


@dataclass
class StepThresholds:
    """Thresholds for all stages at a specific optimization step."""

    step: int
    initial_fwd: StageThresholds
    post_update_fwd: StageThresholds
    param_update: StageThresholds  # Tracks parameter divergence after update


def test_optimizer_workflow():
    """Test complete optimizer workflow with TransformerBridge."""
    # Load model
    bridge = TransformerBridge.boot_transformers("distilgpt2")

    # Verify parameters() returns leaf tensors
    params = list(bridge.parameters())
    assert len(params) > 0, "Should have parameters"
    assert all(p.is_leaf for p in params), "All parameters should be leaf tensors"

    # Verify optimizer creation succeeds
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-4)
    assert optimizer is not None, "Optimizer should be created successfully"

    # Verify tl_parameters() returns TL-style dict
    tl_params = bridge.tl_parameters()
    assert len(tl_params) > 0, "Should have TL-style parameters"
    assert any(
        "blocks." in name and ".attn." in name for name in tl_params.keys()
    ), "Should have TL-style parameter names like 'blocks.0.attn.W_Q'"

    # Verify tl_named_parameters() iterator matches dict
    tl_named_params = list(bridge.tl_named_parameters())
    assert len(tl_named_params) == len(
        tl_params
    ), "Iterator should yield same number of parameters as dict"
    iterator_dict = dict(tl_named_params)
    for name, tensor in tl_params.items():
        assert name in iterator_dict, f"Name {name} should be in iterator output"
        assert torch.equal(iterator_dict[name], tensor), f"Tensor for {name} should match"

    # Verify named_parameters() returns HF-style names
    hf_names = [name for name, _ in bridge.named_parameters()]
    assert len(hf_names) > 0, "Should have HF-style parameters"
    assert any(
        "_original_component" in name for name in hf_names
    ), "Should have HuggingFace-style parameter names"

    # Verify forward pass and backward work
    device = next(bridge.parameters()).device
    input_ids = torch.randint(0, bridge.cfg.d_vocab, (1, 10), device=device)
    logits = bridge(input_ids)
    expected_shape = (1, 10, bridge.cfg.d_vocab)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"

    loss = logits[0, -1].sum()
    loss.backward()

    # Verify gradients were computed
    params_with_grad = [p for p in bridge.parameters() if p.grad is not None]
    assert len(params_with_grad) > 0, "Should have parameters with gradients after backward()"

    # Verify optimizer step updates parameters
    param_before = list(bridge.parameters())[0].clone()
    optimizer.step()
    param_after = list(bridge.parameters())[0]
    assert not torch.allclose(
        param_before, param_after
    ), "Parameters should be updated after optimizer.step()"


def test_optimizer_compatibility_after_compatibility_mode():
    """Test that optimizer still works after enabling compatibility mode."""
    bridge = TransformerBridge.boot_transformers("distilgpt2")
    bridge.enable_compatibility_mode(no_processing=True)

    # Verify parameters are still leaf tensors after compatibility mode
    params = list(bridge.parameters())
    assert all(
        p.is_leaf for p in params
    ), "All parameters should still be leaf tensors after compatibility mode"

    # Verify optimizer works after compatibility mode
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-4)
    device = next(bridge.parameters()).device
    input_ids = torch.randint(0, bridge.cfg.d_vocab, (1, 10), device=device)

    logits = bridge(input_ids)
    loss = logits[0, -1].sum()
    loss.backward()
    optimizer.step()

    # If we got here without errors, the test passed
    assert True, "Optimizer should work after compatibility mode"


def test_bridge_hooked_parity_multi_step_optimization():
    """Test parity between Bridge and HookedTransformer across multiple optimization steps.

    This test validates that both architectures maintain comparable results over
    multiple optimization steps (1, 10), checking:
    - Initial forward pass: logits and loss alignment before any updates
    - Post-update forward pass: logits and loss remain close after each step
    - Parameter updates: unembed weights remain close after each step

    We focus on the unembed layer as it's a directly comparable component between
    both architectures with matching shapes.
    """
    from transformer_lens import HookedTransformer

    # Define thresholds for each step (rounded to next magnitude above observed + 30%)
    step_thresholds = [
        StepThresholds(
            step=1,
            initial_fwd=StageThresholds(logits_max=1e-3, logits_mean=1e-4, loss_relative=1e-6),
            post_update_fwd=StageThresholds(logits_max=2.0, logits_mean=1e-3, loss_relative=1e-5),
            param_update=StageThresholds(params_max=1e-2, params_mean=1e-6),
        ),
        StepThresholds(
            step=10,
            initial_fwd=StageThresholds(logits_max=20.0, logits_mean=0.1, loss_relative=1e-3),
            post_update_fwd=StageThresholds(logits_max=20.0, logits_mean=0.1, loss_relative=1e-4),
            param_update=StageThresholds(params_max=1e-1, params_mean=1e-5),
        ),
    ]

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Load both models with no weight processing for fair comparison
    hooked = HookedTransformer.from_pretrained(
        "distilgpt2",
        device="cpu",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )

    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    bridge.enable_compatibility_mode(no_processing=True)

    # Create optimizers with same settings
    hooked_optimizer = torch.optim.AdamW(hooked.parameters(), lr=1e-3)
    bridge_optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3)

    # Create identical input with fixed seed
    torch.manual_seed(42)
    input_ids = torch.randint(0, bridge.cfg.d_vocab, (1, 10), device="cpu")

    # Access unembed parameters for comparison
    hooked_unembed_param = hooked.unembed.W_U
    bridge_unembed_param = bridge.unembed._original_component.weight

    # Verify shapes are compatible after transpose
    assert hooked_unembed_param.T.shape == bridge_unembed_param.shape, (
        f"Unembed parameter shapes should match after transpose: "
        f"{hooked_unembed_param.T.shape} vs {bridge_unembed_param.shape}"
    )

    # Store initial parameters (should match since loaded from same checkpoint)
    initial_hooked_unembed = hooked_unembed_param.data.clone()
    initial_bridge_unembed = bridge_unembed_param.data.clone()
    param_diff = (initial_hooked_unembed.T - initial_bridge_unembed).abs().max().item()
    assert param_diff < 1e-4, (
        f"Initial unembed parameters should match (loaded from same checkpoint). "
        f"Max diff: {param_diff:.6e}"
    )

    # Track current step for threshold selection
    current_step = 0

    # Run optimization loop
    for step_config in step_thresholds:
        target_step = step_config.step

        # Run optimization steps until we reach the target step
        while current_step < target_step:
            current_step += 1

            # ===== INITIAL FORWARD PASS (before this step) =====
            hooked_logits = hooked(input_ids, return_type="logits")
            bridge_logits = bridge(input_ids, return_type="logits")

            # Only validate initial forward on the target steps
            if current_step == target_step:
                logits_diff = (hooked_logits - bridge_logits).abs()
                logits_max_diff = logits_diff.max().item()
                logits_mean_diff = logits_diff.mean().item()

                # Compare losses
                hooked_loss = hooked_logits[0, -1].sum()
                bridge_loss = bridge_logits[0, -1].sum()
                loss_diff = abs(hooked_loss.item() - bridge_loss.item())
                loss_relative_diff = loss_diff / (abs(hooked_loss.item()) + 1e-8)

                assert logits_max_diff < step_config.initial_fwd.logits_max, (
                    f"Step {current_step}: Initial logits max diff {logits_max_diff:.6f} "
                    f"exceeds threshold {step_config.initial_fwd.logits_max:.6f}"
                )
                assert logits_mean_diff < step_config.initial_fwd.logits_mean, (
                    f"Step {current_step}: Initial logits mean diff {logits_mean_diff:.6f} "
                    f"exceeds threshold {step_config.initial_fwd.logits_mean:.6f}"
                )

                assert loss_relative_diff < step_config.initial_fwd.loss_relative, (
                    f"Step {current_step}: Initial loss relative diff {loss_relative_diff:.6f} "
                    f"exceeds threshold {step_config.initial_fwd.loss_relative:.6f}"
                )

            # Compute loss for backward
            hooked_loss = hooked_logits[0, -1].sum()
            bridge_loss = bridge_logits[0, -1].sum()

            # ===== BACKWARD PASS =====
            hooked_loss.backward()
            bridge_loss.backward()

            # Verify gradients exist and are reasonable (only on target steps)
            if current_step == target_step:
                assert (
                    hooked_unembed_param.grad is not None
                ), "HookedTransformer unembed should have gradients"
                assert bridge_unembed_param.grad is not None, "Bridge unembed should have gradients"

                hooked_grad_mag = hooked_unembed_param.grad.abs().mean().item()
                bridge_grad_mag = bridge_unembed_param.grad.abs().mean().item()

                assert hooked_grad_mag > 1e-6 and hooked_grad_mag < 1e6, (
                    f"Step {current_step}: HookedTransformer gradients should be reasonable: "
                    f"{hooked_grad_mag:.6e}"
                )
                assert bridge_grad_mag > 1e-6 and bridge_grad_mag < 1e6, (
                    f"Step {current_step}: Bridge gradients should be reasonable: "
                    f"{bridge_grad_mag:.6e}"
                )

            # Store parameters before update (for validation on target steps)
            if current_step == target_step:
                hooked_unembed_before = hooked_unembed_param.data.clone()
                bridge_unembed_before = bridge_unembed_param.data.clone()

            # ===== OPTIMIZER STEP =====
            hooked_optimizer.step()
            bridge_optimizer.step()

            # ===== VALIDATE PARAMETER UPDATES (on target steps) =====
            if current_step == target_step:
                hooked_unembed_after = hooked_unembed_param.data
                bridge_unembed_after = bridge_unembed_param.data

                # Verify parameters were updated
                hooked_delta = hooked_unembed_after - hooked_unembed_before
                bridge_delta = bridge_unembed_after - bridge_unembed_before
                assert (
                    hooked_delta.abs().max() > 1e-8
                ), f"Step {current_step}: HookedTransformer unembed should be updated"
                assert (
                    bridge_delta.abs().max() > 1e-8
                ), f"Step {current_step}: Bridge unembed should be updated"

                # Verify parameters remain close
                param_diff = (hooked_unembed_after.T - bridge_unembed_after).abs()
                param_max_diff = param_diff.max().item()
                param_mean_diff = param_diff.mean().item()

                assert param_max_diff < step_config.param_update.params_max, (
                    f"Step {current_step}: Parameter max diff {param_max_diff:.6e} "
                    f"exceeds threshold {step_config.param_update.params_max:.6e}"
                )
                assert param_mean_diff < step_config.param_update.params_mean, (
                    f"Step {current_step}: Parameter mean diff {param_mean_diff:.6e} "
                    f"exceeds threshold {step_config.param_update.params_mean:.6e}"
                )

            # Zero gradients for next iteration
            hooked_optimizer.zero_grad()
            bridge_optimizer.zero_grad()

            # ===== POST-UPDATE FORWARD PASS (on target steps) =====
            if current_step == target_step:
                with torch.no_grad():
                    hooked_logits_after = hooked(input_ids, return_type="logits")
                    bridge_logits_after = bridge(input_ids, return_type="logits")

                logits_diff_after = (hooked_logits_after - bridge_logits_after).abs()
                logits_max_diff_after = logits_diff_after.max().item()
                logits_mean_diff_after = logits_diff_after.mean().item()

                assert logits_max_diff_after < step_config.post_update_fwd.logits_max, (
                    f"Step {current_step}: Post-update logits max diff {logits_max_diff_after:.6f} "
                    f"exceeds threshold {step_config.post_update_fwd.logits_max:.6f}"
                )
                assert logits_mean_diff_after < step_config.post_update_fwd.logits_mean, (
                    f"Step {current_step}: Post-update logits mean diff {logits_mean_diff_after:.6f} "
                    f"exceeds threshold {step_config.post_update_fwd.logits_mean:.6f}"
                )

                # Compare losses after update
                hooked_loss_after = hooked_logits_after[0, -1].sum()
                bridge_loss_after = bridge_logits_after[0, -1].sum()
                loss_diff_after = abs(hooked_loss_after.item() - bridge_loss_after.item())
                loss_relative_diff_after = loss_diff_after / (abs(hooked_loss_after.item()) + 1e-8)

                assert loss_relative_diff_after < step_config.post_update_fwd.loss_relative, (
                    f"Step {current_step}: Post-update loss relative diff "
                    f"{loss_relative_diff_after:.6f} exceeds threshold "
                    f"{step_config.post_update_fwd.loss_relative:.6f}"
                )
