"""Integration test for TransformerBridge optimizer compatibility.

Tests that TransformerBridge works correctly with PyTorch optimizers,
including parameter access, gradient flow, and parameter updates.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from transformers import AutoModelForCausalLM

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


def test_bridge_hooked_parity_multi_step_optimization():
    """Test parity between Bridge and a raw HF model across multiple optimization steps.

    This test validates that the bridge maintains training parity with the plain
    HuggingFace model it wraps over multiple optimization steps (1, 10), checking:
    - Initial forward pass: logits and loss alignment before any updates
    - Post-update forward pass: logits and loss remain close after each step
    - Parameter updates: unembed weights remain close after each step

    We focus on the unembed layer as it's a directly comparable component between
    both models with matching shapes. Runs in fp64: in fp32, AdamW amplifies ulp-level
    gradient differences into a hardware-dependent step-10 gap (0.10 vs 0.21 across CI CPUs).
    """
    from transformers import AutoModelForCausalLM

    # ~1000x above the worst fp64 gap observed across 1-8 BLAS threads
    step_thresholds = [
        StepThresholds(
            step=1,
            initial_fwd=StageThresholds(logits_max=1e-10, logits_mean=1e-11, loss_relative=1e-12),
            post_update_fwd=StageThresholds(
                logits_max=1e-10, logits_mean=1e-11, loss_relative=1e-12
            ),
            param_update=StageThresholds(params_max=1e-13, params_mean=1e-16),
        ),
        StepThresholds(
            step=10,
            initial_fwd=StageThresholds(logits_max=1e-5, logits_mean=1e-6, loss_relative=1e-11),
            post_update_fwd=StageThresholds(logits_max=1e-5, logits_mean=1e-6, loss_relative=1e-10),
            param_update=StageThresholds(params_max=1e-10, params_mean=1e-13),
        ),
    ]

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Load the reference raw HF model exactly as the bridge does (eager attn)
    # eval() matches the bridge (distilgpt2 has dropout; gradients still flow in eval)
    hooked = AutoModelForCausalLM.from_pretrained(
        "distilgpt2", torch_dtype=torch.float64, attn_implementation="eager"
    )
    hooked.eval()

    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu", dtype=torch.float64)
    bridge.enable_compatibility_mode(no_processing=True)

    assert hooked.lm_head.bias is None
    assert not bridge.b_U.requires_grad

    # Create optimizers with same settings
    hooked_optimizer = torch.optim.AdamW(hooked.parameters(), lr=1e-3)
    bridge_optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3)

    # Create identical input with fixed seed
    torch.manual_seed(42)
    input_ids = torch.randint(0, bridge.cfg.d_vocab, (1, 10), device="cpu")

    # Access unembed parameters for comparison (same [d_vocab, d_model] layout)
    hooked_unembed_param = hooked.lm_head.weight
    bridge_unembed_param = bridge.unembed._original_component.weight

    assert hooked_unembed_param.shape == bridge_unembed_param.shape, (
        f"Unembed parameter shapes should match: "
        f"{hooked_unembed_param.shape} vs {bridge_unembed_param.shape}"
    )

    # Store initial parameters (should match since loaded from same checkpoint)
    param_diff = (hooked_unembed_param.data - bridge_unembed_param.data).abs().max().item()
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
            hooked_logits = hooked(input_ids).logits
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
                    f"Step {current_step}: Initial logits max diff {logits_max_diff:.3e} "
                    f"exceeds threshold {step_config.initial_fwd.logits_max:.3e}"
                )
                assert logits_mean_diff < step_config.initial_fwd.logits_mean, (
                    f"Step {current_step}: Initial logits mean diff {logits_mean_diff:.3e} "
                    f"exceeds threshold {step_config.initial_fwd.logits_mean:.3e}"
                )

                assert loss_relative_diff < step_config.initial_fwd.loss_relative, (
                    f"Step {current_step}: Initial loss relative diff {loss_relative_diff:.3e} "
                    f"exceeds threshold {step_config.initial_fwd.loss_relative:.3e}"
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
                ), "HF reference unembed should have gradients"
                assert bridge_unembed_param.grad is not None, "Bridge unembed should have gradients"

                hooked_grad_mag = hooked_unembed_param.grad.abs().mean().item()
                bridge_grad_mag = bridge_unembed_param.grad.abs().mean().item()

                assert hooked_grad_mag > 1e-6 and hooked_grad_mag < 1e6, (
                    f"Step {current_step}: HF reference gradients should be reasonable: "
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
                ), f"Step {current_step}: HF reference unembed should be updated"
                assert (
                    bridge_delta.abs().max() > 1e-8
                ), f"Step {current_step}: Bridge unembed should be updated"

                # Verify parameters remain close
                param_diff = (hooked_unembed_after - bridge_unembed_after).abs()
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
                    hooked_logits_after = hooked(input_ids).logits
                    bridge_logits_after = bridge(input_ids, return_type="logits")

                logits_diff_after = (hooked_logits_after - bridge_logits_after).abs()
                logits_max_diff_after = logits_diff_after.max().item()
                logits_mean_diff_after = logits_diff_after.mean().item()

                assert logits_max_diff_after < step_config.post_update_fwd.logits_max, (
                    f"Step {current_step}: Post-update logits max diff {logits_max_diff_after:.3e} "
                    f"exceeds threshold {step_config.post_update_fwd.logits_max:.3e}"
                )
                assert logits_mean_diff_after < step_config.post_update_fwd.logits_mean, (
                    f"Step {current_step}: Post-update logits mean diff {logits_mean_diff_after:.3e} "
                    f"exceeds threshold {step_config.post_update_fwd.logits_mean:.3e}"
                )

                # Compare losses after update
                hooked_loss_after = hooked_logits_after[0, -1].sum()
                bridge_loss_after = bridge_logits_after[0, -1].sum()
                loss_diff_after = abs(hooked_loss_after.item() - bridge_loss_after.item())
                loss_relative_diff_after = loss_diff_after / (abs(hooked_loss_after.item()) + 1e-8)

                assert loss_relative_diff_after < step_config.post_update_fwd.loss_relative, (
                    f"Step {current_step}: Post-update loss relative diff "
                    f"{loss_relative_diff_after:.3e} exceeds threshold "
                    f"{step_config.post_update_fwd.loss_relative:.3e}"
                )


def test_bridge_hf_float32_training_step():
    """Compare FP32 logits and output-weight gradients across an SGD step."""
    reference = AutoModelForCausalLM.from_pretrained(
        "distilgpt2", torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    bridge.enable_compatibility_mode(no_processing=True)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)
    bridge_optimizer = torch.optim.SGD(bridge.parameters(), lr=1e-3)
    input_ids = torch.tensor([[10, 20, 30, 40, 50]])
    reference_logits = reference(input_ids).logits
    bridge_logits = bridge(input_ids)
    torch.testing.assert_close(bridge_logits, reference_logits, rtol=1e-5, atol=1e-4)
    targets = input_ids[:, 1:].reshape(-1)
    for logits in (reference_logits, bridge_logits):
        torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, bridge.cfg.d_vocab), targets
        ).backward()
    torch.testing.assert_close(
        bridge.unembed.original_component.weight.grad,
        reference.lm_head.weight.grad,
        rtol=1e-4,
        atol=1e-6,
    )
    reference_optimizer.step()
    bridge_optimizer.step()
    with torch.no_grad():
        torch.testing.assert_close(
            bridge(input_ids), reference(input_ids).logits, rtol=1e-5, atol=1e-4
        )
    assert not bridge.b_U.requires_grad


def test_bridge_hf_cross_entropy_multi_step_optimization():
    """Compare next-token training in float64 without FP32 AdamW amplification."""
    torch.manual_seed(42)
    reference = (
        AutoModelForCausalLM.from_pretrained(
            "distilgpt2", torch_dtype=torch.float32, attn_implementation="eager"
        )
        .double()
        .eval()
    )
    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu").double()
    bridge.enable_compatibility_mode(no_processing=True)
    assert reference.lm_head.bias is None
    assert not bridge.b_U.requires_grad
    assert bridge.unembed.original_component.weight is bridge.embed.original_component.weight

    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    bridge_optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3)
    torch.manual_seed(42)
    input_ids = torch.randint(0, bridge.cfg.d_vocab, (1, 10))
    reference_weight = reference.lm_head.weight
    bridge_weight = bridge.unembed.original_component.weight
    torch.testing.assert_close(bridge_weight, reference_weight, rtol=0, atol=0)

    for _ in range(10):
        reference_logits = reference(input_ids).logits
        bridge_logits = bridge(input_ids)
        torch.testing.assert_close(bridge_logits, reference_logits, rtol=1e-8, atol=1e-8)
        targets = input_ids[:, 1:].reshape(-1)
        reference_loss = torch.nn.functional.cross_entropy(
            reference_logits[:, :-1].reshape(-1, bridge.cfg.d_vocab), targets
        )
        bridge_loss = torch.nn.functional.cross_entropy(
            bridge_logits[:, :-1].reshape(-1, bridge.cfg.d_vocab), targets
        )
        torch.testing.assert_close(bridge_loss, reference_loss, rtol=1e-9, atol=1e-9)
        reference_loss.backward()
        bridge_loss.backward()
        torch.testing.assert_close(bridge_weight.grad, reference_weight.grad, rtol=1e-7, atol=1e-9)
        reference_optimizer.step()
        bridge_optimizer.step()
        torch.testing.assert_close(bridge_weight, reference_weight, rtol=1e-8, atol=1e-8)
        reference_optimizer.zero_grad()
        bridge_optimizer.zero_grad()

    with torch.no_grad():
        torch.testing.assert_close(
            bridge(input_ids), reference(input_ids).logits, rtol=1e-8, atol=1e-8
        )
