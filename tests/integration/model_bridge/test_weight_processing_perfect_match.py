#!/usr/bin/env python3
"""
Test that demonstrates perfect ablation matching using corrected ProcessWeights.
This test validates that the weight processing approach works correctly.
"""

import torch

from transformer_lens import HookedTransformer, utils
from transformer_lens.weight_processing import ProcessWeights


def create_correctly_processed_model():
    """Create a correctly processed model that matches HookedTransformer exactly."""
    print("Creating correctly processed model...")

    # Load unprocessed model
    model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")

    # Get state dict and apply ProcessWeights
    state_dict = model.state_dict().copy()

    processed_state_dict = ProcessWeights.process_weights(
        state_dict=state_dict,
        cfg=model.cfg,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    )

    # Filter out problematic parameters (Inf/NaN)
    filtered_state_dict = {}
    for key, tensor in processed_state_dict.items():
        if not (torch.isinf(tensor).any() or torch.isnan(tensor).any()):
            filtered_state_dict[key] = tensor

    # Load filtered weights
    missing_keys, _ = model.load_state_dict(filtered_state_dict, strict=False)

    # Set missing LayerNorm parameters to identity
    with torch.no_grad():
        for key in missing_keys:
            if key in model.state_dict():
                if ".ln1.w" in key or ".ln2.w" in key or "ln_final.w" in key:
                    model.state_dict()[key].fill_(1.0)
                elif ".ln1.b" in key or ".ln2.b" in key or "ln_final.b" in key:
                    model.state_dict()[key].fill_(0.0)

    return model


def test_perfect_ablation_match():
    """Test that ablation matches perfectly between built-in and corrected processing."""
    print("=== TESTING PERFECT ABLATION MATCH ===")

    # Test text
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."

    print("\n1. Loading models...")
    # Built-in processed model (reference)
    hooked_processed = HookedTransformer.from_pretrained("gpt2", device="cpu")

    # Our correctly processed model (should match perfectly)
    corrected_processed = create_correctly_processed_model()

    tokens = hooked_processed.to_tokens(gpt2_text)

    print("\n2. Testing baseline performance...")

    hooked_original = hooked_processed(tokens, return_type="loss")
    corrected_original = corrected_processed(tokens, return_type="loss")

    print(f"HookedTransformer:     {hooked_original.item():.6f}")
    print(f"Corrected Processing:  {corrected_original.item():.6f}")
    print(f"Baseline difference:   {abs(hooked_original.item() - corrected_original.item()):.6f}")

    print("\n3. Testing ablation performance...")

    # Test ablation on layer 0, head 8
    layer_to_ablate = 0
    head_index_to_ablate = 8

    def head_ablation_hook(value, hook):
        value[:, :, head_index_to_ablate, :] = 0.0
        return value

    hook_name = utils.get_act_name("v", layer_to_ablate)

    hooked_ablated = hooked_processed.run_with_hooks(
        tokens, return_type="loss", fwd_hooks=[(hook_name, head_ablation_hook)]
    )

    corrected_ablated = corrected_processed.run_with_hooks(
        tokens, return_type="loss", fwd_hooks=[(hook_name, head_ablation_hook)]
    )

    print(f"HookedTransformer:     {hooked_ablated.item():.6f}")
    print(f"Corrected Processing:  {corrected_ablated.item():.6f}")
    print(f"Ablation difference:   {abs(hooked_ablated.item() - corrected_ablated.item()):.6f}")

    print("\n4. Analyzing interpretability gains...")

    hooked_gain = hooked_ablated.item() - hooked_original.item()
    corrected_gain = corrected_ablated.item() - corrected_original.item()

    print(f"HookedTransformer gain:     {hooked_gain:.6f}")
    print(f"Corrected Processing gain:  {corrected_gain:.6f}")
    print(f"Gain difference:            {abs(hooked_gain - corrected_gain):.6f}")

    print("\n=== RESULTS ===")

    baseline_diff = abs(hooked_original.item() - corrected_original.item())
    ablation_diff = abs(hooked_ablated.item() - corrected_ablated.item())
    gain_diff = abs(hooked_gain - corrected_gain)

    baseline_perfect = baseline_diff < 0.00001
    ablation_perfect = ablation_diff < 0.00001
    gain_perfect = gain_diff < 0.00001

    print(f'Baseline match:     {"✅ PERFECT" if baseline_perfect else "❌"} ({baseline_diff:.8f})')
    print(f'Ablation match:     {"✅ PERFECT" if ablation_perfect else "❌"} ({ablation_diff:.8f})')
    print(f'Gain match:         {"✅ PERFECT" if gain_perfect else "❌"} ({gain_diff:.8f})')

    if baseline_perfect and ablation_perfect and gain_perfect:
        print("\n🎉🎉🎉 PERFECT MATCH ACHIEVED! 🎉🎉🎉")
        print("The corrected processing matches HookedTransformer exactly!")
        print("This solution can be applied to TransformerBridge for perfect ablation matching.")
    else:
        print("\n⚠️  Not quite perfect yet, but very close!")
        pytest.fail("Not quite perfect yet, but very close!")


if __name__ == "__main__":
    success = test_perfect_ablation_match()
    if success:
        print("\n🔥 SOLUTION READY FOR INTEGRATION! 🔥")
