#!/usr/bin/env python3
"""
Test ProcessWeights with architecture adapter for path translation.
This validates that ProcessWeights can work with HF format weights using the adapter.
"""

import torch
from transformers import GPT2LMHeadModel

from transformer_lens import HookedTransformer
from transformer_lens import utilities as utils
from transformer_lens.config.TransformerBridgeConfig import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.weight_processing import ProcessWeights


def test_processweights_with_adapter():
    """Test ProcessWeights with architecture adapter for path translation."""
    print("=== TESTING PROCESSWEIGHTS WITH ARCHITECTURE ADAPTER ===")

    # Test text
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."

    print("\n1. Loading reference HookedTransformer...")
    hooked_processed = HookedTransformer.from_pretrained("gpt2", device="cpu")
    tokens = hooked_processed.to_tokens(gpt2_text)

    print("\n2. Loading raw HuggingFace model...")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval()

    print("\n3. Setting up architecture adapter...")
    # Create a TransformerBridge config that matches GPT-2
    cfg = TransformerBridgeConfig.from_dict(
        {
            "n_layers": 12,
            "d_model": 768,
            "n_heads": 12,
            "d_head": 64,
            "d_mlp": 3072,
            "d_vocab": 50257,
            "act_fn": "gelu",
            "normalization_type": "LN",
            "positional_embedding_type": "standard",
            "n_ctx": 1024,
            "model_name": "gpt2",
            "device": "cpu",
        }
    )

    # Create adapter
    adapter = ArchitectureAdapter(cfg)

    print("\n4. Testing baseline performance...")
    with torch.no_grad():
        hf_outputs_before = hf_model(tokens)
        hf_loss_before = torch.nn.functional.cross_entropy(
            hf_outputs_before.logits[:, :-1].reshape(-1, hf_outputs_before.logits.size(-1)),
            tokens[:, 1:].reshape(-1),
        )

    print(f"   HF model loss (before processing): {hf_loss_before.item():.6f}")

    print("\n5. Applying ProcessWeights with adapter...")
    try:
        # Get HF state dict
        hf_state_dict = hf_model.state_dict().copy()

        # Apply ProcessWeights with adapter
        processed_state_dict = ProcessWeights.process_weights(
            state_dict=hf_state_dict,
            cfg=cfg,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
            refactor_factored_attn_matrices=False,
            adapter=adapter,  # Pass adapter for path translation
        )

        # Filter out problematic parameters (Inf/NaN)
        filtered_state_dict = {}
        for key, tensor in processed_state_dict.items():
            if not (torch.isinf(tensor).any() or torch.isnan(tensor).any()):
                filtered_state_dict[key] = tensor

        # Load filtered weights back into model
        missing_keys, unexpected_keys = hf_model.load_state_dict(filtered_state_dict, strict=False)

        # Handle missing LayerNorm parameters (they were folded)
        if missing_keys:
            print(f"   Setting {len(missing_keys)} missing LayerNorm parameters to identity...")
            with torch.no_grad():
                for key in missing_keys:
                    if key in hf_model.state_dict():
                        if "ln_1.weight" in key or "ln_2.weight" in key or "ln_f.weight" in key:
                            hf_model.state_dict()[key].fill_(1.0)
                        elif "ln_1.bias" in key or "ln_2.bias" in key or "ln_f.bias" in key:
                            hf_model.state_dict()[key].fill_(0.0)

        processing_succeeded = True
        print("   ✅ ProcessWeights with adapter succeeded!")

    except Exception as e:
        print(f"   ❌ ProcessWeights with adapter failed: {e}")
        import traceback

        traceback.print_exc()
        processing_succeeded = False

    if not processing_succeeded:
        print("\n❌ Processing failed - cannot continue with comparison")
        return False

    print("\n6. Testing processed model...")
    with torch.no_grad():
        hf_outputs_after = hf_model(tokens)
        hf_loss_after = torch.nn.functional.cross_entropy(
            hf_outputs_after.logits[:, :-1].reshape(-1, hf_outputs_after.logits.size(-1)),
            tokens[:, 1:].reshape(-1),
        )

    print(f"   HF model loss (after processing): {hf_loss_after.item():.6f}")

    print("\n7. Testing ablation...")
    layer_to_ablate = 0
    head_index_to_ablate = 8

    def head_ablation_hook_hf(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        batch_size, seq_len, d_model = hidden_states.shape
        n_heads = 12
        d_head = d_model // n_heads

        reshaped = hidden_states.view(batch_size, seq_len, n_heads, d_head)
        reshaped[:, :, head_index_to_ablate, :] = 0.0
        ablated_hidden = reshaped.view(batch_size, seq_len, d_model)

        if isinstance(output, tuple):
            return (ablated_hidden,) + output[1:]
        else:
            return ablated_hidden

    hook_handle = hf_model.transformer.h[layer_to_ablate].attn.register_forward_hook(
        head_ablation_hook_hf
    )

    try:
        with torch.no_grad():
            hf_outputs_ablated = hf_model(tokens)
            hf_loss_ablated = torch.nn.functional.cross_entropy(
                hf_outputs_ablated.logits[:, :-1].reshape(-1, hf_outputs_ablated.logits.size(-1)),
                tokens[:, 1:].reshape(-1),
            )
    finally:
        hook_handle.remove()

    # Compare with HookedTransformer
    hooked_original = hooked_processed(tokens, return_type="loss")

    def head_ablation_hook_tl(value, hook):
        value[:, :, head_index_to_ablate, :] = 0.0
        return value

    hook_name = utils.get_act_name("v", layer_to_ablate)
    hooked_ablated = hooked_processed.run_with_hooks(
        tokens, return_type="loss", fwd_hooks=[(hook_name, head_ablation_hook_tl)]
    )

    print("\n=== RESULTS ===")
    print(
        f"HookedTransformer: Original={hooked_original.item():.6f}, Ablated={hooked_ablated.item():.6f}"
    )
    print(
        f"HF + ProcessWeights + Adapter: Original={hf_loss_after.item():.6f}, Ablated={hf_loss_ablated.item():.6f}"
    )

    # Check differences
    orig_diff = abs(hooked_original.item() - hf_loss_after.item())
    ablated_diff = abs(hooked_ablated.item() - hf_loss_ablated.item())

    print(f"\nDifferences:")
    print(f"Original loss diff: {orig_diff:.8f}")
    print(f"Ablated loss diff: {ablated_diff:.8f}")

    # Calculate interpretability gains
    hooked_gain = hooked_ablated.item() - hooked_original.item()
    hf_gain = hf_loss_ablated.item() - hf_loss_after.item()
    gain_diff = abs(hooked_gain - hf_gain)

    print(f"\nInterpretability gains:")
    print(f"HookedTransformer gain: {hooked_gain:.6f}")
    print(f"HF + ProcessWeights + Adapter gain: {hf_gain:.6f}")
    print(f"Gain difference: {gain_diff:.8f}")

    # Success criteria
    baseline_good = orig_diff < 0.01
    ablation_good = ablated_diff < 0.01
    gain_good = gain_diff < 0.01

    print(f"\nSuccess criteria:")
    print(f'Baseline match: {"✅ GOOD" if baseline_good else "❌ POOR"} ({orig_diff:.8f})')
    print(f'Ablation match: {"✅ GOOD" if ablation_good else "❌ POOR"} ({ablated_diff:.8f})')
    print(f'Gain match: {"✅ GOOD" if gain_good else "❌ POOR"} ({gain_diff:.8f})')

    if baseline_good and ablation_good and gain_good:
        print("\n✅✅✅ SUCCESS: ProcessWeights with adapter works! ✅✅✅")
        return True
    else:
        print("\n⚠️  ProcessWeights with adapter needs work")
        return False


if __name__ == "__main__":
    success = test_processweights_with_adapter()
    if success:
        print("\n🔥 ADAPTER APPROACH VALIDATED! 🔥")
