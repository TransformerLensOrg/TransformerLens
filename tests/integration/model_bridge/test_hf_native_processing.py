#!/usr/bin/env python3
"""
Test the HF-native processing approach that's implemented in bridge.py.
This validates the mathematical correctness of the HF-native weight processing methods.
"""

import torch
from transformers import GPT2LMHeadModel

from transformer_lens import HookedTransformer, utils


def apply_hf_native_processing_final(state_dict, cfg):
    """Apply HF-native processing using the exact methods from bridge.py."""

    def _fold_layer_norm_hf_native(state_dict):
        """Fold LayerNorm into subsequent layers using HF tensor formats."""
        # Fold LayerNorm into attention and MLP layers
        for layer_idx in range(cfg.n_layers):
            # === FOLD LN1 INTO ATTENTION ===
            ln1_weight = state_dict[f"transformer.h.{layer_idx}.ln_1.weight"]  # [d_model]
            ln1_bias = state_dict[f"transformer.h.{layer_idx}.ln_1.bias"]  # [d_model]

            # GPT-2 combines Q,K,V into c_attn: [d_model, 3*d_model]
            c_attn_weight = state_dict[
                f"transformer.h.{layer_idx}.attn.c_attn.weight"
            ]  # [d_model, 3*d_model]
            c_attn_bias = state_dict[f"transformer.h.{layer_idx}.attn.c_attn.bias"]  # [3*d_model]

            # Split combined QKV for processing
            d_model = cfg.d_model
            q_weight = c_attn_weight[:, :d_model]  # [d_model, d_model]
            k_weight = c_attn_weight[:, d_model : 2 * d_model]  # [d_model, d_model]
            v_weight = c_attn_weight[:, 2 * d_model :]  # [d_model, d_model]

            q_bias = c_attn_bias[:d_model]  # [d_model]
            k_bias = c_attn_bias[d_model : 2 * d_model]  # [d_model]
            v_bias = c_attn_bias[2 * d_model :]  # [d_model]

            # Apply LayerNorm folding mathematics for HF format [d_model, d_model]:
            # Fold biases: b_new = b_old + sum(W * ln_bias, dim=input_dim)
            q_bias = q_bias + torch.sum(q_weight * ln1_bias[:, None], dim=0)
            k_bias = k_bias + torch.sum(k_weight * ln1_bias[:, None], dim=0)
            v_bias = v_bias + torch.sum(v_weight * ln1_bias[:, None], dim=0)

            # Fold weights: W_new = W * ln_weight (broadcast over input dimension)
            q_weight = q_weight * ln1_weight[:, None]  # [d_model, d_model] * [d_model, 1]
            k_weight = k_weight * ln1_weight[:, None]
            v_weight = v_weight * ln1_weight[:, None]

            # Center weights (remove mean along input dimension)
            q_weight = q_weight - torch.mean(q_weight, dim=0, keepdim=True)
            k_weight = k_weight - torch.mean(k_weight, dim=0, keepdim=True)
            v_weight = v_weight - torch.mean(v_weight, dim=0, keepdim=True)

            # Recombine Q,K,V back into c_attn format
            state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"] = torch.cat(
                [q_weight, k_weight, v_weight], dim=1
            )
            state_dict[f"transformer.h.{layer_idx}.attn.c_attn.bias"] = torch.cat(
                [q_bias, k_bias, v_bias], dim=0
            )

            # Remove LayerNorm parameters (they're now folded in)
            del state_dict[f"transformer.h.{layer_idx}.ln_1.weight"]
            del state_dict[f"transformer.h.{layer_idx}.ln_1.bias"]

            # === FOLD LN2 INTO MLP ===
            ln2_weight = state_dict[f"transformer.h.{layer_idx}.ln_2.weight"]  # [d_model]
            ln2_bias = state_dict[f"transformer.h.{layer_idx}.ln_2.bias"]  # [d_model]

            # MLP input (c_fc): [d_model, 4*d_model]
            c_fc_weight = state_dict[
                f"transformer.h.{layer_idx}.mlp.c_fc.weight"
            ]  # [d_model, 4*d_model]
            c_fc_bias = state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.bias"]  # [4*d_model]

            # Apply LayerNorm folding to MLP input
            c_fc_bias = c_fc_bias + torch.sum(c_fc_weight * ln2_bias[:, None], dim=0)
            c_fc_weight = c_fc_weight * ln2_weight[:, None]
            c_fc_weight = c_fc_weight - torch.mean(c_fc_weight, dim=0, keepdim=True)

            state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.weight"] = c_fc_weight
            state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.bias"] = c_fc_bias

            # Remove LayerNorm parameters
            del state_dict[f"transformer.h.{layer_idx}.ln_2.weight"]
            del state_dict[f"transformer.h.{layer_idx}.ln_2.bias"]

        # === FOLD LN_FINAL INTO UNEMBED ===
        ln_final_weight = state_dict["transformer.ln_f.weight"]  # [d_model]
        ln_final_bias = state_dict["transformer.ln_f.bias"]  # [d_model]

        # Unembedding: [d_vocab, d_model] (HF format)
        lm_head_weight = state_dict["lm_head.weight"]  # [d_vocab, d_model]
        # Note: GPT-2 doesn't have lm_head.bias

        # Apply LayerNorm folding to unembedding
        if "lm_head.bias" in state_dict:
            lm_head_bias = state_dict["lm_head.bias"]  # [d_vocab]
            lm_head_bias = lm_head_bias + torch.sum(lm_head_weight * ln_final_bias[None, :], dim=1)
            state_dict["lm_head.bias"] = lm_head_bias

        lm_head_weight = (
            lm_head_weight * ln_final_weight[None, :]
        )  # [d_vocab, d_model] * [1, d_model]
        state_dict["lm_head.weight"] = lm_head_weight

        # Remove final LayerNorm parameters
        del state_dict["transformer.ln_f.weight"]
        del state_dict["transformer.ln_f.bias"]

    def _center_writing_weights_hf_native(state_dict):
        """Center weights that write to the residual stream using HF tensor formats."""
        # Embedding weights: [vocab_size, d_model]
        wte_weight = state_dict["transformer.wte.weight"]  # [vocab_size, d_model]
        wte_weight = wte_weight - torch.mean(
            wte_weight, dim=1, keepdim=True
        )  # Center over output dim
        state_dict["transformer.wte.weight"] = wte_weight

        # Position embedding weights: [max_pos, d_model]
        if "transformer.wpe.weight" in state_dict:
            wpe_weight = state_dict["transformer.wpe.weight"]  # [max_pos, d_model]
            wpe_weight = wpe_weight - torch.mean(wpe_weight, dim=1, keepdim=True)
            state_dict["transformer.wpe.weight"] = wpe_weight

        # Attention output and MLP output weights (write to residual stream)
        for layer_idx in range(cfg.n_layers):
            # Attention output: [d_model, d_model]
            c_proj_weight = state_dict[f"transformer.h.{layer_idx}.attn.c_proj.weight"]
            c_proj_weight = c_proj_weight - torch.mean(
                c_proj_weight, dim=1, keepdim=True
            )  # Center over output dim
            state_dict[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = c_proj_weight

            # MLP output: [4*d_model, d_model]
            mlp_c_proj_weight = state_dict[f"transformer.h.{layer_idx}.mlp.c_proj.weight"]
            mlp_c_proj_weight = mlp_c_proj_weight - torch.mean(
                mlp_c_proj_weight, dim=1, keepdim=True
            )
            state_dict[f"transformer.h.{layer_idx}.mlp.c_proj.weight"] = mlp_c_proj_weight

    def _center_unembed_hf_native(state_dict):
        """Center unembedding weights using HF tensor formats."""
        # Unembedding weights: [d_vocab, d_model] (HF format)
        lm_head_weight = state_dict["lm_head.weight"]  # [d_vocab, d_model]
        lm_head_weight = lm_head_weight - torch.mean(
            lm_head_weight, dim=1, keepdim=True
        )  # Center over output dim
        state_dict["lm_head.weight"] = lm_head_weight

    def _add_identity_layer_norm_params(state_dict):
        """Add missing LayerNorm parameters as identity (weight=1, bias=0)."""
        # Add identity LayerNorm parameters for each layer
        for layer_idx in range(cfg.n_layers):
            ln1_weight_key = f"transformer.h.{layer_idx}.ln_1.weight"
            ln1_bias_key = f"transformer.h.{layer_idx}.ln_1.bias"
            ln2_weight_key = f"transformer.h.{layer_idx}.ln_2.weight"
            ln2_bias_key = f"transformer.h.{layer_idx}.ln_2.bias"

            if ln1_weight_key not in state_dict:
                state_dict[ln1_weight_key] = torch.ones(cfg.d_model)
            if ln1_bias_key not in state_dict:
                state_dict[ln1_bias_key] = torch.zeros(cfg.d_model)
            if ln2_weight_key not in state_dict:
                state_dict[ln2_weight_key] = torch.ones(cfg.d_model)
            if ln2_bias_key not in state_dict:
                state_dict[ln2_bias_key] = torch.zeros(cfg.d_model)

        # Add identity final LayerNorm parameters
        ln_final_weight_key = "transformer.ln_f.weight"
        ln_final_bias_key = "transformer.ln_f.bias"

        if ln_final_weight_key not in state_dict:
            state_dict[ln_final_weight_key] = torch.ones(cfg.d_model)
        if ln_final_bias_key not in state_dict:
            state_dict[ln_final_bias_key] = torch.zeros(cfg.d_model)

    # Apply the processing steps
    print("    Folding LayerNorm...")
    _fold_layer_norm_hf_native(state_dict)

    print("    Centering writing weights...")
    _center_writing_weights_hf_native(state_dict)

    print("    Centering unembedding weights...")
    _center_unembed_hf_native(state_dict)

    # Add missing LayerNorm parameters as identity (critical fix)
    print("    Adding missing LayerNorm parameters as identity...")
    _add_identity_layer_norm_params(state_dict)


def test_hf_native_processing():
    """Test the HF-native processing approach."""
    print("=== TESTING HF-NATIVE PROCESSING ===")

    # Test text
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."

    print("\n1. Loading reference HookedTransformer...")
    hooked_processed = HookedTransformer.from_pretrained("gpt2", device="cpu")
    tokens = hooked_processed.to_tokens(gpt2_text)

    print("\n2. Loading HuggingFace model...")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval()

    # Test baseline
    with torch.no_grad():
        hf_outputs_before = hf_model(tokens)
        hf_loss_before = torch.nn.functional.cross_entropy(
            hf_outputs_before.logits[:, :-1].reshape(-1, hf_outputs_before.logits.size(-1)),
            tokens[:, 1:].reshape(-1),
        )

    print(f"   HF model loss (before processing): {hf_loss_before.item():.6f}")

    print("\n3. Applying HF-native processing...")
    cfg = hooked_processed.cfg

    try:
        # Extract HF state dict
        print("  Extracting HuggingFace state dict...")
        hf_state_dict = hf_model.state_dict().copy()

        print(f"  Processing {len(hf_state_dict)} parameters with HF-native mathematics...")

        # Apply HF-native processing
        apply_hf_native_processing_final(hf_state_dict, cfg)

        # Load processed weights back into model
        print("  Loading processed weights back into model...")
        hf_model.load_state_dict(hf_state_dict)

        processing_succeeded = True
        print("   ✅ HF-native processing succeeded!")

    except Exception as e:
        print(f"   ❌ HF-native processing failed: {e}")
        import traceback

        traceback.print_exc()
        processing_succeeded = False

    if not processing_succeeded:
        print("\n❌ Processing failed - cannot continue with comparison")
        return False

    print("\n4. Testing processed model...")
    with torch.no_grad():
        hf_outputs_after = hf_model(tokens)
        hf_loss_after = torch.nn.functional.cross_entropy(
            hf_outputs_after.logits[:, :-1].reshape(-1, hf_outputs_after.logits.size(-1)),
            tokens[:, 1:].reshape(-1),
        )

    print(f"   HF model loss (after processing): {hf_loss_after.item():.6f}")

    print("\n5. Testing ablation...")
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
        f"HF + HF-Native: Original={hf_loss_after.item():.6f}, Ablated={hf_loss_ablated.item():.6f}"
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
    print(f"HF + HF-Native gain: {hf_gain:.6f}")
    print(f"Gain difference: {gain_diff:.8f}")

    # Success criteria
    baseline_perfect = orig_diff < 0.00001
    ablation_perfect = ablated_diff < 0.00001
    gain_perfect = gain_diff < 0.00001

    baseline_good = orig_diff < 0.001
    ablation_good = ablated_diff < 0.001
    gain_good = gain_diff < 0.001

    print(f"\nSuccess criteria:")
    print(
        f'Baseline match: {"✅ PERFECT" if baseline_perfect else "👍 GOOD" if baseline_good else "❌ POOR"} ({orig_diff:.8f})'
    )
    print(
        f'Ablation match: {"✅ PERFECT" if ablation_perfect else "👍 GOOD" if ablation_good else "❌ POOR"} ({ablated_diff:.8f})'
    )
    print(
        f'Gain match: {"✅ PERFECT" if gain_perfect else "👍 GOOD" if gain_good else "❌ POOR"} ({gain_diff:.8f})'
    )

    if baseline_perfect and ablation_perfect and gain_perfect:
        print("\n🎉🎉🎉 PERFECT SUCCESS: HF-NATIVE PROCESSING WORKS! 🎉🎉🎉")
        return "PERFECT"
    elif baseline_good and ablation_good and gain_good:
        print("\n👍👍👍 EXCELLENT SUCCESS: HF-NATIVE PROCESSING WORKS WELL! 👍👍👍")
        return "EXCELLENT"
    else:
        print("\n⚠️  HF-native processing needs refinement")
        return "NEEDS_WORK"


if __name__ == "__main__":
    result = test_hf_native_processing()
    print(f"\n🔥 RESULT: {result} 🔥")
