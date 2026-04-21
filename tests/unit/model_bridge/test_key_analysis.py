#!/usr/bin/env python3
"""
Analyze key matching between ProcessWeights and HuggingFace state dict.
This test helps understand key translation patterns for debugging.
"""

from transformers import GPT2LMHeadModel

from transformer_lens import HookedTransformer
from transformer_lens.weight_processing import ProcessWeights


def create_simple_adapter():
    """Create a simple adapter that maps TL paths to HF paths for GPT-2."""

    class SimpleGPT2Adapter:
        def translate_transformer_lens_path(self, tl_path: str) -> str:
            """Translate TransformerLens paths to HuggingFace paths for GPT-2."""

            # Handle embedding weights
            if tl_path == "embed.W_E":
                return "transformer.wte.weight"
            elif tl_path == "pos_embed.W_pos":
                return "transformer.wpe.weight"
            elif tl_path == "unembed.W_U":
                return "lm_head.weight"
            elif tl_path == "unembed.b_U":
                return "lm_head.bias"  # Note: GPT-2 doesn't have this
            elif tl_path == "ln_final.w":
                return "transformer.ln_f.weight"
            elif tl_path == "ln_final.b":
                return "transformer.ln_f.bias"

            # Handle layer-specific weights
            import re

            # Match patterns like "blocks.0.attn.W_Q"
            layer_match = re.match(r"blocks\.(\d+)\.(.+)", tl_path)
            if layer_match:
                layer_idx = layer_match.group(1)
                component_path = layer_match.group(2)

                # Attention weights
                if component_path == "attn.W_Q":
                    return f"transformer.h.{layer_idx}.attn.c_attn.weight"  # GPT-2 combines QKV
                elif component_path == "attn.W_K":
                    return f"transformer.h.{layer_idx}.attn.c_attn.weight"  # GPT-2 combines QKV
                elif component_path == "attn.W_V":
                    return f"transformer.h.{layer_idx}.attn.c_attn.weight"  # GPT-2 combines QKV
                elif component_path == "attn.W_O":
                    return f"transformer.h.{layer_idx}.attn.c_proj.weight"
                elif component_path == "attn.b_Q":
                    return f"transformer.h.{layer_idx}.attn.c_attn.bias"  # GPT-2 combines QKV
                elif component_path == "attn.b_K":
                    return f"transformer.h.{layer_idx}.attn.c_attn.bias"  # GPT-2 combines QKV
                elif component_path == "attn.b_V":
                    return f"transformer.h.{layer_idx}.attn.c_attn.bias"  # GPT-2 combines QKV
                elif component_path == "attn.b_O":
                    return f"transformer.h.{layer_idx}.attn.c_proj.bias"

                # MLP weights
                elif component_path == "mlp.W_in":
                    return f"transformer.h.{layer_idx}.mlp.c_fc.weight"
                elif component_path == "mlp.W_out":
                    return f"transformer.h.{layer_idx}.mlp.c_proj.weight"
                elif component_path == "mlp.b_in":
                    return f"transformer.h.{layer_idx}.mlp.c_fc.bias"
                elif component_path == "mlp.b_out":
                    return f"transformer.h.{layer_idx}.mlp.c_proj.bias"

                # LayerNorm weights
                elif component_path == "ln1.w":
                    return f"transformer.h.{layer_idx}.ln_1.weight"
                elif component_path == "ln1.b":
                    return f"transformer.h.{layer_idx}.ln_1.bias"
                elif component_path == "ln2.w":
                    return f"transformer.h.{layer_idx}.ln_2.weight"
                elif component_path == "ln2.b":
                    return f"transformer.h.{layer_idx}.ln_2.bias"

            # If no match found, return the original path
            return tl_path

    return SimpleGPT2Adapter()


class KeyTrackingProcessWeights:
    """Wrapper around state dict that tracks which keys ProcessWeights tries to access."""

    def __init__(self, state_dict, adapter=None):
        self.state_dict = state_dict
        self.adapter = adapter
        self.accessed_keys = set()
        self.missing_keys = set()

    def __getitem__(self, key):
        """Track key access and translate if adapter is provided."""
        original_key = key
        if self.adapter:
            key = self.adapter.translate_transformer_lens_path(key)

        self.accessed_keys.add(original_key)

        if key in self.state_dict:
            return self.state_dict[key]
        else:
            self.missing_keys.add(key)
            print(f"  Missing key: {original_key} -> {key}")
            raise KeyError(f"Key not found: {key}")

    def __contains__(self, key):
        """Check if key exists, with translation if adapter provided."""
        original_key = key
        if self.adapter:
            key = self.adapter.translate_transformer_lens_path(key)

        return key in self.state_dict

    def __setitem__(self, key, value):
        """Set key with translation if adapter provided."""
        original_key = key
        if self.adapter:
            key = self.adapter.translate_transformer_lens_path(key)

        self.state_dict[key] = value

    def keys(self):
        """Return keys from original state dict."""
        return self.state_dict.keys()

    def items(self):
        """Return items from original state dict."""
        return self.state_dict.items()

    def copy(self):
        """Return a copy of the original state dict."""
        return self.state_dict.copy()


def test_key_analysis():
    """Analyze what keys ProcessWeights tries to access."""
    print("=== ANALYZING PROCESSWEIGHTS KEY ACCESS ===")

    print("\n1. Loading models...")
    hooked_model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    print("\n2. Getting state dicts...")
    tl_state_dict = hooked_model.state_dict()
    hf_state_dict = hf_model.state_dict()

    print(f"   TL state dict keys: {len(tl_state_dict)}")
    print(f"   HF state dict keys: {len(hf_state_dict)}")

    print("\n3. Analyzing TL keys that ProcessWeights expects...")
    print("   Sample TL keys:")
    for i, key in enumerate(sorted(tl_state_dict.keys())):
        if i < 10:
            print(f"     {key}")

    print("\n4. Analyzing HF keys available...")
    print("   Sample HF keys:")
    for i, key in enumerate(sorted(hf_state_dict.keys())):
        if i < 10:
            print(f"     {key}")

    print("\n5. Testing ProcessWeights with TL state dict (should work)...")
    try:
        tracking_tl = KeyTrackingProcessWeights(tl_state_dict)
        processed_tl = ProcessWeights.process_weights(
            state_dict=tracking_tl,
            cfg=hooked_model.cfg,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
            refactor_factored_attn_matrices=False,
        )
        print("   ✅ ProcessWeights succeeded with TL state dict")
        print(f"   Accessed {len(tracking_tl.accessed_keys)} keys")
        print(f"   Missing {len(tracking_tl.missing_keys)} keys")
    except Exception as e:
        print(f"   ❌ ProcessWeights failed with TL state dict: {e}")

    print("\n6. Testing ProcessWeights with HF state dict (will fail)...")
    try:
        tracking_hf = KeyTrackingProcessWeights(hf_state_dict)
        processed_hf = ProcessWeights.process_weights(
            state_dict=tracking_hf,
            cfg=hooked_model.cfg,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
            refactor_factored_attn_matrices=False,
        )
        print("   ✅ ProcessWeights succeeded with HF state dict")
    except Exception as e:
        print(f"   ❌ ProcessWeights failed with HF state dict: {e}")
        print(f"   Accessed {len(tracking_hf.accessed_keys)} keys")
        print(f"   Missing {len(tracking_hf.missing_keys)} keys")

        print("   Keys ProcessWeights tried to access:")
        for key in sorted(tracking_hf.accessed_keys):
            print(f"     {key}")

    print("\n7. Testing ProcessWeights with HF state dict + adapter...")
    try:
        adapter = create_simple_adapter()
        tracking_hf_adapter = KeyTrackingProcessWeights(hf_state_dict, adapter=adapter)
        processed_hf_adapter = ProcessWeights.process_weights(
            state_dict=tracking_hf_adapter,
            cfg=hooked_model.cfg,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
            refactor_factored_attn_matrices=False,
        )
        print("   ✅ ProcessWeights succeeded with HF state dict + adapter")
        print(f"   Accessed {len(tracking_hf_adapter.accessed_keys)} keys")
        print(f"   Missing {len(tracking_hf_adapter.missing_keys)} keys")
    except Exception as e:
        print(f"   ❌ ProcessWeights failed with HF state dict + adapter: {e}")
        print(f"   Accessed {len(tracking_hf_adapter.accessed_keys)} keys")
        print(f"   Missing {len(tracking_hf_adapter.missing_keys)} keys")

    print("\n=== KEY ANALYSIS COMPLETE ===")
    print("\n📋 FINDINGS:")
    print("   • ProcessWeights expects TransformerLens key format")
    print("   • Direct HF state dict fails due to key mismatch")
    print("   • Adapter can bridge the gap by translating keys")
    print("   • Need proper adapter implementation for full compatibility")


if __name__ == "__main__":
    test_key_analysis()
