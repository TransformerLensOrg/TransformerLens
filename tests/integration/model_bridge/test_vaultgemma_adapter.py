"""Integration tests for the VaultGemma architecture adapter.

Seeded tiny from a local VaultGemmaConfig (no hub access). Pins forward
parity and the compatibility-mode runtime gate: the adapter's compat
forward is known to diverge (offset-RMS without post-norms), so
enable_compatibility_mode() must raise instead of silently returning
wrong logits.
"""

import copy

import pytest
import torch


def _tiny_vaultgemma_pair():
    from transformers import AutoModelForCausalLM, VaultGemmaConfig

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    cfg = VaultGemmaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    cfg._attn_implementation = "eager"

    torch.manual_seed(42)
    ref = AutoModelForCausalLM.from_config(cfg).eval()
    hf = AutoModelForCausalLM.from_config(copy.deepcopy(cfg)).eval()
    hf.load_state_dict(ref.state_dict())
    bridge = build_bridge_from_module(
        hf, "VaultGemmaForCausalLM", hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    ).eval()
    return bridge, ref


class TestVaultGemmaBridge:
    def test_forward_matches_hf(self) -> None:
        bridge, ref = _tiny_vaultgemma_pair()
        torch.manual_seed(0)
        ids = torch.randint(3, 128, (1, 12))
        with torch.no_grad():
            out = bridge(ids)
            expected = ref(input_ids=ids).logits
        max_diff = (out - expected).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_compatibility_mode_is_gated(self) -> None:
        """Known-diverging compat forward must raise, not silently mis-answer."""
        bridge, _ = _tiny_vaultgemma_pair()
        with pytest.raises(RuntimeError, match="compatibility mode"):
            bridge.enable_compatibility_mode()
