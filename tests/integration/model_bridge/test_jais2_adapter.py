"""Integration tests for the Jais2 architecture adapter.

Seeded tiny from a local Jais2Config (no hub access). Jais 2 uses plain
nn.LayerNorm in a standard pre-norm layout, so unlike its Nemotron parent
(LayerNorm1P) compatibility-mode folding is valid — the parity test pins
that the re-enabled fold path stays faithful.
"""

import copy

import torch


def _tiny_jais2_pair():
    from transformers import AutoModelForCausalLM, Jais2Config

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    cfg = Jais2Config(
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
        hf, "Jais2ForCausalLM", hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    ).eval()
    return bridge, ref


class TestJais2Bridge:
    def test_forward_matches_hf(self) -> None:
        bridge, ref = _tiny_jais2_pair()
        torch.manual_seed(0)
        ids = torch.randint(3, 128, (1, 12))
        with torch.no_grad():
            out = bridge(ids)
            expected = ref(input_ids=ids).logits
        max_diff = (out - expected).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_fold_flags_reenabled(self) -> None:
        bridge, _ = _tiny_jais2_pair()
        assert bridge.adapter.supports_fold_ln is True
        assert bridge.adapter.supports_center_writing_weights is True

    def test_compatibility_mode_logit_parity(self) -> None:
        """Plain-LN folding must stay faithful to the HF reference.

        Compared on log_softmax: center_unembed shifts raw logits by a
        per-position constant by design, so raw-logit deltas are not a
        parity signal.
        """
        bridge, ref = _tiny_jais2_pair()
        torch.manual_seed(0)
        ids = torch.randint(3, 128, (1, 12))
        bridge.enable_compatibility_mode(disable_warnings=True)
        with torch.no_grad():
            out = torch.log_softmax(bridge(ids), dim=-1)
            expected = torch.log_softmax(ref(input_ids=ids).logits, dim=-1)
        max_diff = (out - expected).abs().max().item()
        assert max_diff < 1e-5, f"Compat-mode vs HF log_softmax max diff = {max_diff}"
