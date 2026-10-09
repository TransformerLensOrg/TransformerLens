"""SVDInterpreter w_in scaling on offset-RMSNorm (Gemma-style) models.

Unfolded offset models expose raw ln2.w (effective scale 1 + w); the w_in
matrix must use the effective scale, so it matches what fold_ln bakes into
W_in. Tiny in-memory models only — no Hub downloads.
"""

import torch
from transformers import AutoModelForCausalLM
from transformers.models.gemma import GemmaConfig

from transformer_lens import SVDInterpreter
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures.gemma1 import (
    Gemma1ArchitectureAdapter,
)


class _Tok:
    pass


def _make_bridge(folded: bool) -> TransformerBridge:
    """Tiny Gemma1 bridge with perturbed norms; same seed, so variants share weights."""
    torch.manual_seed(0)
    cfg = GemmaConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
    )
    cfg.architectures = ["GemmaForCausalLM"]
    hf_model = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if "norm" in name.lower():
                param.copy_(torch.rand_like(param) + 0.5)
            else:
                param.normal_(0.0, 0.1)
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, "GemmaForCausalLM", "gemma-tiny", torch.float32
    )
    bridge = TransformerBridge(hf_model, Gemma1ArchitectureAdapter(bridge_config), tokenizer=_Tok())
    if folded:
        # Compatibility mode (not bare process_weights) writes the folded
        # weights back where tl_parameters() reads them, with ln2.w at identity.
        bridge.enable_compatibility_mode(
            disable_warnings=True,
            fold_ln=True,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )
    return bridge


def test_w_in_offset_norm_matches_folded_w_in():
    """fold_ln bakes (1 + w) into W_in, so it is an independent oracle for the
    unfolded w_in scaling; multiplying by raw w would miss the offset."""
    unfolded = SVDInterpreter(_make_bridge(folded=False))
    folded = SVDInterpreter(_make_bridge(folded=True))
    for layer in range(2):
        m_unfolded = unfolded._get_w_in_matrix(layer)
        m_folded = folded._get_w_in_matrix(layer)
        assert torch.allclose(m_unfolded, m_folded, atol=1e-5), f"layer {layer}"


def test_w_in_offset_norm_uses_effective_scale():
    """The unfolded matrix is w_in * (1 + ln2.w), not w_in * ln2.w."""
    interpreter = SVDInterpreter(_make_bridge(folded=False))
    w_in = interpreter.params["blocks.0.mlp.W_in"].T
    ln_2 = interpreter.params["blocks.0.ln2.w"]
    matrix = interpreter._get_w_in_matrix(0)
    assert torch.allclose(matrix, w_in * (1.0 + ln_2), atol=1e-6)
    # Perturbed norms keep w far from identity, so the raw-w product differs.
    assert not torch.allclose(matrix, w_in * ln_2, atol=1e-3)
