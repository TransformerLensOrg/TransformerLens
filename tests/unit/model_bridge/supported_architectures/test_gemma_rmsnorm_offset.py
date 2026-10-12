"""RMSNorm offset behavior: cfg.rmsnorm_uses_offset alone carries Gemma's +1.

Norm weights deliberately have no weight_processing_conversions entry — weight
processing and the runtime norm read the flag and apply (1 + w) themselves.
These tests perturb the norm weights of tiny in-memory models (so w and 1 + w
differ materially) and pin output parity against HF logits captured before the
bridge mutates the wrapped module.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.gemma import GemmaConfig
from transformers.models.gemma3 import Gemma3TextConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures.gemma1 import (
    Gemma1ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gemma3 import (
    Gemma3ArchitectureAdapter,
)

TOKENS = torch.tensor([[1, 5, 9, 13, 17, 21]])
# Folding is a linear reparametrization; fp32 drift on a tiny model stays ~1e-6.
ATOL = 1e-4


class _Tok:
    pass


def _make_hf(config_cls, architecture):
    """Tiny HF model with perturbed norm weights (w far from 0, so 1+w != w)."""
    torch.manual_seed(0)
    cfg = config_cls(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
    )
    cfg.architectures = [architecture]
    hf_model = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if "norm" in name.lower():
                param.copy_(torch.rand_like(param) + 0.5)
            else:
                param.normal_(0.0, 0.1)
    return hf_model


def _compat_parity(config_cls, architecture, adapter_cls):
    """Max |log_softmax(HF) - log_softmax(bridge)| after enable_compatibility_mode."""
    hf_model = _make_hf(config_cls, architecture)
    with torch.no_grad():
        reference = torch.log_softmax(hf_model(TOKENS).logits, -1).clone()
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, architecture, "tiny", torch.float32
    )
    bridge = TransformerBridge(hf_model, adapter_cls(bridge_config), tokenizer=_Tok())
    bridge.enable_compatibility_mode(disable_warnings=True)
    with torch.no_grad():
        processed = torch.log_softmax(bridge(TOKENS), -1)
    return float((reference - processed).abs().max())


def test_gemma1_offset_fold_log_softmax_parity():
    """Folding reads (1 + w) from the flag; wrong offset handling shifts logits O(1)."""
    parity = _compat_parity(GemmaConfig, "GemmaForCausalLM", Gemma1ArchitectureAdapter)
    assert parity < ATOL, f"log_softmax parity {parity} after fold with perturbed norms"


def test_gemma3_offset_fold_log_softmax_parity():
    """Gemma3's path (incl. q_norm/k_norm) stays HF-exact with perturbed norms."""
    parity = _compat_parity(Gemma3TextConfig, "Gemma3ForCausalLM", Gemma3ArchitectureAdapter)
    assert parity < ATOL, f"log_softmax parity {parity} after fold with perturbed norms"


def test_gemma3_native_forward_parity_with_perturbed_norms():
    """Unprocessed bridge forward matches HF exactly under perturbed norms."""
    hf_model = _make_hf(Gemma3TextConfig, "Gemma3ForCausalLM")
    with torch.no_grad():
        reference = hf_model(TOKENS).logits.clone()
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, "Gemma3ForCausalLM", "tiny", torch.float32
    )
    bridge = TransformerBridge(hf_model, Gemma3ArchitectureAdapter(bridge_config), tokenizer=_Tok())
    with torch.no_grad():
        native = bridge(TOKENS)
    assert torch.allclose(reference, native, atol=1e-6)


@pytest.mark.parametrize(
    "adapter_cls,config_cls,architecture",
    [
        (Gemma1ArchitectureAdapter, GemmaConfig, "GemmaForCausalLM"),
        (Gemma3ArchitectureAdapter, Gemma3TextConfig, "Gemma3ForCausalLM"),
    ],
)
def test_flag_off_breaks_offset_fold(adapter_cls, config_cls, architecture):
    """Control: without the flag, folding uses raw w and parity collapses.

    Guards against the flag read silently becoming a no-op — if processing
    stopped consulting rmsnorm_uses_offset, the positive parity tests above
    could pass vacuously only if folding were skipped entirely; this pins that
    the flag is what carries the offset.
    """
    hf_model = _make_hf(config_cls, architecture)
    with torch.no_grad():
        reference = torch.log_softmax(hf_model(TOKENS).logits, -1).clone()
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, architecture, "tiny", torch.float32
    )
    adapter = adapter_cls(bridge_config)
    adapter.cfg.rmsnorm_uses_offset = False
    bridge = TransformerBridge(hf_model, adapter, tokenizer=_Tok())
    bridge.enable_compatibility_mode(disable_warnings=True)
    with torch.no_grad():
        processed = torch.log_softmax(bridge(TOKENS), -1)
    assert float((reference - processed).abs().max()) > 0.01
