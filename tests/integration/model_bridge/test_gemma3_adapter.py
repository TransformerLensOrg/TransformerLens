"""Integration tests for the Gemma3 adapter on a seeded tiny (no downloads).

Pins the compatibility-mode contract: processed weights must preserve the
predictive distribution (log_softmax parity; raw logits shift by design under
center_unembed).
"""

import copy

import pytest
import torch

pytest.importorskip("transformers", reason="requires transformers")

from transformers import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from transformer_lens.model_bridge.sources._bridge_builder import build_bridge_from_module


def _tiny_gemma3_bridge():
    cfg = Gemma3TextConfig(
        vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=256, sliding_window=8,
        pad_token_id=0, eos_token_id=1, bos_token_id=2,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(42)
    hf = Gemma3ForCausalLM(cfg).eval()
    return build_bridge_from_module(
        hf, "Gemma3ForCausalLM", hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    ).eval()


class TestGemma3CompatibilityMode:
    def test_act_fn_is_tanh_gelu(self) -> None:
        bridge = _tiny_gemma3_bridge()
        assert bridge.cfg.act_fn == "gelu_pytorch_tanh"

    def test_compat_mode_preserves_log_softmax(self) -> None:
        """Seq length deliberately exceeds sliding_window so both mask paths run."""
        bridge = _tiny_gemma3_bridge()
        torch.manual_seed(7)
        tokens = torch.randint(3, 200, (1, 12))
        with torch.no_grad():
            base = bridge(tokens)
        bridge.enable_compatibility_mode()
        with torch.no_grad():
            proc = bridge(tokens)
        ls_base = torch.log_softmax(base.float(), dim=-1)
        ls_proc = torch.log_softmax(proc.float(), dim=-1)
        assert torch.allclose(ls_base, ls_proc, atol=1e-4), (
            f"log_softmax diverged: max={(ls_base - ls_proc).abs().max().item()}"
        )
