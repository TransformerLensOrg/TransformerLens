"""Integration tests for the EXAONE 4.0 architecture adapter.

The 1.2B checkpoint classes are CI-gated for download cost (no tiny mirror);
the hybrid NoPE parity class builds a seeded tiny from a local config and runs
everywhere.
"""

import pytest
import torch

from tests.integration.model_bridge.helpers import make_tiny_pair
from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# EXAONE-4.0 1.2B download/load — excluded from CI tiers via -m "not slow".
slow_download = pytest.mark.slow


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(bridge):
    return bridge.tokenizer("The capital of France is", return_tensors="pt").input_ids


@slow_download
class TestExaone4BridgeCreation:
    def test_adapter_and_hybrid_layers(self, bridge):
        from transformer_lens.model_bridge.supported_architectures.exaone4 import (
            Exaone4ArchitectureAdapter,
        )

        assert isinstance(bridge.adapter, Exaone4ArchitectureAdapter)
        layer_types = bridge.original_model.config.layer_types
        assert "sliding_attention" in layer_types or "full_attention" in layer_types


@slow_download
class TestExaone4ForwardEquivalence:
    def test_forward_matches_hf(self, bridge, sample_tokens):
        hf_model = bridge.original_model
        with torch.no_grad():
            bridge_out = bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"


@slow_download
class TestExaone4Hooks:
    def test_post_norm_and_qk_norm_hooks_fire(self, bridge, sample_tokens):
        d_model = bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = [
            "blocks.0.attn.hook_out",
            "blocks.0.ln1.hook_out",
            "blocks.0.mlp.hook_out",
            "blocks.0.ln2.hook_out",
        ]
        with torch.no_grad():
            bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"


def _tiny_hybrid_pair():
    """Seeded tiny hybrid EXAONE-4.0 (LLLG): layer 3 is full-attention NoPE."""
    from transformers import Exaone4Config

    cfg = Exaone4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        sliding_window=4096,
        sliding_window_pattern=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return make_tiny_pair(cfg, "Exaone4ForCausalLM")


class TestExaone4HybridNoPE:
    """Hybrid checkpoints skip RoPE on full-attention layers (global NoPE).

    The 1.2B fixture above is non-hybrid (sliding_window=null, RoPE everywhere)
    and cannot catch a bridge that rotates unconditionally, so parity is proven
    on a seeded tiny with the 32B family's hybrid LLLG layout.
    """

    def test_layer_types_are_hybrid(self) -> None:
        bridge, ref = _tiny_hybrid_pair()
        assert ref.config.layer_types == ["sliding_attention"] * 3 + ["full_attention"]
        assert ref.model.layers[3].self_attn.is_sliding is False

    def test_forward_matches_hf_on_nope_layers(self) -> None:
        bridge, ref = _tiny_hybrid_pair()
        torch.manual_seed(0)
        ids = torch.randint(3, 128, (1, 12))
        with torch.no_grad():
            out = bridge(ids)
            expected = ref(input_ids=ids).logits
        max_diff = (out - expected).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"
