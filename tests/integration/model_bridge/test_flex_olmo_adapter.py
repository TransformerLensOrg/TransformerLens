"""Integration tests for the FlexOlmo architecture adapter.

All published FlexOlmo checkpoints are 2x7B+ (over the local fp32 ceiling),
so parity is proven on a seeded tiny FlexOlmoForCausalLM built from config —
the audit's 'local tiny FlexOlmoConfig for CI' plan.
"""

import copy

import torch
from transformers import FlexOlmoConfig, FlexOlmoForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import MoEBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


def tiny_flex_olmo_config() -> FlexOlmoConfig:
    return FlexOlmoConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def tiny_flex_olmo_bridge() -> tuple[TransformerBridge, FlexOlmoForCausalLM]:
    torch.manual_seed(42)
    cfg = tiny_flex_olmo_config()
    hf_reference = FlexOlmoForCausalLM(cfg).eval()
    hf_reference.config._attn_implementation = "eager"
    hf_model = FlexOlmoForCausalLM(copy.deepcopy(cfg)).eval()
    hf_model.load_state_dict(hf_reference.state_dict())
    bridge = build_bridge_from_module(
        hf_model,
        "FlexOlmoForCausalLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()
    bridge.adapter.setup_component_testing(hf_model, bridge_model=bridge)
    return bridge, hf_reference


class TestFlexOlmoBridge:
    def test_bridge_structure(self) -> None:
        bridge, _ = tiny_flex_olmo_bridge()
        assert len(bridge.blocks) == 3
        assert isinstance(bridge.blocks[0].mlp, MoEBridge)
        # OLMo-2 post-norm layout carried through the subclass
        assert bridge.blocks[0].ln1.original_component is not None

    def test_forward_matches_hf(self) -> None:
        bridge, hf_reference = tiny_flex_olmo_bridge()
        input_ids = torch.randint(0, 128, (1, 12))
        with torch.no_grad():
            bridge_out = bridge(input_ids)
            hf_out = hf_reference(input_ids=input_ids).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"

    def test_run_with_cache_populates_moe_hooks(self) -> None:
        bridge, _ = tiny_flex_olmo_bridge()
        input_ids = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(input_ids)
        assert any("mlp" in k for k in cache.keys())
        assert any("gate" in k for k in cache.keys())
