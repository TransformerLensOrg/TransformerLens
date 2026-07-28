"""Integration tests for the Laguna architecture adapter.

Published checkpoints are 33B+ (over the local fp32 ceiling), so parity is
proven on a seeded tiny from the remote config with the architecture's two
novelties exercised: heterogeneous per-layer head counts and mixed
dense/sparse MLP layers.
"""

import copy

import torch

from transformer_lens.model_bridge.generalized_components import MoEBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

MODEL_ID = "poolside/Laguna-XS.2"


def _tiny_laguna_pair():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.moe_intermediate_size = 32
    cfg.shared_expert_intermediate_size = 32
    cfg.num_hidden_layers = 4
    cfg.head_dim = 16
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_attention_heads_per_layer = [4, 8, 4, 8]
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.mlp_layer_types = ["dense", "sparse", "sparse", "sparse"]
    cfg.vocab_size = 128
    cfg.pad_token_id = 0
    cfg.bos_token_id = 1
    cfg.eos_token_id = 2
    cfg._attn_implementation = "eager"

    torch.manual_seed(42)
    ref = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).eval()
    hf = AutoModelForCausalLM.from_config(copy.deepcopy(cfg), trust_remote_code=True).eval()
    hf.load_state_dict(ref.state_dict())
    bridge = build_bridge_from_module(
        hf, "LagunaForCausalLM", hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    ).eval()
    return bridge, ref


class TestLagunaBridge:
    def test_forward_matches_hf_with_heterogeneous_heads(self) -> None:
        bridge, ref = _tiny_laguna_pair()
        ids = torch.randint(0, 128, (1, 12))
        with torch.no_grad():
            out = bridge(ids)
            expected = ref(input_ids=ids).logits
        max_diff = (out - expected).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"

    def test_structure_mixed_dense_sparse(self) -> None:
        bridge, _ = _tiny_laguna_pair()
        assert len(bridge.blocks) == 4
        assert isinstance(bridge.blocks[1].mlp, MoEBridge)

    def test_cache_captures_softplus_gate_and_router(self) -> None:
        bridge, _ = _tiny_laguna_pair()
        ids = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(ids)
        assert any("attn.gate" in k for k in cache.keys())
        assert any("mlp.gate" in k for k in cache.keys())
