"""Integration tests for the LLaDA 2.0 MoE architecture adapter.

Every published checkpoint is 16B+ (over the local fp32 ceiling), so parity
is proven on a seeded tiny model built from the remote config — the
FlexOlmo pattern, with the remote code's two quirks exercised: the v4
rope-init shim and the strict 4D block-diffusion attention mask.
"""

import copy

import torch

from transformer_lens.model_bridge.generalized_components import MoEBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)
from transformer_lens.model_bridge.supported_architectures.dream import (
    _register_default_rope_init,
)

MODEL_ID = "inclusionAI/LLaDA2.0-mini"


def _tiny_llada2_pair():
    _register_default_rope_init()
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.moe_intermediate_size = 32
    cfg.num_hidden_layers = 3
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.n_group = 1
    cfg.topk_group = 1
    cfg.vocab_size = 128
    cfg.pad_token_id = 0
    cfg.bos_token_id = 1
    cfg.eos_token_id = 2
    cfg.first_k_dense_replace = 1
    cfg._attn_implementation = "eager"

    torch.manual_seed(42)
    ref = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).eval()
    hf = AutoModelForCausalLM.from_config(copy.deepcopy(cfg), trust_remote_code=True).eval()
    hf.load_state_dict(ref.state_dict())
    bridge = build_bridge_from_module(
        hf, "LLaDA2MoeModelLM", hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    ).eval()
    return bridge, ref


class TestLLaDA2MoeBridge:
    def test_forward_matches_hf_with_block_mask(self) -> None:
        bridge, ref = _tiny_llada2_pair()
        ids = torch.randint(0, 128, (1, 12))
        # The remote forward requires the 4D block-diffusion mask form.
        mask = torch.ones(1, 1, 12, 12)
        with torch.no_grad():
            out = bridge(ids, attention_mask=mask)
            expected = ref(input_ids=ids, attention_mask=mask).logits
        max_diff = (out - expected).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"

    def test_structure_dense_then_moe(self) -> None:
        bridge, _ = _tiny_llada2_pair()
        assert len(bridge.blocks) == 3
        assert isinstance(bridge.blocks[0].mlp, MoEBridge)

    def test_cache_captures_router_and_shared_experts(self) -> None:
        bridge, _ = _tiny_llada2_pair()
        ids = torch.randint(0, 128, (1, 8))
        mask = torch.ones(1, 1, 8, 8)
        with torch.no_grad():
            _, cache = bridge.run_with_cache(ids, attention_mask=mask)
        assert any("gate" in k for k in cache.keys())
        assert any("shared_experts" in k for k in cache.keys())

    def test_2d_mask_guard(self) -> None:
        """All-ones 2D masks translate to the 4D form; padded ones reject clearly."""
        import pytest

        bridge, ref = _tiny_llada2_pair()
        ids = torch.randint(0, 128, (2, 8))
        with torch.no_grad():
            out = bridge(ids, attention_mask=torch.ones(2, 8, dtype=torch.long))
            expected = ref(input_ids=ids, attention_mask=torch.ones(2, 1, 8, 8)).logits
        assert (out - expected).abs().max().item() < 1e-4
        padded = torch.ones(2, 8, dtype=torch.long)
        padded[0, :3] = 0
        with pytest.raises(NotImplementedError, match="padding masks"):
            with torch.no_grad():
                bridge(ids, attention_mask=padded)
