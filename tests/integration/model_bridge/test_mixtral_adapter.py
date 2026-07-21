"""Integration tests for the Mixtral architecture adapter.

Real checkpoints are 8x7B+; parity is proven on a seeded tiny built from a
local MixtralConfig (no hub access). Guards the transformers >= 5.13 layout
where the sparse block moved from layer.block_sparse_moe to layer.mlp —
component setup fails outright if the mapping goes stale again.
"""

import copy

import torch

MODEL_VOCAB = 128


def _tiny_mixtral_pair():
    from transformers import AutoModelForCausalLM, MixtralConfig

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    cfg = MixtralConfig(
        vocab_size=MODEL_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
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
        hf, "MixtralForCausalLM", hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    ).eval()
    return bridge, ref


class TestMixtralBridge:
    def test_sparse_block_resolves_at_mlp(self) -> None:
        """5.13 holds the MoE block at layer.mlp; boot fails if the path is stale."""
        bridge, ref = _tiny_mixtral_pair()
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        assert isinstance(ref.model.layers[0].mlp, MixtralSparseMoeBlock)
        assert bridge.blocks[0].mlp.original_component is not None

    def test_forward_matches_hf(self) -> None:
        bridge, ref = _tiny_mixtral_pair()
        torch.manual_seed(0)
        ids = torch.randint(3, MODEL_VOCAB, (1, 12))
        with torch.no_grad():
            out = bridge(ids)
            expected = ref(input_ids=ids).logits
        max_diff = (out - expected).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_block_hooks_fire(self) -> None:
        bridge, _ = _tiny_mixtral_pair()
        torch.manual_seed(0)
        ids = torch.randint(3, MODEL_VOCAB, (1, 8))
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out", "blocks.1.hook_out"]
        with torch.no_grad():
            bridge.run_with_hooks(ids, fwd_hooks=[(name, grab) for name in hooks])
        for name in hooks:
            assert captured.get(name) == (1, 8, 64), f"{name}: {captured.get(name)}"
