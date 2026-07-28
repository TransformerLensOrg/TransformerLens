"""Integration tests for the Gemma 4 text-only architecture adapter.

No tiny Gemma4ForCausalLM exists on the Hub (all tiny gemma4 uploads are the
multimodal class), so parity is proven on a seeded tiny from a local config
with the hybrid sliding/full attention mix exercised (window < seq).
"""

import copy

import torch

VOCAB = 256


def _tiny_gemma4_text_pair():
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    cfg = Gemma4TextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        sliding_window=8,
        vocab_size_per_layer_input=VOCAB,
        hidden_size_per_layer_input=16,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    cfg._attn_implementation = "eager"

    torch.manual_seed(42)
    ref = Gemma4ForCausalLM(cfg).eval()
    hf = Gemma4ForCausalLM(copy.deepcopy(cfg)).eval()
    hf.load_state_dict(ref.state_dict())
    bridge = build_bridge_from_module(
        hf, "Gemma4ForCausalLM", hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    ).eval()
    return bridge, ref


class TestGemma4TextBridge:
    def test_adapter_selected_and_hybrid_layers(self) -> None:
        from transformer_lens.model_bridge.supported_architectures.gemma4_text import (
            Gemma4TextArchitectureAdapter,
        )

        bridge, ref = _tiny_gemma4_text_pair()
        assert isinstance(bridge.adapter, Gemma4TextArchitectureAdapter)
        assert ref.config.layer_types == ["sliding_attention"] * 3 + ["full_attention"]

    def test_forward_matches_hf(self) -> None:
        """Sliding window (8) < seq (12), so the sliding mask genuinely differs
        from causal on three of four layers."""
        bridge, ref = _tiny_gemma4_text_pair()
        torch.manual_seed(0)
        ids = torch.randint(3, VOCAB, (1, 12))
        with torch.no_grad():
            out = bridge(ids)
            expected = ref(input_ids=ids).logits
        max_diff = (out - expected).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_hooks_fire(self) -> None:
        bridge, _ = _tiny_gemma4_text_pair()
        torch.manual_seed(0)
        ids = torch.randint(3, VOCAB, (1, 8))
        expected = {
            "blocks.0.attn.hook_out": (1, 8, 64),
            "blocks.0.mlp.hook_out": (1, 8, 64),
            "blocks.0.ln1_post.hook_out": (1, 8, 64),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            bridge.run_with_hooks(ids, fwd_hooks=[(name, grab) for name in expected])
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"
