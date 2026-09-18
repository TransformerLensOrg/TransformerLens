"""Integration tests for the StableLM architecture adapter.

Seeded tinies from a local StableLmConfig (no hub access). The qk_layernorm
variant pins the per-head Q/K LayerNorm path: the reimplemented attention
must apply the norms (stablelm-2-12b family) — without it the bridge
silently diverged by ~1.0 on tiny models.
"""

import torch

from tests.integration.model_bridge.helpers import make_tiny_pair


def _tiny_stablelm_pair(qk_layernorm: bool):
    from transformers import StableLmConfig

    cfg = StableLmConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        qk_layernorm=qk_layernorm,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return make_tiny_pair(cfg, "StableLmForCausalLM")


class TestStableLmBridge:
    def _parity(self, qk_layernorm: bool) -> float:
        bridge, ref = _tiny_stablelm_pair(qk_layernorm)
        torch.manual_seed(0)
        ids = torch.randint(3, 128, (1, 12))
        with torch.no_grad():
            out = bridge(ids)
            expected = ref(input_ids=ids).logits
        return (out - expected).abs().max().item()

    def test_forward_matches_hf_plain(self) -> None:
        max_diff = self._parity(qk_layernorm=False)
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_forward_matches_hf_qk_layernorm(self) -> None:
        """stablelm-2-12b layout: per-head Q/K LayerNorm before rotary."""
        max_diff = self._parity(qk_layernorm=True)
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_qk_norm_hooks_fire_on_layernorm_variant(self) -> None:
        bridge, _ = _tiny_stablelm_pair(qk_layernorm=True)
        torch.manual_seed(0)
        ids = torch.randint(3, 128, (1, 8))
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_q_normed", "blocks.0.attn.hook_k_normed"]
        with torch.no_grad():
            bridge.run_with_hooks(ids, fwd_hooks=[(name, grab) for name in hooks])
        # Post-reshape phase: [B, H, S, D] with H = n_heads for Q, n_kv for K.
        assert captured.get(hooks[0]) == (1, 4, 8, 16), captured
        assert captured.get(hooks[1]) == (1, 2, 8, 16), captured
