"""Integration tests for the MiniCPM3 architecture adapter.

Seeded tiny from a local MiniCPM3Config (no hub access). The muP scales are
non-trivial and pairwise distinct, so dropping or swapping any of them breaks
parity, and the LongRoPE case runs past original_max_position_embeddings so the
long factors are actually used.
"""

import pytest
import torch

pytest.importorskip("transformers.models.minicpm3")

from transformers import MiniCPM3Config

from tests.integration.model_bridge.helpers import make_tiny_pair

N_LAYERS = 3
TOL = 1e-5

_LONG_ROPE = {
    "rope_type": "longrope",
    "short_factor": [1.0 + 0.1 * i for i in range(4)],
    "long_factor": [1.0 + 0.5 * i for i in range(4)],
    "original_max_position_embeddings": 32,
    "rope_theta": 10000.0,
}


def _tiny_minicpm3_pair(rope_parameters=None):
    kwargs = {} if rope_parameters is None else {"rope_parameters": rope_parameters}
    cfg = MiniCPM3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        head_dim=16,
        scale_emb=7.0,
        scale_depth=2.3,
        dim_model_base=24,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        **kwargs,
    )
    return make_tiny_pair(cfg, "MiniCPM3ForCausalLM")


def _ids(seq_len: int = 12) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(3, 128, (2, seq_len))


class TestMiniCPM3Bridge:
    @pytest.mark.parametrize(
        ("rope_parameters", "seq_len"),
        [(None, 12), (_LONG_ROPE, 40)],
        ids=["default-rope", "longrope"],
    )
    def test_forward_matches_hf(self, rope_parameters, seq_len) -> None:
        bridge, ref = _tiny_minicpm3_pair(rope_parameters)
        ids = _ids(seq_len)
        with torch.no_grad():
            max_diff = (bridge(ids) - ref(input_ids=ids).logits).abs().max().item()
        assert max_diff < TOL, f"Bridge vs HF max diff = {max_diff}"

    def test_hook_embed_includes_scale_emb(self) -> None:
        bridge, ref = _tiny_minicpm3_pair()
        ids = _ids()
        _, cache = bridge.run_with_cache(ids)
        with torch.no_grad():
            expected = ref.model.embed_tokens(ids)
        torch.testing.assert_close(cache["hook_embed"], expected, atol=TOL, rtol=0)
        torch.testing.assert_close(cache["blocks.0.hook_resid_pre"], expected, atol=TOL, rtol=0)

    def test_residual_stream_decomposes_into_scaled_contributions(self) -> None:
        bridge, _ = _tiny_minicpm3_pair()
        _, cache = bridge.run_with_cache(_ids())
        for layer in range(N_LAYERS):
            rebuilt = (
                cache[f"blocks.{layer}.hook_resid_pre"]
                + cache[f"blocks.{layer}.hook_attn_out"]
                + cache[f"blocks.{layer}.hook_mlp_out"]
            )
            torch.testing.assert_close(
                rebuilt, cache[f"blocks.{layer}.hook_resid_post"], atol=TOL, rtol=0
            )

    def test_zero_ablating_hook_attn_out_matches_hf(self) -> None:
        bridge, ref = _tiny_minicpm3_pair()
        ids = _ids()
        layer = 1

        def zero_attention_output(module, args, output):
            return (torch.zeros_like(output[0]),) + tuple(output[1:])

        with torch.no_grad():
            out = bridge.run_with_hooks(
                ids,
                fwd_hooks=[
                    (f"blocks.{layer}.hook_attn_out", lambda act, hook: torch.zeros_like(act))
                ],
            )
            handle = ref.model.layers[layer].self_attn.register_forward_hook(zero_attention_output)
            try:
                expected = ref(input_ids=ids).logits
            finally:
                handle.remove()
        torch.testing.assert_close(out, expected, atol=TOL, rtol=0)

    def test_mla_rope_hooks_fire(self) -> None:
        bridge, _ = _tiny_minicpm3_pair()
        _, cache = bridge.run_with_cache(_ids())
        assert cache["blocks.0.attn.hook_rot_q"].shape == (2, 12, 4, 8)
        assert cache["blocks.0.attn.hook_rot_k"].shape[-1] == 8

    def test_compatibility_mode_keeps_the_output_distribution(self) -> None:
        # compared on log_softmax: center_unembed shifts raw logits by a per-position constant
        bridge, ref = _tiny_minicpm3_pair()
        ids = _ids()
        bridge.enable_compatibility_mode(disable_warnings=True)
        with torch.no_grad():
            out = torch.log_softmax(bridge(ids), dim=-1)
            expected = torch.log_softmax(ref(input_ids=ids).logits, dim=-1)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=0)

    def test_greedy_generation_matches_hf_with_and_without_cache(self) -> None:
        bridge, ref = _tiny_minicpm3_pair()
        ids = _ids()[:1]
        with torch.no_grad():
            cached = bridge.generate(
                ids, max_new_tokens=6, do_sample=False, use_past_kv_cache=True, verbose=False
            )
            uncached = bridge.generate(
                ids, max_new_tokens=6, do_sample=False, use_past_kv_cache=False, verbose=False
            )
            expected = ref.generate(ids, max_new_tokens=6, do_sample=False)
        assert torch.equal(cached, expected)
        assert torch.equal(uncached, expected)
