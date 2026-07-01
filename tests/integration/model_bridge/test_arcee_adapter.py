"""Integration tests for the Arcee architecture adapter.

Loads the tiny-random ArceeForCausalLM checkpoint and asserts logit parity with
HF plus Arcee-specific structure: an ungated ReLU^2 MLP whose post-activation
neurons are inspectable via ``blocks.{i}.mlp.hook_post``.

Runs at fp32 + eager attention (the adapter forces eager so output_attentions —
and therefore hook_attn_scores / hook_pattern — work).
"""
import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "optimum-intel-internal-testing/tiny-random-ArceeForCausalLM"


@pytest.fixture(scope="module")
def arcee_bridge():
    # The adapter forces eager attention via cfg.attn_implementation, so we do not
    # (and cannot) pass attn_implementation to boot_transformers.
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


class TestArceeBridgeCreation:
    def test_block_count(self, arcee_bridge):
        assert len(arcee_bridge.blocks) == 2

    def test_has_core_components(self, arcee_bridge):
        assert hasattr(arcee_bridge, "embed")
        assert hasattr(arcee_bridge, "unembed")
        assert hasattr(arcee_bridge, "ln_final")

    def test_config_flags(self, arcee_bridge):
        cfg = arcee_bridge.cfg
        assert cfg.normalization_type == "RMS"
        assert cfg.positional_embedding_type == "rotary"
        assert cfg.gated_mlp is False  # ungated ReLU^2 MLP
        assert cfg.act_fn == "relu2"

    def test_gqa_config(self, arcee_bridge):
        """tiny-random config: 4 attention heads, 2 KV heads."""
        assert arcee_bridge.cfg.n_key_value_heads == 2


class TestArceeForwardEquivalence:
    def test_forward_returns_logits(self, arcee_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = arcee_bridge(tokens)
        assert output.shape[0] == 1
        assert output.shape[1] == 4
        assert not torch.isnan(output).any()

    def test_forward_matches_hf(self, arcee_bridge):
        """Bridge delegates to HF native forward — output should match at fp32."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        hf_model = arcee_bridge.original_model
        with torch.no_grad():
            bridge_out = arcee_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"


class TestArceeHFDelegation:
    def test_mlp_delegates_to_hf_relu2_mlp(self, arcee_bridge):
        """The ungated MLP bridge wraps HF's ArceeMLP, whose activation is HF's
        ReLUSquaredActivation (ReLU^2) — i.e. the bridge delegates the MLP forward
        to the real squared-ReLU implementation rather than a gated substitute."""
        hf_mlp = arcee_bridge.blocks[0].mlp.original_component
        assert type(hf_mlp).__name__ == "ArceeMLP"
        assert type(hf_mlp.act_fn).__name__ == "ReLUSquaredActivation"


class TestArceeHookShapes:
    def test_attn_and_mlp_hooks_fire(self, arcee_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = arcee_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.attn.hook_in" in cache
            assert f"blocks.{i}.attn.hook_out" in cache
            assert f"blocks.{i}.mlp.hook_in" in cache
            assert f"blocks.{i}.mlp.hook_out" in cache

    def test_residual_hooks_fire(self, arcee_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = arcee_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.hook_resid_pre" in cache
            assert f"blocks.{i}.hook_resid_post" in cache


class TestArceeReLU2MLP:
    """Arcee's distinguishing quirk: the post-activation MLP neurons (ReLU^2
    output, input to down_proj) are exposed via hook_post and are non-negative
    (ReLU^2 >= 0), which is the sparse-activation structure the issue targets."""

    def test_mlp_post_activation_hook_fires(self, arcee_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = arcee_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.mlp.hook_post" in cache

    def test_mlp_post_activation_shape(self, arcee_bridge):
        """hook_post has the intermediate (d_mlp) width."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = arcee_bridge.run_with_cache(tokens)
        post = cache["blocks.0.mlp.hook_post"]
        assert post.shape[0] == 1
        assert post.shape[1] == 4
        assert post.shape[-1] == arcee_bridge.cfg.d_mlp

    def test_mlp_post_activation_nonnegative(self, arcee_bridge):
        """ReLU^2 output is always >= 0."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = arcee_bridge.run_with_cache(tokens)
        post = cache["blocks.0.mlp.hook_post"]
        assert (post >= 0).all(), "ReLU^2 activations must be non-negative"
