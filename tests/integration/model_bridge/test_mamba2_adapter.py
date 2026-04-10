"""Integration tests for the Mamba-2 architecture adapter.

Verifies wrap-don't-reimplement behavior against AntonV/mamba2-130m-hf
(a community port of Mamba-2 130M with a proper HF config):
- Forward pass matches HF exactly (bridge delegates to Mamba2Mixer.torch_forward)
- Submodule hooks fire for in_proj, conv1d, inner_norm, out_proj
- NO x_proj/dt_proj hooks (those submodules don't exist in Mamba-2)
- inner_norm (MambaRMSNormGated) fires hook_in, hook_gate, hook_out
- Parameter access via __getattr__ fallback (A_log, dt_bias, D — all [num_heads])
- Generation via is_stateful fallback delegates to hf_generate
- d_mlp == 0 assertion: in_proj output features match the 3-way split formula

Cache clone safety: Same guarantee as Phase 1 — the wrap-don't-reimplement
design keeps Mamba2Mixer opaque, so `ssm_states` is never hooked. The only
hooked tensors are projection inputs/outputs, which are per-step and not
mutated by the cache machinery.
"""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    DepthwiseConv1DBridge,
    GatedRMSNormBridge,
    SSM2MixerBridge,
    SSMBlockBridge,
)

MODEL = "AntonV/mamba2-130m-hf"


@pytest.fixture(scope="module")
def mamba2_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


class TestMamba2BridgeCreation:
    def test_block_count(self, mamba2_bridge):
        assert len(mamba2_bridge.blocks) == 24

    def test_config_flags(self, mamba2_bridge):
        assert mamba2_bridge.cfg.normalization_type == "RMS"
        assert mamba2_bridge.cfg.uses_rms_norm is True
        assert mamba2_bridge.cfg.positional_embedding_type == "none"
        assert mamba2_bridge.cfg.gated_mlp is False
        assert mamba2_bridge.cfg.is_stateful is True

    def test_ssm_config_propagated(self, mamba2_bridge):
        """Mamba-2 specific fields must be propagated and computed correctly."""
        assert mamba2_bridge.cfg.state_size == 128
        assert mamba2_bridge.cfg.conv_kernel == 4
        assert mamba2_bridge.cfg.expand == 2
        assert mamba2_bridge.cfg.n_groups == 1
        assert mamba2_bridge.cfg.chunk_size == 256
        assert mamba2_bridge.cfg.n_heads == 24
        assert mamba2_bridge.cfg.d_head == 64
        # intermediate_size is computed (Mamba2Config has no such field)
        assert mamba2_bridge.cfg.intermediate_size == 1536  # 2 * 768
        # conv_dim = intermediate + 2 * n_groups * state_size
        assert mamba2_bridge.cfg.conv_dim == 1792  # 1536 + 2*1*128

    def test_shares_ssm_block_bridge_with_mamba1(self, mamba2_bridge):
        """SSMBlockBridge is the shared block container for both Mamba variants."""
        assert isinstance(mamba2_bridge.blocks[0], SSMBlockBridge)

    def test_uses_mamba2_specific_bridges(self, mamba2_bridge):
        assert isinstance(mamba2_bridge.blocks[0].mixer, SSM2MixerBridge)
        assert isinstance(mamba2_bridge.blocks[0].mixer.conv1d, DepthwiseConv1DBridge)
        assert isinstance(mamba2_bridge.blocks[0].mixer.inner_norm, GatedRMSNormBridge)

    def test_no_x_proj_or_dt_proj(self, mamba2_bridge):
        """Mamba-2 lacks x_proj/dt_proj — they don't exist in the HF mixer either."""
        mixer = mamba2_bridge.blocks[0].mixer
        # The mixer bridge exposes whatever the HF mixer has via __getattr__.
        # Confirm HF Mamba2Mixer doesn't have these attrs.
        hf_mixer = mixer.original_component
        assert not hasattr(hf_mixer, "x_proj")
        assert not hasattr(hf_mixer, "dt_proj")

    def test_d_mlp_is_zero(self, mamba2_bridge):
        """Plan's assertion: projection_size = intermediate + conv_dim + num_heads.

        If a future HF release introduces non-zero d_mlp, this test will fail,
        signaling that the 3-way in_proj split assumption no longer holds.
        """
        mixer = mamba2_bridge.blocks[0].mixer
        actual_out_features = mixer.original_component.in_proj.out_features
        cfg = mamba2_bridge.cfg
        expected = cfg.intermediate_size + cfg.conv_dim + cfg.n_heads
        assert actual_out_features == expected, (
            f"in_proj out_features = {actual_out_features} but expected {expected} "
            f"(intermediate={cfg.intermediate_size}, conv_dim={cfg.conv_dim}, "
            f"n_heads={cfg.n_heads}). If HF has added d_mlp slots, the adapter "
            f"needs to handle a 5-way split instead of the effective 3-way."
        )


class TestMamba2ForwardPass:
    def test_forward_returns_logits(self, mamba2_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = mamba2_bridge(tokens)
        assert output.shape == (1, 4, mamba2_bridge.cfg.d_vocab)
        assert not torch.isnan(output).any()

    def test_forward_matches_hf_exactly(self, mamba2_bridge):
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        hf_model = mamba2_bridge.original_model
        with torch.no_grad():
            bridge_out = mamba2_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff == 0.0, f"Bridge vs HF max diff = {max_diff}"


class TestMamba2HookCoverage:
    @pytest.fixture(scope="class")
    def cache(self, mamba2_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(tokens)
        return cache

    def test_no_transformer_specific_hooks(self, cache):
        forbidden = (
            "hook_resid_mid",
            "hook_attn_",
            "hook_mlp_",
            "hook_q_",
            "hook_k_",
            "hook_v_",
        )
        bad = [k for k in cache if any(f in k for f in forbidden)]
        assert bad == [], f"Unexpected transformer hooks: {bad[:5]}"

    def test_no_x_proj_or_dt_proj_hooks(self, cache):
        """Mamba-2 must not expose x_proj or dt_proj hooks (they don't exist)."""
        bad = [k for k in cache if "x_proj" in k or "dt_proj" in k]
        assert bad == [], f"Unexpected Mamba-1 hooks in Mamba-2 cache: {bad[:5]}"

    def test_mixer_submodule_hooks_fire(self, cache):
        for i in [0, 12, 23]:
            for submod in ("in_proj", "conv1d", "inner_norm", "out_proj"):
                assert f"blocks.{i}.mixer.{submod}.hook_in" in cache
                assert f"blocks.{i}.mixer.{submod}.hook_out" in cache

    def test_gated_norm_fires_gate_hook(self, cache):
        """GatedRMSNormBridge has a dedicated hook_gate for the gate input."""
        for i in [0, 12, 23]:
            key = f"blocks.{i}.mixer.inner_norm.hook_gate"
            assert key in cache
            # Gate has same shape as hidden_states for the norm
            assert cache[key].shape == cache[f"blocks.{i}.mixer.inner_norm.hook_in"].shape

    def test_in_proj_shape_matches_3way_split(self, cache, mamba2_bridge):
        """in_proj.hook_out shape = intermediate + conv_dim + n_heads (d_mlp=0)."""
        cfg = mamba2_bridge.cfg
        expected_features = cfg.intermediate_size + cfg.conv_dim + cfg.n_heads
        assert cache["blocks.0.mixer.in_proj.hook_out"].shape == (1, 4, expected_features)

    def test_conv1d_sees_conv_dim_not_intermediate(self, cache, mamba2_bridge):
        """conv_dim != intermediate_size in Mamba-2; conv1d operates on conv_dim."""
        cfg = mamba2_bridge.cfg
        assert cfg.conv_dim != cfg.intermediate_size  # sanity
        assert cache["blocks.0.mixer.conv1d.hook_in"].shape == (1, cfg.conv_dim, 4)
        assert cache["blocks.0.mixer.conv1d.hook_out"].shape == (
            1,
            cfg.conv_dim,
            4 + cfg.conv_kernel - 1,
        )

    def test_inner_norm_operates_on_intermediate(self, cache, mamba2_bridge):
        """inner_norm takes [batch, seq, intermediate_size] — not conv_dim."""
        cfg = mamba2_bridge.cfg
        assert cache["blocks.0.mixer.inner_norm.hook_in"].shape == (1, 4, cfg.intermediate_size)
        assert cache["blocks.0.mixer.inner_norm.hook_out"].shape == (1, 4, cfg.intermediate_size)


class TestMamba2EffectiveAttention:
    """SSD-based effective attention: materializes M = L ⊙ (C B^T).

    This is the headline Mamba-2 interpretability feature. The reconstruction
    test is the strong invariant: applying M (with dt scaling) to the post-conv
    hidden_states must reproduce the actual SSM output captured at
    inner_norm.hook_in, within float32 precision. If the math is wrong,
    this test will fail loudly rather than silently returning the wrong matrix.
    """

    @pytest.fixture(scope="class")
    def cache_and_mixer(self, mamba2_bridge):
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(tokens)
        return cache, mamba2_bridge.blocks[0].mixer

    def test_shape(self, cache_and_mixer, mamba2_bridge):
        cache, mixer = cache_and_mixer
        M = mixer.compute_effective_attention(cache, layer_idx=0)
        assert M.shape == (1, mamba2_bridge.cfg.n_heads, 8, 8)

    def test_causal_structure(self, cache_and_mixer):
        """Upper triangle (j > i) must be exactly zero by construction."""
        cache, mixer = cache_and_mixer
        M = mixer.compute_effective_attention(cache, layer_idx=0)
        upper = torch.triu(M, diagonal=1)
        assert upper.abs().sum().item() == 0.0

    def test_finite_values(self, cache_and_mixer):
        cache, mixer = cache_and_mixer
        M = mixer.compute_effective_attention(cache, layer_idx=0)
        assert torch.isfinite(M).all()

    def test_single_token_equals_cb_bilinear(self, mamba2_bridge):
        """For seq_len=1, L degenerates to 1 and M reduces to C @ B^T per head.

        This is a concrete algebraic invariant: L[0,0] = exp(0) = 1 (empty
        product), so M[b, h, 0, 0] = sum_s C[b, 0, g, s] * B[b, 0, g, s]
        where g = h // (n_heads // n_groups). Verifying this separately from
        the reconstruction test catches bugs in the L computation without
        relying on HF's SSD path.
        """
        tokens = torch.tensor([[42]])  # single token
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(tokens)
        mixer = mamba2_bridge.blocks[0].mixer
        cfg = mamba2_bridge.cfg

        M = mixer.compute_effective_attention(cache, layer_idx=0)
        assert M.shape == (1, cfg.n_heads, 1, 1)

        # Manually extract C and B for the single position and compute C @ B
        conv_out = cache["blocks.0.mixer.conv1d.hook_out"][..., :1].float()
        conv_activated = torch.nn.functional.silu(conv_out).transpose(1, 2)
        sizes = [
            cfg.intermediate_size,
            cfg.n_groups * cfg.state_size,
            cfg.n_groups * cfg.state_size,
        ]
        _, B_flat, C_flat = conv_activated.split(sizes, dim=-1)
        B = B_flat.view(1, 1, cfg.n_groups, cfg.state_size)
        C = C_flat.view(1, 1, cfg.n_groups, cfg.state_size)

        # Expected: M[0, h, 0, 0] = sum_s C[0, 0, g, s] * B[0, 0, g, s]
        # with g = h // (n_heads // n_groups)
        heads_per_group = cfg.n_heads // cfg.n_groups
        for h in range(cfg.n_heads):
            g = h // heads_per_group
            expected = (C[0, 0, g] * B[0, 0, g]).sum().item()
            actual = M[0, h, 0, 0].item()
            assert (
                abs(actual - expected) < 1e-5
            ), f"head {h}, group {g}: M={actual:.6e} vs C@B={expected:.6e}"

    def test_reconstruction_matches_ssm_output(self, cache_and_mixer, mamba2_bridge):
        """Core invariant: M @ x + D*x reconstructs inner_norm.hook_in.

        This is the strong non-tautological test. The effective attention is
        DEFINED as the matrix such that y = (M @ x) + D_residual equals the
        SSM output. HF's chunked SSD implementation computes y via a completely
        different numerical path (cumsum over chunks, permutes, low-rank
        factorization). If our cumulative-sum-and-exp formulation is correct,
        these two paths must agree to float32 precision.
        """
        cache, mixer = cache_and_mixer
        cfg = mamba2_bridge.cfg
        seq_len = 8

        M_full = mixer.compute_effective_attention(cache, layer_idx=0, include_dt_scaling=True)

        # Extract post-conv, post-SiLU hidden_states per head
        conv_out = cache["blocks.0.mixer.conv1d.hook_out"][..., :seq_len].float()
        conv_activated = torch.nn.functional.silu(conv_out).transpose(1, 2)
        split_sizes = [
            cfg.intermediate_size,
            cfg.n_groups * cfg.state_size,
            cfg.n_groups * cfg.state_size,
        ]
        hidden_x, _, _ = conv_activated.split(split_sizes, dim=-1)
        batch = hidden_x.shape[0]
        x_per_head = hidden_x.view(batch, seq_len, cfg.n_heads, cfg.d_head)

        # Reconstruct: y_pred[b, i, h, d] = sum_j M[b, h, i, j] * x[b, j, h, d] + D[h] * x[b, i, h, d]
        D = mixer.D.float()
        y_pred = torch.einsum("bhij,bjhd->bihd", M_full, x_per_head)
        y_pred = y_pred + D[None, None, :, None] * x_per_head
        y_pred_flat = y_pred.reshape(batch, seq_len, -1)

        y_actual = cache["blocks.0.mixer.inner_norm.hook_in"].float()

        max_diff = (y_actual - y_pred_flat).abs().max().item()
        scale = y_actual.abs().max().item()
        # Relative tolerance: float32 accumulation over 8 positions × 128 state dim
        # plus softplus nonlinearity. 1e-5 relative is a comfortable margin.
        assert max_diff / max(scale, 1e-8) < 1e-5, (
            f"Reconstruction mismatch: max diff {max_diff:.2e} vs scale {scale:.2f} "
            f"(relative {max_diff/max(scale, 1e-8):.2e}). The effective attention "
            "math is inconsistent with HF's chunked SSD output."
        )

    def test_include_dt_scaling_changes_output(self, cache_and_mixer):
        """Toggling dt scaling must actually change the matrix (sanity check)."""
        cache, mixer = cache_and_mixer
        M_att = mixer.compute_effective_attention(cache, layer_idx=0, include_dt_scaling=False)
        M_full = mixer.compute_effective_attention(cache, layer_idx=0, include_dt_scaling=True)
        assert not torch.allclose(M_att, M_full), (
            "include_dt_scaling had no effect — dt is likely all ones or the " "flag was ignored."
        )

    def test_missing_cache_keys_raise(self, mamba2_bridge):
        """Calling with an empty cache should raise a clear RuntimeError."""
        mixer = mamba2_bridge.blocks[0].mixer
        with pytest.raises(RuntimeError, match="in cache"):
            mixer.compute_effective_attention({}, layer_idx=0)

    def test_different_layers_produce_different_attention(self, mamba2_bridge):
        """Each layer's M should be distinct (different weights → different attention)."""
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(tokens)
        M0 = mamba2_bridge.blocks[0].mixer.compute_effective_attention(cache, layer_idx=0)
        M5 = mamba2_bridge.blocks[5].mixer.compute_effective_attention(cache, layer_idx=5)
        assert not torch.allclose(M0, M5)


class TestMamba2ParameterAccess:
    """A_log, dt_bias, D are nn.Parameters of shape [num_heads] on the mixer."""

    def test_a_log_shape(self, mamba2_bridge):
        a_log = mamba2_bridge.blocks[0].mixer.A_log
        # Mamba-2 A_log is 1D [num_heads], unlike Mamba-1 which is [intermediate, state_size]
        assert a_log.shape == (mamba2_bridge.cfg.n_heads,)

    def test_dt_bias_shape(self, mamba2_bridge):
        """dt_bias is unique to Mamba-2 (Mamba-1 uses dt_proj.bias instead)."""
        dt_bias = mamba2_bridge.blocks[0].mixer.dt_bias
        assert dt_bias.shape == (mamba2_bridge.cfg.n_heads,)

    def test_d_shape(self, mamba2_bridge):
        d = mamba2_bridge.blocks[0].mixer.D
        assert d.shape == (mamba2_bridge.cfg.n_heads,)


class TestMamba2Generation:
    """is_stateful fallback in bridge.generate() should delegate to hf_generate."""

    def test_generation_produces_tokens(self, mamba2_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            result = mamba2_bridge.generate(tokens, max_new_tokens=3, do_sample=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1
        assert result.shape[1] == 4 + 3  # input + new tokens


class TestMamba2StopAtLayer:
    """SSMBlockBridge (shared with Mamba-1) handles stop_at_layer for Mamba-2 too."""

    def test_stop_returns_residual(self, mamba2_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(tokens)
            stopped = mamba2_bridge(tokens, stop_at_layer=7)
        expected = cache["blocks.7.hook_in"]
        assert torch.allclose(stopped, expected)
        assert stopped.shape == (1, 4, mamba2_bridge.cfg.d_model)
