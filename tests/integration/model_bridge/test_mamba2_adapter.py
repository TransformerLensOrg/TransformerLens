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

import contextlib

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

    def test_reconstruction_without_dt_scaling_matches_ssm_output(
        self, cache_and_mixer, mamba2_bridge
    ):
        """Correctness check for the include_dt_scaling=False code path.

        The existing reconstruction test verifies include_dt_scaling=True by
        computing y = M_full @ x + D*x and comparing to HF's SSD output. That
        leaves the 'attention-like' variant (M_att) with no direct correctness
        coverage — test_include_dt_scaling_changes_output only proves the two
        matrices differ, not that M_att has the right values.

        This test uses the algebraic identity M_full[i,j] = M_att[i,j] * dt[j]
        to reconstruct y via the False path:
            y[i] = sum_j M_att[i,j] * (dt[j] * x[j]) + D * x[i]
        The ground truth is still HF's chunked SSD output (inner_norm.hook_in),
        so this is non-tautological: it verifies M_att's values against HF's
        independently-computed path, exercising the code branch that skips
        the final dt multiplication.
        """
        cache, mixer = cache_and_mixer
        cfg = mamba2_bridge.cfg
        seq_len = 8

        M_att = mixer.compute_effective_attention(cache, layer_idx=0, include_dt_scaling=False)

        # Extract post-conv hidden_states per head (same as the dt-scaled test)
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

        # Extract dt the same way the mixer does: softplus(raw + bias), clamped.
        in_proj_out = cache["blocks.0.mixer.in_proj.hook_out"].float()
        dt_raw = in_proj_out[..., -cfg.n_heads :]
        dt_bias = mixer.dt_bias.float()
        dt = torch.nn.functional.softplus(dt_raw + dt_bias)
        time_step_limit = getattr(cfg, "time_step_limit", (0.0, float("inf")))
        dt = torch.clamp(dt, float(time_step_limit[0]), float(time_step_limit[1]))
        # dt shape: [batch, seq, num_heads]

        # Scale x by dt per head, then apply M_att + D skip
        x_scaled = dt[:, :, :, None] * x_per_head  # [batch, seq, num_heads, head_dim]
        D = mixer.D.float()
        y_pred = torch.einsum("bhij,bjhd->bihd", M_att, x_scaled)
        y_pred = y_pred + D[None, None, :, None] * x_per_head
        y_pred_flat = y_pred.reshape(batch, seq_len, -1)

        y_actual = cache["blocks.0.mixer.inner_norm.hook_in"].float()
        max_diff = (y_actual - y_pred_flat).abs().max().item()
        scale = y_actual.abs().max().item()
        assert max_diff / max(scale, 1e-8) < 1e-5, (
            f"M_att reconstruction mismatch: max diff {max_diff:.2e} vs scale "
            f"{scale:.2f} (relative {max_diff/max(scale, 1e-8):.2e}). The "
            "include_dt_scaling=False path produces incorrect values — the "
            "L ⊙ (C B^T) computation likely has a bug independent of the "
            "dt multiplication."
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
        from transformer_lens.ActivationCache import ActivationCache

        mixer = mamba2_bridge.blocks[0].mixer
        empty_cache = ActivationCache({}, model=mamba2_bridge)
        with pytest.raises(RuntimeError, match="in cache"):
            mixer.compute_effective_attention(empty_cache, layer_idx=0)

    def test_different_layers_produce_different_attention(self, mamba2_bridge):
        """Each layer's M should be distinct (different weights → different attention)."""
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(tokens)
        M0 = mamba2_bridge.blocks[0].mixer.compute_effective_attention(cache, layer_idx=0)
        M5 = mamba2_bridge.blocks[5].mixer.compute_effective_attention(cache, layer_idx=5)
        assert not torch.allclose(M0, M5)


class TestMamba2EffectiveAttentionDispatch:
    """cache.compute_ssm_effective_attention wraps the mixer method so callers
    don't repeat the layer index and can request all layers at once."""

    @pytest.fixture(scope="class")
    def bridge_cache(self, mamba2_bridge):
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(tokens)
        return mamba2_bridge, cache

    def test_single_layer_matches_mixer_method(self, bridge_cache):
        bridge, cache = bridge_cache
        for layer in [0, 5, 23]:
            M_helper = cache.compute_ssm_effective_attention(layer=layer)
            M_direct = bridge.blocks[layer].mixer.compute_effective_attention(
                cache, layer_idx=layer
            )
            assert torch.equal(M_helper, M_direct)

    def test_all_layers_shape(self, bridge_cache, mamba2_bridge):
        _, cache = bridge_cache
        M_all = cache.compute_ssm_effective_attention()
        assert M_all.shape == (mamba2_bridge.cfg.n_layers, 1, mamba2_bridge.cfg.n_heads, 5, 5)

    def test_all_layers_matches_individual_calls(self, bridge_cache):
        bridge, cache = bridge_cache
        M_all = cache.compute_ssm_effective_attention()
        for layer in [0, 12, 23]:
            M_single = bridge.blocks[layer].mixer.compute_effective_attention(
                cache, layer_idx=layer
            )
            assert torch.equal(M_all[layer], M_single)

    def test_include_dt_scaling_propagates(self, bridge_cache):
        _, cache = bridge_cache
        M_att = cache.compute_ssm_effective_attention(layer=0, include_dt_scaling=False)
        M_full = cache.compute_ssm_effective_attention(layer=0, include_dt_scaling=True)
        assert not torch.allclose(M_att, M_full)

    def test_deprecated_module_wrapper_warns_and_delegates(self, bridge_cache):
        from transformer_lens.model_bridge.supported_architectures.mamba2 import (
            compute_effective_attention,
        )

        bridge, cache = bridge_cache
        with pytest.warns(DeprecationWarning):
            M_dep = compute_effective_attention(bridge, cache, layer=0)
        assert torch.equal(M_dep, cache.compute_ssm_effective_attention(layer=0))


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


class TestMamba2StatefulGeneration:
    """Phase 3: bridge.generate() runs a dedicated stateful loop with
    Mamba2Cache. Tokens match HF's native generate() exactly, and projection
    hooks fire on every step so interventions are possible.
    """

    def test_generation_produces_tokens(self, mamba2_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            result = mamba2_bridge.generate(tokens, max_new_tokens=3, do_sample=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 7)  # input (4) + new tokens (3)

    def test_greedy_matches_hf_exactly(self, mamba2_bridge):
        """Bridge greedy generation must match HF native generate() bit-for-bit."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = mamba2_bridge.generate(tokens, max_new_tokens=5, do_sample=False)
            hf_out = mamba2_bridge.original_model.generate(
                tokens, max_new_tokens=5, do_sample=False, pad_token_id=0
            )
        assert torch.equal(
            bridge_out, hf_out
        ), f"Mismatch: bridge={bridge_out.tolist()} vs HF={hf_out.tolist()}"

    def test_hooks_fire_during_generation(self, mamba2_bridge):
        """Projection hooks fire on every generation step.

        Same caveat as Mamba-1: conv1d fires only on the prefill step because
        HF's torch_forward bypasses `self.conv1d(...)` during decode.
        """
        tokens = torch.tensor([[1, 2, 3, 4]])
        call_counts: dict[str, int] = {"in_proj": 0, "conv1d": 0, "out_proj": 0}

        def make_counter(name: str):
            def hook_fn(t, hook):
                call_counts[name] += 1
                return t

            return hook_fn

        try:
            mamba2_bridge.blocks[0].mixer.in_proj.hook_out.add_hook(make_counter("in_proj"))
            mamba2_bridge.blocks[0].mixer.conv1d.hook_out.add_hook(make_counter("conv1d"))
            mamba2_bridge.blocks[0].mixer.out_proj.hook_out.add_hook(make_counter("out_proj"))
            with torch.no_grad():
                _ = mamba2_bridge.generate(tokens, max_new_tokens=3, do_sample=False)
        finally:
            mamba2_bridge.blocks[0].mixer.in_proj.hook_out.remove_hooks()
            mamba2_bridge.blocks[0].mixer.conv1d.hook_out.remove_hooks()
            mamba2_bridge.blocks[0].mixer.out_proj.hook_out.remove_hooks()

        assert call_counts["in_proj"] == 3
        assert call_counts["out_proj"] == 3
        # conv1d only fires during prefill (HF bypasses forward on decode)
        assert call_counts["conv1d"] == 1

    def test_hook_modification_affects_generation(self, mamba2_bridge):
        """Mutating activations mid-generation must change the generated tokens."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            baseline = mamba2_bridge.generate(tokens, max_new_tokens=3, do_sample=False)

        def scramble(t, hook):
            return t * 3.0

        try:
            mamba2_bridge.blocks[12].mixer.out_proj.hook_out.add_hook(scramble)
            with torch.no_grad():
                perturbed = mamba2_bridge.generate(tokens, max_new_tokens=3, do_sample=False)
        finally:
            mamba2_bridge.blocks[12].mixer.out_proj.hook_out.remove_hooks()

        assert not torch.equal(baseline, perturbed), (
            "Mid-layer mutation had no effect on generation — the stateful "
            "loop may be bypassing the bridge's forward() path."
        )


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


class TestMamba2SSMState:
    """compute_ssm_state reconstructs the recurrent state S read-only from cache.

    S_t = dA_t · S_{t-1} + dt_t · (x_t ⊗ B_t). Two independent checks: a naive
    fp64 eager recurrence (validates the vectorized cumsum/einsum), and the
    y = C·S + D·x reconstruction against HF's chunked SSD output (validates the
    whole reconstruction against a numerically-independent path).
    """

    SEQ_LEN = 8

    @pytest.fixture(scope="class")
    def cache(self, mamba2_bridge):
        tokens = torch.arange(1, self.SEQ_LEN + 1).unsqueeze(0)
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(tokens)
        return cache

    @staticmethod
    def _ssd_inputs(cache, mixer, layer, seq_len):
        """Re-extract dt, per-head x/B/C, A, D from cache — independent of the impl."""
        oc = mixer.original_component
        nh, hd, ns, ng = oc.num_heads, oc.head_dim, oc.ssm_state_size, oc.n_groups
        in_proj = cache[f"blocks.{layer}.mixer.in_proj.hook_out"].float()
        conv = cache[f"blocks.{layer}.mixer.conv1d.hook_out"][..., :seq_len].float()
        dt = torch.nn.functional.softplus(in_proj[..., -nh:] + mixer.dt_bias.float())
        dt = torch.clamp(dt, float(oc.time_step_limit[0]), float(oc.time_step_limit[1]))
        conv_act = torch.nn.functional.silu(conv).transpose(1, 2)
        x_flat, B_flat, C_flat = conv_act.split([oc.intermediate_size, ng * ns, ng * ns], dim=-1)
        x = x_flat.view(1, seq_len, nh, hd)
        B_h = B_flat.view(1, seq_len, ng, ns).repeat_interleave(nh // ng, dim=2)
        C_h = C_flat.view(1, seq_len, ng, ns).repeat_interleave(nh // ng, dim=2)
        A = -torch.exp(mixer.A_log.float())
        return dt, x, B_h, C_h, A, mixer.D.float()

    def test_shape(self, cache, mamba2_bridge):
        mixer = mamba2_bridge.blocks[0].mixer
        oc = mixer.original_component
        S = mixer.compute_ssm_state(cache, layer_idx=0)
        assert S.shape == (1, oc.num_heads, self.SEQ_LEN, oc.head_dim, oc.ssm_state_size)
        assert torch.isfinite(S).all()

    def test_matches_fp64_eager_recurrence(self, cache, mamba2_bridge):
        """The vectorized state must match a naive fp64 step-by-step recurrence."""
        mixer = mamba2_bridge.blocks[0].mixer
        S = mixer.compute_ssm_state(cache, layer_idx=0)

        dt, x, B_h, _, A, _ = self._ssd_inputs(cache, mixer, 0, self.SEQ_LEN)
        dt, x, B_h, A = dt.double(), x.double(), B_h.double(), A.double()
        b, nh = S.shape[0], S.shape[1]
        hd, ns = S.shape[3], S.shape[4]
        state = torch.zeros(b, nh, hd, ns, dtype=torch.float64)
        ref = torch.zeros(b, nh, self.SEQ_LEN, hd, ns, dtype=torch.float64)
        for t in range(self.SEQ_LEN):
            dA = torch.exp(dt[:, t, :] * A[None, :])  # [b, nh]
            write = dt[:, t, :, None, None] * x[:, t, :, :, None] * B_h[:, t, :, None, :]
            state = dA[:, :, None, None] * state + write
            ref[:, :, t] = state

        max_diff = (S.double() - ref).abs().max().item()
        scale = max(ref.abs().max().item(), 1e-8)
        assert (
            max_diff / scale < 1e-5
        ), f"S vs fp64 eager recurrence rel diff {max_diff / scale:.2e}"

    def test_reconstructs_ssm_output(self, cache, mamba2_bridge):
        """y = C·S + D·x must reconstruct HF's SSM output (inner_norm.hook_in)."""
        mixer = mamba2_bridge.blocks[0].mixer
        S = mixer.compute_ssm_state(cache, layer_idx=0)
        _, x, _, C_h, _, D = self._ssd_inputs(cache, mixer, 0, self.SEQ_LEN)

        y = torch.einsum("bthn,bhtpn->bthp", C_h, S) + D[None, None, :, None] * x
        y_pred = y.reshape(1, self.SEQ_LEN, -1)
        y_actual = cache["blocks.0.mixer.inner_norm.hook_in"].float()
        max_diff = (y_actual - y_pred).abs().max().item()
        scale = max(y_actual.abs().max().item(), 1e-8)
        assert max_diff / scale < 1e-5, (
            f"y=C·S+D·x reconstruction rel diff {max_diff / scale:.2e}; the state "
            "is inconsistent with HF's independently-computed SSM output."
        )

    def test_time_step_matches_full_slice(self, cache, mamba2_bridge):
        mixer = mamba2_bridge.blocks[0].mixer
        S = mixer.compute_ssm_state(cache, layer_idx=0)
        S_t = mixer.compute_ssm_state(cache, layer_idx=0, time_step=3)
        assert S_t.shape == (S.shape[0], S.shape[1], S.shape[3], S.shape[4])
        # The full and single-step einsums differ only in fp reduction order.
        assert torch.allclose(S_t, S[:, :, 3], rtol=0, atol=torch.finfo(S.dtype).eps)

    def test_cache_method_matches_mixer(self, cache, mamba2_bridge):
        S_mixer = mamba2_bridge.blocks[0].mixer.compute_ssm_state(cache, layer_idx=0)
        assert torch.equal(cache.compute_ssm_state(layer=0), S_mixer)

    def test_cache_all_layers_stacks(self, cache, mamba2_bridge):
        # Pure Mamba-2 is homogeneous, so all-layers stacks along a new dim 0.
        S_all = cache.compute_ssm_state()
        assert torch.is_tensor(S_all)
        assert S_all.shape[0] == mamba2_bridge.cfg.n_layers

    def test_deprecated_module_wrapper_warns_and_delegates(self, cache, mamba2_bridge):
        from transformer_lens.model_bridge.supported_architectures.mamba2 import (
            compute_ssm_state,
        )

        S_mixer = mamba2_bridge.blocks[0].mixer.compute_ssm_state(cache, layer_idx=0)
        with pytest.warns(DeprecationWarning):
            S_dep = compute_ssm_state(mamba2_bridge, cache, layer=0)
        assert torch.equal(S_dep, S_mixer)

    def test_raises_on_empty_cache(self, mamba2_bridge):
        from transformer_lens.ActivationCache import ActivationCache

        empty = ActivationCache({}, model=mamba2_bridge)
        with pytest.raises(RuntimeError, match="in cache"):
            mamba2_bridge.blocks[0].mixer.compute_ssm_state(empty, layer_idx=0)


@contextlib.contextmanager
def _eager_scan(bridge):
    """Enable the opt-in eager-scan intervention path on every mixer, then reset."""
    for blk in bridge.blocks:
        blk.mixer.eager_scan = True
    try:
        yield bridge
    finally:
        for blk in bridge.blocks:
            blk.mixer.eager_scan = False


class TestMamba2EagerScanIntervention:
    """Phase 4: opt-in eager-scan path exposes hook_ssm_write / hook_ssm_state for
    interventions (write-knockout, state-patch) that propagate to logits, while the
    default run_with_cache path is untouched. Eager scan requires use_cache=False
    (so cache_params is None — prefill only)."""

    TOKENS = torch.tensor([[1, 2, 3, 4, 5, 6]])

    def test_default_path_has_no_eager_hooks(self, mamba2_bridge):
        with torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(self.TOKENS)
        assert "blocks.0.mixer.hook_ssm_write" not in cache
        assert "blocks.0.mixer.hook_ssm_state" not in cache

    def test_eager_requires_use_cache_false(self, mamba2_bridge):
        """With eager on but a default (stateful) forward, cache_params is present
        so the eager path is skipped and the hooks do not fire."""
        with _eager_scan(mamba2_bridge), torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(self.TOKENS)  # default use_cache
        assert "blocks.0.mixer.hook_ssm_write" not in cache

    def test_eager_scan_matches_fused_logits(self, mamba2_bridge):
        with torch.no_grad():
            fused = mamba2_bridge(self.TOKENS)
        with _eager_scan(mamba2_bridge), torch.no_grad():
            eager = mamba2_bridge(self.TOKENS, use_cache=False)
        rel = (eager - fused).abs().max().item() / max(fused.abs().max().item(), 1e-8)
        assert rel < 1e-4, f"eager scan vs fused kernel rel diff {rel:.2e}"

    def test_eager_scan_matches_fp64_step_reference(self, mamba2_bridge):
        """The eager scan's write/state must match an independent fp64 step recurrence.

        ``test_eager_scan_matches_fused_logits`` pins the eager scan to HF's fused
        kernel — but a shared discretization error would pass both. This pins the
        eager scan's own ``hook_ssm_write`` / ``hook_ssm_state`` to a naive fp64
        step-by-step recurrence built independently from the cached in_proj/conv1d
        outputs (``S_t = dA_t·S_{t-1} + dt_t·(x_t⊗B_t)``), so it is a genuine
        correctness gate on the eager scan, not just kernel agreement.
        """
        seq = self.TOKENS.shape[1]
        with _eager_scan(mamba2_bridge), torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(self.TOKENS, use_cache=False)
        mixer = mamba2_bridge.blocks[0].mixer
        write_hook = cache["blocks.0.mixer.hook_ssm_write"].double()
        state_hook = cache["blocks.0.mixer.hook_ssm_state"].double()

        dt, x, B_h, _, A, _ = TestMamba2SSMState._ssd_inputs(cache, mixer, 0, seq)
        dt, x, B_h, A = dt.double(), x.double(), B_h.double(), A.double()
        b, nh, hd, ns = (
            state_hook.shape[0],
            state_hook.shape[2],
            state_hook.shape[3],
            state_hook.shape[4],
        )

        state = torch.zeros(b, nh, hd, ns, dtype=torch.float64)
        ref_write = torch.zeros(b, seq, nh, hd, ns, dtype=torch.float64)
        ref_state = torch.zeros(b, seq, nh, hd, ns, dtype=torch.float64)
        for t in range(seq):
            dA = torch.exp(dt[:, t, :] * A[None, :])  # [b, nh]
            write = dt[:, t, :, None, None] * x[:, t, :, :, None] * B_h[:, t, :, None, :]
            ref_write[:, t] = write
            state = dA[:, :, None, None] * state + write
            ref_state[:, t] = state

        for name, got, ref in (("write", write_hook, ref_write), ("state", state_hook, ref_state)):
            rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-8)
            assert rel < 1e-5, f"eager {name} vs fp64 step reference rel diff {rel:.2e}"

    def test_eager_scan_matches_fused_padded_batch(self, mamba2_bridge):
        """Padded batch: the eager path must mirror HF's padding mask (applied both
        before in_proj and after the conv) to stay at parity."""
        ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        with torch.no_grad():
            fused = mamba2_bridge(ids, attention_mask=mask)
        with _eager_scan(mamba2_bridge), torch.no_grad():
            eager = mamba2_bridge(ids, attention_mask=mask, use_cache=False)
        rel = (eager - fused).abs().max().item() / max(fused.abs().max().item(), 1e-8)
        assert rel < 1e-4, f"padded-batch eager vs fused rel diff {rel:.2e}"

    def test_eager_hooks_fire_with_use_cache_false(self, mamba2_bridge):
        with _eager_scan(mamba2_bridge), torch.no_grad():
            _, cache = mamba2_bridge.run_with_cache(self.TOKENS, use_cache=False)
        oc = mamba2_bridge.blocks[0].mixer.original_component
        seq = self.TOKENS.shape[1]
        expected = (1, seq, oc.num_heads, oc.head_dim, oc.ssm_state_size)
        assert cache["blocks.0.mixer.hook_ssm_write"].shape == expected
        assert cache["blocks.0.mixer.hook_ssm_state"].shape == expected

    def test_write_knockout_changes_logits(self, mamba2_bridge):
        with _eager_scan(mamba2_bridge), torch.no_grad():
            base = mamba2_bridge(self.TOKENS, use_cache=False)

            def knockout(writes, hook):
                writes = writes.clone()
                writes[:, 2] = 0.0  # input position 2 writes nothing — column knockout
                return writes

            patched = mamba2_bridge.run_with_hooks(
                self.TOKENS,
                use_cache=False,
                fwd_hooks=[("blocks.0.mixer.hook_ssm_write", knockout)],
            )
        assert (patched - base).abs().max().item() > 1e-6

    def test_state_patch_changes_logits(self, mamba2_bridge):
        with _eager_scan(mamba2_bridge), torch.no_grad():
            base = mamba2_bridge(self.TOKENS, use_cache=False)

            def patch(state, hook):
                state = state.clone()
                state[:, 3] = 0.0
                return state

            patched = mamba2_bridge.run_with_hooks(
                self.TOKENS,
                use_cache=False,
                fwd_hooks=[("blocks.0.mixer.hook_ssm_state", patch)],
            )
        assert (patched - base).abs().max().item() > 1e-6

    def test_disabling_restores_fused_path(self, mamba2_bridge):
        with torch.no_grad():
            fused_before = mamba2_bridge(self.TOKENS)
        with _eager_scan(mamba2_bridge), torch.no_grad():
            mamba2_bridge(self.TOKENS, use_cache=False)
        with torch.no_grad():
            fused_after = mamba2_bridge(self.TOKENS)
        assert torch.equal(fused_before, fused_after)
