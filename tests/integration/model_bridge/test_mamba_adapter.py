"""Integration tests for the Mamba-1 architecture adapter.

Verifies wrap-don't-reimplement behavior against state-spaces/mamba-130m-hf:
- Forward pass matches HF exactly (bridge delegates to HF slow_forward)
- Submodule hooks fire with expected shapes for in_proj, conv1d, x_proj,
  dt_proj, out_proj
- SSM blocks correctly exclude transformer-specific hook_resid_mid
- Parameter access via __getattr__ fallback (A_log, D)

Cache-clone safety: hooks never expose HF's in-place-mutated `ssm_states`; any
future per-step SSM-state hook MUST `.clone()` captured state.
"""

import contextlib

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    DepthwiseConv1DBridge,
    SSMBlockBridge,
    SSMMixerBridge,
)

MODEL = "state-spaces/mamba-130m-hf"


@pytest.fixture(scope="module")
def mamba_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


class TestMambaBridgeCreation:
    def test_block_count(self, mamba_bridge):
        assert len(mamba_bridge.blocks) == 24

    def test_config_flags(self, mamba_bridge):
        assert mamba_bridge.cfg.normalization_type == "RMS"
        assert mamba_bridge.cfg.uses_rms_norm is True
        assert mamba_bridge.cfg.positional_embedding_type == "none"
        assert mamba_bridge.cfg.gated_mlp is False
        assert mamba_bridge.cfg.is_stateful is True

    def test_ssm_config_propagated(self, mamba_bridge):
        """Mamba-specific SSM config must be propagated from HF config."""
        assert mamba_bridge.cfg.state_size == 16
        assert mamba_bridge.cfg.conv_kernel == 4
        assert mamba_bridge.cfg.expand == 2
        assert mamba_bridge.cfg.intermediate_size == 1536

    def test_block_uses_ssm_bridge(self, mamba_bridge):
        assert isinstance(mamba_bridge.blocks[0], SSMBlockBridge)
        assert isinstance(mamba_bridge.blocks[0].mixer, SSMMixerBridge)
        assert isinstance(mamba_bridge.blocks[0].mixer.conv1d, DepthwiseConv1DBridge)

    def test_has_core_components(self, mamba_bridge):
        assert hasattr(mamba_bridge, "embed")
        assert hasattr(mamba_bridge, "unembed")
        assert hasattr(mamba_bridge, "ln_final")


class TestMambaForwardPass:
    def test_forward_returns_logits(self, mamba_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = mamba_bridge(tokens)
        assert output.shape == (1, 4, mamba_bridge.cfg.d_vocab)
        assert not torch.isnan(output).any()

    def test_forward_matches_hf_exactly(self, mamba_bridge):
        """Wrap-don't-reimplement: bridge output must be bitwise-identical to HF."""
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        hf_model = mamba_bridge.original_model
        with torch.no_grad():
            bridge_out = mamba_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff == 0.0, f"Bridge vs HF max diff = {max_diff}"


class TestMambaHookCoverage:
    @pytest.fixture(scope="class")
    def cache(self, mamba_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            _, cache = mamba_bridge.run_with_cache(tokens)
        return cache

    def test_no_transformer_specific_hooks(self, cache):
        """SSM blocks must not expose attn/mlp/resid_mid hooks."""
        keys = list(cache.keys())
        forbidden = ("hook_resid_mid", "hook_attn_", "hook_mlp_", "hook_q_", "hook_k_", "hook_v_")
        bad = [k for k in keys if any(f in k for f in forbidden)]
        assert bad == [], f"Unexpected transformer hooks in SSM cache: {bad[:5]}"

    def test_mixer_submodule_hooks_fire(self, cache):
        """All projection hooks must fire during the opaque HF forward."""
        for i in [0, 12, 23]:
            for submod in ("in_proj", "conv1d", "x_proj", "dt_proj", "out_proj"):
                assert f"blocks.{i}.mixer.{submod}.hook_in" in cache
                assert f"blocks.{i}.mixer.{submod}.hook_out" in cache

    def test_projection_shapes(self, cache, mamba_bridge):
        """Hook tensor shapes match the expected projection shapes."""
        d_model = mamba_bridge.cfg.d_model  # 768
        intermediate = mamba_bridge.cfg.intermediate_size  # 1536
        conv_kernel = mamba_bridge.cfg.conv_kernel  # 4
        seq = 4

        assert cache["blocks.0.mixer.in_proj.hook_in"].shape == (1, seq, d_model)
        # 2 * intermediate_size (hidden + gate)
        assert cache["blocks.0.mixer.in_proj.hook_out"].shape == (1, seq, 2 * intermediate)
        # Conv1d sees channel-first tensors
        assert cache["blocks.0.mixer.conv1d.hook_in"].shape == (1, intermediate, seq)
        # Output is pre-trim: seq + conv_kernel - 1
        assert cache["blocks.0.mixer.conv1d.hook_out"].shape == (
            1,
            intermediate,
            seq + conv_kernel - 1,
        )
        # x_proj outputs [dt_rank + 2*state_size]
        dt_rank = mamba_bridge.cfg.time_step_rank
        state_size = mamba_bridge.cfg.state_size
        assert cache["blocks.0.mixer.x_proj.hook_out"].shape == (
            1,
            seq,
            dt_rank + 2 * state_size,
        )
        assert cache["blocks.0.mixer.dt_proj.hook_in"].shape == (1, seq, dt_rank)
        assert cache["blocks.0.mixer.dt_proj.hook_out"].shape == (1, seq, intermediate)
        assert cache["blocks.0.mixer.out_proj.hook_out"].shape == (1, seq, d_model)

    def test_residual_hooks_present(self, cache):
        for i in [0, 23]:
            assert f"blocks.{i}.hook_in" in cache
            assert f"blocks.{i}.hook_out" in cache


class TestMambaParameterAccess:
    """A_log and D are nn.Parameters on the HF mixer — accessible via __getattr__."""

    def test_a_log_shape(self, mamba_bridge):
        a_log = mamba_bridge.blocks[0].mixer.A_log
        # [intermediate_size, state_size] for Mamba-1
        assert a_log.shape == (1536, 16)

    def test_d_shape(self, mamba_bridge):
        d = mamba_bridge.blocks[0].mixer.D
        assert d.shape == (1536,)


class TestMambaHookMutation:
    """run_with_hooks can modify activations and affect downstream output."""

    def test_zero_ablation_residual_stream(self, mamba_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            baseline = mamba_bridge(tokens)

        def zero_resid(t, hook):
            return torch.zeros_like(t)

        with torch.no_grad():
            ablated = mamba_bridge.run_with_hooks(
                tokens,
                fwd_hooks=[("blocks.12.hook_in", zero_resid)],
            )
        # Zeroing a mid-layer residual stream should change the output
        assert not torch.allclose(baseline, ablated)


class TestMambaStopAtLayer:
    """SSMBlockBridge reimplements _check_stop_at_layer — verify it matches
    the residual stream at the target layer and doesn't corrupt state.
    """

    @pytest.fixture(scope="class")
    def cached_resids(self, mamba_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            _, cache = mamba_bridge.run_with_cache(tokens)
        return tokens, cache

    def test_stop_returns_residual_at_target_layer(self, mamba_bridge, cached_resids):
        """stop_at_layer=N should return the residual stream entering block N."""
        tokens, cache = cached_resids
        for stop in [0, 5, 12, 23]:
            with torch.no_grad():
                stopped = mamba_bridge(tokens, stop_at_layer=stop)
            expected = cache[f"blocks.{stop}.hook_in"]
            assert torch.allclose(
                stopped, expected
            ), f"stop_at_layer={stop}: max diff = {(stopped - expected).abs().max().item()}"

    def test_output_shape_is_residual_not_logits(self, mamba_bridge):
        """When stopped mid-network, output is [batch, seq, d_model], not logits."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            stopped = mamba_bridge(tokens, stop_at_layer=5)
        assert stopped.shape == (1, 4, mamba_bridge.cfg.d_model)
        assert stopped.shape[-1] != mamba_bridge.cfg.d_vocab

    def test_full_forward_works_after_stop(self, mamba_bridge):
        """Per-call _stop_at_layer_idx must be cleared so later calls run full forward."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            _ = mamba_bridge(tokens, stop_at_layer=5)
            full = mamba_bridge(tokens)
        assert full.shape == (1, 4, mamba_bridge.cfg.d_vocab)
        assert not torch.isnan(full).any()


class TestMambaStatefulGeneration:
    """bridge.generate() runs a proper stateful loop instead of
    delegating to hf_generate(). This gives hook integration during generation.
    """

    def test_greedy_matches_hf_exactly(self, mamba_bridge):
        """Bridge greedy generation must produce the same tokens as HF greedy.

        The bridge runs its own loop with cache_params/cache_position, but the
        underlying numerical path is HF's slow_forward — so the tokens should
        match HF's native generate() bit-for-bit.
        """
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = mamba_bridge.generate(tokens, max_new_tokens=5, do_sample=False)
            hf_out = mamba_bridge.original_model.generate(
                tokens, max_new_tokens=5, do_sample=False, pad_token_id=0
            )
        assert torch.equal(
            bridge_out, hf_out
        ), f"Mismatch: bridge={bridge_out.tolist()} vs HF={hf_out.tolist()}"

    def test_hooks_fire_during_generation(self, mamba_bridge):
        """Projection hooks must fire on each generation forward pass.

        Note: conv1d fires only during the prefill step because HF's
        slow_forward bypasses `self.conv1d(...)` on decode steps (it accesses
        `self.conv1d.weight` directly). in_proj, x_proj, dt_proj, out_proj
        all fire on every step.
        """
        tokens = torch.tensor([[1, 2, 3, 4]])
        call_counts: dict[str, int] = {"in_proj": 0, "conv1d": 0, "out_proj": 0}

        def make_counter(name: str):
            def hook_fn(t, hook):
                call_counts[name] += 1
                return t

            return hook_fn

        try:
            mamba_bridge.blocks[0].mixer.in_proj.hook_out.add_hook(make_counter("in_proj"))
            mamba_bridge.blocks[0].mixer.conv1d.hook_out.add_hook(make_counter("conv1d"))
            mamba_bridge.blocks[0].mixer.out_proj.hook_out.add_hook(make_counter("out_proj"))
            with torch.no_grad():
                _ = mamba_bridge.generate(tokens, max_new_tokens=3, do_sample=False)
        finally:
            mamba_bridge.blocks[0].mixer.in_proj.hook_out.remove_hooks()
            mamba_bridge.blocks[0].mixer.conv1d.hook_out.remove_hooks()
            mamba_bridge.blocks[0].mixer.out_proj.hook_out.remove_hooks()

        # 3 forward passes total: 1 prefill + 2 decode steps (max_new_tokens=3
        # produces 3 tokens via 1 prefill that yields the first new token, plus
        # 2 decode steps for the remaining 2 new tokens).
        assert call_counts["in_proj"] == 3
        assert call_counts["out_proj"] == 3
        # conv1d fires only on prefill — decode bypasses self.conv1d(...)
        assert call_counts["conv1d"] == 1

    @pytest.mark.parametrize("prompt_len", [1, 2, 3, 4, 5, 8, 20])
    def test_greedy_matches_hf_across_prompt_lengths(self, mamba_bridge, prompt_len):
        """Bridge greedy must match HF for any prompt length, including short
        prompts.

        Regression: the initial Phase 3 implementation used
        cache_position=[conv_kernel + step] on decode which worked for
        prompt_len >= conv_kernel via HF's clamp to conv_kernel - 1, but
        silently wrote to the wrong buffer slot for short prompts
        (prompt_len < conv_kernel). The fix uses the actual sequence position
        (prompt_len + step - 1), matching HF's own generate loop. This test
        parametrizes over prompt lengths that cross the conv_kernel boundary
        to catch any regression.
        """
        tokens = torch.arange(1, prompt_len + 1).unsqueeze(0)
        with torch.no_grad():
            bridge_out = mamba_bridge.generate(tokens, max_new_tokens=6, do_sample=False)
            hf_out = mamba_bridge.original_model.generate(
                tokens, max_new_tokens=6, do_sample=False, pad_token_id=0
            )
        assert torch.equal(bridge_out, hf_out), (
            f"prompt_len={prompt_len}: bridge={bridge_out.tolist()} vs " f"HF={hf_out.tolist()}"
        )

    def test_hook_modification_affects_generation(self, mamba_bridge):
        """Mutating activations mid-generation must change the generated tokens.

        If the hook path worked for run_with_hooks but not bridge.generate(),
        this test would fail — proving the stateful loop routes through
        self.forward() and not directly to HF internals.

        Note: state-spaces/mamba-130m-hf ships with a randomly initialized
        lm_head (weight missing from checkpoint), so small perturbations to
        mid-network activations are easily flattened by the random projection.
        We use a large additive perturbation on an early layer's residual
        stream to guarantee the argmax flips.
        """
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            baseline = mamba_bridge.generate(tokens, max_new_tokens=3, do_sample=False)

        def perturb(t, hook):
            return t + 50.0  # additive perturbation survives all downstream ops

        try:
            mamba_bridge.blocks[2].hook_in.add_hook(perturb)
            with torch.no_grad():
                perturbed = mamba_bridge.generate(tokens, max_new_tokens=3, do_sample=False)
        finally:
            mamba_bridge.blocks[2].hook_in.remove_hooks()

        assert not torch.equal(baseline, perturbed), (
            "Mid-layer mutation had no effect on generation — the stateful "
            "loop may be bypassing the bridge's forward() path."
        )


class TestMamba1EffectiveAttention:
    """Mamba-1's S6 scan as per-channel effective attention M.

    Two checks: M·x + D·x matches an fp64 eager recurrence (matrix-algebra
    correctness), and the full mixer output reconstructed through gate + out_proj
    matches HF's cached hook_out (ties M to HF's numerically-independent forward).
    """

    SEQ_LEN = 6

    @pytest.fixture(scope="class")
    def cache(self, mamba_bridge):
        tokens = torch.arange(1, self.SEQ_LEN + 1).unsqueeze(0)
        with torch.no_grad():
            _, cache = mamba_bridge.run_with_cache(tokens)
        return cache

    @staticmethod
    def _inputs(cache, mixer, layer, seq_len):
        """Re-extract x (post-conv SiLU), B, C, dt, A, D, gate — independent of the impl."""
        import torch.nn.functional as F

        oc = mixer.original_component
        d_inner, state, dt_rank = oc.intermediate_size, oc.ssm_state_size, oc.time_step_rank
        conv = cache[f"blocks.{layer}.mixer.conv1d.hook_out"][..., :seq_len].float()
        x_proj = cache[f"blocks.{layer}.mixer.x_proj.hook_out"].float()
        dt_proj = cache[f"blocks.{layer}.mixer.dt_proj.hook_out"].float()
        in_proj = cache[f"blocks.{layer}.mixer.in_proj.hook_out"].float()
        x = oc.act(conv)  # [b, d_inner, seq]
        _ts, B, C = x_proj.split([dt_rank, state, state], dim=-1)
        dt = F.softplus(dt_proj).transpose(1, 2)  # [b, d_inner, seq]
        A = -torch.exp(mixer.A_log.float())
        gate = in_proj.transpose(1, 2)[:, d_inner:, :]  # [b, d_inner, seq]
        return x, B, C, dt, A, mixer.D.float(), gate

    def test_shape(self, cache, mamba_bridge):
        mixer = mamba_bridge.blocks[0].mixer
        oc = mixer.original_component
        M = mixer.compute_effective_attention(cache, layer_idx=0)
        assert M.shape == (1, oc.intermediate_size, self.SEQ_LEN, self.SEQ_LEN)
        assert torch.isfinite(M).all()

    def test_per_state_coord_sums_to_default(self, cache, mamba_bridge):
        mixer = mamba_bridge.blocks[0].mixer
        oc = mixer.original_component
        M = mixer.compute_effective_attention(cache, layer_idx=0, include_dt_scaling=True)
        M_coord = mixer.compute_effective_attention(
            cache, layer_idx=0, include_dt_scaling=True, per_state_coord=True
        )
        assert M_coord.shape == (
            1,
            oc.intermediate_size,
            oc.ssm_state_size,
            self.SEQ_LEN,
            self.SEQ_LEN,
        )
        assert torch.allclose(M_coord.sum(dim=2), M, atol=1e-6)

    def test_causal(self, cache, mamba_bridge):
        M = mamba_bridge.blocks[0].mixer.compute_effective_attention(cache, layer_idx=0)
        upper = torch.triu(torch.ones(self.SEQ_LEN, self.SEQ_LEN, dtype=torch.bool), diagonal=1)
        assert torch.all(M[..., upper] == 0), "effective attention must be causal"

    def test_matches_fp64_eager_recurrence(self, cache, mamba_bridge):
        """M·x + D·x must match a naive fp64 step-by-step S6 recurrence."""
        mixer = mamba_bridge.blocks[0].mixer
        M = mixer.compute_effective_attention(cache, layer_idx=0, include_dt_scaling=True)
        x, B, C, dt, A, D, _ = self._inputs(cache, mixer, 0, self.SEQ_LEN)
        y_pred = torch.einsum("bcij,bcj->bci", M, x) + D[None, :, None] * x

        A, B, C, x, dt = A.double(), B.double(), C.double(), x.double(), dt.double()
        b, d_inner = x.shape[0], x.shape[1]
        ssm = torch.zeros(b, d_inner, A.shape[1], dtype=torch.float64)
        y_ref = torch.zeros(b, d_inner, self.SEQ_LEN, dtype=torch.float64)
        for i in range(self.SEQ_LEN):
            dA = torch.exp(A[None] * dt[:, :, i, None])  # [b, d_inner, state]
            dBu = dt[:, :, i, None] * B[:, i, None, :] * x[:, :, i, None]
            ssm = dA * ssm + dBu
            y_ref[:, :, i] = torch.einsum("bcn,bn->bc", ssm, C[:, i, :])
        y_ref = y_ref + D[None, :, None].double() * x

        rel = (y_pred.double() - y_ref).abs().max().item() / max(y_ref.abs().max().item(), 1e-8)
        assert rel < 1e-5, f"M·x reconstruction vs fp64 eager scan rel diff {rel:.2e}"

    def test_reconstructs_mixer_output(self, cache, mamba_bridge):
        """out_proj((M·x + D·x)·act(gate)) must reconstruct HF's mixer output."""
        mixer = mamba_bridge.blocks[0].mixer
        M = mixer.compute_effective_attention(cache, layer_idx=0, include_dt_scaling=True)
        x, _, _, _, _, D, gate = self._inputs(cache, mixer, 0, self.SEQ_LEN)
        y = torch.einsum("bcij,bcj->bci", M, x) + D[None, :, None] * x
        out = mixer.original_component.out_proj(
            (y * mixer.original_component.act(gate)).transpose(1, 2)
        )
        hook_out = cache["blocks.0.mixer.hook_out"].float()
        rel = (out - hook_out).abs().max().item() / max(hook_out.abs().max().item(), 1e-8)
        assert rel < 1e-5, (
            f"reconstructed mixer output vs hook_out rel diff {rel:.2e}; M is "
            "inconsistent with HF's independently-computed forward."
        )

    def test_include_dt_scaling_changes_output(self, cache, mamba_bridge):
        mixer = mamba_bridge.blocks[0].mixer
        M_att = mixer.compute_effective_attention(cache, layer_idx=0, include_dt_scaling=False)
        M_full = mixer.compute_effective_attention(cache, layer_idx=0, include_dt_scaling=True)
        assert not torch.allclose(M_att, M_full)

    def test_raises_on_empty_cache(self, mamba_bridge):
        from transformer_lens.ActivationCache import ActivationCache

        empty = ActivationCache({}, model=mamba_bridge)
        with pytest.raises(RuntimeError, match="in cache"):
            mamba_bridge.blocks[0].mixer.compute_effective_attention(empty, layer_idx=0)


class TestMamba1SSMState:
    """Mamba-1 recurrent-state reconstruction: read-parity with Mamba-2.

    The vectorized state must match a naive fp64 step-by-step S6 scan, and
    ``y = C·S + D·x`` reconstructed through gate + out_proj must match HF's cached
    ``hook_out`` (ties S to HF's numerically-independent forward).
    """

    SEQ_LEN = 6

    @pytest.fixture(scope="class")
    def cache(self, mamba_bridge):
        tokens = torch.arange(1, self.SEQ_LEN + 1).unsqueeze(0)
        with torch.no_grad():
            _, cache = mamba_bridge.run_with_cache(tokens)
        return cache

    @staticmethod
    def _inputs(cache, mixer, layer, seq_len):
        """Re-extract x (post-conv SiLU), B, C, dt, A, D, gate — independent of the impl."""
        import torch.nn.functional as F

        oc = mixer.original_component
        d_inner, state, dt_rank = oc.intermediate_size, oc.ssm_state_size, oc.time_step_rank
        conv = cache[f"blocks.{layer}.mixer.conv1d.hook_out"][..., :seq_len].float()
        x_proj = cache[f"blocks.{layer}.mixer.x_proj.hook_out"].float()
        dt_proj = cache[f"blocks.{layer}.mixer.dt_proj.hook_out"].float()
        in_proj = cache[f"blocks.{layer}.mixer.in_proj.hook_out"].float()
        x = oc.act(conv)  # [b, d_inner, seq]
        _ts, B, C = x_proj.split([dt_rank, state, state], dim=-1)
        dt = F.softplus(dt_proj).transpose(1, 2)  # [b, d_inner, seq]
        A = -torch.exp(mixer.A_log.float())
        gate = in_proj.transpose(1, 2)[:, d_inner:, :]  # [b, d_inner, seq]
        return x, B, C, dt, A, mixer.D.float(), gate

    def test_shape(self, cache, mamba_bridge):
        mixer = mamba_bridge.blocks[0].mixer
        oc = mixer.original_component
        S = mixer.compute_ssm_state(cache, layer_idx=0)
        assert S.shape == (1, oc.intermediate_size, self.SEQ_LEN, oc.ssm_state_size)
        assert torch.isfinite(S).all()

    def test_matches_fp64_eager_recurrence(self, cache, mamba_bridge):
        """The vectorized state must match a naive fp64 step-by-step S6 scan."""
        mixer = mamba_bridge.blocks[0].mixer
        S = mixer.compute_ssm_state(cache, layer_idx=0)

        x, B, _, dt, A, _, _ = self._inputs(cache, mixer, 0, self.SEQ_LEN)
        A, B, x, dt = A.double(), B.double(), x.double(), dt.double()
        b, d_inner = x.shape[0], x.shape[1]
        ssm = torch.zeros(b, d_inner, A.shape[1], dtype=torch.float64)
        ref = torch.zeros(b, d_inner, self.SEQ_LEN, A.shape[1], dtype=torch.float64)
        for i in range(self.SEQ_LEN):
            dA = torch.exp(A[None] * dt[:, :, i, None])  # [b, d_inner, state]
            dBu = dt[:, :, i, None] * B[:, i, None, :] * x[:, :, i, None]
            ssm = dA * ssm + dBu
            ref[:, :, i] = ssm

        rel = (S.double() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-8)
        assert rel < 1e-5, f"S vs fp64 eager recurrence rel diff {rel:.2e}"

    def test_reconstructs_mixer_output(self, cache, mamba_bridge):
        """out_proj((C·S + D·x)·act(gate)) must reconstruct HF's mixer output."""
        mixer = mamba_bridge.blocks[0].mixer
        S = mixer.compute_ssm_state(cache, layer_idx=0)  # [b, channels, seq, state]
        x, _, C, _, _, D, gate = self._inputs(cache, mixer, 0, self.SEQ_LEN)
        y = torch.einsum("bcts,bts->bct", S, C) + D[None, :, None] * x
        out = mixer.original_component.out_proj(
            (y * mixer.original_component.act(gate)).transpose(1, 2)
        )
        hook_out = cache["blocks.0.mixer.hook_out"].float()
        rel = (out - hook_out).abs().max().item() / max(hook_out.abs().max().item(), 1e-8)
        assert rel < 1e-5, (
            f"C·S+D·x reconstruction vs hook_out rel diff {rel:.2e}; the state is "
            "inconsistent with HF's independently-computed forward."
        )

    def test_time_step_matches_full_slice(self, cache, mamba_bridge):
        mixer = mamba_bridge.blocks[0].mixer
        S = mixer.compute_ssm_state(cache, layer_idx=0)
        S_t = mixer.compute_ssm_state(cache, layer_idx=0, time_step=3)
        assert S_t.shape == (S.shape[0], S.shape[1], S.shape[3])
        # Same math; the full ("bcsij") and single-step ("bcsj") einsums differ only
        # in fp reduction order (measured ~1 fp32 ULP), so allclose, not equal.
        assert torch.allclose(S[:, :, 3], S_t, atol=1e-6)

    def test_cache_method_matches_mixer(self, cache, mamba_bridge):
        direct = mamba_bridge.blocks[0].mixer.compute_ssm_state(cache, layer_idx=0)
        via_cache = cache.compute_ssm_state(layer=0)
        assert torch.equal(direct, via_cache)

    def test_cache_all_layers_stacks(self, cache, mamba_bridge):
        S_all = cache.compute_ssm_state()  # pure Mamba-1 → every block is SSM → stacked
        assert torch.is_tensor(S_all)
        assert S_all.shape[0] == mamba_bridge.cfg.n_layers
        assert torch.equal(S_all[0], cache.compute_ssm_state(layer=0))

    def test_raises_on_empty_cache(self, mamba_bridge):
        from transformer_lens.ActivationCache import ActivationCache

        empty = ActivationCache({}, model=mamba_bridge)
        with pytest.raises(RuntimeError, match="in cache"):
            mamba_bridge.blocks[0].mixer.compute_ssm_state(empty, layer_idx=0)


@contextlib.contextmanager
def _eager_scan(bridge):
    """Enable the opt-in eager-scan intervention path on every Mamba-1 mixer."""
    for block in bridge.blocks:
        block.mixer.eager_scan = True
    try:
        yield bridge
    finally:
        for block in bridge.blocks:
            block.mixer.eager_scan = False


class TestMamba1EagerScanIntervention:
    """Opt-in eager S6 scan exposes hook_ssm_write / hook_ssm_state
    for interventions that propagate to logits, while the default path is untouched.
    Eager scan needs use_cache=False (prefill; cache_params is None)."""

    TOKENS = torch.tensor([[1, 2, 3, 4, 5, 6]])

    def test_default_path_has_no_eager_hooks(self, mamba_bridge):
        with torch.no_grad():
            _, cache = mamba_bridge.run_with_cache(self.TOKENS)
        assert "blocks.0.mixer.hook_ssm_write" not in cache
        assert "blocks.0.mixer.hook_ssm_state" not in cache

    def test_eager_requires_use_cache_false(self, mamba_bridge):
        with _eager_scan(mamba_bridge), torch.no_grad():
            _, cache = mamba_bridge.run_with_cache(self.TOKENS)  # default use_cache
        assert "blocks.0.mixer.hook_ssm_write" not in cache

    def test_eager_scan_matches_fused_logits(self, mamba_bridge):
        with torch.no_grad():
            fused = mamba_bridge(self.TOKENS)
        with _eager_scan(mamba_bridge), torch.no_grad():
            eager = mamba_bridge.run_with_hooks(self.TOKENS, use_cache=False, fwd_hooks=[])
        rel = (eager - fused).abs().max().item() / max(fused.abs().max().item(), 1e-8)
        assert rel < 1e-4, f"eager scan vs HF kernel rel diff {rel:.2e}"

    def test_eager_scan_matches_fused_padded_batch(self, mamba_bridge):
        """Padded batch: eager path mirrors HF's channel-first padding mask (before
        and after the conv)."""
        ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        with torch.no_grad():
            fused = mamba_bridge(ids, attention_mask=mask)
        with _eager_scan(mamba_bridge), torch.no_grad():
            eager = mamba_bridge.run_with_hooks(
                ids, use_cache=False, attention_mask=mask, fwd_hooks=[]
            )
        rel = (eager - fused).abs().max().item() / max(fused.abs().max().item(), 1e-8)
        assert rel < 1e-4, f"padded-batch eager vs fused rel diff {rel:.2e}"

    def test_eager_hooks_fire_with_use_cache_false(self, mamba_bridge):
        with _eager_scan(mamba_bridge), torch.no_grad():
            _, cache = mamba_bridge.run_with_cache(self.TOKENS, use_cache=False)
        oc = mamba_bridge.blocks[0].mixer.original_component
        seq = self.TOKENS.shape[1]
        expected = (1, oc.intermediate_size, seq, oc.ssm_state_size)
        assert cache["blocks.0.mixer.hook_ssm_write"].shape == expected
        assert cache["blocks.0.mixer.hook_ssm_state"].shape == expected

    def test_write_knockout_changes_logits(self, mamba_bridge):
        def knockout(writes, hook):
            writes = writes.clone()
            writes[:, :, 2] = 0.0  # channel-first: zero all channels' write at position 2
            return writes

        with _eager_scan(mamba_bridge), torch.no_grad():
            base = mamba_bridge.run_with_hooks(self.TOKENS, use_cache=False, fwd_hooks=[])
            patched = mamba_bridge.run_with_hooks(
                self.TOKENS,
                use_cache=False,
                fwd_hooks=[("blocks.0.mixer.hook_ssm_write", knockout)],
            )
        assert (patched - base).abs().max().item() > 1e-6

    def test_state_patch_changes_logits(self, mamba_bridge):
        def patch(state, hook):
            state = state.clone()
            state[:, :, 3] = 0.0
            return state

        with _eager_scan(mamba_bridge), torch.no_grad():
            base = mamba_bridge.run_with_hooks(self.TOKENS, use_cache=False, fwd_hooks=[])
            patched = mamba_bridge.run_with_hooks(
                self.TOKENS, use_cache=False, fwd_hooks=[("blocks.0.mixer.hook_ssm_state", patch)]
            )
        assert (patched - base).abs().max().item() > 1e-6

    def test_disabling_restores_fused_path(self, mamba_bridge):
        with torch.no_grad():
            fused_before = mamba_bridge(self.TOKENS)
        with _eager_scan(mamba_bridge), torch.no_grad():
            mamba_bridge.run_with_hooks(self.TOKENS, use_cache=False, fwd_hooks=[])
        with torch.no_grad():
            fused_after = mamba_bridge(self.TOKENS)
        assert torch.equal(fused_before, fused_after)


class _DummyTok:
    pass


def _build_synthetic_mamba_bridge():
    """Tiny random-init Mamba-1 bridge (from_config, no download) for CI-runnable tests."""
    from transformers import AutoModelForCausalLM
    from transformers.models.mamba import MambaConfig

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_config_from_hf,
    )
    from transformer_lens.model_bridge.supported_architectures.mamba import (
        MambaArchitectureAdapter,
    )

    torch.manual_seed(0)
    cfg = MambaConfig(
        vocab_size=128,
        hidden_size=32,
        state_size=16,
        num_hidden_layers=2,
        expand=2,
        time_step_rank=4,
        conv_kernel=4,
    )
    cfg.architectures = ["MambaForCausalLM"]
    hf = AutoModelForCausalLM.from_config(cfg).eval()
    bridge_cfg = build_bridge_config_from_hf(
        hf.config, "MambaForCausalLM", "m1-tiny", torch.float32
    )
    return TransformerBridge(hf, MambaArchitectureAdapter(bridge_cfg), tokenizer=_DummyTok())


class TestMamba1EagerScanFp64Reference:
    """Independent fp64 correctness gate on the Mamba-1 eager scan (_eager_scan_forward).

    Pins the eager scan's hook_ssm_write / hook_ssm_state to a naive fp64 step recurrence
    built from the cached submodule outputs — distinct from both the fused-kernel parity
    test and the read-path fp64 tests, so a shared discretization error can't pass. Uses
    synthetic weights (from_config) so it runs in stock CI without a model download.
    """

    SEQ_LEN = 7
    LAYER = 0

    @pytest.fixture(scope="class")
    def bridge(self):
        return _build_synthetic_mamba_bridge()

    @pytest.fixture(scope="class")
    def eager_cache(self, bridge):
        tokens = torch.randint(1, 128, (2, self.SEQ_LEN))
        with _eager_scan(bridge), torch.no_grad():
            _, cache = bridge.run_with_cache(tokens, use_cache=False)
        return cache

    def _fp64_reference(self, bridge, cache):
        """Reconstruct (writes, state) in fp64 from cached submodule hooks.

        Independent reimplementation of the S6 write term + recurrence; reads the same
        HF submodule outputs the eager scan reuses (so their hooks fire), not the eager
        code. Returns channel-first ``[b, d_inner, seq, state]`` tensors.
        """
        import torch.nn.functional as F

        oc = bridge.blocks[self.LAYER].mixer.original_component
        d_inner, state, dt_rank = oc.intermediate_size, oc.ssm_state_size, oc.time_step_rank
        p = f"blocks.{self.LAYER}.mixer"

        conv = cache[f"{p}.conv1d.hook_out"][..., : self.SEQ_LEN].double()  # [b, d_inner, seq]
        x = oc.act(conv)  # HF act on the trimmed conv output
        xp = cache[f"{p}.x_proj.hook_out"].double()  # [b, seq, dt_rank + 2*state]
        _, B, _C = xp.split([dt_rank, state, state], dim=-1)  # B: [b, seq, state]
        dtp = cache[f"{p}.dt_proj.hook_out"].double()  # [b, seq, d_inner] (pre-softplus)
        dt = F.softplus(dtp).transpose(1, 2)  # [b, d_inner, seq]
        A = -torch.exp(bridge.blocks[self.LAYER].mixer.A_log.double())  # [d_inner, state]

        writes = (dt * x)[:, :, :, None] * B[:, None, :, :]  # [b, d_inner, seq, state]
        b = writes.shape[0]
        ssm = torch.zeros(b, d_inner, state, dtype=torch.float64)
        states = []
        for t in range(self.SEQ_LEN):
            ssm = torch.exp(A[None] * dt[:, :, t, None]) * ssm + writes[:, :, t]
            states.append(ssm)
        return writes, torch.stack(states, dim=2)  # [b, d_inner, seq, state]

    def test_eager_write_matches_fp64_reference(self, bridge, eager_cache):
        writes_ref, _ = self._fp64_reference(bridge, eager_cache)
        got = eager_cache[f"blocks.{self.LAYER}.mixer.hook_ssm_write"].double()
        rel = (got - writes_ref).abs().max().item() / max(writes_ref.abs().max().item(), 1e-12)
        assert rel < 1e-5, f"eager hook_ssm_write vs fp64 reference rel diff {rel:.2e}"

    def test_eager_state_matches_fp64_reference(self, bridge, eager_cache):
        _, state_ref = self._fp64_reference(bridge, eager_cache)
        got = eager_cache[f"blocks.{self.LAYER}.mixer.hook_ssm_state"].double()
        rel = (got - state_ref).abs().max().item() / max(state_ref.abs().max().item(), 1e-12)
        assert rel < 1e-5, f"eager hook_ssm_state vs fp64 reference rel diff {rel:.2e}"

    def test_write_knockout_propagates_forward(self, bridge):
        """Zeroing hook_ssm_write at position k must change the state at every position
        >= k (the recurrence carries it forward) and leave positions < k untouched."""
        tokens = torch.randint(1, 128, (1, self.SEQ_LEN))
        k = 3
        state_key = f"blocks.{self.LAYER}.mixer.hook_ssm_state"
        write_key = f"blocks.{self.LAYER}.mixer.hook_ssm_write"

        def knock(writes, hook):
            writes = writes.clone()
            writes[:, :, k] = 0.0
            return writes

        captured = {}

        def grab(state, hook):
            captured["s"] = state.detach().clone()
            return state

        with _eager_scan(bridge), torch.no_grad():
            _, base_cache = bridge.run_with_cache(tokens, use_cache=False)
            base_state = base_cache[state_key]
            bridge.run_with_hooks(
                tokens, use_cache=False, fwd_hooks=[(write_key, knock), (state_key, grab)]
            )
        patched_state = captured["s"]
        # positions before k unchanged; positions at/after k changed
        assert torch.allclose(patched_state[:, :, :k], base_state[:, :, :k], atol=1e-6)
        assert (patched_state[:, :, k:] - base_state[:, :, k:]).abs().max().item() > 1e-6

    def test_state_patch_is_same_position_only(self, bridge):
        """Patching hook_ssm_state at position k changes only the same-position mixer
        output (y_t = C_t·S_t is a post-scan readout — it does not re-run the scan)."""
        tokens = torch.randint(1, 128, (1, self.SEQ_LEN))
        k = 3
        state_key = f"blocks.{self.LAYER}.mixer.hook_ssm_state"
        out_key = f"blocks.{self.LAYER}.mixer.hook_out"

        def patch(state, hook):
            state = state.clone()
            state[:, :, k] = 0.0
            return state

        base_out, patched_out = {}, {}

        def grab_base(o, hook):
            base_out["o"] = o.detach().clone()
            return o

        def grab_patched(o, hook):
            patched_out["o"] = o.detach().clone()
            return o

        with _eager_scan(bridge), torch.no_grad():
            bridge.run_with_hooks(tokens, use_cache=False, fwd_hooks=[(out_key, grab_base)])
            bridge.run_with_hooks(
                tokens, use_cache=False, fwd_hooks=[(state_key, patch), (out_key, grab_patched)]
            )
        delta = (patched_out["o"] - base_out["o"]).abs()
        assert delta[:, k].max().item() > 1e-6, "same-position output must change"
        other = torch.cat([delta[:, :k], delta[:, k + 1 :]], dim=1)
        assert other.max().item() < 1e-6, "other positions must be untouched (no propagation)"

    def test_batch1_padded_matches_hf(self, bridge):
        """Batch-1 with a padding mask must match HF: MambaMixer masks whenever
        attention_mask is present (no batch>1 guard, unlike Mamba-2)."""
        tokens = torch.tensor([[1, 2, 3, 4, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0]])
        with torch.no_grad():
            fused = bridge(tokens, attention_mask=mask)
            with _eager_scan(bridge):
                eager = bridge.run_with_hooks(
                    tokens, use_cache=False, attention_mask=mask, fwd_hooks=[]
                )
        rel = (eager - fused).abs().max().item() / max(fused.abs().max().item(), 1e-8)
        assert rel < 1e-4, f"batch-1 padded eager vs HF fused rel diff {rel:.2e}"
