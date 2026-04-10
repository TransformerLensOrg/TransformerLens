"""Integration tests for the Mamba-1 architecture adapter.

Verifies wrap-don't-reimplement behavior against state-spaces/mamba-130m-hf:
- Forward pass matches HF exactly (bridge delegates to HF slow_forward)
- Submodule hooks fire with expected shapes for in_proj, conv1d, x_proj,
  dt_proj, out_proj
- SSM blocks correctly exclude transformer-specific hook_resid_mid
- Parameter access via __getattr__ fallback (A_log, D)

Note on cache clone safety: The Mamba adapter plan flagged in-place MambaCache
mutation as a risk — hooks that capture `ssm_states` would see corrupted values
on subsequent decode steps because HF mutates the cache in place. Phase 1 avoids
this risk entirely by design: the wrap-don't-reimplement approach keeps the
MambaMixer opaque, so `ssm_states` is never exposed through a hook. Phase 1
hooks only observe projection inputs/outputs (in_proj, conv1d, x_proj, dt_proj,
out_proj), which are per-step tensors and are never mutated by the cache
machinery. If a future phase adds per-step SSM state hooks (compatibility mode),
those hooks MUST `.clone()` captured state tensors.
"""

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
        """Hook tensor shapes match the plan's Step 1.5 shape summary."""
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
