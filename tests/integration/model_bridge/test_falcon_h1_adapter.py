"""Integration tests for the Falcon-H1 architecture adapter.

Verifies wrap-don't-reimplement behaviour against ``tiiuae/Falcon-H1-0.5B-Base``:
- Forward-pass logits match HF **exactly** (the bridge delegates the whole block
  to HF, which applies Falcon-H1's ~12 scalar multipliers natively).
- Both the attention branch AND the Mamba-2 branch are independently hookable in
  every block — the defining value of a parallel-hybrid adapter.
- Greedy generation matches HF bit-for-bit through the standard unified
  KV-cache path.
- Config propagation: SSM dims surface on ``bridge.cfg``.

The 0.5B checkpoint is not in the CI cached-model allowlist, so this module is
marked ``slow`` and skipped under CI (mirrors the NemotronH heavy-test gating;
see tests/QUARANTINES.md). Run locally with:

    pytest tests/integration/model_bridge/test_falcon_h1_adapter.py -v
"""

import os

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    DepthwiseConv1DBridge,
    PositionEmbeddingsAttentionBridge,
    SSM2MixerBridge,
)

MODEL = "tiiuae/Falcon-H1-0.5B-Base"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        bool(os.getenv("CI")),
        reason="Falcon-H1-0.5B is not in the CI cached-model set; see tests/QUARANTINES.md",
    ),
]


@pytest.fixture(scope="module")
def falcon_h1_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


# ---------------------------------------------------------------------------
# Bridge creation
# ---------------------------------------------------------------------------


class TestFalconH1BridgeCreation:
    def test_block_count(self, falcon_h1_bridge):
        assert len(falcon_h1_bridge.blocks) == 36

    def test_config_flags(self, falcon_h1_bridge):
        cfg = falcon_h1_bridge.cfg
        assert cfg.normalization_type == "RMS"
        assert cfg.uses_rms_norm is True
        assert cfg.positional_embedding_type == "rotary"
        assert cfg.gated_mlp is True
        assert cfg.attn_only is False
        assert cfg.final_rms is True
        assert cfg.n_key_value_heads == 2
        # Standard KV-cache generation path, not the pure-Mamba stateful loop.
        assert getattr(cfg, "is_stateful", False) is False

    def test_ssm_config_propagated(self, falcon_h1_bridge):
        cfg = falcon_h1_bridge.cfg
        assert getattr(cfg, "mamba_d_ssm", None) == 1536
        # conv_dim = d_ssm + 2 * n_groups * d_state = 1536 + 2*1*128
        assert getattr(cfg, "conv_dim", None) == 1536 + 2 * 1 * 128
        # in_proj fuses gate, hidden_BC, dt: 2*d_ssm + conv_dim + mamba_n_heads
        assert getattr(cfg, "expected_in_proj_out_features", None) == 2 * 1536 + 1792 + 24

    def test_both_branches_present(self, falcon_h1_bridge):
        block = falcon_h1_bridge.blocks[0]
        assert isinstance(block, BlockBridge)
        assert isinstance(block.attn, PositionEmbeddingsAttentionBridge)
        assert isinstance(block.mamba, SSM2MixerBridge)

    def test_has_core_components(self, falcon_h1_bridge):
        for comp in ("embed", "unembed", "ln_final", "rotary_emb"):
            assert hasattr(falcon_h1_bridge, comp)


# ---------------------------------------------------------------------------
# Forward equivalence
# ---------------------------------------------------------------------------


class TestFalconH1ForwardEquivalence:
    def test_forward_returns_logits(self, falcon_h1_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            out = falcon_h1_bridge(tokens)
        assert out.shape == (1, 4, falcon_h1_bridge.cfg.d_vocab)
        assert not torch.isnan(out).any()

    def test_forward_matches_hf_exactly(self, falcon_h1_bridge):
        """Passthrough: bridge logits must be bitwise-identical to HF.

        HF applies all of Falcon-H1's scalar multipliers in its own forward, so
        with no weight folding the diff is exactly zero.
        """
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        hf_model = falcon_h1_bridge.original_model
        with torch.no_grad():
            bridge_out = falcon_h1_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff == 0.0, f"Bridge vs HF max diff = {max_diff}"


# ---------------------------------------------------------------------------
# HF delegation (wrap-don't-reimplement)
# ---------------------------------------------------------------------------


class TestFalconH1HFDelegation:
    """Bridge submodules wrap live HF torch modules rather than reimplementing."""

    def test_attn_projections_wrap_linear(self, falcon_h1_bridge):
        attn = falcon_h1_bridge.blocks[0].attn
        for key in ("q", "k", "v", "o"):
            assert isinstance(attn._modules[key].original_component, torch.nn.Linear)

    def test_mamba_submodules_wrap_real_modules(self, falcon_h1_bridge):
        mamba = falcon_h1_bridge.blocks[0].mamba
        assert isinstance(mamba.in_proj.original_component, torch.nn.Linear)
        assert isinstance(mamba.out_proj.original_component, torch.nn.Linear)
        assert isinstance(
            falcon_h1_bridge.blocks[0].mamba.conv1d,
            DepthwiseConv1DBridge,
        )
        assert isinstance(mamba.conv1d.original_component, torch.nn.Conv1d)

    def test_mlp_projections_wrap_linear(self, falcon_h1_bridge):
        mlp = falcon_h1_bridge.blocks[0].mlp
        for key in ("gate", "in", "out"):
            assert isinstance(mlp._modules[key].original_component, torch.nn.Linear)


# ---------------------------------------------------------------------------
# Parallel-branch hook coverage
# ---------------------------------------------------------------------------


class TestFalconH1ParallelHooks:
    @pytest.fixture(scope="class")
    def cache(self, falcon_h1_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            _, cache = falcon_h1_bridge.run_with_cache(tokens)
        return cache

    def test_attention_branch_hooks_fire(self, cache):
        for i in (0, 18, 35):
            for proj in ("q", "k", "v", "o"):
                assert f"blocks.{i}.attn.{proj}.hook_out" in cache

    def test_mamba_branch_hooks_fire(self, cache):
        for i in (0, 18, 35):
            for proj in ("in_proj", "conv1d", "out_proj"):
                assert f"blocks.{i}.mamba.{proj}.hook_out" in cache

    def test_both_branches_fire_in_same_block(self, cache):
        # The parallel-hybrid value proposition: in one block, both paths are
        # observable so a researcher can ablate one and compare.
        assert "blocks.0.attn.hook_out" in cache
        assert "blocks.0.mamba.hook_out" in cache

    def test_residual_and_mlp_hooks_present(self, cache):
        for i in (0, 35):
            assert f"blocks.{i}.hook_in" in cache
            assert f"blocks.{i}.hook_out" in cache
        assert "blocks.0.mlp.hook_out" in cache

    def test_no_nan_in_hooked_activations(self, cache):
        for key in ("blocks.0.attn.hook_out", "blocks.0.mamba.hook_out", "blocks.0.hook_out"):
            assert not torch.isnan(cache[key]).any()


# ---------------------------------------------------------------------------
# Hook ablation
# ---------------------------------------------------------------------------


class TestFalconH1HookAblation:
    def test_zero_ablation_changes_output(self, falcon_h1_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            baseline = falcon_h1_bridge(tokens)

        def zero(t, hook):
            return torch.zeros_like(t)

        with torch.no_grad():
            ablated = falcon_h1_bridge.run_with_hooks(
                tokens, fwd_hooks=[("blocks.18.hook_in", zero)]
            )
        assert not torch.allclose(baseline, ablated)

    def test_mamba_branch_ablation_changes_output(self, falcon_h1_bridge):
        # Ablating just the Mamba branch output must perturb logits, proving the
        # branch is on the live forward path (not a dead hook).
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            baseline = falcon_h1_bridge(tokens)

        def zero(t, hook):
            return torch.zeros_like(t)

        with torch.no_grad():
            ablated = falcon_h1_bridge.run_with_hooks(
                tokens, fwd_hooks=[("blocks.0.mamba.hook_out", zero)]
            )
        assert not torch.allclose(baseline, ablated)


# ---------------------------------------------------------------------------
# Generation parity
# ---------------------------------------------------------------------------


class TestFalconH1Generation:
    @pytest.mark.parametrize("prompt_len", [1, 4])
    def test_greedy_matches_hf(self, falcon_h1_bridge, prompt_len):
        tokens = torch.arange(1, prompt_len + 1).unsqueeze(0)
        with torch.no_grad():
            bridge_out = falcon_h1_bridge.generate(tokens, max_new_tokens=6, do_sample=False)
            hf_out = falcon_h1_bridge.original_model.generate(
                tokens, max_new_tokens=6, do_sample=False, pad_token_id=0
            )
        assert torch.equal(
            bridge_out, hf_out
        ), f"prompt_len={prompt_len}: bridge={bridge_out.tolist()} vs HF={hf_out.tolist()}"


# ---------------------------------------------------------------------------
# Config propagation vs HF
# ---------------------------------------------------------------------------


class TestFalconH1ConfigPropagation:
    def test_mamba_dims_match_hf_config(self, falcon_h1_bridge):
        hf_cfg = falcon_h1_bridge.original_model.config
        assert falcon_h1_bridge.cfg.mamba_d_ssm == hf_cfg.mamba_d_ssm
        assert falcon_h1_bridge.cfg.mamba_n_heads == hf_cfg.mamba_n_heads
        assert falcon_h1_bridge.cfg.mamba_d_state == hf_cfg.mamba_d_state
        assert falcon_h1_bridge.cfg.mamba_n_groups == hf_cfg.mamba_n_groups
