"""Integration tests for the Zamba2 architecture adapter.

Verifies forward-pass and generation parity against Zyphra/Zamba2-1.2B:
- Forward-pass logits match HF exactly (bridge delegates the full forward to HF)
- Greedy multi-token generation matches HF bit-for-bit (exercises the unified
  Zamba2HybridDynamicCache threaded via past_key_values across Mamba-2 and
  shared global-attention layers)
- Sanity checks: config flags, block count, hook coverage on both layer types

Zamba2 has two layer types, surfaced canonically in ``cfg.layers_block_type``:
  - ``"linear_attention"`` — pure Mamba-2 SSM (HF ``"mamba"``)
  - ``"hybrid"`` — Mamba-2 + shared global-attention (Zamba2HybridLayer)

Run with GPU acceleration:
    CUDA_VISIBLE_DEVICES=0 pytest tests/integration/model_bridge/test_zamba2_adapter.py -v -s

On a CPU-only machine:
    pytest tests/integration/model_bridge/test_zamba2_adapter.py -v -s
"""

import gc

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    RMSNormalizationBridge,
    SSM2MixerBridge,
    SSMBlockBridge,
)

pytestmark = pytest.mark.slow

MODEL = "Zyphra/Zamba2-1.2B"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype() -> torch.dtype:
    # bfloat16 on GPU to match HF defaults; float32 on CPU for numerical safety
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


# ---------------------------------------------------------------------------
# Session fixture — load once, share across all test classes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zamba2_bridge():
    device = _device()
    dtype = _dtype()
    bridge = TransformerBridge.boot_transformers(MODEL, device=device, dtype=dtype)
    yield bridge
    del bridge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for _ in range(3):
        gc.collect()


# ---------------------------------------------------------------------------
# Config and bridge structure
# ---------------------------------------------------------------------------


class TestZamba2BridgeCreation:
    """Smoke-test that the bridge loads with the right config flags."""

    def test_norm_bridges_are_rms(self, zamba2_bridge: TransformerBridge) -> None:
        """normalization_type='RMS' must wire RMSNormalizationBridge, not LayerNorm.

        The block pre-norm (on Mamba layers) and the final norm should both be
        RMS. Asserting the concrete bridge type catches a regression where the
        adapter wires the wrong normalization component.
        """
        lbt = getattr(zamba2_bridge.cfg, "layers_block_type", [])
        first_mamba = next(i for i, t in enumerate(lbt) if t == "linear_attention")
        assert isinstance(zamba2_bridge.blocks[first_mamba].norm, RMSNormalizationBridge)
        assert isinstance(zamba2_bridge.ln_final, RMSNormalizationBridge)

    def test_block_count(self, zamba2_bridge: TransformerBridge) -> None:
        # Zamba2-1.2B has 38 layers (32 linear_attention + 6 hybrid); the block count
        # must match the per-layer type list rather than a hard-coded magic
        # number so the test tracks the checkpoint's actual config.
        lbt = getattr(zamba2_bridge.cfg, "layers_block_type", [])
        assert len(zamba2_bridge.blocks) == len(lbt) == 38

    def test_blocks_are_ssm_block_bridge(self, zamba2_bridge: TransformerBridge) -> None:
        assert isinstance(zamba2_bridge.blocks[0], SSMBlockBridge)

    def test_mamba_layers_have_mixer(self, zamba2_bridge: TransformerBridge) -> None:
        """Mamba (mamba) layers should expose SSM2MixerBridge as .mixer."""
        lbt = getattr(zamba2_bridge.cfg, "layers_block_type", [])
        first_mamba = next(i for i, t in enumerate(lbt) if t == "linear_attention")
        assert isinstance(zamba2_bridge.blocks[first_mamba].mixer, SSM2MixerBridge)

    def test_hybrid_layers_have_no_mixer(self, zamba2_bridge: TransformerBridge) -> None:
        """Hybrid layers have no top-level .mamba; .mixer should be absent or None."""
        lbt = getattr(zamba2_bridge.cfg, "layers_block_type", [])
        first_hybrid = next(i for i, t in enumerate(lbt) if t == "hybrid")
        # The optional=True submodule should not be set up by component_setup
        mixer = getattr(zamba2_bridge.blocks[first_hybrid], "mixer", None)
        assert mixer is None or getattr(
            mixer, "optional", False
        ), "Hybrid layer should not have a wired .mixer (no top-level .mamba attribute)"

    def test_layers_block_type_populated(self, zamba2_bridge: TransformerBridge) -> None:
        lbt = getattr(zamba2_bridge.cfg, "layers_block_type", [])
        assert len(lbt) == len(zamba2_bridge.blocks)
        # Both layer types must appear
        assert "linear_attention" in lbt, "No linear_attention (Mamba) layers found"
        assert "hybrid" in lbt, "No hybrid (Mamba + attention) layers found"

    def test_mamba_intermediate_size_positive(self, zamba2_bridge: TransformerBridge) -> None:
        assert getattr(zamba2_bridge.cfg, "mamba_intermediate_size", 0) > 0

    def test_conv_dim_positive(self, zamba2_bridge: TransformerBridge) -> None:
        assert getattr(zamba2_bridge.cfg, "conv_dim", 0) > 0

    def test_uses_standard_kv_cache_path(self, zamba2_bridge: TransformerBridge) -> None:
        # Zamba2 threads its unified cache via past_key_values (standard KV
        # path), NOT the Mamba cache_params path. is_stateful=True would select
        # the Mamba path, whose cache_params kwarg collides with Zamba2's own
        # forward. Keeping it False routes generation through the past_key_values
        # path, which matches HF generate() bit-for-bit (see TestZamba2Generation).
        assert zamba2_bridge.cfg.is_stateful is False

    def test_positional_embedding_none(self, zamba2_bridge: TransformerBridge) -> None:
        # RoPE is handled inside the HF attention block; no model-level embedding
        assert zamba2_bridge.cfg.positional_embedding_type == "none"


# ---------------------------------------------------------------------------
# Forward-pass parity
# ---------------------------------------------------------------------------


class TestZamba2ForwardPass:
    """Bridge logits must match HF logits exactly.

    Zamba2ArchitectureAdapter uses SSMBlockBridge with a pure passthrough
    forward, so the bridge never reimplements any computation. Parity with
    HF should be exact (diff == 0), not just close.
    """

    @pytest.fixture(scope="class")
    def tokens(self) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    def test_forward_returns_logits(
        self, zamba2_bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        tokens = tokens.to(_device())
        with torch.no_grad():
            out = zamba2_bridge(tokens)
        assert out.shape == (1, 8, zamba2_bridge.cfg.d_vocab)
        assert not torch.isnan(out).any(), "NaN in bridge logits"
        assert not torch.isinf(out).any(), "Inf in bridge logits"

    def test_forward_matches_hf_exactly(
        self, zamba2_bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        tokens = tokens.to(_device())
        hf_model = zamba2_bridge.original_model
        with torch.no_grad():
            bridge_out = zamba2_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out.float() - hf_out.float()).abs().max().item()
        assert max_diff == 0.0, (
            f"Bridge vs HF forward max diff = {max_diff:.2e}. "
            "Expected 0 because SSMBlockBridge.forward() is a pure passthrough."
        )

    def test_forward_no_nan_on_longer_sequence(self, zamba2_bridge: TransformerBridge) -> None:
        # 32 tokens exercises multiple Mamba SSM steps and shared attention passes
        tokens = torch.arange(1, 33).unsqueeze(0).to(_device())
        with torch.no_grad():
            out = zamba2_bridge(tokens)
        assert not torch.isnan(out).any(), "NaN in logits for 32-token sequence"


# ---------------------------------------------------------------------------
# Multi-token generation parity (exercises DynamicCache state handling)
# ---------------------------------------------------------------------------


class TestZamba2Generation:
    """Bridge greedy generation must match HF native generate() exactly.

    DynamicCache carries both KV-cache entries (from the shared attention
    blocks in hybrid layers) and Mamba-2 conv/recurrent states. Token-level
    equality with HF confirms the state threading is correct across both
    layer types.
    """

    @pytest.fixture(scope="class")
    def prompt(self) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, 4]])

    def test_generation_produces_tokens(
        self, zamba2_bridge: TransformerBridge, prompt: torch.Tensor
    ) -> None:
        prompt = prompt.to(_device())
        with torch.no_grad():
            result = zamba2_bridge.generate(prompt, max_new_tokens=5, do_sample=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 9)  # 4 prompt + 5 new

    def test_greedy_matches_hf_exactly(
        self, zamba2_bridge: TransformerBridge, prompt: torch.Tensor
    ) -> None:
        """Bit-for-bit equality with HF generate() over 8 new tokens."""
        prompt = prompt.to(_device())
        hf_model = zamba2_bridge.original_model
        with torch.no_grad():
            bridge_out = zamba2_bridge.generate(prompt, max_new_tokens=8, do_sample=False)
            hf_out = hf_model.generate(prompt, max_new_tokens=8, do_sample=False, pad_token_id=0)
        assert torch.equal(bridge_out, hf_out), (
            f"Token mismatch between bridge and HF.\n"
            f"  bridge : {bridge_out.tolist()}\n"
            f"  hf     : {hf_out.tolist()}\n"
            "DynamicCache state threading across Mamba-2 and hybrid attention layers "
            "is likely wrong."
        )

    def test_generation_is_deterministic(
        self, zamba2_bridge: TransformerBridge, prompt: torch.Tensor
    ) -> None:
        prompt = prompt.to(_device())
        with torch.no_grad():
            out1 = zamba2_bridge.generate(prompt, max_new_tokens=4, do_sample=False)
            out2 = zamba2_bridge.generate(prompt, max_new_tokens=4, do_sample=False)
        assert torch.equal(out1, out2), "Greedy generation is not deterministic"


# ---------------------------------------------------------------------------
# Hook coverage: bridge hooks fire for both Mamba and hybrid layers
# ---------------------------------------------------------------------------


class TestZamba2HookCoverage:
    """run_with_cache captures residual stream and mixer hooks."""

    @pytest.fixture(scope="class")
    def cache(self, zamba2_bridge: TransformerBridge):
        tokens = torch.tensor([[1, 2, 3, 4, 5]]).to(_device())
        with torch.no_grad():
            _, cache = zamba2_bridge.run_with_cache(tokens)
        return cache

    def test_block_hooks_fire_on_all_layers(self, cache, zamba2_bridge: TransformerBridge) -> None:
        """hook_in and hook_out must fire on every layer regardless of type."""
        n_blocks = len(zamba2_bridge.blocks)
        # Sample first, an early, the middle, and the last layer (indices derived
        # from the actual block count, not a hard-coded magic number).
        for i in sorted({0, 1, n_blocks // 2, n_blocks - 1}):
            assert f"blocks.{i}.hook_in" in cache, f"Missing hook_in for block {i}"
            assert f"blocks.{i}.hook_out" in cache, f"Missing hook_out for block {i}"

    def test_mamba_mixer_submodule_hooks_fire(
        self, cache, zamba2_bridge: TransformerBridge
    ) -> None:
        """Mamba (mamba) layers must expose in_proj / conv1d / out_proj hooks."""
        lbt = getattr(zamba2_bridge.cfg, "layers_block_type", [])
        mamba_indices = [i for i, t in enumerate(lbt) if t == "linear_attention"]
        assert mamba_indices, "No mamba layers found in layers_block_type"
        for i in mamba_indices[:3]:
            for submod in ("in_proj", "conv1d", "out_proj"):
                key_in = f"blocks.{i}.mixer.{submod}.hook_in"
                key_out = f"blocks.{i}.mixer.{submod}.hook_out"
                assert key_in in cache, f"Missing {key_in}"
                assert key_out in cache, f"Missing {key_out}"

    def test_hybrid_layers_no_mixer_hooks(self, cache, zamba2_bridge: TransformerBridge) -> None:
        """Hybrid layers have no top-level .mamba, so no mixer submodule hooks."""
        lbt = getattr(zamba2_bridge.cfg, "layers_block_type", [])
        hybrid_indices = [i for i, t in enumerate(lbt) if t == "hybrid"]
        assert hybrid_indices, "No hybrid layers found in layers_block_type"
        for i in hybrid_indices[:3]:
            # mixer hooks must not appear for hybrid layers
            assert (
                f"blocks.{i}.mixer.in_proj.hook_in" not in cache
            ), f"Unexpected mixer hook on hybrid layer {i}"

    def test_no_transformer_specific_hooks(self, cache) -> None:
        """SSMBlockBridge must not inject transformer-shaped hook names."""
        forbidden = ("hook_resid_mid", "hook_attn_out", "hook_mlp_out")
        bad = [k for k in cache if any(f in k for f in forbidden)]
        assert bad == [], f"Unexpected transformer-shaped hooks: {bad[:5]}"

    def test_no_nan_in_cache(self, cache) -> None:
        for key, val in cache.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                assert not torch.isnan(val).any(), f"NaN in cache['{key}']"
