"""Integration tests for the Lfm2 architecture adapter.

Verifies forward-pass and generation parity against LiquidAI/LFM2.5-230M:
- Forward-pass logits match HF exactly (bridge delegates the full forward to HF)
- Greedy multi-token generation matches HF bit-for-bit (exercises DynamicCache
  state handling across attention and conv layers)
- Sanity checks: config flags, block count, hook coverage
"""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    DepthwiseConv1DBridge,
    Lfm2ShortConvBridge,
    RMSNormalizationBridge,
)

MODEL = "LiquidAI/LFM2.5-230M"


# ---------------------------------------------------------------------------
# Session fixture — load once, share across all test classes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lfm2_bridge():
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
    return bridge


# ---------------------------------------------------------------------------
# Config and bridge structure
# ---------------------------------------------------------------------------


class TestLfm2BridgeCreation:
    """Smoke-test that the bridge loads with the right config flags."""

    def test_norm_bridges_are_rms(self, lfm2_bridge: TransformerBridge) -> None:
        assert isinstance(lfm2_bridge.blocks[0].ln1, RMSNormalizationBridge)
        assert isinstance(lfm2_bridge.blocks[0].ln2, RMSNormalizationBridge)
        assert isinstance(lfm2_bridge.ln_final, RMSNormalizationBridge)

    def test_block_count(self, lfm2_bridge: TransformerBridge) -> None:
        # LiquidAI/LFM2.5-230M has 14 layers
        assert len(lfm2_bridge.blocks) == 14

    def test_blocks_are_block_bridge(self, lfm2_bridge: TransformerBridge) -> None:
        assert isinstance(lfm2_bridge.blocks[0], BlockBridge)

    def test_conv_is_lfm2_short_conv_bridge(self, lfm2_bridge: TransformerBridge) -> None:
        assert isinstance(lfm2_bridge.blocks[0].conv, Lfm2ShortConvBridge)

    def test_conv_conv_is_depthwise_conv_bridge(self, lfm2_bridge: TransformerBridge) -> None:
        assert isinstance(lfm2_bridge.blocks[0].conv.conv, DepthwiseConv1DBridge)

    def test_layers_block_type_populated(self, lfm2_bridge: TransformerBridge) -> None:
        lt = getattr(lfm2_bridge.cfg, "layer_types", [])
        assert len(lt) == len(lfm2_bridge.blocks)
        # Should contain at least one attention and one conv layer
        assert "full_attention" in lt
        assert "conv" in lt


# ---------------------------------------------------------------------------
# Forward-pass parity
# ---------------------------------------------------------------------------


class TestLfm2ForwardPass:
    """Bridge logits must match HF logits exactly.

    Lfm2 uses Lfm2ShortConvBridge delegating forward (original_component(*args, **kwargs))
    to HF model, so the bridge never reimplements any computation.
    Parity with HF should be exact (diff == 0),
    not just close.
    """

    @pytest.fixture(scope="class")
    def tokens(self) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    def test_forward_returns_logits(
        self, lfm2_bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        with torch.no_grad():
            out = lfm2_bridge(tokens)
        assert out.shape == (1, 8, lfm2_bridge.cfg.d_vocab)
        assert not torch.isnan(out).any(), "NaN in bridge logits"
        assert not torch.isinf(out).any(), "Inf in bridge logits"

    def test_forward_matches_hf_exactly(
        self, lfm2_bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        hf_model = lfm2_bridge.original_model
        with torch.no_grad():
            bridge_out = lfm2_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out.float() - hf_out.float()).abs().max().item()
        assert max_diff == 0.0, (
            f"Bridge vs HF forward max diff = {max_diff:.2e}. "
            "Expected 0 because Lfm2ShortConvBridge.forward() delegates to HF."
        )

    def test_forward_no_nan_on_longer_sequence(self, lfm2_bridge: TransformerBridge) -> None:
        tokens = torch.arange(1, 33).unsqueeze(0)
        with torch.no_grad():
            out = lfm2_bridge(tokens)
        assert not torch.isnan(out).any(), "NaN in logits for 32-token sequence"


# ---------------------------------------------------------------------------
# Multi-token generation parity (exercises DynamicCache state handling)
# ---------------------------------------------------------------------------


class TestLfm2Generation:
    """Bridge greedy generation must match HF native generate() exactly.

    This exercises the DynamicCache stateful loop: attention layers write KV
    entries, conv layers write conv1D states, all via the same
    unified cache object. Token-level equality with HF confirms the state
    threading is correct across both layer types (conv / attention).
    """

    @pytest.fixture(scope="class")
    def prompt(self) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, 4]])

    def test_generation_produces_tokens(
        self, lfm2_bridge: TransformerBridge, prompt: torch.Tensor
    ) -> None:
        with torch.no_grad():
            result = lfm2_bridge.generate(prompt, max_new_tokens=5, do_sample=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 9)

    def test_greedy_matches_hf_exactly(
        self, lfm2_bridge: TransformerBridge, prompt: torch.Tensor
    ) -> None:
        """Bit-for-bit equality with HF generate() over 8 new tokens."""
        hf_model = lfm2_bridge.original_model
        with torch.no_grad():
            bridge_out = lfm2_bridge.generate(prompt, max_new_tokens=8, do_sample=False)
            hf_out = hf_model.generate(prompt, max_new_tokens=8, do_sample=False)
        assert torch.equal(bridge_out, hf_out), (
            f"Token mismatch between bridge and HF.\n"
            f"  bridge : {bridge_out.tolist()}\n"
            f"  hf     : {hf_out.tolist()}\n"
            "DynamicCache state threading across layer types is likely wrong."
        )

    def test_generation_is_deterministic(
        self, lfm2_bridge: TransformerBridge, prompt: torch.Tensor
    ) -> None:
        """Two identical greedy calls must produce identical tokens."""
        with torch.no_grad():
            out1 = lfm2_bridge.generate(prompt, max_new_tokens=4, do_sample=False)
            out2 = lfm2_bridge.generate(prompt, max_new_tokens=4, do_sample=False)
        assert torch.equal(out1, out2), "Greedy generation is not deterministic"


# ---------------------------------------------------------------------------
# Hook coverage: bridge hooks fire for both conv and attention layers
# ---------------------------------------------------------------------------


class TestLfm2HookCoverage:
    """run_with_cache captures residual stream and conv/attention hooks on relevant layer types."""

    @pytest.fixture(scope="class")
    def cache(self, lfm2_bridge: TransformerBridge):
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            _, cache = lfm2_bridge.run_with_cache(tokens)
        return cache

    def test_block_hooks_fire(self, cache) -> None:
        for i in [0, 6, 13]:
            assert f"blocks.{i}.hook_in" in cache, f"Missing hook_in for block {i}"
            assert f"blocks.{i}.hook_out" in cache, f"Missing hook_out for block {i}"

    def test_lfm2_short_conv_submodule_hooks_fire(
        self, cache, lfm2_bridge: TransformerBridge
    ) -> None:
        """Lfm2ShortConv layers should expose in / conv / out hooks."""
        lt = getattr(lfm2_bridge.cfg, "layer_types", [])
        conv_indices = [i for i, t in enumerate(lt) if t == "conv"]
        assert conv_indices, "No conv layers found in layer_types"
        # Check a few conv layers
        for i in conv_indices[:3]:
            for submod in ("in", "conv", "out"):
                key_in = f"blocks.{i}.conv.{submod}.hook_in"
                key_out = f"blocks.{i}.conv.{submod}.hook_out"
                assert key_in in cache, f"Missing {key_in}"
                assert key_out in cache, f"Missing {key_out}"

    def test_no_nan_in_cache(self, cache) -> None:
        for key, val in cache.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                assert not torch.isnan(val).any(), f"NaN in cache['{key}']"
