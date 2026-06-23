"""Integration tests for the NemotronH architecture adapter.

Verifies forward-pass and generation parity against nvidia/Nemotron-H-8B-Base:
- Forward-pass logits match HF exactly (bridge delegates the full forward to HF)
- Greedy multi-token generation matches HF bit-for-bit (exercises DynamicCache
  state handling across attention, Mamba-2, MLP, and MoE layers)
- Sanity checks: config flags, block count, hook coverage

Note: requires ~18 GB RAM (CPU) or ~16 GB VRAM (GPU) to load the 8B checkpoint.
On a machine with less memory, skip with:
    pytest -m "not slow" tests/integration/model_bridge/test_nemotron_h_adapter.py

Run with GPU acceleration:
    CUDA_VISIBLE_DEVICES=0 pytest tests/integration/model_bridge/test_nemotron_h_adapter.py -v -s
"""

import gc

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    SSM2MixerBridge,
    SSMBlockBridge,
)

MODEL = "nvidia/Nemotron-H-8B-Base"

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
def nemotron_bridge():
    device = _device()
    dtype = _dtype()
    bridge = TransformerBridge.boot_transformers(MODEL, device=device, dtype=dtype)
    yield bridge
    # Cleanup
    del bridge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for _ in range(3):
        gc.collect()


# ---------------------------------------------------------------------------
# Config and bridge structure
# ---------------------------------------------------------------------------


class TestNemotronHBridgeCreation:
    """Smoke-test that the bridge loads with the right config flags."""

    def test_config_flags(self, nemotron_bridge: TransformerBridge) -> None:
        cfg = nemotron_bridge.cfg
        assert cfg.normalization_type == "RMS"
        assert cfg.uses_rms_norm is True
        assert cfg.positional_embedding_type == "none"
        assert cfg.gated_mlp is False
        assert cfg.is_stateful is True

    def test_block_count(self, nemotron_bridge: TransformerBridge) -> None:
        # Nemotron-H-8B has 56 layers
        assert len(nemotron_bridge.blocks) == 56

    def test_blocks_are_ssm_block_bridge(self, nemotron_bridge: TransformerBridge) -> None:
        assert isinstance(nemotron_bridge.blocks[0], SSMBlockBridge)

    def test_mixer_is_ssm2_mixer_bridge(self, nemotron_bridge: TransformerBridge) -> None:
        assert isinstance(nemotron_bridge.blocks[0].mixer, SSM2MixerBridge)

    def test_layers_block_type_populated(self, nemotron_bridge: TransformerBridge) -> None:
        lbt = getattr(nemotron_bridge.cfg, "layers_block_type", [])
        assert len(lbt) == len(nemotron_bridge.blocks)
        # Should contain at least one attention and one mamba layer
        assert "attention" in lbt
        assert "mamba" in lbt

    def test_mamba_intermediate_size_positive(self, nemotron_bridge: TransformerBridge) -> None:
        assert getattr(nemotron_bridge.cfg, "mamba_intermediate_size", 0) > 0

    def test_conv_dim_positive(self, nemotron_bridge: TransformerBridge) -> None:
        assert getattr(nemotron_bridge.cfg, "conv_dim", 0) > 0


# ---------------------------------------------------------------------------
# Forward-pass parity
# ---------------------------------------------------------------------------


class TestNemotronHForwardPass:
    """Bridge logits must match HF logits exactly.

    NemotronHArchitectureAdapter uses SSM2MixerBridge with a pure passthrough
    forward (original_component(*args, **kwargs)), so the bridge never
    reimplements any computation. Parity with HF should be exact (diff == 0),
    not just close.
    """

    @pytest.fixture(scope="class")
    def tokens(self) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    def test_forward_returns_logits(
        self, nemotron_bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        tokens = tokens.to(_device())
        with torch.no_grad():
            out = nemotron_bridge(tokens)
        assert out.shape == (1, 8, nemotron_bridge.cfg.d_vocab)
        assert not torch.isnan(out).any(), "NaN in bridge logits"
        assert not torch.isinf(out).any(), "Inf in bridge logits"

    def test_forward_matches_hf_exactly(
        self, nemotron_bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        tokens = tokens.to(_device())
        hf_model = nemotron_bridge.original_model
        with torch.no_grad():
            bridge_out = nemotron_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out.float() - hf_out.float()).abs().max().item()
        assert max_diff == 0.0, (
            f"Bridge vs HF forward max diff = {max_diff:.2e}. "
            "Expected 0 because SSM2MixerBridge.forward() is a pure passthrough."
        )

    def test_forward_no_nan_on_longer_sequence(
        self, nemotron_bridge: TransformerBridge
    ) -> None:
        # Exercise more SSM steps to catch state accumulation issues
        tokens = torch.arange(1, 33).unsqueeze(0).to(_device())
        with torch.no_grad():
            out = nemotron_bridge(tokens)
        assert not torch.isnan(out).any(), "NaN in logits for 32-token sequence"


# ---------------------------------------------------------------------------
# Multi-token generation parity (exercises DynamicCache state handling)
# ---------------------------------------------------------------------------


class TestNemotronHGeneration:
    """Bridge greedy generation must match HF native generate() exactly.

    This exercises the DynamicCache stateful loop: attention layers write KV
    entries, Mamba-2 layers write recurrent SSM states, all via the same
    unified cache object. Token-level equality with HF confirms the state
    threading is correct across all four layer types (mamba / attention /
    moe / mlp).
    """

    @pytest.fixture(scope="class")
    def prompt(self) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, 4]])

    def test_generation_produces_tokens(
        self, nemotron_bridge: TransformerBridge, prompt: torch.Tensor
    ) -> None:
        prompt = prompt.to(_device())
        with torch.no_grad():
            result = nemotron_bridge.generate(prompt, max_new_tokens=5, do_sample=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 9)  # 4 prompt + 5 new

    def test_greedy_matches_hf_exactly(
        self, nemotron_bridge: TransformerBridge, prompt: torch.Tensor
    ) -> None:
        """Bit-for-bit equality with HF generate() over 8 new tokens."""
        prompt = prompt.to(_device())
        hf_model = nemotron_bridge.original_model
        with torch.no_grad():
            bridge_out = nemotron_bridge.generate(
                prompt, max_new_tokens=8, do_sample=False
            )
            hf_out = hf_model.generate(
                prompt, max_new_tokens=8, do_sample=False, pad_token_id=0
            )
        assert torch.equal(bridge_out, hf_out), (
            f"Token mismatch between bridge and HF.\n"
            f"  bridge : {bridge_out.tolist()}\n"
            f"  hf     : {hf_out.tolist()}\n"
            "DynamicCache state threading across layer types is likely wrong."
        )

    def test_generation_is_deterministic(
        self, nemotron_bridge: TransformerBridge, prompt: torch.Tensor
    ) -> None:
        """Two identical greedy calls must produce identical tokens."""
        prompt = prompt.to(_device())
        with torch.no_grad():
            out1 = nemotron_bridge.generate(prompt, max_new_tokens=4, do_sample=False)
            out2 = nemotron_bridge.generate(prompt, max_new_tokens=4, do_sample=False)
        assert torch.equal(out1, out2), "Greedy generation is not deterministic"


# ---------------------------------------------------------------------------
# Hook coverage: bridge hooks fire for both Mamba and attention layers
# ---------------------------------------------------------------------------


class TestNemotronHHookCoverage:
    """run_with_cache captures residual stream and mixer hooks on all layer types."""

    @pytest.fixture(scope="class")
    def cache(self, nemotron_bridge: TransformerBridge):
        tokens = torch.tensor([[1, 2, 3, 4, 5]]).to(_device())
        with torch.no_grad():
            _, cache = nemotron_bridge.run_with_cache(tokens)
        return cache

    def test_block_hooks_fire(self, cache, nemotron_bridge: TransformerBridge) -> None:
        for i in [0, 28, 55]:
            assert f"blocks.{i}.hook_in" in cache, f"Missing hook_in for block {i}"
            assert f"blocks.{i}.hook_out" in cache, f"Missing hook_out for block {i}"

    def test_mamba_mixer_submodule_hooks_fire(
        self, cache, nemotron_bridge: TransformerBridge
    ) -> None:
        """Mamba layers should expose in_proj / conv1d / out_proj hooks."""
        lbt = getattr(nemotron_bridge.cfg, "layers_block_type", [])
        mamba_indices = [i for i, t in enumerate(lbt) if t == "mamba"]
        assert mamba_indices, "No mamba layers found in layers_block_type"
        # Check a few mamba layers
        for i in mamba_indices[:3]:
            for submod in ("in_proj", "conv1d", "out_proj"):
                key_in = f"blocks.{i}.mixer.{submod}.hook_in"
                key_out = f"blocks.{i}.mixer.{submod}.hook_out"
                assert key_in in cache, f"Missing {key_in}"
                assert key_out in cache, f"Missing {key_out}"

    def test_no_transformer_specific_hooks(self, cache) -> None:
        """SSMBlockBridge must not inject transformer-shaped hook names."""
        forbidden = ("hook_resid_mid", "hook_attn_out", "hook_mlp_out")
        bad = [k for k in cache if any(f in k for f in forbidden)]
        assert bad == [], f"Unexpected transformer-shaped hooks: {bad[:5]}"

    def test_no_nan_in_cache(self, cache) -> None:
        for key, val in cache.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                assert not torch.isnan(val).any(), f"NaN in cache['{key}']"
