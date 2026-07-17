"""Integration tests for the Raven / Huginn architecture adapter (RavenForCausalLM).

Model: tomg-group-umd/huginn-0125
  - Remote-code (auto_map → raven_modeling_minimal.RavenForCausalLM), so no
    tiny random checkpoint exists; this loads the real ~3.5B weights (~14GB
    download, ~28GB RAM for the two fp32 copies). Gated out of CI on
    network/memory budget; run locally with:
        uv run pytest tests/integration/model_bridge/test_raven_adapter.py -v -m slow

Huginn is a depth-recurrent decoder: prelude (2 blocks) → weight-tied
recurrent core (4 blocks applied N times, default N = mean_recurrence = 32) →
coda (2 blocks). The recurrence, the per-step prelude re-injection, the
sandwich norms and RoPE all live inside the remote-code forward, which the
bridge delegates to, so logit parity holds unchanged.

Two behaviours make Huginn different from a flat decoder and shape these tests:

1. Random initial latent state. ``iterate_forward`` seeds the recurrence with
   ``torch.randn_like`` (``initialize_state``), so the forward is
   non-deterministic across calls. Parity is only meaningful with the RNG
   pinned identically before the bridge and HF calls; ``_seeded`` does that.

2. Weight-tied recurrent core. The four ``core_block`` blocks run once per
   recurrence step, so their hooks fire N times per forward and
   ``run_with_cache`` keeps the final step (pinned by TestRavenRecurrentCore).

Comparing against an independent HF load (never ``bridge.original_model``,
whose modules are hook-wrapped) keeps the parity check honest.
"""

import os
import platform

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge

MODEL = "tomg-group-umd/huginn-0125"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        bool(os.getenv("CI")),
        reason="tomg-group-umd/huginn-0125: ~14GB download + ~28GB RAM, too large for CI",
    ),
]

# Depth-recurrent sandwich-norm model with 100+ effective layer applications at
# fp32; allow a wider op-order noise floor on GH Actions macOS-arm64 (same idiom
# as the Ouro and SmolLM3 bridge-vs-HF parity tests).
_MACOS_ARM64 = platform.system() == "Darwin" and platform.machine() == "arm64"
FP32_NOISE_TOL = 1e-2 if _MACOS_ARM64 else 1e-3

# Reduced recurrence for a tractable local run; passed identically to both the
# bridge and HF forward so parity is exact regardless of the value.
NUM_STEPS = 8


def _seeded(fn, seed: int = 0):
    """Run ``fn`` with the global torch RNG pinned.

    Huginn's ``initialize_state`` draws the initial latent from
    ``torch.randn_like``; pinning the seed makes the bridge and HF forwards
    start from the same latent so their logits are comparable.
    """
    torch.manual_seed(seed)
    return fn()


@pytest.fixture(scope="module")
def bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(
        MODEL, device="cpu", dtype=torch.float32, trust_remote_code=True
    )


@pytest.fixture(scope="module")
def hf_eager(bridge: TransformerBridge) -> torch.nn.Module:
    """HF model loaded independently of the bridge's wrapped instance.

    Depends on the bridge fixture only for ordering: booting the bridge runs
    the adapter's prepare_loading patch (guarding transformers v5 weight
    re-init) on the cached raven modeling module, and this plain
    from_pretrained load benefits from that patch too. The weights are still a
    separate copy.
    """
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    return AutoModelForCausalLM.from_pretrained(
        MODEL,
        config=config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).eval()


@pytest.fixture(scope="module")
def tokens(hf_eager: torch.nn.Module) -> torch.Tensor:
    """A short, deterministic random token sequence."""
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, hf_eager.config.vocab_size, (1, 16), generator=generator)


class TestRavenBridgeCreation:
    """The bridge loads the remote-code model and wires the three phases."""

    def test_boot_transformers_succeeds(self, bridge: TransformerBridge) -> None:
        assert bridge is not None

    def test_phase_block_lists_present(self, bridge: TransformerBridge) -> None:
        """Prelude / core_block / coda are exposed as separate block lists."""
        assert len(bridge.prelude) == bridge.cfg.n_layers_in_prelude
        assert len(bridge.core_block) == bridge.cfg.n_layers_in_recurrent_block
        assert len(bridge.coda) == bridge.cfg.n_layers_in_coda

    def test_physical_layer_split(
        self, bridge: TransformerBridge, hf_eager: torch.nn.Module
    ) -> None:
        """The three lists sum to num_hidden_layers (the physical block count)."""
        physical = len(bridge.prelude) + len(bridge.core_block) + len(bridge.coda)
        assert physical == hf_eager.config.num_hidden_layers

    def test_config_flags(self, bridge: TransformerBridge) -> None:
        assert bridge.cfg.normalization_type == "RMS"
        assert bridge.cfg.positional_embedding_type == "rotary"
        assert bridge.cfg.gated_mlp is True
        assert bridge.cfg.final_rms is True

    def test_sandwich_norms_wired(self, bridge: TransformerBridge) -> None:
        """All four per-block RMSNorms are bridged on each phase."""
        for phase in (bridge.prelude, bridge.core_block, bridge.coda):
            block = phase[0]
            for ln in ("norm_1", "norm_2", "norm_3", "norm_4"):
                assert hasattr(block, ln), f"missing {ln}"

    def test_recurrence_config_propagates(
        self, bridge: TransformerBridge, hf_eager: torch.nn.Module
    ) -> None:
        assert bridge.cfg.mean_recurrence == hf_eager.config.mean_recurrence


class TestRavenHFDelegation:
    """The bridge wraps the live remote-code modules in place (no copies)."""

    def test_core_block_is_the_hf_tree_module(self, bridge: TransformerBridge) -> None:
        assert bridge.core_block[0] is bridge.original_model.transformer.core_block[0]
        assert type(bridge.core_block[0].original_component).__name__ == "SandwichBlock"

    def test_attention_combined_qkv_wraps_live_hf_modules(self, bridge: TransformerBridge) -> None:
        attn = bridge.core_block[0].attn
        assert type(attn.original_component).__name__ == "CausalSelfAttention"
        # Huginn uses a combined Wqkv projection (not split q/k/v).
        assert attn.qkv is attn.original_component.Wqkv
        assert attn.o is attn.original_component.proj

    def test_mlp_combined_fc_wraps_live_hf_modules(self, bridge: TransformerBridge) -> None:
        mlp = bridge.core_block[0].mlp
        assert type(mlp.original_component).__name__ == "GatedMLP"
        # Combined gate+up "fc" and output "proj".
        assert getattr(mlp, "in") is mlp.original_component.fc
        assert mlp.out is mlp.original_component.proj

    def test_sandwich_norms_wrap_live_hf_modules(self, bridge: TransformerBridge) -> None:
        block = bridge.core_block[0]
        layer = block.original_component
        assert block.norm_1 is layer.norm_1
        assert block.norm_2 is layer.norm_2
        assert block.norm_3 is layer.norm_3
        assert block.norm_4 is layer.norm_4


class TestRavenForwardEquivalence:
    """Bridge output reproduces the HF eager reference through the recurrence.

    Both forwards are given the same reduced ``num_steps`` and the same pinned
    RNG seed, so the random initial latent state and recurrence depth match and
    the logits are directly comparable.
    """

    def test_forward_logits_match_hf_eager(
        self, bridge: TransformerBridge, hf_eager: torch.nn.Module, tokens: torch.Tensor
    ) -> None:
        with torch.inference_mode():
            bridge_logits = _seeded(lambda: bridge(tokens, num_steps=NUM_STEPS))
            hf_logits = _seeded(lambda: hf_eager(tokens, num_steps=NUM_STEPS).logits)
        max_diff = (bridge_logits - hf_logits).abs().max().item()
        assert max_diff < FP32_NOISE_TOL, (
            f"Raven bridge vs HF eager logit drift={max_diff:.2e} exceeds the "
            f"fp32-noise tolerance {FP32_NOISE_TOL:.0e}."
        )


class TestRavenRecurrentCore:
    """Pin the weight-tied recurrent-core semantics the adapter documents."""

    def test_core_block_hook_fires_once_per_recurrence_step(
        self, bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        """The same physical core block executes num_steps times per forward.

        This is the load-bearing difference from a flat decoder: hooks on
        core_block.{i} fire once per recurrence step and run_with_cache keeps
        the final step. If HF's remote code changes the loop (or the bridge
        stops delegating it), this fails.
        """
        fired: list[bool] = []
        _seeded(
            lambda: bridge.run_with_hooks(
                tokens,
                num_steps=NUM_STEPS,
                fwd_hooks=[("core_block.0.hook_out", lambda value, hook: fired.append(True))],
            )
        )
        assert len(fired) == NUM_STEPS

    def test_prelude_hook_fires_once(self, bridge: TransformerBridge, tokens: torch.Tensor) -> None:
        """Prelude blocks are non-recurrent: their hooks fire exactly once."""
        fired: list[bool] = []
        _seeded(
            lambda: bridge.run_with_hooks(
                tokens,
                num_steps=NUM_STEPS,
                fwd_hooks=[("prelude.0.hook_out", lambda value, hook: fired.append(True))],
            )
        )
        assert len(fired) == 1


class TestRavenHookShapes:
    """Residual-stream hooks fire with the expected shape on each phase."""

    @pytest.mark.parametrize("phase", ["prelude", "core_block", "coda"])
    def test_hook_out_shape_matches_residual_stream(
        self, bridge: TransformerBridge, tokens: torch.Tensor, phase: str
    ) -> None:
        captured: list[torch.Tensor] = []

        def _capture(value, hook):
            captured.append(value[0] if isinstance(value, tuple) else value)
            return value

        _seeded(
            lambda: bridge.run_with_hooks(
                tokens,
                num_steps=NUM_STEPS,
                fwd_hooks=[(f"{phase}.0.hook_out", _capture)],
            )
        )
        assert captured
        assert captured[-1].shape == (1, tokens.shape[1], bridge.cfg.d_model)
