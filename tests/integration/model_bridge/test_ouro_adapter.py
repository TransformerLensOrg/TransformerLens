"""Integration tests for the Ouro architecture adapter (OuroForCausalLM).

Model: ByteDance/Ouro-1.4B
  - Remote-code (auto_map to modeling_ouro), so no tiny random checkpoint
    exists; this loads the real 1.4B weights (~2.8GB download, ~11GB RAM for
    the two fp32 copies). Gated out of CI on network/memory budget; run
    locally with:
        uv run pytest tests/integration/model_bridge/test_ouro_adapter.py -v -m slow

Ouro is a looped-depth (Universal Transformer) model: the remote-code
OuroModel.forward applies the shared 24-layer stack total_ut_steps=4 times,
with model.norm after each pass. The bridge delegates the loop to HF's own
forward, so logit parity holds unchanged; the one observable difference from
a standard decoder is that each physical block's hooks fire once per UT step
(pinned by TestOuroLoopedDepth below) and a cache keeps the final step only.

The bridge always reimplements attention in eager mode (so the score and
pattern hooks fire), so the reference HF model is loaded independently with
attn_implementation="eager" too. Comparing against an independent HF load
(never bridge.original_model, whose modules are hook-wrapped) keeps the
parity check honest.
"""

import os
import platform

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge

MODEL = "ByteDance/Ouro-1.4B"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        bool(os.getenv("CI")),
        reason="ByteDance/Ouro-1.4B: 2.8GB download + ~11GB RAM, too large for CI",
    ),
]

# Sandwich-norm model with 96 effective layer applications at fp32; allow a
# wider op-order noise floor on GH Actions macOS-arm64 (same idiom as the
# SmolLM3 and general bridge-vs-HF parity tests).
_MACOS_ARM64 = platform.system() == "Darwin" and platform.machine() == "arm64"
FP32_NOISE_TOL = 1e-2 if _MACOS_ARM64 else 1e-3


@pytest.fixture(scope="module")
def bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(
        MODEL, device="cpu", dtype=torch.float32, trust_remote_code=True
    )


@pytest.fixture(scope="module")
def hf_eager(bridge: TransformerBridge) -> torch.nn.Module:
    """HF model loaded independently of the bridge's wrapped instance.

    Depends on the bridge fixture only for ordering: booting the bridge runs
    the adapter's prepare_loading patch (restoring the "default" RoPE init
    that transformers v5 removed) on the cached modeling_ouro module, and
    this plain from_pretrained load needs that patch too. The weights are
    still a separate copy.

    The pad_token_id shim mirrors the bridge load path
    (sources/transformers.py): transformers v5 no longer materializes a
    default pad_token_id on configs, but Ouro's remote code reads
    config.pad_token_id unconditionally. Mirroring the same eos fallback
    keeps both copies' embedding padding_idx identical.
    """
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if "pad_token_id" not in config.__dict__:
        fallback_pad = getattr(config, "eos_token_id", None)
        if isinstance(fallback_pad, list):
            fallback_pad = fallback_pad[0] if fallback_pad else None
        config.pad_token_id = fallback_pad
    return AutoModelForCausalLM.from_pretrained(
        MODEL,
        config=config,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval()


@pytest.fixture(scope="module")
def tokens(hf_eager: torch.nn.Module) -> torch.Tensor:
    """A short, deterministic random token sequence.

    Random ids keep the test tokenizer-free and cover the vocab beyond the
    few ids a short prompt would produce.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, hf_eager.config.vocab_size, (1, 16), generator=generator)


class TestOuroBridgeCreation:
    """The bridge loads the remote-code model and wires the sandwich norms."""

    def test_boot_transformers_succeeds(self, bridge: TransformerBridge) -> None:
        assert bridge is not None

    def test_block_count_is_physical_layers(
        self, bridge: TransformerBridge, hf_eager: torch.nn.Module
    ) -> None:
        """n_layers counts the 24 physical layers, not the 96 loop applications."""
        assert len(bridge.blocks) == hf_eager.config.num_hidden_layers
        assert bridge.cfg.n_layers == hf_eager.config.num_hidden_layers

    def test_config_flags(self, bridge: TransformerBridge) -> None:
        assert bridge.cfg.normalization_type == "RMS"
        assert bridge.cfg.positional_embedding_type == "rotary"
        assert bridge.cfg.gated_mlp is True

    def test_sandwich_norms_wired(self, bridge: TransformerBridge) -> None:
        """All four per-layer RMSNorms are bridged."""
        block = bridge.blocks[0]
        for ln_name in ("ln1", "ln1_post", "ln2", "ln2_post"):
            assert hasattr(block, ln_name), f"missing {ln_name} on block 0"


class TestOuroHFDelegation:
    """The bridge wraps the live remote-code modules in place (no copies).

    Wrapping is in-place: the HF tree slot holds the bridge component itself,
    and the bridge's submodule IS the module the wrapped OuroDecoderLayer /
    OuroAttention executes. Assert those identities directly.
    """

    def test_blocks_are_the_hf_tree_layers(self, bridge: TransformerBridge) -> None:
        assert bridge.blocks[0] is bridge.original_model.model.layers[0]
        assert type(bridge.blocks[0].original_component).__name__ == "OuroDecoderLayer"

    def test_attention_projections_wrap_live_hf_modules(self, bridge: TransformerBridge) -> None:
        attn = bridge.blocks[0].attn
        assert type(attn.original_component).__name__ == "OuroAttention"
        assert attn.q is attn.original_component.q_proj
        assert attn.k is attn.original_component.k_proj
        assert attn.v is attn.original_component.v_proj
        assert attn.o is attn.original_component.o_proj

    def test_sandwich_norms_wrap_live_hf_modules(self, bridge: TransformerBridge) -> None:
        block = bridge.blocks[0]
        layer = block.original_component
        assert block.ln1 is layer.input_layernorm
        assert block.ln1_post is layer.input_layernorm_2
        assert block.ln2 is layer.post_attention_layernorm
        assert block.ln2_post is layer.post_attention_layernorm_2


class TestOuroForwardEquivalence:
    """Bridge output reproduces the HF eager reference through the full UT loop."""

    def test_forward_logits_match_hf_eager(
        self, bridge: TransformerBridge, hf_eager: torch.nn.Module, tokens: torch.Tensor
    ) -> None:
        with torch.inference_mode():
            bridge_logits = bridge(tokens)
            hf_logits = hf_eager(tokens).logits
        max_diff = (bridge_logits - hf_logits).abs().max().item()
        assert max_diff < FP32_NOISE_TOL, (
            f"Ouro bridge vs HF eager logit drift={max_diff:.2e} exceeds the "
            f"fp32-noise tolerance {FP32_NOISE_TOL:.0e}."
        )


class TestOuroLoopedDepth:
    """Pin the Universal-Transformer loop semantics the adapter documents."""

    def test_block_hook_fires_once_per_ut_step(
        self, bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        """The same physical block executes total_ut_steps times per forward.

        This is the load-bearing behavioural difference from a standard
        decoder: hooks on blocks.{i} fire once per loop step, and
        run_with_cache keeps the final step's value. If HF's remote code ever
        changes the loop (or the bridge stops delegating it), this fails.
        """
        fired: list[bool] = []
        bridge.run_with_hooks(
            tokens,
            fwd_hooks=[("blocks.0.hook_out", lambda value, hook: fired.append(True))],
        )
        assert len(fired) == bridge.cfg.total_ut_steps

    def test_hook_out_shape_matches_residual_stream(
        self, bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        captured: list[torch.Tensor] = []

        def _capture(value, hook):
            captured.append(value[0] if isinstance(value, tuple) else value)
            return value

        bridge.run_with_hooks(tokens, fwd_hooks=[("blocks.0.hook_out", _capture)])
        assert captured
        assert captured[-1].shape == (1, tokens.shape[1], bridge.cfg.d_model)

    def test_bridge_runs_its_own_attention_reconstruction(
        self, bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        """Anti-tautology guard: prove the bridge's custom attention path executes.

        If a future refactor made the bridge delegate to HF attention directly,
        the parity test above would pass trivially. Assert a bridge-specific
        hook fires during the forward pass instead.
        """
        fired: list[bool] = []
        bridge.run_with_hooks(
            tokens,
            fwd_hooks=[
                ("blocks.0.attn.hook_attn_scores", lambda value, hook: fired.append(True)),
            ],
        )
        assert fired, (
            "blocks.0.attn.hook_attn_scores did not fire, so the bridge no longer "
            "runs its own attention reconstruction and the parity test is tautological."
        )


class TestOuroConfigPropagation:
    """Loop-related HF config attrs surface on bridge.cfg via _HF_PASSTHROUGH_ATTRS."""

    def test_total_ut_steps_propagates(
        self, bridge: TransformerBridge, hf_eager: torch.nn.Module
    ) -> None:
        assert bridge.cfg.total_ut_steps == hf_eager.config.total_ut_steps

    def test_early_exit_threshold_propagates(
        self, bridge: TransformerBridge, hf_eager: torch.nn.Module
    ) -> None:
        assert bridge.cfg.early_exit_threshold == hf_eager.config.early_exit_threshold
