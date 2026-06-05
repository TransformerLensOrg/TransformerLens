"""Integration tests for the SmolLM3 architecture adapter (SmolLM3ForCausalLM).

Model: yujiepan/smollm3-tiny-random
  - 2 layers, CPU-safe random init.
  - no_rope_layers = [1, 0]: layer 0 applies RoPE, layer 1 is a NoPE layer.
    This is the smallest checkpoint that exercises BOTH attention paths, so the
    per-layer parity test below is a direct check on _SmolLM3AttentionBridge.
  - sliding_window = 128: a no-op for the short prompts used here (sequence
    length stays well under the window), so these tests isolate the NoPE
    behaviour without a sliding-window confound. The released SmolLM3-3B sets
    sliding_window = null anyway.

The bridge always reimplements attention in eager mode (so the score and pattern
hooks fire), so the reference HF model is loaded with attn_implementation="eager"
too. Bridge vs HF eager matches to fp32 op-order noise. See the general parity
test in test_bridge_vs_hf_eager_parity.py for the same idiom on Pythia.
"""

import platform

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.supported_architectures.smollm3 import (
    _SmolLM3AttentionBridge,
)

MODEL = "yujiepan/smollm3-tiny-random"

# Wider fp32 op-order noise floor on GH Actions macOS-arm64, matching the existing
# bridge-vs-HF parity test. Locally this comparison is bit-exact (0.0).
_MACOS_ARM64 = platform.system() == "Darwin" and platform.machine() == "arm64"
FP32_NOISE_TOL = 1e-2 if _MACOS_ARM64 else 1e-5


@pytest.fixture(scope="module")
def bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def hf_eager() -> torch.nn.Module:
    """HF model loaded independently of the bridge's wrapped instance."""
    return AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()


@pytest.fixture(scope="module")
def tokens(hf_eager: torch.nn.Module) -> torch.Tensor:
    """A short, deterministic random token sequence.

    Random ids avoid depending on a tokenizer for a randomly-initialised
    checkpoint. Length 16 keeps the sequence under the 128 sliding window.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, hf_eager.config.vocab_size, (1, 16), generator=generator)


class TestSmolLM3BridgeCreation:
    """The bridge loads cleanly and wires the NoPE-aware attention bridge."""

    def test_boot_transformers_succeeds(self, bridge: TransformerBridge) -> None:
        assert bridge is not None

    def test_block_count_matches_hf(
        self, bridge: TransformerBridge, hf_eager: torch.nn.Module
    ) -> None:
        assert len(bridge.blocks) == hf_eager.config.num_hidden_layers

    def test_attention_bridge_is_nope_aware(self, bridge: TransformerBridge) -> None:
        """Every block uses the NoPE-aware subclass, not the plain base bridge."""
        for block in bridge.blocks:
            assert type(block.attn) is _SmolLM3AttentionBridge

    def test_checkpoint_actually_mixes_rope_and_nope(self, hf_eager: torch.nn.Module) -> None:
        """Guard the premise of the parity tests: this model has one of each layer.

        If a future revision of the checkpoint dropped the NoPE layer,
        the per-layer parity test would still pass
        but would no longer prove the NoPE path matches HF.
        Fail loudly here instead.
        """
        use_rope = [int(layer.self_attn.use_rope) for layer in hf_eager.model.layers]
        assert 1 in use_rope, "expected at least one RoPE layer"
        assert 0 in use_rope, "expected at least one NoPE layer (use_rope == 0)"


class TestSmolLM3MatchesHuggingFace:
    """Bridge output reproduces the HF eager reference, including the NoPE layer."""

    def test_forward_logits_match_hf_eager(
        self, bridge: TransformerBridge, hf_eager: torch.nn.Module, tokens: torch.Tensor
    ) -> None:
        with torch.inference_mode():
            bridge_logits = bridge(tokens)
            hf_logits = hf_eager(tokens).logits
        max_diff = (bridge_logits - hf_logits).abs().max().item()
        assert max_diff < FP32_NOISE_TOL, (
            f"SmolLM3 bridge vs HF eager logit drift={max_diff:.2e} exceeds the "
            f"fp32-noise tolerance {FP32_NOISE_TOL:.0e}."
        )

    def test_per_layer_residual_matches_hf_eager(
        self, bridge: TransformerBridge, hf_eager: torch.nn.Module, tokens: torch.Tensor
    ) -> None:
        """Per-layer parity, which pins the NoPE layer specifically.

        The NoPE layer (use_rope == 0) is where _SmolLM3AttentionBridge suppresses
        position embeddings. If that handling were wrong, this layer's residual
        would diverge from HF even when the final logits happened to wash it out.
        """
        n_layers = hf_eager.config.num_hidden_layers

        hf_layer_out: dict[int, torch.Tensor] = {}

        def _make_hf_hook(idx: int):
            def _hook(_module, _inputs, output):
                hf_layer_out[idx] = (output[0] if isinstance(output, tuple) else output).detach()

            return _hook

        handles = [
            layer.register_forward_hook(_make_hf_hook(i))
            for i, layer in enumerate(hf_eager.model.layers)
        ]
        try:
            with torch.inference_mode():
                hf_eager(tokens)
        finally:
            for handle in handles:
                handle.remove()

        bridge_layer_out: dict[int, torch.Tensor] = {}
        fwd_hooks = [
            (
                f"blocks.{i}.hook_resid_post",
                lambda value, hook, idx=i: bridge_layer_out.__setitem__(idx, value.detach()),
            )
            for i in range(n_layers)
        ]
        with torch.inference_mode():
            bridge.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        for i, layer in enumerate(hf_eager.model.layers):
            drift = (hf_layer_out[i] - bridge_layer_out[i]).abs().max().item()
            kind = "RoPE" if layer.self_attn.use_rope else "NoPE"
            assert drift < FP32_NOISE_TOL, (
                f"layer {i} ({kind}) residual drift={drift:.2e} exceeds the "
                f"fp32-noise tolerance {FP32_NOISE_TOL:.0e}."
            )

    def test_bridge_runs_its_own_attention_reconstruction(
        self, bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        """Anti-tautology guard: prove the bridge's custom attention path executes.

        If a future refactor made the bridge delegate to HF attention directly,
        the parity tests above would pass trivially. This fails fast in that case
        by asserting a bridge-specific hook fires during the forward pass.
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
            "runs its own attention reconstruction and the parity tests are tautological."
        )
