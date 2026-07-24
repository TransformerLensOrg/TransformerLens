"""Integration tests for the RWKV-7 architecture adapter (RWKV7ForCausalLM).

Model: fla-hub/rwkv7-0.1B-g1
  - Remote-code (``trust_remote_code=True``); the modeling classes live in the
    flash-linear-attention (``fla``) library, which must be installed:
        pip install flash-linear-attention
  - ~191M params (~0.8GB download). Gated out of CI on the network/dependency
    budget; run locally with:
        uv run pytest tests/integration/model_bridge/test_rwkv7_adapter.py -v -m slow

RWKV-7 is an attention-free recurrent decoder: a flat stack of pre-norm blocks,
each a generalized-delta-rule time-mixing sublayer (``attn``) plus a
token-shifted squared-ReLU channel-mixing sublayer (``ffn``), wrapped by standard
biased LayerNorm. The recurrence, the token shift, and the cross-block
``v_first`` threading all live inside the ``fla`` remote-code forward, which the
bridge delegates to, so logit parity holds unchanged.

Comparing against an independent HF load (never ``bridge.original_model``, whose
modules are hook-wrapped) keeps the parity check honest.
"""

import os
import platform

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

pytest.importorskip("fla", reason="RWKV-7 needs flash-linear-attention (pip install it)")

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

MODEL = "fla-hub/rwkv7-0.1B-g1"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        bool(os.getenv("CI")),
        reason="fla-hub/rwkv7-0.1B-g1: remote code + flash-linear-attention dep, skipped in CI",
    ),
]

# Recurrent fp32 model; allow a wider op-order noise floor on GH Actions
# macOS-arm64 (same idiom as the Ouro / Raven bridge-vs-HF parity tests).
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

    Depends on the bridge fixture only for ordering: booting the bridge runs the
    adapter's prepare_loading patch (guarding transformers v5 weight re-init) on
    the cached rwkv7 modeling module, and this plain from_pretrained load benefits
    from that patch too. The weights are still a separate copy.
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


class TestRWKV7BridgeCreation:
    """The bridge loads the remote-code model and wires the flat block stack."""

    def test_boot_transformers_succeeds(self, bridge: TransformerBridge) -> None:
        assert bridge is not None

    def test_block_count_matches_config(
        self, bridge: TransformerBridge, hf_eager: torch.nn.Module
    ) -> None:
        assert len(bridge.blocks) == hf_eager.config.num_hidden_layers

    def test_config_flags(self, bridge: TransformerBridge) -> None:
        assert bridge.cfg.normalization_type == "LN"
        assert bridge.cfg.positional_embedding_type == "none"
        assert bridge.cfg.gated_mlp is False
        assert bridge.cfg.final_rms is False

    def test_block_submodules_wired(self, bridge: TransformerBridge) -> None:
        """attn_norm / attn / ffn_norm / ffn are bridged on each block."""
        block = bridge.blocks[0]
        for sub in ("attn_norm", "attn", "ffn_norm", "ffn"):
            assert hasattr(block, sub), f"missing {sub}"

    def test_attn_projections_wrap_live_hf_modules(self, bridge: TransformerBridge) -> None:
        attn = bridge.blocks[0].attn
        hf_attn = bridge.original_model.model.layers[0].attn
        assert attn.r_proj is hf_attn.r_proj
        assert attn.k_proj is hf_attn.k_proj
        assert attn.v_proj is hf_attn.v_proj
        assert attn.o_proj is hf_attn.o_proj

    def test_ffn_projections_wrap_live_hf_modules(self, bridge: TransformerBridge) -> None:
        ffn = bridge.blocks[0].ffn
        hf_ffn = bridge.original_model.model.layers[0].ffn
        assert ffn.key is hf_ffn.key
        assert ffn.value is hf_ffn.value


class TestRWKV7ForwardEquivalence:
    """Bridge output reproduces the HF eager reference through the recurrence."""

    def test_forward_logits_match_hf_eager(
        self, bridge: TransformerBridge, hf_eager: torch.nn.Module, tokens: torch.Tensor
    ) -> None:
        with torch.inference_mode():
            bridge_logits = bridge(tokens)
            hf_logits = hf_eager(tokens).logits
        max_diff = (bridge_logits - hf_logits).abs().max().item()
        assert max_diff < FP32_NOISE_TOL, (
            f"RWKV-7 bridge vs HF eager logit drift={max_diff:.2e} exceeds the "
            f"fp32-noise tolerance {FP32_NOISE_TOL:.0e}."
        )


class TestRWKV7Hooks:
    """Residual-stream and sublayer hooks fire with the expected shape."""

    @pytest.mark.parametrize(
        "hook_name",
        ["blocks.0.hook_out", "blocks.0.attn.hook_out", "blocks.0.ffn.hook_out"],
    )
    def test_hook_fires_with_residual_shape(
        self, bridge: TransformerBridge, tokens: torch.Tensor, hook_name: str
    ) -> None:
        captured: list[torch.Tensor] = []

        def _capture(value, hook):
            captured.append(value[0] if isinstance(value, tuple) else value)
            return value

        bridge.run_with_hooks(tokens, fwd_hooks=[(hook_name, _capture)])
        assert captured, f"{hook_name} did not fire"
        assert captured[-1].shape == (1, tokens.shape[1], bridge.cfg.d_model)

    def test_run_with_cache_has_block_hooks(
        self, bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        _, cache = bridge.run_with_cache(tokens)
        for key in ("blocks.0.hook_out", "blocks.0.attn.hook_out", "blocks.0.ffn.hook_out"):
            assert key in cache, f"{key} missing from cache"
