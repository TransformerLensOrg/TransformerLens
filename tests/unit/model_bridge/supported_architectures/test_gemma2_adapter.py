"""Unit tests for Gemma2ArchitectureAdapter.

Post-fix invariant: the adapter MUST NOT override setup_hook_compatibility.
Gemma2TextScaledWordEmbedding scales by sqrt(d_model) inside its own forward(),
so the bridge's wrapped layer already returns the scaled value. An override
that installed a hook_conversion would double-scale (cache R * d_model instead
of R * sqrt(d_model)) — the dev-4.x vLLM-comparison investigation surfaced this.
"""

import os

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.model_bridge.supported_architectures.gemma2 import (
    Gemma2ArchitectureAdapter,
)


def _make_cfg(d_model: int = 32) -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // 4,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_vocab=256,
        d_mlp=64,
        architecture="Gemma2ForCausalLM",
    )


@pytest.fixture(scope="module")
def adapter() -> Gemma2ArchitectureAdapter:
    return Gemma2ArchitectureAdapter(_make_cfg())


class TestGemma2HookCompatibility:
    def test_adapter_does_not_override_setup_hook_compatibility(
        self, adapter: Gemma2ArchitectureAdapter
    ) -> None:
        # bridge.py:763 uses hasattr() to decide whether to call the override.
        # Absence is the contract — the base ArchitectureAdapter has no such
        # method, so the bridge skips installation entirely.
        assert "setup_hook_compatibility" not in vars(type(adapter))


@pytest.mark.skipif(bool(os.getenv("CI")), reason="Network/disk fetch of tiny Gemma2 — skip in CI")
def test_gemma2_embed_hook_out_magnitude_matches_sqrt_d_model_scaling():
    """End-to-end regression for the embed double-scale bug.

    Before the fix: embed.hook_out held R * d_model (double-scaled).
    After:         embed.hook_out holds R * sqrt(d_model) — i.e. exactly what
                   Gemma2TextScaledWordEmbedding.forward returns and what flows
                   into block 0.
    """
    import torch.nn.functional as F

    from transformer_lens.model_bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers(
        "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
        dtype=torch.float32,
        device="cpu",
    )
    assert bridge.embed.hook_out.hook_conversion is None

    tokens = torch.tensor([[1, 2, 3, 4]])
    raw_R = F.embedding(tokens, bridge.embed.original_component.weight)
    scaled_direct = bridge.embed.original_component(tokens)
    _, cache = bridge.run_with_cache(tokens)

    # embed.hook_out should equal the ScaledWordEmbedding's actual output
    assert torch.allclose(cache["embed.hook_out"], scaled_direct, atol=1e-6)
    # And the ratio to the raw lookup is sqrt(d_model), not d_model
    ratio = cache["embed.hook_out"].abs().max().item() / raw_R.abs().max().item()
    assert ratio == pytest.approx(bridge.cfg.d_model**0.5, rel=1e-4)
