"""Unit tests for Gemma1ArchitectureAdapter.

Mirrors test_gemma2_adapter.py: GemmaTextScaledWordEmbedding scales internally,
so the adapter must NOT override setup_hook_compatibility (an override that
installed a hook_conversion would double-scale embed.hook_out).
"""

import pytest

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.model_bridge.supported_architectures.gemma1 import (
    Gemma1ArchitectureAdapter,
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
        architecture="GemmaForCausalLM",
    )


@pytest.fixture(scope="module")
def adapter() -> Gemma1ArchitectureAdapter:
    return Gemma1ArchitectureAdapter(_make_cfg())


class TestGemma1HookCompatibility:
    def test_adapter_does_not_override_setup_hook_compatibility(
        self, adapter: Gemma1ArchitectureAdapter
    ) -> None:
        # bridge.py:763 uses hasattr() to decide whether to call the override.
        assert "setup_hook_compatibility" not in vars(type(adapter))
