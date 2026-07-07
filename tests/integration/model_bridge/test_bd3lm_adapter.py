"""Integration tests for the BD3LM architecture adapter.

Verifies forward-pass and hook parity against kuleshov-group/bd3lm-owt-block_size4:
- Forward-pass logits match HF exactly (bridge delegates the full forward to HF)
- Sanity checks: config flags, block count, hook coverage

Note: runs on slow mark to avoid hitting HF on standard runs.
"""

import gc

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.supported_architectures.bd3lm import (
    BD3LMArchitectureAdapter,
)

pytestmark = pytest.mark.slow

MODEL = "kuleshov-group/bd3lm-owt-block_size4"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def bd3lm_bridge():
    device = _device()
    # Override model_length to 4 so that we can run sequence length 8 (model_length * 2) forward passes
    bridge = TransformerBridge.boot_transformers(
        MODEL,
        device=device,
        trust_remote_code=True,
        hf_config_overrides={"model_length": 4},
    )
    yield bridge
    # Cleanup
    del bridge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for _ in range(3):
        gc.collect()


class TestBD3LMBridgeCreation:
    """Smoke-test that the bridge loads with the right config flags."""

    def test_block_count(self, bd3lm_bridge: TransformerBridge) -> None:
        assert len(bd3lm_bridge.blocks) == 12

    def test_adapter_type(self, bd3lm_bridge: TransformerBridge) -> None:
        assert isinstance(bd3lm_bridge.adapter, BD3LMArchitectureAdapter)


class TestBD3LMForwardPass:
    """Bridge logits must match HF logits exactly."""

    @pytest.fixture(scope="class")
    def tokens(self) -> torch.Tensor:
        # Sequence length must be model_length * 2 = 8
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    def test_forward_matches_hf_exactly(
        self, bd3lm_bridge: TransformerBridge, tokens: torch.Tensor
    ) -> None:
        tokens = tokens.to(_device())
        hf_model = bd3lm_bridge.original_model

        timesteps = torch.tensor([10.0]).to(_device())

        with torch.no_grad():
            bridge_out = bd3lm_bridge(tokens, timesteps=timesteps)
            hf_out = hf_model(tokens, timesteps=timesteps)

        if hasattr(hf_out, "logits"):
            hf_logits = hf_out.logits
        else:
            hf_logits = hf_out

        max_diff = (bridge_out.float() - hf_logits.float()).abs().max().item()
        assert max_diff == 0.0, (
            f"Bridge vs HF forward max diff = {max_diff:.2e}. "
            "Expected 0 because DelegatedAttentionBlockBridge.forward() is a pure passthrough."
        )


class TestBD3LMHookCoverage:
    """run_with_cache captures residual stream, attention, MLP, and other hooks."""

    @pytest.fixture(scope="class")
    def cache(self, bd3lm_bridge: TransformerBridge):
        # Sequence length must be model_length * 2 = 8
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]).to(_device())
        timesteps = torch.tensor([10.0]).to(_device())
        with torch.no_grad():
            _, cache = bd3lm_bridge.run_with_cache(tokens, timesteps=timesteps)
        return cache

    def test_block_hooks_fire(self, cache, bd3lm_bridge: TransformerBridge) -> None:
        for i in range(len(bd3lm_bridge.blocks)):
            assert f"blocks.{i}.hook_in" in cache
            assert f"blocks.{i}.hook_out" in cache

    def test_expected_hook_aliases_fire(self, cache, bd3lm_bridge: TransformerBridge) -> None:
        """Verify that every registered hook alias in the block actually fires with real data."""
        for i in range(len(bd3lm_bridge.blocks)):
            for alias in (
                "hook_resid_pre",
                "hook_resid_mid",
                "hook_resid_post",
                "hook_attn_out",
                "hook_mlp_out",
            ):
                key = f"blocks.{i}.{alias}"
                assert key in cache, f"Missing hook alias {key} in cache"
                val = cache[key]
                assert val is not None
                assert isinstance(val, torch.Tensor)
                assert val.shape[0] == 1  # batch size
                assert val.shape[1] == 8  # seq length (model_length * 2)
                assert val.shape[2] == bd3lm_bridge.cfg.d_model  # d_model
                assert not torch.isnan(val).any(), f"NaN in cache['{key}']"

    def test_no_nan_in_cache(self, cache) -> None:
        for key, val in cache.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                assert not torch.isnan(val).any(), f"NaN in cache['{key}']"
