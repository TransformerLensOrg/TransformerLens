"""Integration tests for BLOOM residual-branch hook semantics."""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

MODEL = "trl-internal-testing/tiny-BloomForCausalLM"


@pytest.fixture(scope="module")
def bloom_bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


def test_residual_branch_hooks_decompose_stream(bloom_bridge: TransformerBridge) -> None:
    tokens = bloom_bridge.to_tokens("The capital of France is Paris.")

    with torch.no_grad():
        _, cache = bloom_bridge.run_with_cache(tokens)

    for layer in range(bloom_bridge.cfg.n_layers):
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_mid"],
            cache[f"blocks.{layer}.hook_resid_pre"] + cache[f"blocks.{layer}.hook_attn_out"],
        )
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_post"],
            cache[f"blocks.{layer}.hook_resid_mid"] + cache[f"blocks.{layer}.hook_mlp_out"],
        )


def test_residual_branch_hooks_are_writable(bloom_bridge: TransformerBridge) -> None:
    tokens = bloom_bridge.to_tokens("The capital of France is Paris.")

    with torch.no_grad():
        baseline = bloom_bridge(tokens)
        ablated = bloom_bridge.run_with_hooks(
            tokens,
            fwd_hooks=[("blocks.0.hook_attn_out", lambda tensor, hook: torch.zeros_like(tensor))],
        )

    assert not torch.equal(ablated, baseline)
