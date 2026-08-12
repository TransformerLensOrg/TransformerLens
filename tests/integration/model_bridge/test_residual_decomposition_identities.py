"""Residual-stream decomposition identities across sequential-residual architectures.

Guards the HookedTransformer contract that ``hook_attn_out`` / ``hook_mlp_out``
are additive contributions (#1639: BLOOM's HF modules add the residual
internally, so the default block aliases exposed accumulated states):

    resid_pre + attn_out == resid_mid
    resid_mid + mlp_out  == resid_post

The fixture list is curated, not exhaustive — one tiny checkpoint per
residual-wiring pattern the bridge handles:

- gpt2: block-level residual adds, the HookedTransformer reference wiring
- mistral: Llama-style pre-RMSNorm block-level adds
- bloom: residual added *inside* the HF attention/MLP modules (the #1639 case)
- olmo2: post-norm inside the residual branch — RMSNorm applies to the sublayer
  output before the add, so the contributions are the norm outputs (the #1648 case)
- mpt: residual added inside the HF MLP module only (attention adds at block level)

Parallel-residual architectures (Falcon, GPT-J, NeoX, Cohere) are out of scope:
they have no ``hook_resid_mid``.
"""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

SEQUENTIAL_RESIDUAL_MODELS = [
    pytest.param("hf-internal-testing/tiny-random-gpt2", id="gpt2"),
    pytest.param("trl-internal-testing/tiny-MistralForCausalLM-0.2", id="mistral"),
    pytest.param("trl-internal-testing/tiny-BloomForCausalLM", id="bloom"),
    pytest.param("hf-internal-testing/tiny-random-Olmo2ForCausalLM", id="olmo2"),
    pytest.param("hf-internal-testing/tiny-random-MptForCausalLM", id="mpt"),
]


@pytest.mark.parametrize("model_name", SEQUENTIAL_RESIDUAL_MODELS)
def test_residual_decomposition_identities(model_name: str) -> None:
    bridge = TransformerBridge.boot_transformers(model_name, device="cpu", dtype=torch.float32)
    tokens = bridge.to_tokens("The capital of France is Paris.")

    with torch.no_grad():
        _, cache = bridge.run_with_cache(tokens)

    for layer in range(bridge.cfg.n_layers):
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_mid"],
            cache[f"blocks.{layer}.hook_resid_pre"] + cache[f"blocks.{layer}.hook_attn_out"],
        )
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_post"],
            cache[f"blocks.{layer}.hook_resid_mid"] + cache[f"blocks.{layer}.hook_mlp_out"],
        )
