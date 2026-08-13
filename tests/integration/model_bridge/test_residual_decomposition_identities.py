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
- granite: the scaled-residual block wiring end to end. NOTE: the hub tiny ships
  residual_multiplier=1.0, so this entry does NOT exercise the multiplier —
  test_granite_hook_semantics.py builds a fixture with a real multiplier and is
  the coverage for the #1648 Granite case.
- gemma2: sandwich norms — a post-sublayer norm inside each residual branch, so
  the contributions are the post-norm outputs

Parallel-residual architectures (Falcon, GPT-J, NeoX, Cohere) are out of scope
here: they have no ``hook_resid_mid``. Their two-term identity
(``resid_post == resid_pre + attn_out + mlp_out``) is covered by
``test_parallel_residual_identities.py``.
"""

import pytest
import torch

from tests.tiny_checkpoints import sequential_residual_params
from transformer_lens.model_bridge import TransformerBridge

SEQUENTIAL_RESIDUAL_MODELS = sequential_residual_params()


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
