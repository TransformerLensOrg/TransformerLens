"""BitNet in compatibility mode, on a tiny random model built without hub access.

BitNet applies an RMSNorm (attn_sub_norm / ffn_sub_norm) right before each output
projection. The processed-weights path rebuilds attention and the MLP from weights,
so it has to apply both norms to match HF.
"""
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)

TOKENS = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])


def _hf_bitnet():
    from transformers import AutoModelForCausalLM, BitNetConfig

    torch.manual_seed(0)
    c = BitNetConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    c.architectures = ["BitNetForCausalLM"]
    hf = AutoModelForCausalLM.from_config(c, attn_implementation="eager").eval()
    with torch.no_grad():
        # Non-trivial sub-norm weights, so skipping a norm changes the output.
        for name, param in hf.named_parameters():
            if "norm" in name:
                param.normal_(1.0, 0.3)
    return hf


def _bridge(hf) -> TransformerBridge:
    from transformer_lens.factories.architecture_adapter_factory import (
        ArchitectureAdapterFactory,
    )

    cfg = build_bridge_config_from_hf(hf.config, "BitNetForCausalLM", "tiny", torch.float32)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    return TransformerBridge(model=hf, adapter=adapter, tokenizer=MagicMock())


@pytest.fixture(scope="module")
def reference():
    hf = _hf_bitnet()
    with torch.no_grad():
        logits = hf(TOKENS).logits
    _, cache = _bridge(hf).run_with_cache(TOKENS)
    return logits, cache


@pytest.fixture(scope="module")
def compat():
    bridge = _bridge(_hf_bitnet())
    bridge.enable_compatibility_mode()
    with torch.no_grad():
        logits, cache = bridge.run_with_cache(TOKENS)
    return bridge, logits, cache


def test_compatibility_mode_matches_hf(reference, compat):
    ref_logits, _ = reference
    _, logits, _ = compat
    torch.testing.assert_close(
        logits.log_softmax(-1), ref_logits.log_softmax(-1), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize(
    "hook", ["blocks.0.mlp.hook_post", "blocks.0.hook_mlp_out", "blocks.1.hook_resid_post"]
)
def test_mlp_hooks_match_the_unprocessed_bridge(reference, compat, hook):
    """hook_post is the down_proj input, after ffn_sub_norm, on both paths."""
    _, ref_cache = reference
    _, _, cache = compat
    torch.testing.assert_close(cache[hook], ref_cache[hook], atol=1e-5, rtol=1e-5)


def test_neuron_decomposition_sums_to_residual(compat):
    bridge, _, cache = compat
    final = cache[f"blocks.{bridge.cfg.n_layers - 1}.hook_resid_post"]
    stack = cache.get_full_resid_decomposition(expand_neurons=True)
    torch.testing.assert_close(stack.sum(0), final, atol=1e-5, rtol=0)
