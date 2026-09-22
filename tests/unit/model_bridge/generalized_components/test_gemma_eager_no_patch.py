"""Guard against the PositionEmbeddingsAttentionBridge patching HF's eager attention.

The bridge used to monkey-patch transformers' module-level
``eager_attention_forward`` for gemma2/gemma3 (once per process, never reverted)
to fire hook_rot_q/hook_rot_k. Since the bridge's own forward reimplements
attention and fires those hooks itself, the patch was dead and leaked into any
process that built a bridge — breaking downstream source-inspection tooling
(e.g. nnsight/circuit-tracer). This test builds a tiny Gemma3 bridge, confirms
the rotary hooks still fire, and asserts HF's eager_attention_forward is left
untouched on both gemma modules.
"""

import copy

import pytest
import torch

pytest.importorskip("transformers", reason="requires transformers")

from transformers import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


def _tiny_gemma3_bridge():
    cfg = Gemma3TextConfig(
        vocab_size=200,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        sliding_window=8,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(42)
    hf = Gemma3ForCausalLM(cfg).eval()
    return build_bridge_from_module(
        hf, "Gemma3ForCausalLM", hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    ).eval()


def test_rotary_hook_fires_without_patching_hf_eager_attention():
    bridge = _tiny_gemma3_bridge()
    tokens = torch.randint(3, 200, (1, 6))
    _, cache = bridge.run_with_cache(tokens)

    assert "blocks.0.attn.hook_rot_q" in cache

    import transformers.models.gemma2.modeling_gemma2 as gemma2_module
    import transformers.models.gemma3.modeling_gemma3 as gemma3_module

    assert gemma2_module.eager_attention_forward.__module__.startswith("transformers.")
    assert gemma3_module.eager_attention_forward.__module__.startswith("transformers.")
