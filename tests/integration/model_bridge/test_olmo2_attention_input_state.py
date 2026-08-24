"""OLMo 2 attention-input fork regression tests using a tiny local HF model."""

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.olmo2 import Olmo2Config

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures.olmo2 import (
    Olmo2ArchitectureAdapter,
)


class _Tokenizer:
    pass


def _tiny_olmo2_bridge() -> TransformerBridge:
    torch.manual_seed(0)
    config = Olmo2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config.architectures = ["Olmo2ForCausalLM"]
    hf_model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, "Olmo2ForCausalLM", "olmo2-tiny", torch.float32
    )
    return TransformerBridge(
        hf_model, Olmo2ArchitectureAdapter(bridge_config), tokenizer=_Tokenizer()
    )


@pytest.mark.parametrize(
    ("setter_name", "input_hook_names"),
    [
        (
            "set_use_split_qkv_input",
            (
                "blocks.1.attn.hook_q_input",
                "blocks.1.attn.hook_k_input",
                "blocks.1.attn.hook_v_input",
            ),
        ),
        ("set_use_attn_in", ("blocks.1.attn.hook_attn_in",)),
    ],
)
def test_attention_input_fork_does_not_leak_state_between_forwards(
    setter_name: str, input_hook_names: tuple[str, ...]
) -> None:
    bridge = _tiny_olmo2_bridge()
    bridge.enable_compatibility_mode()
    getattr(bridge, setter_name)(True)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.no_grad():
        first_logits, first_cache = bridge.run_with_cache(tokens)
        second_logits, second_cache = bridge.run_with_cache(tokens)

    torch.testing.assert_close(second_logits, first_logits, rtol=0, atol=0)
    for hook_name in input_hook_names:
        torch.testing.assert_close(second_cache[hook_name], first_cache[hook_name], rtol=0, atol=0)
        for cache in (first_cache, second_cache):
            expected = cache["blocks.1.hook_resid_pre"].unsqueeze(2).expand_as(cache[hook_name])
            torch.testing.assert_close(cache[hook_name], expected, rtol=0, atol=0)
