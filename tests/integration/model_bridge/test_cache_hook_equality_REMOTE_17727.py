import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge

MODEL = "gpt2"
prompt = "Hello World!"


@pytest.fixture(scope="module")
def bridge():
    """Load TransformerBridge once per module."""
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu")
    bridge.enable_compatibility_mode(disable_warnings=False)
    return bridge


@pytest.fixture(scope="module")
def hooked_transformer():
    """Load HookedTransformer once per module."""
    return HookedTransformer.from_pretrained(MODEL, device="cpu")


act_names_in_cache = [
    # "hook_embed",
    # "hook_pos_embed",
    "blocks.0.hook_resid_pre",
    # "blocks.0.ln1.hook_scale",
    "blocks.0.ln1.hook_normalized",
    # "blocks.0.attn.hook_q",
    # "blocks.0.attn.hook_k",
    # "blocks.0.attn.hook_v",
    # "blocks.0.attn.hook_attn_scores",
    # "blocks.0.attn.hook_pattern",
    # "blocks.0.attn.hook_z",
    "blocks.0.hook_attn_out",
    "blocks.0.hook_resid_mid",
    # "blocks.0.ln2.hook_scale",
    "blocks.0.ln2.hook_normalized",
    "blocks.0.mlp.hook_pre",
    # "blocks.0.mlp.hook_post",
    "blocks.0.hook_mlp_out",
    "blocks.0.hook_resid_post",
    # "ln_final.hook_scale",
    "ln_final.hook_normalized",
]


def test_cache_hook_names(bridge, hooked_transformer):
    """Test that TransformerBridge cache contains the expected hook names."""
    _, bridge_cache = bridge.run_with_cache(prompt)
    _, hooked_transformer_cache = hooked_transformer.run_with_cache(prompt)

    for hook in act_names_in_cache:
        hooked_transformer_activation = hooked_transformer_cache[hook]
        bridge_activation = bridge_cache[hook]
        assert hooked_transformer_activation.shape == bridge_activation.shape, (
            f"Shape mismatch for hook {hook}: "
            f"HookedTransformer shape {hooked_transformer_activation.shape}, "
            f"TransformerBridge shape {bridge_activation.shape}"
        )

        assert (
            torch.mean(torch.abs(hooked_transformer_activation - bridge_activation)) < 0.5
        ), f"Hook {hook} does not match between old HookedTransformer and new TransformerBridge."
