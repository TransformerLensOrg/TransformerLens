"""Bloom's gated attention hooks must fire like every other joint-QKV bridge.

`BloomAttentionBridge` overrides both `forward` and `_reconstruct_attention`, and
the overrides projected Q/K/V directly and always took the plain output
projection. `hook_result`, `hook_q_input`, `hook_k_input`, `hook_v_input` and
`hook_attn_in` therefore never fired — enabling `use_attn_result` on a Bloom
model silently produced nothing.
"""

import pytest
import torch

from transformer_lens.benchmarks import benchmark_gated_hooks_fire
from transformer_lens.model_bridge import TransformerBridge

MODEL = "bigscience/bloom-560m"
PROMPT = "The theory of relativity explains that the speed of light"


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


def test_every_gated_hook_fires(bridge) -> None:
    result = benchmark_gated_hooks_fire(bridge, PROMPT)
    assert result.passed, result.message
    fired = (result.details or {}).get("fired_counts", {})
    assert fired and all(count > 0 for count in fired.values()), fired


@pytest.mark.parametrize("flag", ["use_attn_result", "use_split_qkv_input", "use_attn_in"])
def test_gated_paths_preserve_the_output(bridge, flag: str) -> None:
    """The fork re-parameterizes the same math, so logits must not move beyond
    the op-order noise a correct implementation shows (gpt2 1e-4, pythia 4e-3)."""
    tokens = bridge.to_tokens(PROMPT)
    with torch.no_grad():
        baseline = bridge(tokens).float()
        setattr(bridge.cfg, flag, True)
        try:
            gated = bridge(tokens).float()
        finally:
            setattr(bridge.cfg, flag, False)
    torch.testing.assert_close(gated, baseline, atol=5e-3, rtol=1e-3)
