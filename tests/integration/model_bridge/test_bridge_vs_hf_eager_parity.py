"""Asserts ``TransformerBridge`` reproduces ``AutoModelForCausalLM`` eager-attention logits.

Issue #385 reported drift between bridge and HF for rotary models like Pythia. The drift
was an attention-implementation mismatch — bridge always uses eager, default HF loads use
SDPA, which reorders ops in a fused kernel. Bridge vs HF *eager* matches to fp32-noise.
"""

from typing import Callable

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens.model_bridge import TransformerBridge

MODEL_NAME = "EleutherAI/pythia-70m"

# Op-reorder noise floor for fp32 transformer forward passes. We currently
# measure 0.0 on this model, but allow a small epsilon so harmless refactors
# (intermediate allocations, equivalent op reorderings) don't break the test.
FP32_NOISE_TOL = 1e-5


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL_NAME, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def hf_eager():
    """HF model loaded independently of the bridge's wrapped instance."""
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()


@pytest.fixture
def tokenize(tokenizer) -> Callable[[str], torch.Tensor]:
    def _tok(prompt: str) -> torch.Tensor:
        return tokenizer(prompt, return_tensors="pt").input_ids

    return _tok


@pytest.mark.parametrize("prompt", ["Hello, world!", "The quick brown fox jumps"])
def test_bridge_logits_match_hf_eager(bridge, hf_eager, tokenize, prompt):
    tokens = tokenize(prompt)
    with torch.inference_mode():
        bridge_logits = bridge(tokens)
        hf_logits = hf_eager(tokens).logits
    max_diff = (bridge_logits - hf_logits).abs().max().item()
    assert max_diff < FP32_NOISE_TOL, (
        f"{MODEL_NAME!r} bridge vs HF eager drift={max_diff:.2e} on {prompt!r} "
        f"exceeds fp32-noise tolerance {FP32_NOISE_TOL:.0e} — bridge's "
        f"_reconstruct_attention may have regressed (see issue #385)."
    )


def test_bridge_residual_stream_matches_hf_eager(bridge, hf_eager, tokenize):
    """Per-layer parity catches compensating errors that wash out at the final logits."""
    tokens = tokenize("Hello, world!")
    n_layers = len(hf_eager.gpt_neox.layers)

    hf_layer_out: dict[int, torch.Tensor] = {}

    def _make_hf_hook(idx):
        def _h(_m, _i, o):
            hf_layer_out[idx] = (o[0] if isinstance(o, tuple) else o).detach()

        return _h

    handles = [
        layer.register_forward_hook(_make_hf_hook(i))
        for i, layer in enumerate(hf_eager.gpt_neox.layers)
    ]
    try:
        with torch.inference_mode():
            hf_eager(tokens)
    finally:
        for h in handles:
            h.remove()

    bridge_layer_out: dict[int, torch.Tensor] = {}
    fwd_hooks = [
        (
            f"blocks.{i}.hook_resid_post",
            lambda v, hook, idx=i: bridge_layer_out.__setitem__(idx, v.detach()),
        )
        for i in range(n_layers)
    ]
    with torch.inference_mode():
        bridge.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    for i in range(n_layers):
        d = (hf_layer_out[i] - bridge_layer_out[i]).abs().max().item()
        assert d < FP32_NOISE_TOL, (
            f"layer {i} residual drift={d:.2e} exceeds fp32-noise tolerance "
            f"{FP32_NOISE_TOL:.0e} — bridge layer output diverges from HF eager."
        )


def test_bridge_attention_reconstruction_actually_runs(bridge, tokenize):
    """Guard against tautology: prove bridge's custom attention path executes.

    If a future refactor made the bridge delegate to HF directly, the previous
    parity tests would pass trivially. This one fails fast in that case by
    asserting bridge-specific hooks fire during forward.
    """
    tokens = tokenize("Hello, world!")
    attn_scores_fired: list[bool] = []
    bridge.run_with_hooks(
        tokens,
        fwd_hooks=[
            ("blocks.0.attn.hook_attn_scores", lambda v, hook: attn_scores_fired.append(True)),
        ],
    )
    assert attn_scores_fired, (
        "blocks.0.attn.hook_attn_scores did not fire — bridge no longer runs its "
        "own attention reconstruction, making the parity tests tautological."
    )
