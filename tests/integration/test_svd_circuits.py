"""Integration test: OV vocab readout and the causal patch gate on a real GPT-2 head.

Drives the whole read-then-patch path against layer 9 head 9 (the paper's canonical
name-mover head) on a minimal IOI-style prompt, to check that the readout and the causal gate
agree on a real Bridge rather than only on the unit suite's synthetic and tiny-model fixtures.
Tiny MPT, Bamba and T5 checkpoints pin the architecture refusals on real bridges.
"""

from __future__ import annotations

import math

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.SVDInterpreter import SVDInterpreter
from transformer_lens.tools.analysis.svd_circuits import (
    decompose_head,
    patch_along_directions,
    project_activations,
    vocab_readout,
)

CLEAN_PROMPT = "When Mary and John went to the store, John gave a drink to"
LAYER, HEAD = 9, 9


@pytest.fixture(scope="module")
def gpt2_bridge():
    model = TransformerBridge.boot_transformers("gpt2", device="cpu", dtype=torch.float32)
    model.enable_compatibility_mode()
    return model


def _logit_diff_metric(model):
    mary_token = model.to_single_token(" Mary")
    john_token = model.to_single_token(" John")

    def metric(logits: torch.Tensor) -> float:
        return float(logits[0, -1, mary_token] - logits[0, -1, john_token])

    return metric


def test_svd_circuits_readout_and_patch_gate_on_name_mover_head(gpt2_bridge) -> None:
    decomposition = decompose_head(gpt2_bridge, layer=LAYER, head=HEAD, which=("OV",))
    ov = decomposition.OV
    assert ov is not None

    readout = vocab_readout(gpt2_bridge, ov, k=10)
    assert readout.shape == (gpt2_bridge.cfg.d_vocab, 10)
    assert torch.isfinite(readout).all()

    # Pick the first non-degenerate direction rather than assuming index 0 is isolated.
    top_direction = next(row.idx for row in ov.rank_report if not row.is_degenerate)

    # readout reproduces the shipped SVDInterpreter reference up to sign, so a U-for-V swap
    # or an all-zeros return in vocab_readout fails here rather than passing a finiteness check.
    reference = SVDInterpreter(gpt2_bridge).get_singular_vectors(
        "OV", LAYER, head_index=HEAD, num_vectors=top_direction + 1
    )[:, 0, top_direction]
    assert torch.allclose(readout[:, top_direction], reference, atol=1e-4) or torch.allclose(
        readout[:, top_direction], -reference, atol=1e-4
    )

    prompt = gpt2_bridge.to_tokens(CLEAN_PROMPT)
    metric = _logit_diff_metric(gpt2_bridge)

    result = patch_along_directions(gpt2_bridge, ov, prompt, metric, keep=[top_direction])

    assert result.retained == [top_direction]
    assert math.isfinite(result.original_metric)
    assert math.isfinite(result.patched_metric)
    assert math.isfinite(result.delta_metric)
    assert math.isfinite(result.baseline_delta_metric)
    assert isinstance(result.gated, bool)

    rank = ov.V.shape[1]

    # Keeping every direction reconstructs the head onto its own full OV span: a no-op, so the
    # metric barely moves. A width-rank in-span control also spans span(V), so it ties this
    # kept projector by construction and the gate needs an explicit threshold. A hook never
    # installed, or one projecting the wrong basis, breaks the delta assertion.
    full_keep = patch_along_directions(
        gpt2_bridge, ov, prompt, metric, keep=list(range(rank)), threshold=1e-4
    )
    assert abs(full_keep.delta_metric) < 1e-4

    # Ablating every direction zeroes the head's whole output, which must move the logit diff.
    # An empty retained set ties the baseline by construction, so gate it with an explicit
    # threshold; only delta_metric's magnitude is asserted here.
    full_ablate = patch_along_directions(
        gpt2_bridge, ov, prompt, metric, ablate=list(range(rank)), threshold=0.0
    )
    assert abs(full_ablate.delta_metric) > 1e-2


TINY_MPT = "hf-internal-testing/tiny-random-MptForCausalLM"
TINY_BAMBA = "hmellor/tiny-random-BambaForCausalLM"
TINY_T5 = "hf-internal-testing/tiny-random-t5"


@pytest.fixture(scope="module")
def tiny_mpt():
    model = TransformerBridge.boot_transformers(TINY_MPT, device="cpu", dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_bamba():
    return TransformerBridge.boot_transformers(TINY_BAMBA, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def tiny_t5():
    return TransformerBridge.boot_transformers(TINY_T5, device="cpu", dtype=torch.float32)


def _hook_result_fires(model, layer: int, prompt: torch.Tensor) -> bool:
    fired = []
    model.set_use_attn_result(True)
    try:
        model.run_with_hooks(
            prompt,
            fwd_hooks=[(f"blocks.{layer}.attn.hook_result", lambda act, hook: fired.append(act))],
        )
    finally:
        model.set_use_attn_result(False)
    return bool(fired)


def test_mpt_activation_tools_refuse_unless_hook_result_fires(tiny_mpt) -> None:
    """MPT's ALiBi attention accepts use_attn_result; the activation tools either read a fired
    hook_result or refuse, never a KeyError or a gate verdict from a no-op patch.

    Which branch runs follows the bridge: an MPT attention that skips hook_result gets the
    refusal, one that fires it gets real coefficients and a patch that moves the metric.
    """
    layer, head = 0, 0
    ov = decompose_head(tiny_mpt, layer, head, which=("OV",)).OV
    assert ov is not None
    assert torch.isfinite(ov.S).all()
    prompt = torch.tensor([[1, 5, 7, 9, 11]])
    metric = lambda logits: float(logits[0, -1, 0] - logits[0, -1, 1])

    if _hook_result_fires(tiny_mpt, layer, prompt):
        projection = project_activations(tiny_mpt, ov, prompt)
        assert projection.coefficients.shape == (prompt.shape[1], ov.V.shape[1])
        assert torch.isfinite(projection.coefficients).all()
        rank = ov.V.shape[1]
        ablated = patch_along_directions(
            tiny_mpt, ov, prompt, metric, ablate=list(range(rank)), threshold=0.0
        )
        assert ablated.delta_metric != 0.0
    else:
        message = (
            r"MPTALiBiAttentionBridge\) accepted use_attn_result but never fired "
            rf"blocks\.{layer}\.attn\.hook_result"
        )
        with pytest.raises(ValueError, match=message):
            project_activations(tiny_mpt, ov, prompt)
        with pytest.raises(ValueError, match=message):
            patch_along_directions(tiny_mpt, ov, prompt, metric, keep=[0])
    assert tiny_mpt.cfg.use_attn_result is False


def test_decompose_head_refuses_ssm_layer_on_hybrid(tiny_bamba) -> None:
    """Bamba's mamba layers carry no attn submodule; decompose_head names the layer and the
    attention layers instead of the bridge's bare AttributeError, and those still decompose."""
    blocks = tiny_bamba.blocks
    ssm_layers = [i for i, block in enumerate(blocks) if "attn" not in block._modules]
    attn_layers = [i for i, block in enumerate(blocks) if "attn" in block._modules]
    assert ssm_layers and attn_layers

    with pytest.raises(ValueError, match=f"attention block at layer {ssm_layers[0]}") as excinfo:
        decompose_head(tiny_bamba, ssm_layers[0], 0)
    assert f"Layers with an `attn` submodule: {attn_layers}" in str(excinfo.value)

    decomposition = decompose_head(tiny_bamba, attn_layers[0], 0)
    assert decomposition.QK is not None and decomposition.OV is not None
    assert torch.isfinite(decomposition.OV.S).all()


def test_decompose_head_refuses_encoder_decoder_bridge(tiny_t5) -> None:
    """T5 keeps separate encoder and decoder stacks and no `blocks`; the refusal names the layer."""
    with pytest.raises(ValueError, match="layer 0's head weights.*no `blocks` stack"):
        decompose_head(tiny_t5, 0, 0)
