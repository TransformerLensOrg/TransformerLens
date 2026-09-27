"""Real-Bridge integration checks for Projection Kernel head affinity."""

import pytest
import torch

from transformer_lens.tools.analysis.projection_kernel import (
    attention_head_subspace_affinity,
    orthonormal_subspace,
    projection_kernel,
)


@pytest.fixture(scope="module")
def gpt2_bridge_bfloat16():
    """Fresh gpt2 bridge loaded with bfloat16 weights."""
    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers("gpt2", device="cpu", dtype=torch.bfloat16)


@pytest.mark.parametrize(("role", "attribute"), [("Q", "W_Q"), ("K", "W_K"), ("V", "W_V")])
def test_gpt2_head_affinity_contract_and_sample_parity(gpt2_bridge, role, attribute):
    result = attention_head_subspace_affinity(gpt2_bridge, target_role=role)

    assert result.scores.shape == (12, 12, 12, 12)
    assert result.source_layer_indices == tuple(range(12))
    assert result.target_layer_indices == tuple(range(12))
    assert int(result.valid_mask.sum()) == 9504
    assert result.source_head_kind == "query"
    assert result.target_head_kind == ("query" if role == "Q" else "kv")
    assert bool((result.scores[result.valid_mask] >= -1e-5).all())
    assert bool((result.scores[result.valid_mask] <= 64 + 1e-4).all())
    assert bool((result.normalized[result.valid_mask] >= -1e-6).all())
    assert bool((result.normalized[result.valid_mask] <= 1 + 1e-5).all())

    source = orthonormal_subspace(gpt2_bridge.blocks[0].attn.W_O[0].T)
    target_weight = getattr(gpt2_bridge.blocks[1].attn, attribute)[1]
    expected = projection_kernel(source, orthonormal_subspace(target_weight))
    assert result.scores[0, 0, 1, 1].item() == pytest.approx(
        expected.score.item(), rel=1e-5, abs=1e-5
    )
    assert result.normalized[0, 0, 1, 1].item() == pytest.approx(
        expected.normalized.item(), rel=1e-5, abs=1e-6
    )


def test_bfloat16_weights_measure_full_rank_without_explicit_rtol(
    gpt2_bridge, gpt2_bridge_bfloat16
):
    """Half-precision storage must not loosen the default rank tolerance.

    gpt2-small is dense MHA with no GQA grouping, so this isolates the storage
    dtype effect from the head-cardinality effect.

    The discriminating assertion is the tolerance itself. gpt2-small's
    worst-conditioned head sits near 40, well below the reciprocal of the
    bfloat16 epsilon, so its measured ranks survive even the storage-dtype
    floor; the rank and bound checks below guard the end-to-end contract rather
    than reproducing the collapse. A head whose spectrum decays past that
    reciprocal is covered by the synthetic unit cases.
    """
    reference = attention_head_subspace_affinity(gpt2_bridge, target_role="Q")
    result = attention_head_subspace_affinity(gpt2_bridge_bfloat16, target_role="Q")

    assert result.rtol < torch.finfo(torch.bfloat16).eps
    assert torch.equal(result.source_ranks, reference.source_ranks)
    assert torch.equal(result.target_ranks, reference.target_ranks)
    assert bool(torch.isfinite(result.scores[result.valid_mask]).all())
    assert bool((result.scores[result.valid_mask] >= -1e-5).all())
    assert bool((result.scores[result.valid_mask] <= 64 + 1e-4).all())
    assert bool((result.normalized[result.valid_mask] >= -1e-6).all())
    assert bool((result.normalized[result.valid_mask] <= 1 + 1e-5).all())
