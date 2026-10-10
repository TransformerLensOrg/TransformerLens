"""Real-Bridge integration checks for Projection Kernel head affinity."""

import pytest
import torch

from transformer_lens.tools.analysis.projection_kernel import (
    attention_head_subspace_affinity,
    orthonormal_subspace,
    projection_kernel,
)


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


def test_bfloat16_weights_measure_full_rank_without_explicit_rtol(gpt2_bridge):
    """Half-precision storage must not loosen the default rank tolerance.

    Casting a real checkpoint head to bfloat16 exercises the storage-dtype path
    without booting a second model. The default tolerance must track the float32
    dtype the SVD actually runs in: a storage-dtype floor would land at the
    bfloat16 epsilon instead of ``max(shape) * float32_eps`` and detach the rank
    decision from the computed singular values.
    """
    weight = gpt2_bridge.blocks[0].attn.W_O[0].T
    reference = orthonormal_subspace(weight)
    result = orthonormal_subspace(weight.to(torch.bfloat16))

    assert result.rtol < torch.finfo(torch.bfloat16).eps
    assert result.rtol == pytest.approx(max(weight.shape) * torch.finfo(torch.float32).eps)
    assert result.measured_rank == reference.measured_rank
    assert result.basis.dtype == torch.float32
