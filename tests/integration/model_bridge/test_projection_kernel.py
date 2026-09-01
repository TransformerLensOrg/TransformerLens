"""Real-Bridge integration checks for Projection Kernel head affinity."""

import pytest

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
