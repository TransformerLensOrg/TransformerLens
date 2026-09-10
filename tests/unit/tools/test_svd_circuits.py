"""Unit tests for per-head QK/OV singular-vector decomposition.

Model-free: they build synthetic weight tensors and exercise the factored SVD and the
degeneracy guard directly, so no model is loaded and no pretrained weights are downloaded.
"""

import warnings

import pytest
import torch

from transformer_lens.tools.analysis.svd_circuits import (
    DegenerateDirectionError,
    HeadSVD,
    RankReportRow,
    _degeneracy_blocks,
    _factored_head_svd,
)

D_MODEL, D_HEAD = 12, 4


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #
def _random_ov(seed=0):
    """A non-degenerate OV factoring: W_V [d_model, d_head], W_O [d_head, d_model]."""
    g = torch.Generator().manual_seed(seed)
    W_V = torch.randn(D_MODEL, D_HEAD, generator=g)
    W_O = torch.randn(D_HEAD, D_MODEL, generator=g)
    return W_V, W_O


def _factored_with_spectrum(spectrum, seed=0):
    """Build (A, B) so that A @ B == U0 @ diag(s) @ V0.T with U0, V0 orthonormal columns.

    ``spectrum`` is a descending sequence of singular values; passing a repeated value plants
    an exactly-degenerate block, so the recovered SVD has a known, controllable spectrum.
    """
    s = torch.as_tensor(spectrum, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    U0, _ = torch.linalg.qr(torch.randn(D_MODEL, len(s), generator=g))
    V0, _ = torch.linalg.qr(torch.randn(D_MODEL, len(s), generator=g))
    root = s.sqrt()
    A = U0 @ torch.diag(root)  # [d_model, r]
    B = torch.diag(root) @ V0.transpose(-2, -1)  # [r, d_model]
    return A, B


# --------------------------------------------------------------------------- #
# Oracle / correctness of the factored SVD
# --------------------------------------------------------------------------- #
def test_singular_values_match_torch_linalg_svd_on_ov():
    """The factored OV SVD reproduces the singular values of the materialized W_V @ W_O."""
    W_V, W_O = _random_ov()
    S_ref = torch.linalg.svd(W_V @ W_O).S
    res = _factored_head_svd(W_V, W_O, which="OV", layer=0, head=0, eps=1e-2)
    assert isinstance(res, HeadSVD)
    assert torch.allclose(res.S, S_ref[:D_HEAD], atol=1e-5)


def test_reconstruction_matches_materialized_matrix():
    """U @ diag(S) @ V.T rebuilds W_V @ W_O, sidestepping the U/V column-sign ambiguity."""
    W_V, W_O = _random_ov()
    res = _factored_head_svd(W_V, W_O, which="OV", layer=0, head=0, eps=1e-2)
    reconstruction = res.U @ torch.diag(res.S) @ res.V.transpose(-2, -1)
    assert torch.allclose(reconstruction, W_V @ W_O, atol=1e-4)


def test_singular_values_sorted_descending():
    W_V, W_O = _random_ov()
    res = _factored_head_svd(W_V, W_O, which="OV", layer=0, head=0, eps=1e-2)
    assert bool(torch.all(res.S[:-1] + 1e-6 >= res.S[1:]))


def test_rank_bounded_by_d_head():
    """A product of d_head-rank factors has at most d_head singular values, all reported."""
    W_V, W_O = _random_ov()
    res = _factored_head_svd(W_V, W_O, which="OV", layer=0, head=0, eps=1e-2)
    assert res.S.shape[0] == D_HEAD
    assert int((res.S > 1e-6).sum()) <= D_HEAD


def test_qk_svd_matches_materialized():
    """The QK path factors W_Q and W_K.T and matches torch.linalg.svd(W_Q @ W_K.T)."""
    g = torch.Generator().manual_seed(1)
    W_Q = torch.randn(D_MODEL, D_HEAD, generator=g)
    W_K = torch.randn(D_MODEL, D_HEAD, generator=g)
    S_ref = torch.linalg.svd(W_Q @ W_K.transpose(-1, -2)).S
    res = _factored_head_svd(W_Q, W_K.transpose(-1, -2), which="QK", layer=0, head=0, eps=1e-2)
    assert torch.allclose(res.S, S_ref[:D_HEAD], atol=1e-5)


def test_uses_V_not_Vh_no_deprecation_warning():
    """FactoredMatrix.Vh is a deprecated alias that warns; the decomposition must read .V only."""
    W_V, W_O = _random_ov()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _factored_head_svd(W_V, W_O, which="OV", layer=0, head=0, eps=1e-2)


def test_rank_report_shape_and_ratio():
    """Each direction gets one report row; sigma_ratio is sigma normalized by the top value."""
    W_V, W_O = _random_ov()
    res = _factored_head_svd(W_V, W_O, which="OV", layer=0, head=0, eps=1e-2)
    assert len(res.rank_report) == D_HEAD
    assert all(isinstance(row, RankReportRow) for row in res.rank_report)
    top = float(res.S.max())
    for row in res.rank_report:
        assert row.sigma_ratio == pytest.approx(row.sigma / top, rel=1e-5)
    assert res.rank_report[0].sigma_ratio == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Degeneracy guard: refuse per-direction attribution inside a degenerate block
# --------------------------------------------------------------------------- #
def test_degeneracy_block_flagged_on_equal_singular_values():
    """Equal singular values (5, 3, 3, 1) make directions 1 and 2 rotation-ambiguous."""
    res = _factored_head_svd(
        *_factored_with_spectrum([5.0, 3.0, 3.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    assert res.is_degenerate(1) and res.is_degenerate(2)
    assert not res.is_degenerate(0) and not res.is_degenerate(3)


def test_block_of_returns_full_degenerate_run():
    """block_of returns the whole degenerate run for a member, and a singleton otherwise."""
    res = _factored_head_svd(
        *_factored_with_spectrum([5.0, 3.0, 3.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    assert set(res.block_of(1)) == {1, 2} == set(res.block_of(2))
    assert res.block_of(0) == [0]


def test_require_isolated_raises_inside_block():
    """require_isolated raises for a degenerate direction and returns None for an isolated one."""
    res = _factored_head_svd(
        *_factored_with_spectrum([5.0, 3.0, 3.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    with pytest.raises(DegenerateDirectionError):
        res.require_isolated(1)
    assert res.require_isolated(0) is None


def test_degenerate_error_message_points_to_block():
    """The refusal names the block indices and steers the caller to subspace attribution."""
    res = _factored_head_svd(
        *_factored_with_spectrum([5.0, 3.0, 3.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    with pytest.raises(DegenerateDirectionError) as excinfo:
        res.require_isolated(1)
    message = str(excinfo.value)
    assert "1" in message and "2" in message
    assert "subspace" in message.lower()


def test_well_separated_spectrum_flags_nothing():
    """A well-separated spectrum (8, 4, 2, 1) has no degenerate block and no flagged row."""
    res = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    assert res.degenerate_blocks() == []
    assert not any(row.is_degenerate for row in res.rank_report)


def test_near_but_not_equal_respects_eps():
    """A relative gap of 5e-3 groups under eps=1e-2 but stays isolated under eps=1e-3."""
    spectrum = [3.0, 3.0 * (1 - 5e-3), 1.0, 0.5]
    grouped = _factored_head_svd(
        *_factored_with_spectrum(spectrum), which="OV", layer=0, head=0, eps=1e-2
    )
    assert grouped.is_degenerate(0) and grouped.is_degenerate(1)
    separated = _factored_head_svd(
        *_factored_with_spectrum(spectrum), which="OV", layer=0, head=0, eps=1e-3
    )
    assert not separated.is_degenerate(0) and not separated.is_degenerate(1)


def test_null_run_grouped():
    """A run of near-zero singular values forms one degenerate null block."""
    res = _factored_head_svd(
        *_factored_with_spectrum([5.0, 1.0, 1e-9, 1e-9]), which="OV", layer=0, head=0, eps=1e-2
    )
    assert res.is_degenerate(2) and res.is_degenerate(3)
    with pytest.raises(DegenerateDirectionError):
        res.require_isolated(2)
    with pytest.raises(DegenerateDirectionError):
        res.require_isolated(3)


def test_degeneracy_blocks_partition_all_indices():
    """Degenerate blocks plus the remaining singletons cover every direction exactly once."""
    res = _factored_head_svd(
        *_factored_with_spectrum([5.0, 3.0, 3.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    covered = [i for block in res.degenerate_blocks() for i in block]
    singletons = [i for i in range(D_HEAD) if i not in covered]
    assert sorted(covered + singletons) == list(range(D_HEAD))
    assert len(covered + singletons) == D_HEAD  # no direction counted twice


# --------------------------------------------------------------------------- #
# Degeneracy grouping on a raw spectrum (fully model-free)
# --------------------------------------------------------------------------- #
def test_degeneracy_blocks_groups_equal_run_directly():
    """_degeneracy_blocks groups a repeated value and isolates the well-separated neighbours."""
    blocks = _degeneracy_blocks(torch.tensor([5.0, 3.0, 3.0, 1.0]), eps=1e-2)
    assert blocks == [[0], [1, 2], [3]]


def test_degeneracy_blocks_on_well_separated_are_all_singletons():
    blocks = _degeneracy_blocks(torch.tensor([8.0, 4.0, 2.0, 1.0]), eps=1e-2)
    assert blocks == [[0], [1], [2], [3]]
