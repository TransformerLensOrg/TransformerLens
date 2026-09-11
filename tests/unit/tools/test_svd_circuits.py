"""Unit tests for per-head QK/OV singular-vector decomposition.

Model-free: they build synthetic weight tensors and exercise the factored SVD and the
degeneracy guard directly, so no model is loaded and no pretrained weights are downloaded.
"""

import warnings
from types import SimpleNamespace

import pytest
import torch

from transformer_lens.tools.analysis.svd_circuits import (
    DegenerateDirectionError,
    HeadSVD,
    RankReportRow,
    _degeneracy_blocks,
    _factored_head_svd,
    decompose_head,
)

D_MODEL, D_HEAD = 12, 4


class _StubAttn:
    def __init__(self, W_Q, W_K, W_V, W_O):
        self.W_Q, self.W_K, self.W_V, self.W_O = W_Q, W_K, W_V, W_O


class _StubBlock:
    def __init__(self, attn):
        self.attn = attn


class _StubModel:
    """Model-free stand-in exposing cfg.n_layers/n_heads and blocks[i].attn.W_*.

    W_K/W_V carry ``n_kv_heads`` rows (the grouped-query layout); W_Q/W_O carry
    ``n_heads`` rows, matching the per-block shapes ``decompose_head`` reads.
    """

    def __init__(
        self,
        n_heads,
        n_kv_heads,
        n_layers=1,
        d_model=D_MODEL,
        d_head=D_HEAD,
        seed=0,
    ):
        g = torch.Generator().manual_seed(seed)
        W_Q = torch.randn(n_heads, d_model, d_head, generator=g)
        W_K = torch.randn(n_kv_heads, d_model, d_head, generator=g)
        W_V = torch.randn(n_kv_heads, d_model, d_head, generator=g)
        W_O = torch.randn(n_heads, d_head, d_model, generator=g)
        self.cfg = SimpleNamespace(n_layers=n_layers, n_heads=n_heads)
        self.blocks = [_StubBlock(_StubAttn(W_Q, W_K, W_V, W_O)) for _ in range(n_layers)]


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


def test_degenerate_blocks_returns_equal_run():
    """degenerate_blocks pins the actual grouping for a repeated value, defeating a
    "return [] unconditionally" mutation that would otherwise slip through a test that
    only checks non-degeneracy elsewhere."""
    res = _factored_head_svd(
        *_factored_with_spectrum([5.0, 3.0, 3.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    assert res.degenerate_blocks() == [[1, 2]]


def test_slow_decay_does_not_form_one_block():
    """A spectrum decaying ~0.9% per step, just under eps=1e-2, no longer chains every
    direction into one block spanning far more than eps: each block's spread is capped to
    eps of its own anchor, so the slowly decaying tail breaks into several small blocks
    instead of one, and the well-separated top direction stays isolated."""
    tail = [0.5 * (0.991**i) for i in range(7)]
    spectrum = [1.0] + tail
    res = _factored_head_svd(
        *_factored_with_spectrum(spectrum), which="OV", layer=0, head=0, eps=1e-2
    )
    assert res.require_isolated(0) is None
    assert res.degenerate_blocks() == [[1, 2], [3, 4], [5, 6]]


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


# --------------------------------------------------------------------------- #
# decompose_head wiring and input guards (model-free, via _StubModel)
# --------------------------------------------------------------------------- #
def test_decompose_head_mha_reads_requested_head():
    """MHA stub (n_heads == n_kv_heads): each head's OV decomposes its own W_V/W_O."""
    model = _StubModel(n_heads=4, n_kv_heads=4)
    attn = model.blocks[0].attn
    for h in range(4):
        result = decompose_head(model, layer=0, head=h, which=("OV",))
        S_ref = torch.linalg.svd(attn.W_V[h] @ attn.W_O[h]).S
        assert torch.allclose(result.OV.S, S_ref[:D_HEAD], atol=1e-4)


def test_decompose_head_gqa_maps_query_to_kv_head():
    """GQA stub (4 query heads, 2 kv heads): every head decomposes without IndexError,
    and heads 0,1 recover kv head 0's W_K/W_V while heads 2,3 recover kv head 1's."""
    model = _StubModel(n_heads=4, n_kv_heads=2)
    attn = model.blocks[0].attn
    n_heads = model.cfg.n_heads
    n_kv_heads = attn.W_K.shape[0]
    for h in range(n_heads):
        result = decompose_head(model, layer=0, head=h, which=("QK", "OV"))
        kv_head = h // (n_heads // n_kv_heads)
        ov_ref = torch.linalg.svd(attn.W_V[kv_head] @ attn.W_O[h]).S
        qk_ref = torch.linalg.svd(attn.W_Q[h] @ attn.W_K[kv_head].transpose(-1, -2)).S
        assert torch.allclose(result.OV.S, ov_ref[:D_HEAD], atol=1e-4)
        assert torch.allclose(result.QK.S, qk_ref[:D_HEAD], atol=1e-4)


def test_decompose_head_qk_transpose_wiring():
    """QK factors W_Q_h and W_K_h.T; res.QK.S matches svd(W_Q_h @ W_K_h.T)."""
    model = _StubModel(n_heads=4, n_kv_heads=4)
    attn = model.blocks[0].attn
    result = decompose_head(model, layer=0, head=2, which=("QK",))
    S_ref = torch.linalg.svd(attn.W_Q[2] @ attn.W_K[2].transpose(-1, -2)).S
    assert torch.allclose(result.QK.S, S_ref[:D_HEAD], atol=1e-4)


def test_decompose_head_which_filter():
    """which=("OV",) returns only OV; which=("QK",) returns only QK."""
    model = _StubModel(n_heads=2, n_kv_heads=2)
    ov_only = decompose_head(model, layer=0, head=0, which=("OV",))
    assert ov_only.OV is not None and ov_only.QK is None
    qk_only = decompose_head(model, layer=0, head=0, which=("QK",))
    assert qk_only.QK is not None and qk_only.OV is None


def test_decompose_head_rejects_bad_layer():
    model = _StubModel(n_heads=2, n_kv_heads=2, n_layers=1)
    with pytest.raises(ValueError):
        decompose_head(model, layer=1, head=0)


def test_decompose_head_rejects_bad_head():
    model = _StubModel(n_heads=2, n_kv_heads=2)
    with pytest.raises(ValueError):
        decompose_head(model, layer=0, head=2)


def test_decompose_head_rejects_empty_which():
    model = _StubModel(n_heads=2, n_kv_heads=2)
    with pytest.raises(ValueError):
        decompose_head(model, layer=0, head=0, which=())


def test_decompose_head_rejects_unknown_which():
    """An unknown which entry is rejected; the jaxtyping/beartype import hook enforced
    by this suite's pytest config raises its own violation before the manual ValueError
    guard runs, so the check is on the exception class name rather than ValueError."""
    model = _StubModel(n_heads=2, n_kv_heads=2)
    with pytest.raises(Exception) as exc_info:
        decompose_head(model, layer=0, head=0, which=("QK", "XX"))
    exc_name = type(exc_info.value).__name__
    assert "TypeCheckError" in exc_name or "Beartype" in exc_name
