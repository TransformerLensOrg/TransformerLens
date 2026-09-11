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
    ``n_heads`` rows, matching the per-block shapes ``decompose_head`` reads. Each
    layer draws its own weights so a read from the wrong layer is visible.
    """

    def __init__(
        self,
        n_heads,
        n_kv_heads,
        n_layers=1,
        d_model=D_MODEL,
        d_head=D_HEAD,
        seed=0,
        dtype=torch.float32,
        requires_grad=False,
    ):
        g = torch.Generator().manual_seed(seed)
        self.cfg = SimpleNamespace(n_layers=n_layers, n_heads=n_heads)
        self.blocks = []
        for _ in range(n_layers):
            shapes = [
                (n_heads, d_model, d_head),
                (n_kv_heads, d_model, d_head),
                (n_kv_heads, d_model, d_head),
                (n_heads, d_head, d_model),
            ]
            weights = [torch.randn(*shape, generator=g).to(dtype) for shape in shapes]
            if requires_grad:
                weights = [torch.nn.Parameter(w) for w in weights]
            self.blocks.append(_StubBlock(_StubAttn(*weights)))


def _reconstruct(res):
    return res.U @ torch.diag(res.S) @ res.V.transpose(-2, -1)


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
    assert torch.allclose(_reconstruct(res), W_V @ W_O, atol=1e-4)


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


def test_refusal_message_names_the_rule_that_formed_the_block():
    """A gap-formed block is refused for its eps gap; a null block is refused as null."""
    gap = _factored_head_svd(
        *_factored_with_spectrum([5.0, 3.0, 3.0, 1.0]), which="QK", layer=2, head=3, eps=1e-2
    )
    with pytest.raises(
        DegenerateDirectionError, match=r"QK SVD of head L2H3.*eps=0\.01"
    ) as gap_exc:
        gap.require_isolated(2)
    assert "null" not in str(gap_exc.value)
    null = _factored_head_svd(
        *_factored_with_spectrum([5.0, 2e-9, 1e-9]), which="OV", layer=0, head=0, eps=1e-2
    )
    with pytest.raises(DegenerateDirectionError, match="numerically null"):
        null.require_isolated(1)


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
    """Two directions the gap rule cannot group (8e-3 is 40x above 2e-4) form one null
    block under a null_rtol wide enough to cover both."""
    res = _factored_head_svd(
        *_factored_with_spectrum([1.0, 8e-3, 2e-4]),
        which="OV",
        layer=0,
        head=0,
        eps=1e-2,
        null_rtol=1e-2,
    )
    assert res.degenerate_blocks() == [[1, 2]]
    with pytest.raises(DegenerateDirectionError):
        res.require_isolated(1)
    with pytest.raises(DegenerateDirectionError):
        res.require_isolated(2)


def test_null_default_does_not_overgroup():
    """Under the default null_rtol, 8e-3 and 2e-4 are both far above the null cutoff and 40x
    apart, so neither is degenerate."""
    res = _factored_head_svd(
        *_factored_with_spectrum([1.0, 8e-3, 2e-4]), which="OV", layer=0, head=0, eps=1e-2
    )
    assert not res.is_degenerate(1)
    assert not res.is_degenerate(2)


def test_null_run_groups_true_null_tail():
    """The default null_rtol still catches a genuinely near-zero tail as a null block."""
    res = _factored_head_svd(
        *_factored_with_spectrum([5.0, 1.0, 1e-9, 1e-9]), which="OV", layer=0, head=0, eps=1e-2
    )
    assert res.degenerate_blocks() == [[2, 3]]
    with pytest.raises(DegenerateDirectionError):
        res.require_isolated(2)
    with pytest.raises(DegenerateDirectionError):
        res.require_isolated(3)


def test_lone_null_direction_is_degenerate():
    """A single null direction is refused on its own: its singular vector is an arbitrary
    null-space vector even though no neighbour groups with it."""
    res = _factored_head_svd(
        *_factored_with_spectrum([5.0, 3.0, 1.0, 0.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    assert res.rank_report[3].is_null and res.is_degenerate(3)
    assert res.degenerate_blocks() == [[3]]
    with pytest.raises(DegenerateDirectionError, match="null"):
        res.require_isolated(3)
    assert all(res.require_isolated(i) is None for i in range(3))


def test_degenerate_blocks_returns_equal_run():
    """degenerate_blocks returns the repeated-value run and nothing else."""
    res = _factored_head_svd(
        *_factored_with_spectrum([5.0, 3.0, 3.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    assert res.degenerate_blocks() == [[1, 2]]


def test_slow_decay_forms_one_gap_separated_block():
    """A tail decaying 0.9% per step has no eps=1e-2 gap anywhere inside it, so it is one
    block, and only the well-separated top direction is isolated."""
    tail = [0.5 * (0.991**i) for i in range(7)]
    spectrum = [1.0] + tail
    res = _factored_head_svd(
        *_factored_with_spectrum(spectrum), which="OV", layer=0, head=0, eps=1e-2
    )
    assert res.require_isolated(0) is None
    assert res.degenerate_blocks() == [[1, 2, 3, 4, 5, 6, 7]]


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
    blocks = _degeneracy_blocks(torch.tensor([5.0, 3.0, 3.0, 1.0]), eps=1e-2, null_rtol=1e-2)
    assert blocks == [[0], [1, 2], [3]]


def test_degeneracy_blocks_on_well_separated_are_all_singletons():
    blocks = _degeneracy_blocks(torch.tensor([8.0, 4.0, 2.0, 1.0]), eps=1e-2, null_rtol=1e-2)
    assert blocks == [[0], [1], [2], [3]]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("eps", [1e-2, 5e-2])
def test_every_block_boundary_is_an_eps_gap(seed, eps):
    """Blocks end exactly where the spectrum has a relative gap of at least eps, so every
    boundary is eps-separated and no singleton has a neighbour within eps."""
    g = torch.Generator().manual_seed(seed)
    S = torch.sort(torch.rand(64, generator=g) * 10 + 0.1, descending=True).values
    S[10:20] = S[10]  # a planted exactly-equal run
    blocks = _degeneracy_blocks(S, eps=eps, null_rtol=0.0)
    gap = lambda i, j: 1.0 - float(S[j]) / float(S[i])  # i < j
    assert [i for block in blocks for i in block] == list(range(64))
    for left, right in zip(blocks, blocks[1:]):
        assert gap(left[-1], right[0]) >= eps
    for block in blocks:
        for i, j in zip(block, block[1:]):
            assert gap(i, j) < eps


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
        assert torch.allclose(_reconstruct(result.OV), attn.W_V[h] @ attn.W_O[h], atol=1e-4)


def test_decompose_head_gqa_maps_query_to_kv_head():
    """GQA stub (4 query heads, 2 kv heads): each query head reads its own kv head, with
    U/V orientation and the head identity pinned as well."""
    model = _StubModel(n_heads=4, n_kv_heads=2)
    attn = model.blocks[0].attn
    n_heads = model.cfg.n_heads
    n_kv_heads = attn.W_K.shape[0]
    for h in range(n_heads):
        result = decompose_head(model, layer=0, head=h, which=("QK", "OV"))
        kv_head = h // (n_heads // n_kv_heads)
        ov_ref = attn.W_V[kv_head] @ attn.W_O[h]
        qk_ref = attn.W_Q[h] @ attn.W_K[kv_head].transpose(-1, -2)
        assert torch.allclose(result.OV.S, torch.linalg.svd(ov_ref).S[:D_HEAD], atol=1e-4)
        assert torch.allclose(result.QK.S, torch.linalg.svd(qk_ref).S[:D_HEAD], atol=1e-4)
        assert torch.allclose(_reconstruct(result.OV), ov_ref, atol=1e-4)
        assert torch.allclose(_reconstruct(result.QK), qk_ref, atol=1e-4)
        assert (result.QK.which, result.QK.layer, result.QK.head) == ("QK", 0, h)
        assert (result.OV.which, result.OV.layer, result.OV.head) == ("OV", 0, h)


def test_decompose_head_reads_requested_layer():
    """On a two-layer stub, layer 1's decomposition matches layer 1's weights, not layer 0's."""
    model = _StubModel(n_heads=2, n_kv_heads=2, n_layers=2)
    result = decompose_head(model, layer=1, head=1, which=("OV",))
    own = model.blocks[1].attn.W_V[1] @ model.blocks[1].attn.W_O[1]
    other = model.blocks[0].attn.W_V[1] @ model.blocks[0].attn.W_O[1]
    assert torch.allclose(_reconstruct(result.OV), own, atol=1e-4)
    assert not torch.allclose(_reconstruct(result.OV), other, atol=1e-2)
    assert result.OV.layer == 1


def test_decompose_head_qk_transpose_wiring():
    """QK factors W_Q_h and W_K_h.T; res.QK.S matches svd(W_Q_h @ W_K_h.T)."""
    model = _StubModel(n_heads=4, n_kv_heads=4)
    attn = model.blocks[0].attn
    result = decompose_head(model, layer=0, head=2, which=("QK",))
    S_ref = torch.linalg.svd(attn.W_Q[2] @ attn.W_K[2].transpose(-1, -2)).S
    assert torch.allclose(result.QK.S, S_ref[:D_HEAD], atol=1e-4)


def test_decompose_head_detaches_from_model_weights():
    """The returned factors carry no autograd graph back to the model's parameters."""
    model = _StubModel(n_heads=2, n_kv_heads=2, requires_grad=True)
    result = decompose_head(model, layer=0, head=0)
    for res in (result.QK, result.OV):
        for tensor in (res.U, res.S, res.V):
            assert not tensor.requires_grad and tensor.grad_fn is None


def test_decompose_head_keeps_float64_and_promotes_half():
    """float64 weights stay float64 (and null_rtol follows), fp16 weights are promoted."""
    result = decompose_head(_StubModel(n_heads=2, n_kv_heads=2, dtype=torch.float64), 0, 0)
    assert result.OV.S.dtype == torch.float64
    assert result.OV.null_rtol == pytest.approx(D_MODEL * torch.finfo(torch.float64).eps)
    result = decompose_head(_StubModel(n_heads=2, n_kv_heads=2, dtype=torch.float16), 0, 0)
    assert result.OV.S.dtype == torch.float32
    assert result.OV.null_rtol == pytest.approx(D_MODEL * torch.finfo(torch.float32).eps)


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
    model = _StubModel(n_heads=2, n_kv_heads=2)
    with pytest.raises(ValueError, match="which entries"):
        decompose_head(model, layer=0, head=0, which=("QK", "XX"))


def test_decompose_head_rejects_bare_string_which():
    """A bare string is refused instead of being iterated into its characters."""
    model = _StubModel(n_heads=2, n_kv_heads=2)
    with pytest.raises(ValueError, match="sequence"):
        decompose_head(model, layer=0, head=0, which="QK")
