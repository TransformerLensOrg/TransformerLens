"""Unit tests for per-head QK/OV singular-vector decomposition.

Mostly model-free: most tests build synthetic weight tensors and exercise the factored SVD
and the degeneracy guard directly, so no model is loaded and no pretrained weights are
downloaded. A few tests instantiate a tiny, randomly-initialized TransformerBridge (CPU-only,
no Hub access) where a real per-head decomposition is needed for a cross-check.
"""

import math
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.tools.analysis.svd_circuits import (
    ActivationProjection,
    DegenerateDirectionError,
    HeadSVD,
    LogitSignature,
    PatchResult,
    RankReportRow,
    _degeneracy_blocks,
    _factored_head_svd,
    _make_subspace_hook,
    _resolve_retained,
    decompose_head,
    logit_signature,
    patch_along_directions,
    project_activations,
    vocab_readout,
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


# --------------------------------------------------------------------------- #
# OV output-direction convention (cross-check against SVDInterpreter)
# --------------------------------------------------------------------------- #
def _make_tiny_bridge():
    """Build a tiny GPT-2 bridge: MHA (n_heads == n_kv_heads), CPU-only, no Hub access."""
    from transformers import GPT2Config, GPT2LMHeadModel

    from transformer_lens.config.transformer_bridge_config import (
        TransformerBridgeConfig,
    )
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.model_bridge.supported_architectures.gpt2 import (
        GPT2ArchitectureAdapter,
    )

    torch.manual_seed(0)
    hf_model = GPT2LMHeadModel(
        GPT2Config(n_embd=32, n_layer=1, n_head=2, vocab_size=64, n_positions=32)
    ).eval()
    cfg = TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=1,
        n_ctx=32,
        d_vocab=64,
        architecture="GPT2LMHeadModel",
    )
    tokenizer = MagicMock()
    # A real tokenizer isn't available (no-download fixture), but to_str_tokens's return
    # value is jaxtyped-checked as List[str], so batch_decode must return actual strings.
    tokenizer.batch_decode = lambda tokens_list, **kwargs: [str(ids[0]) for ids in tokens_list]
    return TransformerBridge(hf_model, GPT2ArchitectureAdapter(cfg), tokenizer=tokenizer)


@pytest.fixture(scope="module")
def tiny_bridge():
    """Module-scoped tiny bridge. Compatibility mode folds weights in place and has no
    inverse, so tests that enable it must build their own bridge via ``_make_tiny_bridge``
    rather than mutate this shared instance."""
    return _make_tiny_bridge()


def test_ov_output_direction_matches_svd_interpreter(tiny_bridge):
    """OV's write/vocab-readout direction is ``.V``, not ``.U`` - checked against the
    already-shipped ``SVDInterpreter``, not just internal self-consistency, so the
    assertion cannot pass under either U/V labeling by construction."""
    from transformer_lens.SVDInterpreter import SVDInterpreter

    layer, head = 0, 0
    decomposition = decompose_head(tiny_bridge, layer, head, which=("OV",))
    ov = decomposition.OV
    W_U = tiny_bridge.W_U

    interpreter = SVDInterpreter(tiny_bridge)
    reference = interpreter.get_singular_vectors("OV", layer, head_index=head, num_vectors=1)

    projected_via_V = ov.V[:, 0] @ W_U
    assert torch.allclose(projected_via_V, reference[:, 0, 0], atol=1e-4) or torch.allclose(
        projected_via_V, -reference[:, 0, 0], atol=1e-4
    )

    projected_via_U = ov.U[:, 0] @ W_U
    assert not torch.allclose(projected_via_U, reference[:, 0, 0], atol=1e-2)
    assert not torch.allclose(projected_via_U, -reference[:, 0, 0], atol=1e-2)


def test_vocab_readout_matches_svd_interpreter_through_public_api():
    """The public vocab_readout, not just a hand-rolled V @ W_U, reproduces SVDInterpreter's
    OV readout up to sign.

    The shape-only check elsewhere passes a U-for-V swap or an all-zeros return inside
    vocab_readout; matching the shipped reference through the public function fails both.
    Needs compatibility mode for the W_U-folding contract, so this builds its own bridge
    rather than using the shared no-compat fixture.
    """
    from transformer_lens.SVDInterpreter import SVDInterpreter

    model = _make_tiny_bridge()
    model.enable_compatibility_mode()
    layer, head = 0, 0
    ov = decompose_head(model, layer, head, which=("OV",)).OV
    top = next(row.idx for row in ov.rank_report if not row.is_degenerate)

    readout = vocab_readout(model, ov, k=top + 1)
    reference = SVDInterpreter(model).get_singular_vectors(
        "OV", layer, head_index=head, num_vectors=top + 1
    )[:, 0, top]

    assert torch.allclose(readout[:, top], reference, atol=1e-4) or torch.allclose(
        readout[:, top], -reference, atol=1e-4
    )


# --------------------------------------------------------------------------- #
# Compatibility-mode binding: a decomposition is tied to the state it was built under
# --------------------------------------------------------------------------- #
def test_decompose_head_records_compatibility_mode():
    """Each HeadSVD records the model's compatibility-mode state at decomposition time,
    for both maps and both toggle values."""
    off = decompose_head(_make_tiny_bridge(), 0, 0, which=("QK", "OV"))
    assert off.OV.compatibility_mode is False
    assert off.QK.compatibility_mode is False

    on_model = _make_tiny_bridge()
    on_model.enable_compatibility_mode()
    on = decompose_head(on_model, 0, 0, which=("QK", "OV"))
    assert on.OV.compatibility_mode is True
    assert on.QK.compatibility_mode is True


@pytest.mark.parametrize(
    "call_consumer",
    [
        lambda model, ov: vocab_readout(model, ov),
        lambda model, ov: logit_signature(model, ov, direction=0, tokens=torch.tensor([0])),
        lambda model, ov: project_activations(model, ov, torch.tensor([[5, 63, 7, 9]])),
        lambda model, ov: patch_along_directions(
            model, ov, torch.tensor([[5, 63, 7, 9]]), lambda logits: float(logits.sum()), keep=[0]
        ),
    ],
    ids=["vocab_readout", "logit_signature", "project_activations", "patch_along_directions"],
)
def test_consumers_reject_stale_compatibility_state(call_consumer):
    """A HeadSVD decomposed before enable_compatibility_mode() describes the pre-folding OV
    map, so every consumer refuses it once the model's state has changed and points the
    caller back at decompose_head. The refusal fires before any forward pass runs."""
    bridge = _make_tiny_bridge()
    stale = decompose_head(bridge, 0, 0, which=("OV",)).OV
    bridge.enable_compatibility_mode()
    with pytest.raises(ValueError, match="decompose_head"):
        call_consumer(bridge, stale)


# --------------------------------------------------------------------------- #
# vocab_readout / logit_signature (OV output-direction readout)
# --------------------------------------------------------------------------- #
def test_vocab_readout_shape_and_which_guard():
    """vocab_readout returns [d_vocab, k] for OV, and rejects QK input and out-of-range k."""
    ov = _factored_head_svd(*_random_ov(), which="OV", layer=0, head=0, eps=1e-2)
    vocab_size = 20
    model = SimpleNamespace(W_U=torch.randn(D_MODEL, vocab_size))

    result = vocab_readout(model, ov, k=3)
    assert result.shape == (vocab_size, 3)

    qk = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="QK", layer=0, head=0, eps=1e-2
    )
    with pytest.raises(ValueError, match="OV"):
        vocab_readout(model, qk, k=1)

    with pytest.raises(ValueError, match="k must be"):
        vocab_readout(model, ov, k=D_HEAD + 1)


def test_vocab_readout_raises_without_compatibility_mode(tiny_bridge):
    """A TransformerBridge without compatibility mode enabled refuses to project through W_U."""
    decomposition = decompose_head(tiny_bridge, layer=0, head=0, which=("OV",))
    with pytest.raises(ValueError, match="enable_compatibility_mode"):
        vocab_readout(tiny_bridge, decomposition.OV)


def test_logit_signature_matches_manual_projection():
    """logit_signature's reconstruction matches a hand-computed S[i] * V[:, i] projection."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    vocab_size = 16
    W_U = torch.randn(D_MODEL, vocab_size)
    model = SimpleNamespace(W_U=W_U)
    token_ids = torch.tensor([1, 5, 9])

    result = logit_signature(model, ov, direction=0, tokens=token_ids)
    assert isinstance(result, LogitSignature)
    expected = (ov.S[0] * ov.V[:, 0]) @ W_U[:, token_ids]
    assert torch.allclose(result.values, expected, atol=1e-5)
    assert result.direction == 0


def test_logit_signature_raises_on_degenerate_direction():
    """A rotation-ambiguous direction's 'signature' is not attributable, so this must raise."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([5.0, 3.0, 3.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    model = SimpleNamespace(W_U=torch.randn(D_MODEL, 8))
    with pytest.raises(DegenerateDirectionError):
        logit_signature(model, ov, direction=1, tokens=torch.tensor([0]))


def test_logit_signature_rejects_out_of_range_direction():
    """A direction outside [0, rank) is a clear ValueError, not a wrap-around (direction=-1
    reading the last column) or a bare IndexError (direction=rank)."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    model = SimpleNamespace(W_U=torch.randn(D_MODEL, 8))
    rank = ov.V.shape[1]
    for bad in (rank, -1):
        with pytest.raises(ValueError, match="out of range"):
            logit_signature(model, ov, direction=bad, tokens=torch.tensor([0]))


def _ov_headsvd_in_dtype(dtype):
    """A minimal, well-separated OV HeadSVD with U/S/V cast to ``dtype``.

    Built by casting a float32 decomposition rather than decomposing in ``dtype`` directly,
    since torch.linalg.svd does not run in half precision on CPU; the readout paths under
    test only read U/S/V and W_U, so the cast stands in for a genuine reduced-precision model.
    """
    base = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    return HeadSVD(
        which="OV",
        layer=base.layer,
        head=base.head,
        U=base.U.to(dtype),
        S=base.S.to(dtype),
        V=base.V.to(dtype),
        rank_report=base.rank_report,
        eps=base.eps,
        null_rtol=base.null_rtol,
        compatibility_mode=False,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_vocab_readout_dtype_parametrised(dtype):
    """vocab_readout promotes V to W_U's dtype, so the projection matmul runs on every model
    precision instead of raising a dtype mismatch on bf16, fp16, or fp64."""
    ov = _ov_headsvd_in_dtype(dtype)
    vocab_size = 20
    model = SimpleNamespace(W_U=torch.randn(D_MODEL, vocab_size, dtype=dtype))
    result = vocab_readout(model, ov, k=3)
    assert result.shape == (vocab_size, 3)
    assert result.dtype == dtype
    assert torch.isfinite(result.float()).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_logit_signature_dtype_parametrised(dtype):
    """logit_signature casts its rank-1 reconstruction to W_U's dtype, so it returns one
    finite value per token on every model precision instead of raising a dtype mismatch."""
    ov = _ov_headsvd_in_dtype(dtype)
    vocab_size = 16
    model = SimpleNamespace(W_U=torch.randn(D_MODEL, vocab_size, dtype=dtype))
    token_ids = torch.tensor([1, 5, 9])
    result = logit_signature(model, ov, direction=0, tokens=token_ids)
    assert result.values.shape == (token_ids.numel(),)
    assert result.values.dtype == dtype
    assert torch.isfinite(result.values.float()).all()


# --------------------------------------------------------------------------- #
# project_activations (per-position firing coefficients in the OV output basis)
# --------------------------------------------------------------------------- #
def test_project_activations_which_guard():
    """QK has no write direction to project onto, so a QK HeadSVD is refused."""
    qk = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="QK", layer=0, head=0, eps=1e-2
    )
    with pytest.raises(ValueError, match="OV"):
        project_activations(SimpleNamespace(), qk, "hello world")


@pytest.mark.parametrize("head", [0, 1])
def test_project_activations_reconstructs_head_output(tiny_bridge, head):
    """Coefficients recovered against V reconstruct the actual cached hook_result slice.

    Runs on both heads of the two-head bridge, so a hardcoded head-0 read cannot pass by
    only ever being asked about head 0.
    """
    layer = 0
    decomposition = decompose_head(tiny_bridge, layer, head, which=("OV",))
    # A raw token tensor, not a string: tiny_bridge's tokenizer is a MagicMock and cannot
    # tokenize text, but a token-id tensor bypasses that path entirely.
    prompt = torch.tensor([[5, 63, 7, 9]])

    projection = project_activations(tiny_bridge, decomposition.OV, prompt)
    assert isinstance(projection, ActivationProjection)
    assert projection.head_svd is decomposition.OV

    previous = tiny_bridge.cfg.use_attn_result
    tiny_bridge.set_use_attn_result(True)
    try:
        _, cache = tiny_bridge.run_with_cache(prompt)
    finally:
        tiny_bridge.set_use_attn_result(previous)
    expected = cache[("result", layer, "attn")][0, :, head, :]

    reconstructed = projection.coefficients @ decomposition.OV.V.transpose(-2, -1)
    assert torch.allclose(reconstructed, expected, atol=1e-4)
    assert projection.str_tokens == tiny_bridge.to_str_tokens(prompt)


def test_project_activations_restores_use_attn_result(tiny_bridge):
    """use_attn_result is restored to its prior value regardless of what it started as."""
    decomposition = decompose_head(tiny_bridge, 0, 0, which=("OV",))
    original = tiny_bridge.cfg.use_attn_result
    try:
        for initial in (False, True):
            tiny_bridge.set_use_attn_result(initial)
            project_activations(tiny_bridge, decomposition.OV, torch.tensor([[5, 63, 7, 9]]))
            assert tiny_bridge.cfg.use_attn_result == initial
    finally:
        tiny_bridge.set_use_attn_result(original)


def test_project_activations_caches_only_the_read_hook(tiny_bridge, monkeypatch):
    """Only the one hook_result being read is cached, not every hook point of the forward.

    run_with_cache retains every activation it is not told to filter, so reading one head's
    output without a names_filter would keep the whole forward pass. Pin that a filter is
    passed and that it selects exactly the read hook and nothing else.
    """
    layer, head = 0, 0
    ov = decompose_head(tiny_bridge, layer, head, which=("OV",)).OV
    captured = {}
    original = tiny_bridge.run_with_cache

    def spy(*args, **kwargs):
        captured["names_filter"] = kwargs.get("names_filter")
        return original(*args, **kwargs)

    monkeypatch.setattr(tiny_bridge, "run_with_cache", spy)
    project_activations(tiny_bridge, ov, torch.tensor([[5, 63, 7, 9]]))

    names_filter = captured["names_filter"]
    assert names_filter is not None
    assert names_filter(f"blocks.{layer}.attn.hook_result")
    assert not names_filter(f"blocks.{layer}.attn.hook_z")


def test_project_activations_rejects_batched_tensor_before_forward(tiny_bridge):
    """A batched token tensor is refused before the forward pass, not after caching it.

    The ids are out of vocab range, so a check that ran only after the forward would surface
    the embedding's IndexError instead of this refusal; catching the batch dimension first both
    saves the forward and gives the caller the actionable message.
    """
    ov = decompose_head(tiny_bridge, 0, 0, which=("OV",)).OV
    batched = torch.full((2, 4), 10**6, dtype=torch.long)
    with pytest.raises(ValueError, match="single prompt"):
        project_activations(tiny_bridge, ov, batched)


@pytest.mark.parametrize("head", [0, 1])
def test_subspace_hook_leaves_sibling_head_untouched(tiny_bridge, head):
    """The patch hook rewrites only its own head's slice of the [batch, pos, head, d_model]
    activation: the sibling head's block is bit-for-bit unchanged while the patched head's
    block moves.

    Pins _make_subspace_hook's per-head scope on the two-head bridge, which a hook that
    wrote every head, or hardcoded head 0, would break. The clean hook_result comes from a
    real forward pass; applying the hook directly makes the per-head write the sole variable.
    """
    ov = decompose_head(tiny_bridge, 0, head, which=("OV",)).OV
    other = 1 - head
    prompt = torch.tensor([[5, 63, 7, 9]])

    previous = tiny_bridge.cfg.use_attn_result
    tiny_bridge.set_use_attn_result(True)
    try:
        _, cache = tiny_bridge.run_with_cache(prompt)
    finally:
        tiny_bridge.set_use_attn_result(previous)
    clean = cache[("result", 0, "attn")]

    keep = [ov.rank_report[0].idx]
    projector = ov.V[:, keep] @ ov.V[:, keep].transpose(-2, -1)
    patched = _make_subspace_hook(head, projector)(clean, hook=SimpleNamespace(name="hook_result"))

    assert torch.equal(patched[:, :, other, :], clean[:, :, other, :])
    assert not torch.allclose(patched[:, :, head, :], clean[:, :, head, :])


# --------------------------------------------------------------------------- #
# patch_along_directions (mandatory causal gate)
# --------------------------------------------------------------------------- #
class _PatchStubModel:
    """Model-free stand-in for patch_along_directions' run_with_hooks/metric protocol.

    Exposes a single fixed per-head "hook_result" activation that a hook can intercept
    exactly as ``run_with_hooks`` would present it (the hook receives the activation and
    a hook object, and returns the replacement), then reduces it to a scalar the same
    way for the unmodified, patched, and baseline calls, so the block guard and gating
    arithmetic can be tested without a real forward pass.
    """

    def __init__(self, d_model, n_heads, pos=3, seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.cfg = SimpleNamespace(use_attn_result=False)
        self._result = torch.randn(1, pos, n_heads, d_model, generator=g).to(device)
        self._readout = torch.randn(d_model, generator=g).to(device)

    def set_use_attn_result(self, value):
        self.cfg.use_attn_result = value

    def _logits(self, result):
        return result.sum(dim=2) @ self._readout  # [batch, pos]

    def __call__(self, prompt):
        return self._logits(self._result)

    def run_with_hooks(self, prompt, fwd_hooks):
        activation = self._result
        for name, hook_fn in fwd_hooks:
            activation = hook_fn(activation, hook=SimpleNamespace(name=name))
        return self._logits(activation)


def test_patch_along_directions_which_guard():
    """QK has no write direction to reconstruct onto, so a QK HeadSVD is refused."""
    qk = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="QK", layer=0, head=0, eps=1e-2
    )
    with pytest.raises(ValueError, match="OV"):
        patch_along_directions(SimpleNamespace(), qk, "prompt", lambda logits: 0.0, keep=[0])


def test_patch_along_directions_requires_keep_xor_ablate():
    """Passing both keep and ablate, or neither, is refused before any model access."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    metric = lambda logits: 0.0
    with pytest.raises(ValueError, match="exactly one"):
        patch_along_directions(SimpleNamespace(), ov, "prompt", metric, keep=[0], ablate=[1])
    with pytest.raises(ValueError, match="exactly one"):
        patch_along_directions(SimpleNamespace(), ov, "prompt", metric)


def test_patch_along_directions_rejects_partial_degenerate_block():
    """keep must take a degenerate block whole or not at all; a partial slice is refused,
    and the whole block plus an isolated direction is accepted."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([5.0, 3.0, 3.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    metric = lambda logits: 0.0
    with pytest.raises(DegenerateDirectionError):
        patch_along_directions(SimpleNamespace(), ov, "prompt", metric, keep=[0, 1])

    stub = _PatchStubModel(d_model=D_MODEL, n_heads=1)
    result = patch_along_directions(
        stub, ov, "prompt", lambda logits: float(logits.sum()), keep=[0, 1, 2]
    )
    assert isinstance(result, PatchResult)
    assert result.retained == [0, 1, 2]


def test_resolve_retained_rejects_out_of_range():
    """keep/ablate indices outside [0, rank) raise ValueError, not a silent wrap-around for a
    negative index or a silent no-op for an out-of-range ablate."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    for bad in ([999], [-1]):
        with pytest.raises(ValueError, match="out of range"):
            _resolve_retained(ov, keep=bad, ablate=None)
        with pytest.raises(ValueError, match="out of range"):
            _resolve_retained(ov, keep=None, ablate=bad)


def test_patch_rejects_empty_retained_without_threshold():
    """An empty retained set (keep=[] or ablate over the full rank) reconstructs onto the zero
    subspace and ties the baseline by construction, so it is refused unless the caller passes
    an explicit threshold, and accepted when one is."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    rank = ov.V.shape[1]
    stub = _PatchStubModel(d_model=D_MODEL, n_heads=1)
    metric = lambda logits: float(logits.sum())
    for empty in (dict(keep=[]), dict(ablate=list(range(rank)))):
        with pytest.raises(ValueError, match="threshold"):
            patch_along_directions(stub, ov, "prompt", metric, **empty)
        result = patch_along_directions(
            stub,
            ov,
            "prompt",
            metric,
            threshold=0.0,
            rng=torch.Generator().manual_seed(0),
            **empty,
        )
        assert isinstance(result, PatchResult)
        assert result.retained == []


def test_patch_ablate_rejects_partial_degenerate_block():
    """ablate must take a degenerate block whole: ablating one member of block [1, 2] leaves
    the other in the retained complement, splitting the block, and is refused before any model
    access the same way the keep path is."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([5.0, 3.0, 3.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    with pytest.raises(DegenerateDirectionError):
        patch_along_directions(SimpleNamespace(), ov, "prompt", lambda logits: 0.0, ablate=[1])


def test_patch_ablate_reports_retained():
    """An ablate call retains the complement of the ablated set and reports it, so an ablate
    path is asserted end to end rather than only the keep path."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    stub = _PatchStubModel(d_model=D_MODEL, n_heads=1)
    result = patch_along_directions(
        stub,
        ov,
        "prompt",
        lambda logits: float(logits.sum()),
        ablate=[1, 3],
        rng=torch.Generator().manual_seed(0),
    )
    assert isinstance(result, PatchResult)
    assert result.retained == [0, 2]


def test_patch_along_directions_reports_delta_and_restores_use_attn_result():
    """delta_metric is patched minus original metric, the averaged baseline is finite, and
    use_attn_result is restored to its prior value regardless of what it started as.

    The gate's outcome is not recomputed from the same result's fields here (that would hold
    for any self-consistent implementation); the mode-specific outcomes are pinned under a
    fixed seed by the dedicated gate-semantics tests below.
    """
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    stub = _PatchStubModel(d_model=D_MODEL, n_heads=1)
    metric = lambda logits: float(logits.sum())
    for initial in (False, True):
        stub.set_use_attn_result(initial)
        result = patch_along_directions(
            stub, ov, "prompt", metric, keep=[0], rng=torch.Generator().manual_seed(0)
        )
        assert result.delta_metric == pytest.approx(result.patched_metric - result.original_metric)
        assert math.isfinite(result.baseline_delta_metric)
        assert isinstance(result.gated, bool)
        assert stub.cfg.use_attn_result == initial


def _span_aligned_stub(ov, weights):
    """A patch stub whose single head writes ``ov.V @ weights`` at every position and whose
    readout is that same in-span vector.

    Because the head output lies entirely in ``span(ov.V)`` and the readout equals it, removing
    any in-span component can only reduce the metric, so the per-draw baseline deltas share a
    sign and their average does not cancel toward zero. That makes the gate outcomes analytic:
    the requested subspace's delta and the averaged random-subspace delta are both exact
    functions of ``weights``, so a chosen weight vector fixes which side of the gate a mode
    lands on with a wide margin rather than by numerical luck.
    """
    V = ov.V
    head_out = V @ torch.tensor(weights, dtype=V.dtype)
    stub = _PatchStubModel(d_model=V.shape[0], n_heads=1)
    stub._result = torch.zeros_like(stub._result)
    stub._result[:, :, 0, :] = head_out
    stub._readout = head_out
    return stub


def test_patch_ablate_weak_direction_gates_false():
    """Ablating the weakest isolated direction moves the metric less than removing an
    equally-sized random in-span subspace does, so in ablate mode the gate is False."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    weak = ov.rank_report[-1].idx
    assert not ov.is_degenerate(weak)
    # Starve the weakest direction of energy while the strong ones carry it, so removing the
    # weak one barely moves the metric relative to removing an arbitrary in-span direction.
    weights = [1.0, 1.0, 1.0, 1.0]
    weights[weak] = 1e-3
    stub = _span_aligned_stub(ov, weights)
    metric = lambda logits: float(logits.sum())
    result = patch_along_directions(
        stub, ov, "prompt", metric, ablate=[weak], rng=torch.Generator().manual_seed(0)
    )
    assert result.gated is False
    assert abs(result.delta_metric) < abs(result.baseline_delta_metric)


def test_patch_threshold_above_delta_gates_false():
    """An explicit threshold above abs(delta_metric) overrides the averaged baseline and gates
    False in ablate mode, even for a direction that gates True against the baseline alone."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    strong = ov.rank_report[0].idx
    assert not ov.is_degenerate(strong)
    # Concentrate the head output in the direction to be ablated so removing it beats the
    # random in-span baseline; the gate is then True until an explicit threshold overrides it.
    weights = [1e-3, 1e-3, 1e-3, 1e-3]
    weights[strong] = 1.0
    stub = _span_aligned_stub(ov, weights)
    metric = lambda logits: float(logits.sum())
    default = patch_along_directions(
        stub, ov, "prompt", metric, ablate=[strong], rng=torch.Generator().manual_seed(0)
    )
    assert default.gated is True
    raised = patch_along_directions(
        stub,
        ov,
        "prompt",
        metric,
        ablate=[strong],
        threshold=abs(default.delta_metric) + 1.0,
        rng=torch.Generator().manual_seed(0),
    )
    assert raised.gated is False


def test_patch_baseline_is_reproducible():
    """The averaged baseline is drawn from the passed generator, so the same seed reproduces it
    bit-for-bit and a different seed gives a different average."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    stub = _span_aligned_stub(ov, [1.0, 1.0, 1.0, 1.0])
    metric = lambda logits: float(logits.sum())
    first = patch_along_directions(
        stub, ov, "prompt", metric, ablate=[0], rng=torch.Generator().manual_seed(0)
    )
    same_seed = patch_along_directions(
        stub, ov, "prompt", metric, ablate=[0], rng=torch.Generator().manual_seed(0)
    )
    other_seed = patch_along_directions(
        stub, ov, "prompt", metric, ablate=[0], rng=torch.Generator().manual_seed(1)
    )
    assert same_seed.baseline_delta_metric == first.baseline_delta_metric
    assert other_seed.baseline_delta_metric != first.baseline_delta_metric


def test_patch_keep_mode_gate_semantics():
    """In keep mode the gate asks whether the retained subspace reconstructs the head: keeping
    the single direction the head output lies along preserves the metric better than keeping an
    equally-sized random in-span subspace, so it gates True, where the ablate-style single-sided
    "moved more than baseline" test would have gated the same retained set False."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    strong = ov.rank_report[0].idx
    assert not ov.is_degenerate(strong)
    weights = [0.0, 0.0, 0.0, 0.0]
    weights[strong] = 1.0
    stub = _span_aligned_stub(ov, weights)
    metric = lambda logits: float(logits.sum())
    result = patch_along_directions(
        stub, ov, "prompt", metric, keep=[strong], rng=torch.Generator().manual_seed(0)
    )
    assert result.gated is True
    assert abs(result.delta_metric) < abs(result.baseline_delta_metric)
    assert not abs(result.delta_metric) > abs(result.baseline_delta_metric)


def test_patch_rejects_non_positive_n_baseline():
    """The averaged baseline needs at least one draw, so n_baseline below 1 is refused before
    any model access."""
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    with pytest.raises(ValueError, match="n_baseline"):
        patch_along_directions(
            SimpleNamespace(), ov, "prompt", lambda logits: 0.0, keep=[0], n_baseline=0
        )


@pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="needs an MPS or CUDA device to exercise cross-device projector placement",
)
def test_patch_completes_on_accelerator_device():
    """The kept and baseline projectors are drawn on CPU, but the activation lives on the
    model's device, so the hook must move the projector before the matmul. On an
    accelerator the patch completes without a device-mismatch RuntimeError."""
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    ov = _factored_head_svd(
        *_factored_with_spectrum([8.0, 4.0, 2.0, 1.0]), which="OV", layer=0, head=0, eps=1e-2
    )
    stub = _PatchStubModel(d_model=D_MODEL, n_heads=1, device=device)
    metric = lambda logits: float(logits.sum())
    result = patch_along_directions(
        stub, ov, "prompt", metric, keep=[0], rng=torch.Generator().manual_seed(0)
    )
    assert isinstance(result, PatchResult)
    assert math.isfinite(result.delta_metric)
    assert math.isfinite(result.baseline_delta_metric)


def test_patch_along_directions_discriminates_causal_direction(tiny_bridge):
    """Ablating the head's strongest OV direction should move a metric aligned with that
    direction materially more than ablating its weakest, least load-bearing direction."""
    layer, head = 0, 0
    decomposition = decompose_head(tiny_bridge, layer, head, which=("OV",))
    ov = decomposition.OV
    strong = ov.rank_report[0].idx
    weak = ov.rank_report[-1].idx
    assert not ov.is_degenerate(strong)
    assert not ov.is_degenerate(weak)

    target_token = int((ov.V[:, strong] @ tiny_bridge.W_U).argmax())
    prompt = torch.tensor([[5, 63, 7, 9]])

    def metric(logits):
        return logits[0, -1, target_token].item()

    strong_result = patch_along_directions(
        tiny_bridge, ov, prompt, metric, ablate=[strong], rng=torch.Generator().manual_seed(0)
    )
    weak_result = patch_along_directions(
        tiny_bridge, ov, prompt, metric, ablate=[weak], rng=torch.Generator().manual_seed(0)
    )
    assert isinstance(strong_result, PatchResult)
    assert math.isfinite(strong_result.baseline_delta_metric)
    assert math.isfinite(weak_result.baseline_delta_metric)
    # A wide margin, not a bare ordering: swapping the projected basis inside the patch hook
    # collapses the strong-vs-weak ratio toward 1 while still ordering the two, so a bare `>`
    # would pass the swap this decomposition's OV convention exists to prevent.
    assert abs(strong_result.delta_metric) > 5 * abs(weak_result.delta_metric)


def test_patch_along_directions_restores_use_attn_result(tiny_bridge):
    """use_attn_result is restored to its prior value regardless of what it started as."""
    decomposition = decompose_head(tiny_bridge, 0, 0, which=("OV",))
    ov = decomposition.OV
    metric = lambda logits: float(logits.sum())
    original = tiny_bridge.cfg.use_attn_result
    try:
        for initial in (False, True):
            tiny_bridge.set_use_attn_result(initial)
            patch_along_directions(
                tiny_bridge, ov, torch.tensor([[5, 63, 7, 9]]), metric, keep=[ov.rank_report[0].idx]
            )
            assert tiny_bridge.cfg.use_attn_result == initial
    finally:
        tiny_bridge.set_use_attn_result(original)
