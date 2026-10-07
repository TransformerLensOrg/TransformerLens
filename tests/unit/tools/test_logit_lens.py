"""Unit tests for the logit lens and logit readout.

Everything here runs on a tiny random GPT-2 built in memory (no Hub download);
real-model exactness lives in ``tests/integration/test_logit_lens.py``.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.tools.analysis import (
    LogitLensResult,
    LogitReadout,
    logit_lens,
    logit_readout,
)

D_VOCAB = 64
D_MODEL = 32
TOKENS = torch.tensor([[1, 7, 11, 3, 5], [2, 9, 4, 4, 6]])


class _Tok:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=D_VOCAB, n_positions=32, n_embd=D_MODEL, n_layer=2, n_head=4)
    hf = GPT2LMHeadModel(cfg).eval()
    model = build_bridge_from_module(
        hf, "GPT2LMHeadModel", hf_config=cfg, dtype=torch.float32, device="cpu"
    )
    with torch.no_grad():
        ref = model(TOKENS)
    _, cache = model.run_with_cache(TOKENS)
    return model, ref, cache


def test_final_entry_reproduces_model_logits(tiny):
    model, ref, _ = tiny
    result = logit_lens(model, TOKENS, positions=None)
    assert isinstance(result, LogitLensResult)
    assert result.labels == ["0_pre", "1_pre", "final_post"]
    assert result.values.shape == (3, 2, 5, D_VOCAB)
    torch.testing.assert_close(result.values[-1], ref, atol=1e-5, rtol=1e-5)
    assert result.readout.applied_ln and result.readout.applied_output_transform


def test_skipping_the_final_norm_does_not_reproduce_the_model(tiny):
    """Teeth for the exactness assertion: the norm is load-bearing."""
    model, ref, _ = tiny
    raw = logit_lens(model, TOKENS, positions=None, apply_ln=False)
    assert not raw.readout.applied_ln
    assert (raw.values[-1] - ref).abs().max() > 1e-2


def test_cache_delegate_matches_function(tiny):
    model, ref, cache = tiny
    via_cache = cache.logit_lens(positions=None)
    via_fn = logit_lens(cache, positions=None)
    torch.testing.assert_close(via_cache.values, via_fn.values)
    torch.testing.assert_close(via_cache.values[-1], ref, atol=1e-5, rtol=1e-5)


def test_readout_on_final_residual_matches_model(tiny):
    model, ref, cache = tiny
    out = logit_readout(model, cache["blocks.1.hook_resid_post"])
    assert isinstance(out, LogitReadout)
    torch.testing.assert_close(out.values, ref, atol=1e-5, rtol=1e-5)
    assert out.entropy.shape == (2, 5)
    expected_entropy = -(torch.softmax(ref, -1) * torch.log_softmax(ref, -1)).sum(-1)
    torch.testing.assert_close(out.entropy, expected_entropy, atol=1e-5, rtol=1e-5)


def test_vocab_subset_is_a_gather_of_the_full_result(tiny):
    model, ref, cache = tiny
    resid = cache["blocks.1.hook_resid_post"]
    subset = [3, 5, 9]
    logits = logit_readout(model, resid, vocab=subset)
    torch.testing.assert_close(logits.values, ref[..., subset])
    assert logits.vocab_ids.tolist() == subset
    # log-probs over a subset keep the full-vocabulary normalizer
    log_probs = logit_readout(model, resid, vocab=subset, return_type="log_probs")
    torch.testing.assert_close(log_probs.values, torch.log_softmax(ref, -1)[..., subset])
    probs = logit_readout(model, resid, vocab=subset, return_type="probs")
    torch.testing.assert_close(probs.values, torch.softmax(ref, -1)[..., subset])


def test_target_ranks_are_full_vocabulary_competition_ranks(tiny):
    model, ref, cache = tiny
    out = logit_readout(model, cache["blocks.1.hook_resid_post"], targets=[3, 5])
    assert out.target_ids.tolist() == [3, 5]
    for column, token in enumerate([3, 5]):
        expected = (ref > ref[..., token : token + 1]).sum(-1)
        assert torch.equal(out.target_ranks[..., column], expected)
    argmax = ref[0, -1].argmax().item()
    top = logit_readout(model, cache["blocks.1.hook_resid_post"][0, -1], targets=[argmax])
    assert top.target_ranks.tolist() == [0]


def test_top_k_matches_torch_topk(tiny):
    model, ref, cache = tiny
    out = logit_readout(model, cache["blocks.1.hook_resid_post"], top_k=3)
    expected = ref.topk(3, dim=-1)
    assert torch.equal(out.top_ids, expected.indices)
    torch.testing.assert_close(out.values, expected.values)
    assert out.decode(_Tok())[0][0] == [f"<{i}>" for i in expected.indices[0, 0].tolist()]


@pytest.mark.parametrize("chunk_size", [1, 3, 1000])
def test_chunking_does_not_change_results(tiny, chunk_size):
    """Only matmul rounding may differ between chunkings; 3 is not a divisor of 10 rows."""
    model, _, cache = tiny
    resid = cache["blocks.1.hook_resid_post"]
    reference = logit_readout(model, resid, chunk_size=10)
    chunked = logit_readout(model, resid, chunk_size=chunk_size, targets=[3])
    torch.testing.assert_close(chunked.values, reference.values[..., [3]], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(chunked.entropy, reference.entropy, atol=1e-6, rtol=1e-6)
    # Per-row targets must advance with the chunks, not restart at row 0.
    nxt = torch.roll(TOKENS, -1, dims=1)
    per_row = logit_readout(model, resid, chunk_size=chunk_size, targets=nxt)
    expected = torch.gather(reference.values, -1, nxt.unsqueeze(-1))
    torch.testing.assert_close(per_row.values, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(per_row.target_ranks, (reference.values > expected).sum(-1, keepdim=True))


def test_layers_select_stack_entries(tiny):
    model, _, cache = tiny
    full = logit_lens(cache, positions=None)
    picked = logit_lens(cache, positions=None, layers=[0, -1])
    assert picked.labels == ["0_pre", "final_post"]
    torch.testing.assert_close(picked.values[0], full.values[0])
    torch.testing.assert_close(picked.values[1], full.values[-1])
    with pytest.raises(ValueError, match="valid entries are 0..2"):
        logit_lens(cache, layers=[3])


def test_incl_mid_adds_mid_entries(tiny):
    model, ref, cache = tiny
    result = logit_lens(model, TOKENS, incl_mid=True, positions=None)
    assert result.labels == ["0_pre", "0_mid", "1_pre", "1_mid", "final_post"]
    torch.testing.assert_close(result.values[-1], ref, atol=1e-5, rtol=1e-5)


def test_positions_and_batch_slices(tiny):
    model, ref, cache = tiny
    last = logit_lens(cache)  # default positions=-1
    assert last.values.shape == (3, 2, D_VOCAB)
    torch.testing.assert_close(last.values[-1], ref[:, -1], atol=1e-5, rtol=1e-5)
    window = logit_lens(cache, positions=(1, 4))
    torch.testing.assert_close(window.values[-1], ref[:, 1:4], atol=1e-5, rtol=1e-5)
    listed = logit_lens(cache, positions=[0, 4])
    torch.testing.assert_close(listed.values[-1], ref[:, [0, 4]], atol=1e-5, rtol=1e-5)
    one_example = logit_lens(cache, positions=None, batch_slice=1)
    assert not one_example.has_batch_dim
    assert one_example.values.shape == (3, 5, D_VOCAB)
    torch.testing.assert_close(one_example.values[-1], ref[1], atol=1e-5, rtol=1e-5)


def test_batchless_cache(tiny):
    model, ref, _ = tiny
    _, cache = model.run_with_cache(TOKENS[:1], remove_batch_dim=True)
    assert not cache.has_batch_dim
    result = logit_lens(cache, positions=None)
    assert not result.has_batch_dim
    assert result.values.shape == (3, 5, D_VOCAB)
    torch.testing.assert_close(result.values[-1], ref[0], atol=1e-5, rtol=1e-5)


def test_lens_callable_is_applied_before_the_norm(tiny):
    model, ref, cache = tiny
    identity = logit_lens(cache, lens=lambda entry, index: entry)
    torch.testing.assert_close(identity.values[-1], ref[:, -1], atol=1e-5, rtol=1e-5)
    seen = []

    def shift(entry, index):
        seen.append(index)
        return entry + 1.0

    shifted = logit_lens(cache, lens=shift)
    assert seen == [0, 1, 2]
    # LayerNorm removes a constant offset, so the shift must not change the readout.
    torch.testing.assert_close(shifted.values, identity.values, atol=1e-5, rtol=1e-5)


def test_rank_trajectory_and_top_tokens(tiny):
    model, ref, cache = tiny
    by_id = logit_lens(cache, targets=[3, 5])
    assert by_id.rank_trajectory(5).shape == (3, 2)
    assert torch.equal(by_id.rank_trajectory(5)[-1], (ref[:, -1] > ref[:, -1, 5:6]).sum(-1))
    with pytest.raises(ValueError, match="pass target="):
        by_id.rank_trajectory()
    with pytest.raises(ValueError, match="not among"):
        by_id.rank_trajectory(7)
    single = logit_lens(cache, targets=3)
    assert torch.equal(single.rank_trajectory(), by_id.rank_trajectory(3))
    top = logit_lens(cache, top_k=2)
    decoded = top.top_tokens(_Tok())
    assert list(decoded) == ["0_pre", "1_pre", "final_post"]
    assert decoded["final_post"][0] == [f"<{i}>" for i in ref[0, -1].topk(2).indices.tolist()]
    with pytest.raises(ValueError, match="top_k"):
        by_id.top_tokens(_Tok())


def test_to_dataframe(tiny):
    model, ref, cache = tiny
    frame = logit_lens(cache, top_k=2).to_dataframe(_Tok())
    assert list(frame.columns) == ["layer", "batch", "token_index", "token_string", "logit"]
    assert len(frame) == 3 * 2 * 2
    full = logit_lens(cache, positions=None, batch_slice=0).to_dataframe(top_k=1)
    assert list(full.columns) == ["layer", "pos", "token_index", "logit", "log_prob", "probability"]
    assert len(full) == 3 * 5


def test_readout_rejects_bad_arguments(tiny):
    model, _, cache = tiny
    resid = cache["blocks.1.hook_resid_post"]
    with pytest.raises(ValueError, match="return_type"):
        logit_readout(model, resid, return_type="ranks")
    with pytest.raises(ValueError, match="at most one"):
        logit_readout(model, resid, vocab=[1], top_k=2)
    with pytest.raises(ValueError, match="chunk_size"):
        logit_readout(model, resid, chunk_size=0)
    with pytest.raises(ValueError, match="d_model"):
        logit_readout(model, resid[..., :-1])
    with pytest.raises(ValueError, match="top_k must be"):
        logit_readout(model, resid, top_k=D_VOCAB + 1)
    with pytest.raises(ValueError, match="within"):
        logit_readout(model, resid, vocab=[D_VOCAB])
    with pytest.raises(TypeError):
        logit_readout(model, resid.long())


def test_lens_rejects_ambiguous_inputs(tiny):
    model, _, cache = tiny
    with pytest.raises(ValueError, match="not both"):
        logit_lens(cache, TOKENS)
    with pytest.raises(ValueError, match="needs an input"):
        logit_lens(model)


def test_requires_eval_mode_before_any_forward(tiny, monkeypatch):
    model, _, cache = tiny

    def forward_ran(*args, **kwargs):
        raise AssertionError("the model was run before the eval-mode check")

    monkeypatch.setattr(model, "run_with_cache", forward_ran)
    model.train()
    try:
        with pytest.raises(ValueError, match="evaluation mode"):
            logit_lens(model, TOKENS)
        with pytest.raises(ValueError, match="evaluation mode"):
            logit_readout(model, cache["blocks.1.hook_resid_post"])
    finally:
        model.eval()


def _fake_model(**cfg_overrides):
    cfg = SimpleNamespace(normalization_type="LN", d_model=D_MODEL, model_name="fake")
    for key, value in cfg_overrides.items():
        setattr(cfg, key, value)
    model = SimpleNamespace(cfg=cfg, unembed=object(), _modules={})
    return model


def test_norm_without_ln_final_component_raises(tiny):
    model = _fake_model()
    with pytest.raises(ValueError, match="no `ln_final` component"):
        logit_readout(model, torch.zeros(2, D_MODEL))


def test_unknown_normalization_type_raises():
    model = _fake_model(normalization_type="BatchNorm")
    with pytest.raises(ValueError, match="normalization_type='BatchNorm'"):
        logit_readout(model, torch.zeros(2, D_MODEL))


def test_encoder_decoder_and_headless_bridges_are_refused():
    model = _fake_model()
    model._modules = {"encoder_blocks": object(), "decoder_blocks": object()}
    with pytest.raises(NotImplementedError, match="encoder-decoder"):
        logit_readout(model, torch.zeros(2, D_MODEL))
    headless = _fake_model()
    headless.unembed = None
    with pytest.raises(NotImplementedError, match="unembed"):
        logit_readout(headless, torch.zeros(2, D_MODEL))


def test_per_row_targets_next_token(tiny):
    """[batch, pos] next-token ids give one rank per (layer, batch, pos) without a full-vocab tensor."""
    model, ref, cache = tiny
    nxt = torch.roll(TOKENS, -1, dims=1)
    result = logit_lens(cache, positions=None, targets=nxt)
    assert result.readout.per_row_targets
    assert result.values.shape == (3, 2, 5, 1)
    expected_values = torch.gather(ref, -1, nxt.unsqueeze(-1))
    torch.testing.assert_close(result.values[-1], expected_values, atol=1e-5, rtol=1e-5)
    expected_ranks = (ref > expected_values).sum(-1)
    assert torch.equal(result.rank_trajectory()[-1], expected_ranks)
    assert result.rank_trajectory().shape == (3, 2, 5)
    assert result.readout.target_ids.shape == (3, 2, 5)
    assert result.readout.decode(_Tok())[0][1][2] == f"<{nxt[1, 2].item()}>"
    with pytest.raises(ValueError, match="one target per row"):
        result.rank_trajectory(3)
    # log-probs per row use the full-vocabulary normalizer
    lp = logit_lens(cache, positions=None, targets=nxt, return_type="log_probs")
    torch.testing.assert_close(
        lp.values[-1],
        torch.gather(torch.log_softmax(ref, -1), -1, nxt.unsqueeze(-1)),
        atol=1e-5,
        rtol=1e-5,
    )


def test_per_row_targets_broadcasting(tiny):
    model, ref, cache = tiny
    per_example = torch.tensor([3, 5])
    last = logit_lens(cache, targets=per_example)  # default positions=-1 -> lead [layers, batch]
    torch.testing.assert_close(
        last.values[-1, :, 0], ref[torch.arange(2), -1, per_example], atol=1e-5, rtol=1e-5
    )
    every_pos = logit_lens(cache, positions=None, targets=per_example.unsqueeze(-1))  # [batch, 1]
    assert every_pos.values.shape == (3, 2, 5, 1)
    torch.testing.assert_close(every_pos.values[-1, 1, :, 0], ref[1, :, 5], atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="do not broadcast"):
        logit_lens(cache, positions=None, targets=torch.tensor([3, 5, 7]))
    with pytest.raises(ValueError, match="more dims"):
        logit_readout(model, cache["blocks.1.hook_resid_post"][0, -1], targets=torch.tensor([[3]]))
    scalar = logit_readout(model, cache["blocks.1.hook_resid_post"], targets=torch.tensor(3))
    assert not scalar.per_row_targets and scalar.target_ids.tolist() == [3]


def test_to_dataframe_full_vocab_log_probs_and_probs(tiny):
    model, ref, cache = tiny
    for return_type, column in (("log_probs", "log_prob"), ("probs", "probability")):
        frame = logit_lens(cache, batch_slice=0, return_type=return_type).to_dataframe(top_k=3)
        assert list(frame.columns) == ["layer", "token_index", column]
        assert len(frame) == 3 * 3
        final = frame[frame.layer == "final_post"]
        assert final.token_index.tolist() == ref[0, -1].topk(3).indices.tolist()
        assert final[column].is_monotonic_decreasing


def test_to_dataframe_selected_rows_sort_by_value(tiny):
    model, ref, cache = tiny
    frame = logit_lens(cache, batch_slice=0, vocab=[3, 5, 9]).to_dataframe(top_k=2)
    assert list(frame.columns) == ["layer", "token_index", "logit"]
    final = frame[frame.layer == "final_post"]
    order = ref[0, -1, [3, 5, 9]].argsort(descending=True)[:2]
    assert final.token_index.tolist() == torch.tensor([3, 5, 9])[order].tolist()
    per_row = logit_lens(cache, positions=None, targets=torch.roll(TOKENS, -1, 1)).to_dataframe(
        _Tok()
    )
    assert list(per_row.columns) == [
        "layer",
        "batch",
        "pos",
        "token_index",
        "token_string",
        "logit",
    ]
    assert len(per_row) == 3 * 2 * 5


def test_enable_grad_reaches_a_lens_translator(tiny):
    model, _, cache = tiny
    translator = torch.nn.Linear(D_MODEL, D_MODEL)
    fitted = logit_lens(
        cache, lens=lambda entry, index: translator(entry), targets=[3], enable_grad=True
    )
    assert fitted.values.requires_grad
    fitted.values.sum().backward()
    assert translator.weight.grad is not None and translator.weight.grad.abs().sum() > 0
    frozen = logit_lens(cache, lens=lambda entry, index: translator(entry), targets=[3])
    assert not frozen.values.requires_grad


def test_arguments_are_validated_before_the_forward_pass(tiny, monkeypatch):
    model, _, _ = tiny

    def forward_ran(*args, **kwargs):
        raise AssertionError("the model was run before argument validation")

    monkeypatch.setattr(model, "run_with_cache", forward_ran)
    with pytest.raises(ValueError, match="return_type"):
        logit_lens(model, TOKENS, return_type="ranks")
    with pytest.raises(ValueError, match="at most one"):
        logit_lens(model, TOKENS, vocab=[1], top_k=2)
    with pytest.raises(ValueError, match="chunk_size"):
        logit_lens(model, TOKENS, chunk_size=0)
    with pytest.raises(ValueError, match="top_k must be"):
        logit_lens(model, TOKENS, top_k=D_VOCAB + 1)
    with pytest.raises(ValueError, match="within"):
        logit_lens(model, TOKENS, targets=[D_VOCAB])


def test_mixed_targets_keep_positional_names(tiny, monkeypatch):
    model, _, cache = tiny
    monkeypatch.setattr(model, "to_single_token", lambda string: 3)
    result = logit_lens(cache, targets=[" x", 5])
    assert result.target_names == [" x", None]
    assert torch.equal(result.rank_trajectory(" x"), result.rank_trajectory(3))
    with pytest.raises(ValueError, match=r"known: \[' x'\]"):
        result.rank_trajectory(" y")


def test_tensor_targets_and_vocab_need_integer_dtypes(tiny):
    model, _, cache = tiny
    resid = cache["blocks.1.hook_resid_post"]
    with pytest.raises(TypeError, match="integer token ids"):
        logit_readout(model, resid, targets=torch.tensor([7.9, 3.0]))
    with pytest.raises(TypeError, match="integer token ids"):
        logit_readout(model, resid, vocab=torch.tensor([True, False]))
    ok = logit_readout(model, resid, vocab=torch.tensor([3, 5], dtype=torch.int32))
    assert ok.vocab_ids.tolist() == [3, 5]


def test_empty_vectors_raise(tiny):
    model, _, _ = tiny
    with pytest.raises(ValueError, match="no rows"):
        logit_readout(model, torch.zeros(0, D_MODEL))
