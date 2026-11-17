"""Converted lens provenance and optional vocabulary exclusions."""

import pytest
import torch

from tests.unit.tools.conftest import D_MODEL, D_VOCAB, N_LAYERS, _lens
from transformer_lens.tools.analysis import JacobianLens


def test_converted_provenance_round_trips_without_becoming_model_checks(tmp_path, toy_model):
    provenance = {
        "method": "jpp",
        "source_config": {"hf_model_name": "a/mirror"},
        "capture_point": "block_output",
        "stream_reduce": "none",
    }
    path = tmp_path / "converted.pt"
    torch.save(
        {
            "J": {0: torch.eye(D_MODEL)},
            "d_model": D_MODEL,
            "n_prompts": 96,
            "provenance": provenance,
            "metadata": {"note": "retained"},
        },
        path,
    )
    lens = JacobianLens.load(str(path))
    assert lens.metadata == {"note": "retained", "provenance": provenance}
    lens.validate_model(toy_model)
    lens.save(str(path))
    assert JacobianLens.load(str(path)).metadata == lens.metadata


def test_different_imported_provenance_uses_existing_merge_guard(tmp_path):
    lenses = []
    for method in ("jpp", "jacobian"):
        path = tmp_path / f"{method}.pt"
        torch.save(
            {
                "J": {0: torch.eye(D_MODEL)},
                "d_model": D_MODEL,
                "n_prompts": 2,
                "provenance": {"method": method},
            },
            path,
        )
        lenses.append(JacobianLens.load(str(path)))
    with pytest.raises(ValueError, match="provenance"):
        JacobianLens.merge(lenses)
    merged = JacobianLens.merge([lenses[0], lenses[0]])
    assert merged.metadata["provenance"] == {"method": "jpp"}


@pytest.mark.parametrize("use_jacobian", [True, False])
def test_exclusions_rank_before_topk_and_leave_raw_outputs_unchanged(toy_model, use_jacobian):
    lens = _lens()
    kwargs = dict(
        layers=[0, N_LAYERS - 1],
        positions=[-1],
        top_k=3,
        return_full_logits=True,
        use_jacobian=use_jacobian,
    )
    baseline = lens.readout(toy_model, "a prompt", **kwargs)
    excluded = sorted({int(ids[0, 0]) for ids in baseline.lens_topk_indices.values()})
    filtered = lens.readout(toy_model, "a prompt", exclude_token_ids=excluded + excluded, **kwargs)
    assert filtered.exclude_token_ids == excluded
    assert baseline.exclude_token_ids == []
    torch.testing.assert_close(filtered.model_logits, baseline.model_logits, rtol=0, atol=0)
    assert torch.equal(filtered.model_topk_indices, baseline.model_topk_indices)
    assert torch.equal(filtered.model_topk_values, baseline.model_topk_values)
    for layer in baseline.lens_logits:
        original = baseline.lens_logits[layer]
        assert torch.equal(filtered.lens_logits[layer], original)
        ranked = original.clone()
        ranked[:, excluded] = -torch.inf
        expected = ranked.topk(3, dim=-1)
        assert torch.equal(filtered.lens_topk_indices[layer], expected.indices)
        assert torch.equal(filtered.lens_topk_values[layer], original.gather(-1, expected.indices))


def test_empty_exclusions_preserve_default_results(toy_model):
    lens = _lens()
    baseline = lens.readout(toy_model, "a prompt", top_k=3)
    actual = lens.readout(toy_model, "a prompt", top_k=3, exclude_token_ids=[])
    for layer in baseline.lens_topk_indices:
        assert torch.equal(actual.lens_topk_indices[layer], baseline.lens_topk_indices[layer])
        assert torch.equal(actual.lens_topk_values[layer], baseline.lens_topk_values[layer])


def test_excluded_ids_cannot_win_negative_infinity_ties(toy_model, monkeypatch):
    from types import SimpleNamespace

    import transformer_lens.tools.analysis.jacobian_lens as module

    original = toy_model.run_with_cache

    def infinite_logits(*args, **kwargs):
        logits, cache = original(*args, **kwargs)
        return torch.full_like(logits, -torch.inf), cache

    monkeypatch.setattr(toy_model, "run_with_cache", infinite_logits)
    monkeypatch.setattr(
        module,
        "logit_readout",
        lambda *args, **kwargs: SimpleNamespace(values=torch.full((1, D_VOCAB), -torch.inf)),
    )
    result = _lens().readout(
        toy_model,
        "prompt",
        layers=[0, N_LAYERS - 1],
        positions=[-1],
        top_k=2,
        exclude_token_ids=list(range(D_VOCAB - 2)),
    )
    for ids in result.lens_topk_indices.values():
        assert set(ids.flatten().tolist()) == {D_VOCAB - 2, D_VOCAB - 1}


def test_conflicting_provenance_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "conflict.pt"
    torch.save(
        {
            "J": {0: torch.eye(D_MODEL)},
            "d_model": D_MODEL,
            "metadata": {"provenance": {"method": "j"}},
            "provenance": {"method": "jpp"},
        },
        path,
    )
    with pytest.raises(ValueError, match="conflicting"):
        JacobianLens.load(str(path))


@pytest.mark.parametrize(
    "excluded", [[-1], [D_VOCAB], [True], list(range(D_VOCAB)), list(range(D_VOCAB - 1))]
)
def test_invalid_or_insufficient_exclusions_raise(toy_model, excluded):
    with pytest.raises(ValueError, match="exclude|eligible"):
        _lens().readout(toy_model, "a prompt", top_k=2, exclude_token_ids=excluded)
