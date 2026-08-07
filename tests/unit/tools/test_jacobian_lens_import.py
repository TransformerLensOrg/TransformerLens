"""Unit tests for JacobianLens.load() checkpoint support.

All tests use synthetic in-memory fixtures — no model, no Hub access.
"""

import pytest
import torch

from transformer_lens.tools.analysis.jacobian_lens import JacobianLens

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _save_artifact(path, *, n_layers=3, d_model=8, n_prompts=10, metadata=None):
    """Write a minimal lens artifact (J-key format) to *path*."""
    payload = {
        "J": {i: torch.randn(d_model, d_model).half() for i in range(n_layers)},
        "n_prompts": n_prompts,
        "source_layers": list(range(n_layers)),
        "d_model": d_model,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, path)


def _save_checkpoint(
    path,
    *,
    n_layers=3,
    d_model=8,
    n_prompts=5,
    metadata=None,
    flat_provenance=None,
    n_prompts_override=None,
):
    """Write a minimal fit-checkpoint (jacobian_sum format) to *path*.

    *n_prompts_override* lets callers inject a bad value without changing the
    sums (useful for zero/negative n_prompts tests).
    """
    payload = {
        "jacobian_sum": {i: torch.randn(d_model, d_model) * n_prompts for i in range(n_layers)},
        "n_prompts": n_prompts if n_prompts_override is None else n_prompts_override,
        "source_layers": list(range(n_layers)),
        "d_model": d_model,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    if flat_provenance is not None:
        payload.update(flat_provenance)
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Artifact loading (regression tests — existing behaviour must be preserved)
# ---------------------------------------------------------------------------


class TestLoadArtifact:
    def test_basic_load(self, tmp_path):
        p = str(tmp_path / "lens.pt")
        _save_artifact(p, n_layers=3, d_model=8, n_prompts=10)
        lens = JacobianLens.load(p)
        assert lens.d_model == 8
        assert lens.n_prompts == 10
        assert lens.source_layers == [0, 1, 2]

    def test_matrices_cast_to_fp32(self, tmp_path):
        p = str(tmp_path / "lens.pt")
        _save_artifact(p, d_model=8)
        lens = JacobianLens.load(p)
        assert all(j.dtype == torch.float32 for j in lens.jacobians.values())

    def test_metadata_preserved(self, tmp_path):
        p = str(tmp_path / "lens.pt")
        meta = {"model_name": "gpt2", "corpus": "wikitext"}
        _save_artifact(p, metadata=meta)
        lens = JacobianLens.load(p)
        assert lens.metadata["model_name"] == "gpt2"
        assert lens.metadata["corpus"] == "wikitext"

    def test_missing_metadata_gives_empty_dict(self, tmp_path):
        p = str(tmp_path / "lens.pt")
        _save_artifact(p)
        lens = JacobianLens.load(p)
        assert lens.metadata == {}

    def test_unknown_format_raises(self, tmp_path):
        p = str(tmp_path / "garbage.pt")
        torch.save({"random_key": 42}, str(p))
        with pytest.raises(ValueError, match="does not look like"):
            JacobianLens.load(p)


# ---------------------------------------------------------------------------
# Checkpoint loading (new functionality)
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    def test_basic_load(self, tmp_path):
        p = str(tmp_path / "ckpt.pt")
        _save_checkpoint(p, n_layers=3, d_model=8, n_prompts=5)
        lens = JacobianLens.load(p)
        assert lens.d_model == 8
        assert lens.n_prompts == 5
        assert lens.source_layers == [0, 1, 2]

    def test_means_reconstructed_correctly(self, tmp_path):
        """jacobian_sum / n_prompts must equal the stored average."""
        d_model, n_prompts = 8, 5
        sums = {i: torch.ones(d_model, d_model) * n_prompts for i in range(2)}
        p = str(tmp_path / "ckpt.pt")
        payload = {
            "jacobian_sum": sums,
            "n_prompts": n_prompts,
            "source_layers": [0, 1],
            "d_model": d_model,
        }
        torch.save(payload, p)
        lens = JacobianLens.load(p)
        for j in lens.jacobians.values():
            assert torch.allclose(j, torch.ones(d_model, d_model), atol=1e-6)

    def test_converted_from_key_set(self, tmp_path):
        p = str(tmp_path / "ckpt.pt")
        _save_checkpoint(p)
        lens = JacobianLens.load(p)
        assert lens.metadata.get("converted_from") == "jacobian_lens_checkpoint"

    def test_fit_reserved_key_stripped(self, tmp_path):
        p = str(tmp_path / "ckpt.pt")
        meta = {
            "transformer_lens_fit": True,
            "transformer_lens_version": "1.2.3",
            "model_system": "TransformerBridge",
            "corpus": "wiki",
        }
        _save_checkpoint(p, metadata=meta)
        lens = JacobianLens.load(p)
        assert "transformer_lens_fit" not in lens.metadata
        assert "transformer_lens_version" not in lens.metadata
        assert "model_system" not in lens.metadata
        # safe keys survive
        assert lens.metadata.get("corpus") == "wiki"

    def test_safe_scalar_metadata_preserved(self, tmp_path):
        p = str(tmp_path / "ckpt.pt")
        meta = {
            "model_name": "gpt2",
            "corpus": "wikitext",
            "extra_int": 42,
            "extra_float": 3.14,
            "extra_list": [1, 2, 3],
        }
        _save_checkpoint(p, metadata=meta)
        lens = JacobianLens.load(p)
        assert lens.metadata["model_name"] == "gpt2"
        assert lens.metadata["corpus"] == "wikitext"
        assert lens.metadata["extra_int"] == 42
        assert lens.metadata["extra_float"] == pytest.approx(3.14)
        assert lens.metadata["extra_list"] == [1, 2, 3]

    def test_tensor_valued_metadata_dropped_and_recorded(self, tmp_path):
        """Tensor fields that fail _validate_metadata are logged in dropped_fields."""
        d_model, n_prompts = 8, 5
        p = str(tmp_path / "ckpt.pt")
        payload = {
            "jacobian_sum": {0: torch.randn(d_model, d_model) * n_prompts},
            "n_prompts": n_prompts,
            "source_layers": [0],
            "d_model": d_model,
            "metadata": {
                "model_name": "gpt2",
                "embedding_stats": torch.randn(4),  # tensor — will be dropped
            },
        }
        torch.save(payload, p)
        lens = JacobianLens.load(p)
        assert "embedding_stats" not in lens.metadata
        dropped = lens.metadata.get("dropped_fields", [])
        assert any("embedding_stats" in f for f in dropped)
        # safe field survives
        assert lens.metadata.get("model_name") == "gpt2"

    def test_flat_provenance_harvested(self, tmp_path):
        """Checkpoints that store model_name / corpus at top level are handled."""
        p = str(tmp_path / "ckpt.pt")
        _save_checkpoint(
            p, flat_provenance={"model_name": "llama3", "corpus": "pile", "model_revision": "abc"}
        )
        lens = JacobianLens.load(p)
        assert lens.metadata.get("model_name") == "llama3"
        assert lens.metadata.get("corpus") == "pile"
        assert lens.metadata.get("model_revision") == "abc"

    def test_nested_metadata_overrides_flat_provenance(self, tmp_path):
        """Explicit metadata dict takes precedence over flat payload keys."""
        p = str(tmp_path / "ckpt.pt")
        _save_checkpoint(
            p,
            metadata={"model_name": "nested-name"},
            flat_provenance={"model_name": "flat-name"},
        )
        lens = JacobianLens.load(p)
        assert lens.metadata.get("model_name") == "nested-name"

    def test_zero_n_prompts_raises(self, tmp_path):
        p = str(tmp_path / "ckpt.pt")
        _save_checkpoint(p, n_prompts_override=0)
        with pytest.raises(ValueError, match="n_prompts=0"):
            JacobianLens.load(p)

    def test_negative_n_prompts_raises(self, tmp_path):
        p = str(tmp_path / "ckpt.pt")
        _save_checkpoint(p, n_prompts_override=-1)
        with pytest.raises(ValueError, match="n_prompts=-1"):
            JacobianLens.load(p)

    def test_single_layer_checkpoint(self, tmp_path):
        p = str(tmp_path / "ckpt.pt")
        _save_checkpoint(p, n_layers=1, d_model=4)
        lens = JacobianLens.load(p)
        assert lens.source_layers == [0]

    def test_load_then_save_roundtrip(self, tmp_path):
        """A checkpoint loaded and re-saved produces a valid artifact."""
        ckpt_path = str(tmp_path / "ckpt.pt")
        art_path = str(tmp_path / "artifact.pt")
        _save_checkpoint(ckpt_path, n_layers=2, d_model=8, n_prompts=3, metadata={"corpus": "wiki"})
        lens = JacobianLens.load(ckpt_path)
        lens.save(art_path)
        reloaded = JacobianLens.load(art_path)
        assert reloaded.n_prompts == 3
        assert reloaded.source_layers == [0, 1]
        assert reloaded.metadata.get("converted_from") == "jacobian_lens_checkpoint"


# ---------------------------------------------------------------------------
# merge() refuses to mix converted and TL-fitted lenses
# ---------------------------------------------------------------------------


class TestMergeProvenance:
    def test_merge_two_checkpoints_succeeds(self, tmp_path):
        p1, p2 = str(tmp_path / "c1.pt"), str(tmp_path / "c2.pt")
        _save_checkpoint(p1, metadata={"corpus": "wiki"})
        _save_checkpoint(p2, metadata={"corpus": "wiki"})
        l1 = JacobianLens.load(p1)
        l2 = JacobianLens.load(p2)
        merged = JacobianLens.merge([l1, l2])
        assert merged.metadata.get("converted_from") == "jacobian_lens_checkpoint"
        assert merged.n_prompts == l1.n_prompts + l2.n_prompts

    def test_merge_two_artifacts_succeeds(self, tmp_path):
        shared_meta = {"model_name": "gpt2", "corpus": "wiki", "transformer_lens_fit": True}
        p1, p2 = str(tmp_path / "a1.pt"), str(tmp_path / "a2.pt")
        _save_artifact(p1, metadata=dict(shared_meta))
        _save_artifact(p2, metadata=dict(shared_meta))
        l1 = JacobianLens.load(p1)
        l2 = JacobianLens.load(p2)
        # Both have transformer_lens_fit=True — provenance matches
        merged = JacobianLens.merge([l1, l2])
        assert merged.n_prompts == l1.n_prompts + l2.n_prompts

    def test_merge_checkpoint_and_artifact_raises(self, tmp_path):
        """A converted checkpoint and a TL-fitted artifact must not merge."""
        p_ckpt = str(tmp_path / "ckpt.pt")
        p_art = str(tmp_path / "art.pt")
        _save_checkpoint(p_ckpt, metadata={"corpus": "wiki"})
        _save_artifact(
            p_art,
            metadata={
                "transformer_lens_fit": True,
                "corpus": "wiki",
                "model_name": "gpt2",
            },
        )
        ckpt_lens = JacobianLens.load(p_ckpt)
        art_lens = JacobianLens.load(p_art)
        with pytest.raises(ValueError, match="provenance metadata"):
            JacobianLens.merge([ckpt_lens, art_lens])

    def test_merge_checkpoints_different_corpus_raises(self, tmp_path):
        p1, p2 = str(tmp_path / "c1.pt"), str(tmp_path / "c2.pt")
        _save_checkpoint(p1, metadata={"corpus": "wiki"})
        _save_checkpoint(p2, metadata={"corpus": "pile"})
        l1 = JacobianLens.load(p1)
        l2 = JacobianLens.load(p2)
        with pytest.raises(ValueError, match="provenance metadata"):
            JacobianLens.merge([l1, l2])

    def test_merge_weighted_average_correct(self, tmp_path):
        """Merged matrices are prompt-count-weighted averages."""
        d_model = 4
        p1, p2 = str(tmp_path / "c1.pt"), str(tmp_path / "c2.pt")

        n1, n2 = 3, 7
        # jacobian_sum = n * mean, so here mean = ones for both
        payload1 = {
            "jacobian_sum": {0: torch.ones(d_model, d_model) * n1},
            "n_prompts": n1,
            "d_model": d_model,
            "source_layers": [0],
        }
        payload2 = {
            "jacobian_sum": {0: torch.ones(d_model, d_model) * n2 * 2},
            "n_prompts": n2,
            "d_model": d_model,
            "source_layers": [0],
        }
        torch.save(payload1, p1)
        torch.save(payload2, p2)
        l1 = JacobianLens.load(p1)  # mean = ones
        l2 = JacobianLens.load(p2)  # mean = 2*ones
        merged = JacobianLens.merge([l1, l2])
        expected_mean = (n1 * 1.0 + n2 * 2.0) / (n1 + n2)
        assert torch.allclose(
            merged.jacobians[0],
            torch.full((d_model, d_model), expected_mean),
            atol=1e-5,
        )
