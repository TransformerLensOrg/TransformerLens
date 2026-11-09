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


# ---------------------------------------------------------------------------
# J++ artifacts: converted (J + provenance) and native (parameters.jacobians)
# ---------------------------------------------------------------------------


def _author_config(
    *,
    d_model=8,
    n_layers=3,
    n_prompts=10,
    lrp_mode="rlens",
    lens_type="jacobian",
    hf_model_name="Qwen/Qwen3.5-0.8B",
    relative_end_transport_layer=-1,
):
    """A ``LensConfig.to_dict()`` payload as written by the J++ Lens package."""
    return {
        "hf_model_name": hf_model_name,
        "checkpoint_name": "recipe_lens",
        "lens_type": lens_type,
        "checkpoint_every_n_prompts": 4,
        "creation_time": "2026-10-07_00-51-51",
        "artifacts_base_dir": "artifacts",
        "checkpoint_path": "artifacts/2026-10-07/recipe_lens",
        "source_layers": list(range(n_layers)),
        "relative_end_transport_layer": relative_end_transport_layer,
        "d_model": d_model,
        "num_prompts_trained_on": n_prompts,
        "next_prompt_idx": 0,
        "jacobian_rows_per_pass": 16,
        "max_seq_len": 128,
        "skip_first_n_positions": 16,
        "num_clusters": 0,
        "cluster_projection_dim": 0,
        "lrp_mode": lrp_mode,
    }


def _converted_provenance(config, **overrides):
    """The ``provenance`` dict Neuronpedia's convert-jpp-lens.py writes."""
    provenance = {
        "source": "neuronpedia fit, koayon/jpp_lens recipe, 96 WikiText prompts",
        "method": "jpp",
        "lrp_mode": config["lrp_mode"],
        "source_config": config,
        "capture_point": "block_output",
        "stream_reduce": "none",
        "stream_index": None,
    }
    provenance.update(overrides)
    return provenance


def _jacobians(n_layers=3, d_model=8):
    return {i: torch.randn(d_model, d_model) for i in range(n_layers)}


def _save_converted_artifact(path, jacobians, provenance, *, n_prompts=10, metadata=None):
    d_model = next(iter(jacobians.values())).shape[0]
    payload = {
        "J": {layer: matrix.half() for layer, matrix in jacobians.items()},
        "n_prompts": n_prompts,
        "source_layers": sorted(jacobians),
        "d_model": d_model,
        "provenance": provenance,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, path)


def _save_author_artifact(path, jacobians, config):
    torch.save(
        {
            "parameters": {
                "jacobians": {layer: matrix.half() for layer, matrix in jacobians.items()}
            },
            "source_layers": sorted(jacobians),
            "config": config,
        },
        path,
    )


class TestLoadConvertedProvenance:
    def test_provenance_preserved_in_metadata(self, tmp_path):
        p = str(tmp_path / "jpp.pt")
        provenance = _converted_provenance(_author_config())
        _save_converted_artifact(p, _jacobians(), provenance)
        lens = JacobianLens.load(p)
        assert lens.metadata["provenance"] == provenance

    def test_recorded_values_normalized_without_inventing_revision(self, tmp_path):
        p = str(tmp_path / "jpp.pt")
        _save_converted_artifact(p, _jacobians(), _converted_provenance(_author_config()))
        lens = JacobianLens.load(p)
        assert lens.metadata["model_name"] == "Qwen/Qwen3.5-0.8B"
        assert lens.metadata["method"] == "jpp"
        assert lens.metadata["lrp_mode"] == "rlens"
        assert lens.metadata["capture_point"] == "block_output"
        assert lens.metadata["stream_reduce"] == "none"
        assert lens.metadata["relative_end_transport_layer"] == -1
        assert "model_revision" not in lens.metadata
        assert "target_layer" not in lens.metadata
        assert lens.n_prompts == 10
        assert lens.d_model == 8

    def test_recorded_revision_and_target_layer_harvested(self, tmp_path):
        p = str(tmp_path / "rlens.pt")
        provenance = _converted_provenance(
            _author_config(), target_layer=1, model_revision="2fc0636"
        )
        _save_converted_artifact(p, _jacobians(), provenance)
        lens = JacobianLens.load(p)
        assert lens.metadata["target_layer"] == 1
        assert lens.metadata["model_revision"] == "2fc0636"

    def test_explicit_metadata_wins_over_provenance(self, tmp_path):
        p = str(tmp_path / "jpp.pt")
        provenance = _converted_provenance(_author_config())
        _save_converted_artifact(
            p, _jacobians(), provenance, metadata={"model_name": "explicit", "corpus": "wiki"}
        )
        lens = JacobianLens.load(p)
        assert lens.metadata["model_name"] == "explicit"
        assert lens.metadata["corpus"] == "wiki"
        assert lens.metadata["method"] == "jpp"
        assert lens.metadata["provenance"] == provenance

    def test_unsafe_provenance_leaves_dropped_and_recorded(self, tmp_path):
        p = str(tmp_path / "jpp.pt")
        provenance = _converted_provenance(_author_config(), fit_stats=torch.randn(3))
        _save_converted_artifact(p, _jacobians(), provenance)
        lens = JacobianLens.load(p)
        assert "fit_stats" not in lens.metadata["provenance"]
        assert any("fit_stats" in field for field in lens.metadata["dropped_fields"])
        assert lens.metadata["provenance"]["method"] == "jpp"
        assert lens.metadata["model_name"] == "Qwen/Qwen3.5-0.8B"

    def test_save_load_roundtrip_preserves_provenance(self, tmp_path):
        p = str(tmp_path / "jpp.pt")
        q = str(tmp_path / "resaved.pt")
        _save_converted_artifact(p, _jacobians(), _converted_provenance(_author_config()))
        lens = JacobianLens.load(p)
        lens.save(q)
        reloaded = JacobianLens.load(q)
        assert reloaded.metadata == lens.metadata
        assert reloaded.n_prompts == lens.n_prompts
        for layer in lens.source_layers:
            torch.testing.assert_close(reloaded.jacobians[layer], lens.jacobians[layer])

    def test_merge_refuses_jpp_maps(self, tmp_path):
        p1, p2 = str(tmp_path / "a.pt"), str(tmp_path / "b.pt")
        provenance = _converted_provenance(_author_config())
        _save_converted_artifact(p1, _jacobians(), provenance)
        _save_converted_artifact(p2, _jacobians(), provenance)
        with pytest.raises(ValueError, match="additive"):
            JacobianLens.merge([JacobianLens.load(p1), JacobianLens.load(p2)])

    def test_plain_artifact_without_provenance_unchanged(self, tmp_path):
        p = str(tmp_path / "plain.pt")
        _save_artifact(p, metadata={"model_name": "gpt2"})
        lens = JacobianLens.load(p)
        assert lens.metadata == {"model_name": "gpt2"}


class TestLoadAuthorSchema:
    def test_loads_same_jacobians_as_native_schema(self, tmp_path):
        jacobians = _jacobians()
        native, author = str(tmp_path / "native.pt"), str(tmp_path / "author.pt")
        torch.save(
            {
                "J": {layer: matrix.half() for layer, matrix in jacobians.items()},
                "n_prompts": 10,
                "source_layers": sorted(jacobians),
                "d_model": 8,
            },
            native,
        )
        _save_author_artifact(author, jacobians, _author_config())
        native_lens = JacobianLens.load(native)
        author_lens = JacobianLens.load(author)
        assert author_lens.source_layers == native_lens.source_layers
        assert author_lens.d_model == native_lens.d_model
        assert author_lens.n_prompts == native_lens.n_prompts
        for layer in native_lens.source_layers:
            torch.testing.assert_close(author_lens.jacobians[layer], native_lens.jacobians[layer])
            assert author_lens.jacobians[layer].dtype == torch.float32

    def test_config_normalized_into_metadata(self, tmp_path):
        p = str(tmp_path / "author.pt")
        config = _author_config()
        _save_author_artifact(p, _jacobians(), config)
        lens = JacobianLens.load(p)
        assert lens.metadata["converted_from"] == "jpp_lens_artifact"
        assert lens.metadata["provenance"]["source_config"] == config
        assert lens.metadata["model_name"] == "Qwen/Qwen3.5-0.8B"
        assert lens.metadata["lrp_mode"] == "rlens"
        assert lens.metadata["relative_end_transport_layer"] == -1
        assert "model_revision" not in lens.metadata
        assert "method" not in lens.metadata

    def test_j_with_config_variant(self, tmp_path):
        p = str(tmp_path / "j_config.pt")
        jacobians = _jacobians()
        torch.save(
            {
                "J": {layer: matrix.half() for layer, matrix in jacobians.items()},
                "config": _author_config(n_prompts=7),
            },
            p,
        )
        lens = JacobianLens.load(p)
        assert lens.d_model == 8
        assert lens.n_prompts == 7
        assert lens.metadata["model_name"] == "Qwen/Qwen3.5-0.8B"
        assert lens.metadata["converted_from"] == "jpp_lens_artifact"

    def test_d_model_inferred_when_config_omits_it(self, tmp_path):
        p = str(tmp_path / "author.pt")
        config = _author_config()
        del config["d_model"]
        _save_author_artifact(p, _jacobians(d_model=4), config)
        assert JacobianLens.load(p).d_model == 4

    def test_rejects_non_jacobian_lens_type(self, tmp_path):
        p = str(tmp_path / "logit.pt")
        torch.save(
            {
                "parameters": {},
                "source_layers": [0, 1],
                "config": _author_config(lens_type="logit"),
            },
            p,
        )
        with pytest.raises(ValueError, match="lens_type='logit'"):
            JacobianLens.load(p)

    def test_rejects_parameters_without_jacobians(self, tmp_path):
        p = str(tmp_path / "no_jac.pt")
        torch.save(
            {"parameters": {"other": {0: 1.0}}, "source_layers": [0], "config": _author_config()},
            p,
        )
        with pytest.raises(ValueError, match="parameters.jacobians"):
            JacobianLens.load(p)

    def test_save_load_roundtrip(self, tmp_path):
        p, q = str(tmp_path / "author.pt"), str(tmp_path / "resaved.pt")
        _save_author_artifact(p, _jacobians(), _author_config())
        lens = JacobianLens.load(p)
        lens.save(q)
        reloaded = JacobianLens.load(q)
        assert reloaded.metadata == lens.metadata
        for layer in lens.source_layers:
            torch.testing.assert_close(reloaded.jacobians[layer], lens.jacobians[layer])

    def test_merge_refuses_imported_author_artifacts(self, tmp_path):
        p1, p2 = str(tmp_path / "a.pt"), str(tmp_path / "b.pt")
        _save_author_artifact(p1, _jacobians(), _author_config())
        _save_author_artifact(p2, _jacobians(), _author_config())
        with pytest.raises(ValueError, match="additive"):
            JacobianLens.merge([JacobianLens.load(p1), JacobianLens.load(p2)])
