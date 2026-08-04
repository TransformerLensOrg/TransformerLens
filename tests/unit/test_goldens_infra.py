"""Golden-fixture infrastructure: loader resolution + capture-script pure helpers.

Network-free and model-free: exercises path resolution, artifact round-trips,
and the deterministic checksum/sampling helpers on synthetic tensors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import capture_ht_goldens as capture  # noqa: E402

from tests import goldens  # noqa: E402


@pytest.fixture()
def synthetic_goldens(tmp_path, monkeypatch):
    """A minimal on-disk golden cell for model 'org/tiny' config 'no_processing'."""
    from safetensors.torch import save_file

    cell = tmp_path / "org__tiny" / "no_processing"
    cell.mkdir(parents=True)
    (cell / "provenance.json").write_text(json.dumps({"schema_version": 1, "model": "org/tiny"}))
    (cell / "scalars.json").write_text(json.dumps({"long_text_ce_loss": 3.5}))
    save_file({"W_E": torch.ones(4, 2)}, str(cell / "views.safetensors"))

    monkeypatch.setenv("TL_GOLDENS_DIR", str(tmp_path))
    goldens.resolve_goldens_dir.cache_clear()
    yield tmp_path
    goldens.resolve_goldens_dir.cache_clear()


class TestLoaderResolution:
    def test_env_dir_resolves_and_cell_loads(self, synthetic_goldens):
        assert goldens.goldens_available()
        assert goldens.goldens_available("org/tiny", "no_processing")
        views = goldens.load_golden_tensors("org/tiny", "no_processing", "views")
        assert torch.equal(views["W_E"], torch.ones(4, 2))
        assert goldens.load_golden_json("org/tiny", "no_processing", "scalars")[
            "long_text_ce_loss"
        ] == pytest.approx(3.5)

    def test_missing_cell_raises_with_guidance(self, synthetic_goldens):
        with pytest.raises(FileNotFoundError, match="provenance.json"):
            goldens.golden_path("org/tiny", "full_defaults")
        assert not goldens.goldens_available("org/tiny", "full_defaults")

    def test_unset_env_and_unreachable_hub_degrade_to_unavailable(self, monkeypatch):
        monkeypatch.setenv("TL_GOLDENS_DIR", "/nonexistent/path")
        goldens.resolve_goldens_dir.cache_clear()
        assert goldens.resolve_goldens_dir() is None
        assert not goldens.goldens_available()
        goldens.resolve_goldens_dir.cache_clear()


class TestCaptureHelpers:
    def test_checksum_is_deterministic_and_value_sensitive(self):
        t = torch.arange(12.0).reshape(3, 4)
        assert capture._tensor_checksum(t) == capture._tensor_checksum(t.clone())
        t2 = t.clone()
        t2[0, 0] += 1e-3
        assert capture._tensor_checksum(t) != capture._tensor_checksum(t2)

    def test_seeded_sample_deterministic_and_bounded(self):
        big = torch.randn(10_000, generator=torch.Generator().manual_seed(0))
        s1 = capture._seeded_sample(big, capture.SAMPLE_SEED)
        s2 = capture._seeded_sample(big, capture.SAMPLE_SEED)
        assert torch.equal(s1, s2)
        assert s1.numel() == capture.SAMPLE_COUNT
        small = torch.randn(10)
        assert capture._seeded_sample(small, capture.SAMPLE_SEED).numel() == 10

    def test_config_matrix_shape(self):
        # refactor_factored is gpt2-only; every other config applies to all models.
        assert set(capture.CONFIGS["refactor_factored"]["models"]) == {"gpt2"}
        full = [c for c, v in capture.CONFIGS.items() if v["full_state_dict"]]
        assert sorted(full) == ["full_defaults", "no_processing"]

    def test_model_dir_name_roundtrip_safe(self):
        assert capture._model_dir_name("EleutherAI/pythia-14m") == "EleutherAI__pythia-14m"
        assert goldens._model_dir_name("EleutherAI/pythia-14m") == "EleutherAI__pythia-14m"
