"""_clear_hf_cache must never delete the pinned Phase-4 judge: the sweep clears
the HF cache after every model family, and re-downloading the judge each time
defeats the batch preload."""

import pytest

pytest.importorskip("transformers")


def test_clear_hf_cache_preserves_judge_snapshot(tmp_path, monkeypatch):
    from pathlib import Path

    from transformer_lens.benchmarks.text_quality import JUDGE_MODEL_ID
    from transformer_lens.tools.model_registry import verify_models

    hub = tmp_path / ".cache" / "huggingface" / "hub"
    judge_dir = hub / ("models--" + JUDGE_MODEL_ID.replace("/", "--")) / "blobs"
    other_dir = hub / "models--someone--other-model" / "blobs"
    judge_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    judge_blob = judge_dir / "aaaa"
    other_blob = other_dir / "bbbb"
    judge_blob.write_bytes(b"judge-weights")
    other_blob.write_bytes(b"other-weights")

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    verify_models._clear_hf_cache(quiet=True)

    assert judge_blob.exists(), "judge blob must survive the per-family cache clear"
    assert not other_blob.exists(), "non-judge blobs must still be cleared"
