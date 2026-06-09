"""Unit tests for SGLang version parsing + the ``assert_sglang_supported`` gate."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import pytest

from transformer_lens.model_bridge.sources.sglang import internals


class TestParseVersion:
    def test_simple_triplet(self):
        assert internals._parse_version("0.5.12") == (0, 5, 12)

    def test_post_suffix_drops_to_triplet(self):
        # ``0.5.12.post1`` should compare equal-or-greater than (0,5,12).
        assert internals._parse_version("0.5.12.post1") == (0, 5, 12)

    def test_rc_suffix_drops_chunk_at_nondigit(self):
        # ``0.5.12rc1`` → (0, 5, 12); rc-suffix doesn't lower the patch.
        assert internals._parse_version("0.5.12rc1") == (0, 5, 12)

    def test_short_version_pads_with_zeros(self):
        assert internals._parse_version("0.5") == (0, 5, 0)

    def test_patch_level_compare_catches_pre_forward_hooks(self):
        """Old versions in the 0.5 line predate ``forward_hooks`` (landed at .12)."""
        assert internals._parse_version("0.5.11") < internals._parse_version("0.5.12")
        assert internals._parse_version("0.5.0") < internals._parse_version("0.5.12")


class TestAssertSglangSupported:
    def test_missing_raises(self, monkeypatch):
        def _missing(_):
            raise PackageNotFoundError("sglang")

        monkeypatch.setattr("importlib.metadata.version", _missing)
        with pytest.raises(RuntimeError, match="sglang is not installed"):
            internals.assert_sglang_supported()

    def test_too_old_raises(self, monkeypatch):
        monkeypatch.setattr("importlib.metadata.version", lambda _: "0.5.11")
        with pytest.raises(RuntimeError, match="sglang>=0.5.12 required"):
            internals.assert_sglang_supported()

    def test_too_old_at_minor_split(self, monkeypatch):
        monkeypatch.setattr("importlib.metadata.version", lambda _: "0.4.99")
        with pytest.raises(RuntimeError, match="sglang>=0.5.12 required"):
            internals.assert_sglang_supported()

    def test_pinned_version_passes(self, monkeypatch):
        monkeypatch.setattr("importlib.metadata.version", lambda _: "0.5.12.post1")
        internals.assert_sglang_supported()  # no raise

    def test_newer_passes(self, monkeypatch):
        monkeypatch.setattr("importlib.metadata.version", lambda _: "0.6.0")
        internals.assert_sglang_supported()  # no raise
