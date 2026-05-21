"""Tests for the HuggingFace Hub 429 retry helper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import pytest

from transformer_lens.utilities import hf_utils
from transformer_lens.utilities.hf_utils import call_hf_with_retry


class _FakeHTTPError(Exception):
    """Stand-in for HfHubHTTPError / requests.HTTPError — exposes .response.status_code."""

    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        headers: dict[str, str] = {}
        if retry_after is not None:
            headers["Retry-After"] = retry_after
        self.response = SimpleNamespace(status_code=status_code, headers=headers)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    """Capture sleep calls and don't actually sleep — keeps tests fast."""
    waits: List[float] = []
    monkeypatch.setattr(hf_utils.time, "sleep", lambda s: waits.append(s))
    return waits


@pytest.fixture
def _deterministic_random(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force random.random() == 0.5 so jitter factor (0.8 + 0.4*r) == 1.0 exactly.

    Lets backoff tests assert exact values instead of ranges, without coupling
    to the specific jitter window.
    """
    monkeypatch.setattr(hf_utils.random, "random", lambda: 0.5)


def _make_flaky(fail_times: int, exc_factory: Any) -> Any:
    """Build a callable that raises `fail_times` then returns 'ok'."""
    state = {"calls": 0}

    def _inner(*args: Any, **kwargs: Any) -> str:
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise exc_factory()
        return "ok"

    _inner.state = state  # type: ignore[attr-defined]
    return _inner


class TestCallHfWithRetry:
    def test_returns_immediately_on_success(self) -> None:
        func = _make_flaky(0, lambda: _FakeHTTPError(429))
        assert call_hf_with_retry(func) == "ok"
        assert func.state["calls"] == 1

    def test_retries_on_429_then_succeeds(self, _no_sleep: List[float]) -> None:
        func = _make_flaky(2, lambda: _FakeHTTPError(429))
        assert call_hf_with_retry(func, max_attempts=3, base_delay=1.0) == "ok"
        assert func.state["calls"] == 3
        assert len(_no_sleep) == 2

    def test_raises_after_max_attempts(self, _no_sleep: List[float]) -> None:
        func = _make_flaky(99, lambda: _FakeHTTPError(429))
        with pytest.raises(_FakeHTTPError):
            call_hf_with_retry(func, max_attempts=3, base_delay=1.0)
        assert func.state["calls"] == 3
        # Sleeps happen between attempts, not after the final one.
        assert len(_no_sleep) == 2

    def test_non_429_propagates_immediately(self, _no_sleep: List[float]) -> None:
        func = _make_flaky(99, lambda: _FakeHTTPError(503))
        with pytest.raises(_FakeHTTPError):
            call_hf_with_retry(func, max_attempts=3, base_delay=1.0)
        assert func.state["calls"] == 1
        assert _no_sleep == []

    def test_non_http_exception_propagates_immediately(self, _no_sleep: List[float]) -> None:
        def boom() -> None:
            raise ValueError("not a network error")

        with pytest.raises(ValueError):
            call_hf_with_retry(boom, max_attempts=3, base_delay=1.0)
        assert _no_sleep == []

    def test_honors_retry_after_header(self, _no_sleep: List[float]) -> None:
        func = _make_flaky(1, lambda: _FakeHTTPError(429, retry_after="7.5"))
        assert call_hf_with_retry(func, max_attempts=3, base_delay=1.0) == "ok"
        assert func.state["calls"] == 2
        assert _no_sleep == [7.5]

    def test_falls_back_to_backoff_when_retry_after_unparseable(
        self, _no_sleep: List[float], _deterministic_random: None
    ) -> None:
        func = _make_flaky(1, lambda: _FakeHTTPError(429, retry_after="soon"))
        call_hf_with_retry(func, max_attempts=3, base_delay=10.0)
        # base_delay * 2**0 * jitter_factor(0.5) = 10 * 1 * 1.0 = 10.0 exactly
        assert _no_sleep == [10.0]

    def test_exponential_backoff_grows(
        self, _no_sleep: List[float], _deterministic_random: None
    ) -> None:
        func = _make_flaky(3, lambda: _FakeHTTPError(429))
        with pytest.raises(_FakeHTTPError):
            call_hf_with_retry(func, max_attempts=3, base_delay=10.0)
        # Two backoffs between three attempts; last attempt has no sleep.
        # attempt 0: 10 * 2**0 * 1.0 = 10; attempt 1: 10 * 2**1 * 1.0 = 20.
        assert _no_sleep == [10.0, 20.0]

    def test_backoff_capped_at_max_delay(
        self, _no_sleep: List[float], _deterministic_random: None
    ) -> None:
        """A huge base_delay must be clamped by _HF_RETRY_MAX_DELAY_SECONDS."""
        func = _make_flaky(1, lambda: _FakeHTTPError(429))
        call_hf_with_retry(func, max_attempts=2, base_delay=10_000.0)
        # Without cap: 10000 * 2**0 * 1.0 = 10000s. With 120s cap: exactly 120.0.
        assert _no_sleep == [hf_utils._HF_RETRY_MAX_DELAY_SECONDS]


class TestEnableHfRetry:
    """Verify the global Auto*.from_pretrained wrapper installed by enable_hf_retry."""

    def test_session_fixture_wraps_autoconfig(self) -> None:
        """tests/conftest.py:_enable_hf_retry_for_tests must have wrapped AutoConfig."""
        from transformers import AutoConfig

        assert getattr(
            AutoConfig.from_pretrained, hf_utils._TL_RETRY_WRAPPED_ATTR, False
        ), "enable_hf_retry was not applied to AutoConfig — check conftest fixture"

    def test_session_fixture_wraps_autotokenizer(self) -> None:
        from transformers import AutoTokenizer

        assert getattr(
            AutoTokenizer.from_pretrained, hf_utils._TL_RETRY_WRAPPED_ATTR, False
        )

    def test_idempotent(self) -> None:
        """A second enable_hf_retry call must not re-wrap (or otherwise break) the classes."""
        from transformers import AutoConfig

        before = AutoConfig.from_pretrained.__func__
        hf_utils.enable_hf_retry()
        after = AutoConfig.from_pretrained.__func__
        assert before is after


class TestDownloadFileFromHf:
    """End-to-end coverage: download_file_from_hf must actually use the retry helper.

    Without this, a refactor that calls hf_hub_download directly again — exactly the
    regression this change is meant to prevent — would slip past the unit tests above.
    """

    def test_retries_underlying_hf_hub_download_on_429(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _no_sleep: List[float],
        tmp_path: Any,
    ) -> None:
        fake_file = tmp_path / "data.json"
        fake_file.write_text('{"ok": true}')
        state = {"calls": 0}

        def fake_hub_download(**kwargs: Any) -> str:
            state["calls"] += 1
            if state["calls"] < 2:
                raise _FakeHTTPError(429)
            return str(fake_file)

        monkeypatch.setattr(hf_utils, "hf_hub_download", fake_hub_download)

        result = hf_utils.download_file_from_hf("any/repo", "data.json")

        assert result == {"ok": True}
        assert state["calls"] == 2
        assert len(_no_sleep) == 1
