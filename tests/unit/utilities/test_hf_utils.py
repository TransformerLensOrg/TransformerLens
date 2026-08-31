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

        assert getattr(AutoTokenizer.from_pretrained, hf_utils._TL_RETRY_WRAPPED_ATTR, False)

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


class TestRemotePostInitCompat:
    """4.x-era remote configs define argless __post_init__, which >=5.x calls
    with the class's own fields as kwargs — crashing, and (even if not) never
    setting those fields, since 4.x classes relied on the base to setattr them.
    """

    @staticmethod
    def _legacy_class():
        from transformers.configuration_utils import PretrainedConfig

        class _LegacyRemoteConfig(PretrainedConfig):
            model_type = "tl-test-legacy"

            def __post_init__(self) -> None:
                # 4.x style: derives from fields the base was expected to set.
                self.derived = getattr(self, "model_dim", 0) * 2

        return _LegacyRemoteConfig

    def test_legacy_class_crashes_without_the_shim(self) -> None:
        """The premise: if this stops failing, transformers changed the
        contract and the shim should be retired."""
        cls = self._legacy_class()
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            cls(model_dim=8, use_cache=True)

    def test_tolerant_wrapper_sets_fields_then_derives(self) -> None:
        from transformer_lens.utilities.hf_utils import make_post_init_kwarg_tolerant

        cls = self._legacy_class()
        make_post_init_kwarg_tolerant(cls)
        config = cls(model_dim=8, use_cache=True)
        # Base-first ordering is the load-bearing part: the original body must
        # see model_dim already set, or every derived field silently zeroes.
        assert config.model_dim == 8
        assert config.derived == 16

    def test_wrapper_is_idempotent(self) -> None:
        from transformer_lens.utilities.hf_utils import make_post_init_kwarg_tolerant

        cls = self._legacy_class()
        make_post_init_kwarg_tolerant(cls)
        first = cls.__post_init__
        make_post_init_kwarg_tolerant(cls)
        assert cls.__post_init__ is first


class TestNumericTowerCompat:
    """huggingface_hub's strict dataclasses type-check with a bare isinstance, so
    an int in a float field is rejected where PEP 484 accepts it. Real configs
    write integral values -- ai-sage/GigaChat3-10B-A1.8B ships
    ``"routed_scaling_factor": 1`` -- and without the shim they cannot load.
    """

    @staticmethod
    def _strict_class():
        from dataclasses import make_dataclass

        from huggingface_hub.dataclasses import strict

        # make_dataclass, not a class body: this module postpones annotations, and
        # hub's validator skips any field whose type arrives as a string.
        return strict(make_dataclass("_Scaled", [("scale", float), ("enabled", int, 0)]))

    @pytest.fixture
    def unpatched(self):
        """Hub's own validator, restored for the duration of the test.

        transformer_lens installs the shim at import, so the premise can only be
        shown by putting the original back.
        """
        from huggingface_hub import dataclasses as hub_dataclasses

        installed = hub_dataclasses._validate_simple_type
        original = getattr(installed, "__wrapped__", installed)
        hub_dataclasses._validate_simple_type = original
        yield
        hub_dataclasses._validate_simple_type = installed

    def test_int_in_float_field_is_rejected_without_the_shim(self, unpatched) -> None:
        """The premise: if this stops failing, huggingface_hub fixed it upstream
        and the shim should be retired."""
        from huggingface_hub.errors import StrictDataclassFieldValidationError

        with pytest.raises(StrictDataclassFieldValidationError, match="expected float, got int"):
            self._strict_class()(scale=1)

    def test_int_in_float_field_is_accepted(self) -> None:
        from transformer_lens.utilities.hf_utils import enable_hf_numeric_tower

        enable_hf_numeric_tower()
        assert self._strict_class()(scale=1).scale == 1

    def test_float_fields_still_reject_non_numbers(self) -> None:
        from huggingface_hub.errors import StrictDataclassFieldValidationError
        from transformer_lens.utilities.hf_utils import enable_hf_numeric_tower

        enable_hf_numeric_tower()
        with pytest.raises(StrictDataclassFieldValidationError):
            self._strict_class()(scale="1.0")

    def test_bool_is_not_widened_into_an_int_field(self) -> None:
        """bool is an int subclass; widening floats must not loosen that check."""
        from huggingface_hub.errors import StrictDataclassFieldValidationError
        from transformer_lens.utilities.hf_utils import enable_hf_numeric_tower

        enable_hf_numeric_tower()
        with pytest.raises(StrictDataclassFieldValidationError):
            self._strict_class()(scale=1.0, enabled=True)

    def test_is_idempotent(self) -> None:
        from huggingface_hub import dataclasses as hub_dataclasses
        from transformer_lens.utilities.hf_utils import enable_hf_numeric_tower

        enable_hf_numeric_tower()
        first = hub_dataclasses._validate_simple_type
        enable_hf_numeric_tower()
        assert hub_dataclasses._validate_simple_type is first


class TestOutputAttentionsCompat:
    """A repo that pins flash_attention_2 in its config cannot also be asked for
    output_attentions (zstanjj/HTML-Pruner-Phi-3.8B). The bridge needs attention
    outputs and loads the model eager anyway, so the config request should follow.
    """

    @staticmethod
    def _raiser(message, validator="validate_output_attentions", sentinel="config"):
        """An AutoConfig double that fails until asked for eager attention."""
        from huggingface_hub.errors import StrictDataclassClassValidationError

        calls = []

        class _AutoConfig:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                calls.append(kwargs)
                if kwargs.get("attn_implementation") == "eager":
                    return sentinel
                raise StrictDataclassClassValidationError(
                    validator=validator, cause=ValueError(message)
                )

        return _AutoConfig, calls

    def test_retries_with_eager_attention(self) -> None:
        from transformer_lens.utilities.hf_utils import (
            autoconfig_with_remote_post_init_compat,
        )

        auto_config, calls = self._raiser(
            "The `output_attentions` attribute is not supported when using the "
            "`attn_implementation` set to flash_attention_2."
        )
        result = autoconfig_with_remote_post_init_compat(
            "repo/pinned-fa2", auto_config=auto_config, output_attentions=True
        )
        assert result == "config"
        assert calls[0] == {"output_attentions": True}
        assert calls[1]["attn_implementation"] == "eager"

    def test_unrelated_strict_error_propagates(self) -> None:
        from huggingface_hub.errors import StrictDataclassError
        from transformer_lens.utilities.hf_utils import (
            autoconfig_with_remote_post_init_compat,
        )

        auto_config, _ = self._raiser(
            "some unrelated validator complaint", validator="validate_rope_parameters"
        )
        with pytest.raises(StrictDataclassError):
            autoconfig_with_remote_post_init_compat(
                "repo/other", auto_config=auto_config, output_attentions=True
            )

    def test_caller_choice_of_implementation_is_not_overridden(self) -> None:
        """An explicit attn_implementation means the caller has already decided."""
        from huggingface_hub.errors import StrictDataclassError
        from transformer_lens.utilities.hf_utils import (
            autoconfig_with_remote_post_init_compat,
        )

        auto_config, _ = self._raiser("output_attentions is not supported with fa2")
        with pytest.raises(StrictDataclassError):
            autoconfig_with_remote_post_init_compat(
                "repo/pinned-fa2",
                auto_config=auto_config,
                output_attentions=True,
                attn_implementation="sdpa",
            )


class TestSpecialTokenCompat:
    """Some repos declare special tokens their own tokenizer.json already contains
    (catherinearnett/B-GPT lists ~1200 of them), and re-adding raises TypeError.
    The serialized fast tokenizer is intact, so rebuild from it.
    """

    @pytest.fixture
    def repo_files(self, tmp_path, monkeypatch):
        """A tokenizer.json + config on disk, standing in for a hub repo."""
        import json as json_mod

        from tokenizers import Tokenizer, models, pre_tokenizers

        vocab = {"[CLS]": 0, "[SEP]": 1, "<pad>": 2, "hello": 3, "world": 4}
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[CLS]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer_file = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_file))
        config_file = tmp_path / "tokenizer_config.json"
        config_file.write_text(
            json_mod.dumps(
                {
                    "bos_token": "[CLS]",
                    "eos_token": {"content": "[SEP]", "special": True},
                    "pad_token": "<pad>",
                    "model_max_length": 512,
                }
            )
        )
        monkeypatch.setattr(
            hf_utils,
            "hf_hub_download",
            lambda repo, filename, **kw: str(tmp_path / filename),
        )
        return tmp_path

    @staticmethod
    def _rejecting_tokenizer(message):
        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                raise TypeError(message)

        return _AutoTokenizer

    def test_rebuilds_from_tokenizer_json(self, repo_files) -> None:
        from transformer_lens.utilities.hf_utils import (
            autotokenizer_with_special_token_compat,
        )

        auto = self._rejecting_tokenizer(
            "argument 'special_tokens': Expected Union[Tuple[str, int], Tuple[int, str], dict]"
        )
        tokenizer = autotokenizer_with_special_token_compat("repo/b-gpt", auto_tokenizer=auto)
        assert tokenizer("hello world")["input_ids"] == [3, 4]
        # Roles from tokenizer_config must survive the rebuild, dict form included.
        assert tokenizer.bos_token == "[CLS]"
        assert tokenizer.eos_token == "[SEP]"
        assert tokenizer.pad_token == "<pad>"

    def test_unrelated_type_error_propagates(self, repo_files) -> None:
        from transformer_lens.utilities.hf_utils import (
            autotokenizer_with_special_token_compat,
        )

        auto = self._rejecting_tokenizer("unexpected keyword argument 'foo'")
        with pytest.raises(TypeError, match="foo"):
            autotokenizer_with_special_token_compat("repo/other", auto_tokenizer=auto)

    def test_repo_without_tokenizer_json_reraises(self, monkeypatch) -> None:
        """Nothing to rebuild from: the original error must stand."""
        from transformer_lens.utilities.hf_utils import (
            autotokenizer_with_special_token_compat,
        )

        def _missing(*args, **kwargs):
            raise OSError("404")

        monkeypatch.setattr(hf_utils, "hf_hub_download", _missing)
        auto = self._rejecting_tokenizer("argument 'special_tokens': Expected Union[...]")
        with pytest.raises(TypeError, match="special_tokens"):
            autotokenizer_with_special_token_compat("repo/no-json", auto_tokenizer=auto)
