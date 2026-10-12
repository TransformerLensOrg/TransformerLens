"""boot()'s code_revision plumbing: config load, model kwargs, concrete-class pop."""

from unittest.mock import patch

import pytest
from transformers import GPTNeoXForCausalLM


class _AbortBoot(Exception):
    """Raised by the model-load patch to short-circuit ``boot()`` before any real load."""


def _patched_boot(**boot_kwargs):
    """Capture AutoConfig and model-load kwargs, aborting before any real model load."""
    from transformer_lens.model_bridge.sources import transformers as bridge_src

    captured: dict = {}
    real_autoconfig = bridge_src.AutoConfig.from_pretrained

    def capture_autoconfig(name, **kwargs):
        captured["autoconfig_kwargs"] = dict(kwargs)
        # Strip the fake pins so the real call hits the local cache.
        kwargs.pop("revision", None)
        kwargs.pop("code_revision", None)
        return real_autoconfig(name, **kwargs)

    def capture_model_load(*args, **kwargs):
        captured["model_kwargs"] = kwargs
        raise _AbortBoot()

    with patch.object(
        bridge_src.AutoConfig, "from_pretrained", side_effect=capture_autoconfig
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=capture_model_load,
    ), patch.object(
        GPTNeoXForCausalLM, "from_pretrained", side_effect=capture_model_load
    ):
        with pytest.raises(_AbortBoot):
            bridge_src.boot(model_name="EleutherAI/pythia-70m", device="cpu", **boot_kwargs)

    return captured


def test_code_revision_forwarded_to_autoconfig_and_model_load():
    captured = _patched_boot(revision="step3000", code_revision="deadbeef")
    assert captured["autoconfig_kwargs"].get("code_revision") == "deadbeef"
    # code_revision rides alongside revision, never in place of it.
    assert captured["autoconfig_kwargs"].get("revision") == "step3000"
    assert captured["model_kwargs"].get("code_revision") == "deadbeef"
    assert captured["model_kwargs"].get("revision") == "step3000"


def test_default_code_revision_is_absent_from_model_kwargs():
    captured = _patched_boot(revision="step3000")
    assert "code_revision" not in captured["model_kwargs"]


def test_code_revision_popped_for_concrete_model_class():
    # A concrete PreTrainedModel subclass rejects code_revision with a TypeError;
    # boot must strip it after prepare_loading consumed it.
    captured = _patched_boot(code_revision="deadbeef", model_class=GPTNeoXForCausalLM)
    assert "code_revision" not in captured["model_kwargs"]
