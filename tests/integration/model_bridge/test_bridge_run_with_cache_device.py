"""Regression coverage for `TransformerBridge.run_with_cache(device=...)`.

The `device=` kwarg is a cache-offload knob: cached activations are stored on
that device, but the underlying model and inputs must stay where the caller
put them, matching `ActivationCache.to` and the legacy `get_caching_hooks`
("device to store on") contract.
"""

from unittest.mock import patch

import pytest


@pytest.fixture()
def bridge(distilgpt2_bridge):
    """Alias the session fixture for concise test signatures."""
    return distilgpt2_bridge


def test_run_with_cache_device_does_not_move_model(bridge):
    """`run_with_cache(device=...)` must not relocate the underlying model.

    CPU runners cannot reproduce the original cross-device crash directly
    (`to('cpu')` is a no-op there), so we spy on `original_model.to` with
    `Mock(wraps=...)` and assert it isn't invoked during the call. That
    catches the regression on any platform.
    """
    with patch.object(bridge.original_model, "to", wraps=bridge.original_model.to) as to_spy:
        _, cache = bridge.run_with_cache(bridge.to_tokens("hello"), device="cpu")

    assert to_spy.call_count == 0, (
        f"run_with_cache(device=...) moved the underlying model "
        f"({to_spy.call_count} call(s): {to_spy.call_args_list})."
    )
    # And the cache itself still lands on the requested device.
    assert next(iter(cache.values())).device.type == "cpu"
