"""Tests for `benchmark_gated_hooks_fire`.

The benchmark's job: exercise every cfg-gated attention hook on every
verified architecture and fail if the hook was registered but never fired.
This replaces the prior approach of silently filtering cfg-gated hooks from
the registration benchmark (which swallowed real regressions).

These tests cover the function's contract; the regression signal on real
adapters comes from `verify_models` runs, which invoke this benchmark on
every supported architecture.
"""
from __future__ import annotations

import pytest

from transformer_lens.benchmarks import benchmark_gated_hooks_fire
from transformer_lens.benchmarks.utils import BenchmarkSeverity
from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="module")
def gpt2_bridge():
    return TransformerBridge.boot_transformers("openai-community/gpt2", device="cpu")


def test_gated_hooks_fire_on_supported_architecture(gpt2_bridge):
    """JointQKVAttentionBridge supports all three flags; every hook stem fires."""
    result = benchmark_gated_hooks_fire(gpt2_bridge, "The quick brown fox")
    assert result.passed, f"Gated-hooks benchmark failed: {result.message} / {result.details}"
    assert result.severity == BenchmarkSeverity.INFO
    assert result.details is not None
    fired_counts = result.details["fired_counts"]
    for stem in ("hook_result", "hook_q_input", "hook_k_input", "hook_v_input", "hook_attn_in"):
        assert fired_counts.get(stem, 0) > 0, (
            f"stem {stem!r} did not fire on any layer. This is exactly the "
            "regression the benchmark was added to catch."
        )
    assert result.details["skipped"] == []
    assert set(result.details["tested_flags"]) == {
        "use_attn_result",
        "use_split_qkv_input",
        "use_attn_in",
    }


def test_gated_hooks_benchmark_restores_default_flags(gpt2_bridge):
    """Benchmark toggles flags internally but must restore them to False on exit."""
    gpt2_bridge.cfg.use_attn_result = False
    gpt2_bridge.cfg.use_attn_in = False
    gpt2_bridge.cfg.use_split_qkv_input = False
    benchmark_gated_hooks_fire(gpt2_bridge, "hi")
    assert gpt2_bridge.cfg.use_attn_result is False
    assert gpt2_bridge.cfg.use_attn_in is False
    assert gpt2_bridge.cfg.use_split_qkv_input is False


def test_gated_hooks_skip_on_unsupported_architecture(monkeypatch, gpt2_bridge):
    """Arch with no fork-capable attention: flags skip, benchmark returns INFO, not FAIL."""
    from torch import nn

    class _FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Identity()

    fake_blocks = nn.ModuleList([_FakeBlock()])
    monkeypatch.setattr(gpt2_bridge, "blocks", fake_blocks, raising=True)
    result = benchmark_gated_hooks_fire(gpt2_bridge, "hi")
    assert result.passed, f"Unexpected failure: {result.message}"
    assert result.severity == BenchmarkSeverity.INFO
    assert result.details is not None
    skipped_flags = [s[0] for s in result.details["skipped"]]
    assert set(skipped_flags) == {"use_attn_result", "use_split_qkv_input", "use_attn_in"}


def test_gated_hooks_benchmark_fails_when_hook_registered_but_never_fires(gpt2_bridge, monkeypatch):
    """Regression signal: if a hook is in hook_dict but forward never calls it,
    the benchmark reports a DANGER failure (not INFO).

    Patches the attention bridge's `hook_result` attribute to a passthrough
    callable so the forward path still runs but `hook_result.add_hook(...)`
    won't fire (the HookPoint in `hook_dict` is the *original*, which never
    gets invoked). This is exactly the regression the prior filter-based
    approach would have hidden.
    """
    # Replace hook_result on each attn bridge with a passthrough that bypasses
    # the original HookPoint. hook_dict still holds the *original* HookPoint,
    # so the benchmark registers its capture hook on an object that forward
    # no longer calls — the activation dict stays empty for hook_result.
    from transformer_lens.hook_points import HookPoint

    originals: list[tuple[object, HookPoint]] = []
    try:
        for b in gpt2_bridge.blocks:
            attn = b.attn
            originals.append((attn, attn.hook_result))

            def _noop(t, *a, **kw):
                return t

            object.__setattr__(attn, "hook_result", _noop)

        result = benchmark_gated_hooks_fire(gpt2_bridge, "hi")
    finally:
        for attn, orig in originals:
            object.__setattr__(attn, "hook_result", orig)

    assert not result.passed, (
        "Benchmark should have failed: hook_result was registered in "
        "hook_dict but the forward bypassed it."
    )
    assert result.severity == BenchmarkSeverity.DANGER
    assert result.details is not None
    failed = result.details["failed"]
    assert any(
        flag == "use_attn_result" and stem == "hook_result" for flag, stem in failed
    ), f"Expected (use_attn_result, hook_result) in failure list; got {failed}"
