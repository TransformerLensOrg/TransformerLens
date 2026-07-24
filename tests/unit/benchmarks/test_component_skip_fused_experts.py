"""The component-isolation benchmark must skip un-isolatable fused experts.

transformers 5.x routes MoE experts (Qwen2/Qwen3.5) through a fused
grouped-matmul whose forward requires the router's ``top_k_index`` /
``top_k_weights``. The isolated harness only has a single hidden-state probe,
so it cannot call these standalone — and ``forward_pass_logits`` already covers
their parity. The skip is arity-driven (any mlp component whose bound forward
needs more than ``hidden_states``), so it must fire for fused experts yet leave
a plain MLP tested. The prior guard inspected a nonexistent ``hf_component``
attribute and was permanently dead, which surfaced as a P1 regression the day
transformers changed the experts signature.
"""

import pytest

pytest.importorskip("transformers")


def _report(model_id: str):
    from transformer_lens.benchmarks.component_outputs import benchmark_model

    try:
        return benchmark_model(model_id, device="cpu")
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"fixture unavailable offline: {exc}")


def test_fused_experts_are_skipped_not_failed() -> None:
    """The fused ``mlp.experts`` nodes must be skipped, so every tested
    component passes — pre-fix they errored with a missing-router-args
    TypeError and P1 cratered to 50%."""
    report = _report("hyper-accel/ci-random-qwen2-moe-a3b")

    tested = {r.component_path for r in report.component_results}
    assert not any(
        p.endswith("mlp.experts") for p in tested
    ), f"fused experts must be skipped, but were tested: {sorted(tested)}"
    failed = [r.component_path for r in report.component_results if not r.passed]
    assert not failed, f"no isolatable component should fail: {failed}"


def test_plain_mlp_is_still_tested() -> None:
    """The arity gate must be surgical: a standard single-arg MLP is
    isolatable and must remain under test (guards against over-skipping)."""
    report = _report("openai-community/gpt2")

    mlp_tested = [
        r.component_path for r in report.component_results if r.component_path.endswith(".mlp")
    ]
    assert mlp_tested, "gpt2's plain MLPs must still be isolation-tested"
    assert all(r.passed for r in report.component_results), "gpt2 components must all pass"
