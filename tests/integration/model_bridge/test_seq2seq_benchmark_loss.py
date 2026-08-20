"""Benchmark loss calls must supply labels and a resolvable ablation hook on seq2seq.

The bridge refuses label-less return_type="loss" for encoder-decoder models
(encoder input_ids are not decoder targets). forward_pass.py was updated when
that guard landed; hook_registration and weight_processing kept the bare call,
so P2 hook_functionality errored on all seven seq2seq architectures. The
ablation hook also targeted blocks.0.* which does not exist on encoder-decoder
bridges, silently no-opping the whole check.
"""

import pytest

from transformer_lens.benchmarks.hook_registration import benchmark_hook_functionality
from transformer_lens.benchmarks.utils import BenchmarkSeverity, bridge_self_target_loss
from transformer_lens.model_bridge import TransformerBridge

TEXT = "translate English to German: Hello world"


@pytest.fixture(scope="module")
def t5():
    return TransformerBridge.boot_transformers("google-t5/t5-small", device="cpu")


def test_self_target_loss_is_finite_on_seq2seq(t5) -> None:
    loss = bridge_self_target_loss(t5, TEXT)
    assert loss.ndim == 0 and loss.isfinite()


def test_hook_functionality_runs_and_the_ablation_bites(t5) -> None:
    result = benchmark_hook_functionality(t5, TEXT)
    assert result.passed, result.message
    assert result.severity != BenchmarkSeverity.ERROR, result.message
    # A vacuous run (unresolvable hook) reports "minimal effect: 0.000000".
    assert "minimal effect" not in result.message, result.message
