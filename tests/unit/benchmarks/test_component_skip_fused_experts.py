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
import torch

pytest.importorskip("transformers")

# Config/tokenizer source only — weights are random-initialized at tiny dims
# below (the real repo is 1.76B params; fp32 double-load swamps CI).
MOE_ID = "hyper-accel/ci-random-qwen2-moe-a3b"


def _report(model_id: str):
    from transformer_lens.benchmarks.component_outputs import benchmark_model

    try:
        return benchmark_model(model_id, device="cpu")
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"fixture unavailable offline: {exc}")


@pytest.fixture(scope="module")
def tiny_moe_path(tmp_path_factory):
    """Local snapshot of a shrunken random Qwen2-MoE; fused-experts routing is
    structural, so the arity-driven skip fires at any size."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    try:
        cfg = AutoConfig.from_pretrained(MOE_ID)
        tok = AutoTokenizer.from_pretrained(MOE_ID)
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"fixture unavailable offline: {exc}")
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.moe_intermediate_size = 32
    cfg.shared_expert_intermediate_size = 32
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    path = tmp_path_factory.mktemp("tiny_qwen2_moe")
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def test_fused_experts_are_skipped_not_failed(tiny_moe_path) -> None:
    """Fused mlp.experts are skipped and every remaining isolated component passes."""
    report = _report(tiny_moe_path)

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
