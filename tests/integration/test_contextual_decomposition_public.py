"""Public single-source CD workflow and result ownership on local GPT-2."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.tools.analysis import (
    CDBaseline,
    CDResult,
    CDSource,
    CDTarget,
    ContextualDecomposition,
)


@pytest.fixture
def model():
    with torch.random.fork_rng():
        torch.manual_seed(5)
        config = GPT2Config(vocab_size=16, n_positions=8, n_embd=8, n_layer=2, n_head=2)
        config._attn_implementation = "eager"
        return build_bridge_from_module(
            GPT2LMHeadModel(config).eval(),
            "GPT2LMHeadModel",
            hf_config=config,
            dtype=torch.float32,
            device="cpu",
        ).eval()


TOKENS = torch.tensor([[1, 2, 3], [3, 2, 1]])


def test_public_head_workflow_and_selective_diagnostics(model):
    analysis = ContextualDecomposition(model)
    result = analysis.decompose(
        TOKENS,
        source=CDSource.attention_head(0, 1, positions=(0,)),
        target=CDTarget.logit_difference(
            positive_token_id=(2, 3), negative_token_id=(1, 2), position=(1, 2)
        ),
        retain_intermediates=("blocks.1.ln1.hook_out",),
    )
    assert isinstance(result, CDResult)
    assert result.target_beta.shape == (2,)
    torch.testing.assert_close(
        result.target_beta + result.target_gamma, result.target_total, atol=2e-6, rtol=2e-5
    )
    assert set(result.intermediates) == {"blocks.1.ln1.hook_out"}
    assert result.diagnostics and result.coverage
    assert result.baseline_description == "zero"
    assert result.policy == analysis.policy
    assert not hasattr(result, "faithful")


def test_default_does_not_retain_intermediate_tensors(model):
    result = ContextualDecomposition(model).decompose(
        TOKENS,
        source=CDSource.residual(0),
        target=CDTarget.logit(token_id=1, position=2),
    )
    assert not result.intermediates
    assert not result.target_beta.requires_grad


def test_reference_snapshot_and_result_mutation_do_not_affect_repeat_calls(model):
    reference = torch.full((2, 3, 8), 0.001)
    baseline = CDBaseline.from_tensor(reference, description="contrast reference")
    reference.zero_()
    analysis = ContextualDecomposition(model)
    kwargs = dict(
        source=CDSource.residual(1, location="post"),
        target=CDTarget.logit(token_id=2, position=2),
        baseline=baseline,
    )
    first = analysis.decompose(TOKENS, **kwargs)
    expected = first.target_beta.clone()
    first.target_beta.zero_()
    first.target_gamma.fill_(10)
    second = analysis.decompose(TOKENS, **kwargs)
    torch.testing.assert_close(second.target_beta, expected)
    assert second.baseline_description == "contrast reference"
    with pytest.raises(TypeError):
        second.diagnostics["new"] = None


def test_internal_target_and_invalid_retention(model):
    analysis = ContextualDecomposition(model)
    result = analysis.decompose(
        TOKENS,
        source=CDSource.residual(0),
        target=CDTarget.internal("blocks.1.attn.q.hook_out", position=2, head=1),
    )
    assert result.target_beta.shape == (2, 4)
    with pytest.raises(ValueError, match="intermediate"):
        analysis.decompose(
            TOKENS,
            source=CDSource.residual(1),
            target=CDTarget.logit(token_id=1, position=2),
            retain_intermediates=("blocks.0.hook_out",),
        )


def test_mutating_retained_intermediates_does_not_affect_repeat_call(model):
    analysis = ContextualDecomposition(model)
    kwargs = dict(
        source=CDSource.residual(0),
        target=CDTarget.logit(token_id=1, position=2),
        retain_intermediates=("ln_final.hook_out",),
    )
    first = analysis.decompose(TOKENS, **kwargs)
    expected = first.intermediates["ln_final.hook_out"].beta.clone()
    first.intermediates["ln_final.hook_out"].beta.zero_()
    second = analysis.decompose(TOKENS, **kwargs)
    torch.testing.assert_close(second.intermediates["ln_final.hook_out"].beta, expected)


def test_duplicate_retention_names_rejected(model):
    with pytest.raises(ValueError, match="unique"):
        ContextualDecomposition(model).decompose(
            TOKENS,
            source=CDSource.residual(0),
            target=CDTarget.logit(token_id=1, position=2),
            retain_intermediates=("ln_final.hook_out", "ln_final.hook_out"),
        )


def test_documented_example_executes():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/source/content/contextual_decomposition.md").read_text()
    example = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    assert example is not None
    exec(compile(example.group(1), "contextual_decomposition_example", "exec"), {})


def test_policy_and_model_validation(model):
    with pytest.raises(ValueError, match="policy"):
        ContextualDecomposition(model, policy="frozen_attention")
    with pytest.raises((ValueError, TypeError), match="TransformerBridge|model"):
        ContextualDecomposition(torch.nn.Linear(2, 2))
    analysis = ContextualDecomposition(model)
    model.blocks[0].mlp.train()
    with pytest.raises(ValueError, match="evaluation"):
        analysis.decompose(
            TOKENS, source=CDSource.residual(0), target=CDTarget.logit(token_id=1, position=2)
        )
