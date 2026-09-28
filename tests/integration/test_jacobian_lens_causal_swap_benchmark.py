"""GPT-2 integration smoke test for the causal coordinate-swap benchmark.

Structural assertions only -- no specific success-rate claim, matching the policy already
established for ``coordinate_patch_hooks`` itself. A deliberately tiny corpus (one function,
two concepts) keeps this fast enough for the regular cached-model integration suite; the full
country corpus runs only in the frozen-artifact generation script.
"""

import pytest
import torch

from transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark import (
    SCHEMA_VERSION,
    BenchmarkCorpus,
    FunctionSpec,
    build_protocol_manifest,
    corpus_definition,
    run_causal_swap_benchmark,
    serialize_artifact,
    success_rate_ci,
)

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
GPT2_LENS_FILE = "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"


@pytest.fixture(scope="module")
def gpt2_bridge():
    from transformer_lens.model_bridge import TransformerBridge

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TransformerBridge.boot_transformers("gpt2", dtype=torch.float32, device=device)


@pytest.fixture(scope="module")
def published_gpt2_lens(gpt2_bridge):
    from transformer_lens.tools.analysis import JacobianLens

    return JacobianLens.from_pretrained(
        LENS_REPO,
        filename=GPT2_LENS_FILE,
        revision=LENS_REVISION,
        model=gpt2_bridge,
    )


def test_causal_swap_benchmark_gpt2_smoke(published_gpt2_lens, gpt2_bridge) -> None:
    # The shipped corpus only executes the continent template with source Egypt, at layers 9
    # and 10. A fixture outside that set runs zero trials: ok_trials stays empty, the
    # control-displacement loop never runs, and selector breakages pass unnoticed.
    corpus = BenchmarkCorpus(
        name="smoke",
        concepts=("Egypt", "France"),
        functions=(
            FunctionSpec(
                name="continent",
                template="{arg} is a country on the continent of",
                answers={"Egypt": "Africa", "France": "Europe"},
            ),
        ),
    )
    trials, excluded = run_causal_swap_benchmark(
        published_gpt2_lens,
        gpt2_bridge,
        corpus,
        layers=[9],
        control_seeds=(0, 1),
    )
    assert len(trials) + len(excluded) >= 1

    ok_trials = [t for t in trials if t.status == "ok"]
    assert ok_trials, (
        "no trial executed: the published lens revision no longer has Egypt active in "
        "layer 9's support for the continent prompt, so this fixture needs a new "
        "(template, source, layer) combination"
    )
    assert any(t.source == "Egypt" and t.target == "France" for t in ok_trials)

    dictionary = published_gpt2_lens.lens_vector_dictionary(gpt2_bridge, 9)
    for trial in ok_trials:
        source_id = gpt2_bridge.to_single_token(f" {trial.source}")
        target_id = gpt2_bridge.to_single_token(f" {trial.target}")
        source_atom = dictionary[source_id].float()
        target_displacement = (dictionary[target_id].float() - source_atom).norm()
        control_displacement = (dictionary[trial.control_token_id].float() - source_atom).norm()
        assert (control_displacement - target_displacement).abs() <= (
            0.1 * target_displacement + 1e-6
        )

    manifest = build_protocol_manifest(
        model_id="gpt2",
        model_revision="n/a",
        lens_repo=LENS_REPO,
        lens_file=GPT2_LENS_FILE,
        lens_revision=LENS_REVISION,
        corpus_name=corpus.name,
        corpus=corpus_definition(corpus),
        corpus_repo="anthropics/jacobian-lens",
        corpus_path="data/experiments/flexible-generalization.json",
        corpus_revision="581d398613e5602a5af361e1c34d3a92ea82ba8e",
        layers_swept=[9],
        alpha=1.0,
        k=8,
        control_tolerance=0.1,
        control_seeds=[0, 1],
        success_definition="target token id equals deterministic argmax token id",
        baseline_definition="source answer token id equals deterministic argmax token id",
        rank_definition="1 + count(logits strictly greater than target logit)",
    )
    real_ci = success_rate_ci([t.real_target_metrics.target_is_top1 for t in ok_trials] or [False])
    control_ci = success_rate_ci(
        [t.control_target_metrics.target_is_top1 for t in ok_trials] or [False]
    )
    artifact = serialize_artifact(manifest, trials, excluded, real_ci, control_ci)
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["protocol_fingerprint"]
