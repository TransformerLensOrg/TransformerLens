"""Integration tests for the Jacobian lens tool on gpt2 (cached).

These exercise the real-model paths: loading the published gpt2-small lens from
the Hugging Face Hub (``neuronpedia/jacobian-lens``, ~13 MB), readout invariants
(the final-layer row equals the model's own logits; late-layer readouts align
with the model output), HookedTransformer/TransformerBridge agreement in the raw
weight basis, an exact fit/merge consistency identity, and a causal steering
smoke test.

gpt2 is in the CI model cache, so these live in ``integration/`` per
``tests/AGENTS.md``; the small fits are marked ``slow``.
"""

import pytest
import torch

PROMPT = "The Eiffel Tower is in the city of"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"

# Long enough to clear the default 16-position source skip.
FIT_PROMPTS = [
    "The industrial revolution transformed the economies of Europe during the "
    "nineteenth century, as steam power, railways, and mechanized factories "
    "reorganized both production and daily life in the growing cities.",
    "Astronomers studying distant galaxies rely on the redshift of spectral "
    "lines to estimate how quickly those galaxies recede, which in turn "
    "constrains the expansion history of the observable universe.",
]


@pytest.fixture(scope="module")
def ht_gpt2():
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")


@pytest.fixture(scope="module")
def bridge_gpt2():
    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers("gpt2", dtype=torch.float32, device="cpu")


@pytest.fixture(scope="module")
def published_lens(ht_gpt2):
    from transformer_lens.tools.analysis import JacobianLens

    return JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, model=ht_gpt2)


def test_published_lens_loads_and_validates(published_lens, ht_gpt2):
    assert published_lens.d_model == 768
    assert published_lens.source_layers == list(range(11))
    assert published_lens.n_prompts > 0
    assert published_lens.validate_model(ht_gpt2) is published_lens


def test_final_layer_readout_equals_model_logits(published_lens, ht_gpt2):
    result = published_lens.readout(ht_gpt2, PROMPT, layers=[11], positions=[-1])
    torch.testing.assert_close(result.lens_logits[11], result.model_logits)


def test_late_layer_readout_aligns_with_model_output(published_lens, ht_gpt2):
    """J_l -> I near the output: the rank of the model's top-1 token under the
    J-lens readout must collapse from thousands early to near zero late.
    (Top-k overlap is deliberately not asserted mid-stack — there the lens
    surfaces verbalizable content, not output predictions.)

    Measured on gpt2 for this prompt: rank 8577 at L0, 12 at L10, 0 at L11.
    """
    result = published_lens.readout(ht_gpt2, PROMPT, layers=[0, 10, 11], positions=[-1])
    model_top1 = result.model_logits[-1].argmax().item()

    def rank_of_model_top1(layer):
        logits = result.lens_logits[layer][-1]
        return int((logits > logits[model_top1]).sum().item())

    assert rank_of_model_top1(0) >= 1000
    assert rank_of_model_top1(10) <= 50
    assert rank_of_model_top1(11) == 0


def test_logit_lens_baseline_differs_mid_stack(published_lens, ht_gpt2):
    """use_jacobian=False is the logit lens; mid-stack the two must diverge
    (the whole point of the lens) while sharing the identical code path."""
    lens_result = published_lens.readout(ht_gpt2, PROMPT, layers=[5], positions=[-1])
    baseline = published_lens.readout(
        ht_gpt2, PROMPT, layers=[5], positions=[-1], use_jacobian=False
    )
    assert not torch.allclose(lens_result.lens_logits[5], baseline.lens_logits[5])


def test_bridge_and_ht_readouts_agree(published_lens, ht_gpt2, bridge_gpt2):
    """Raw-basis Bridge and no-processing HT must produce near-identical
    readouts (same weights, fp32); allow one near-tie swap in the top-8."""
    ht_result = published_lens.readout(ht_gpt2, PROMPT, positions=[-1])
    bridge_result = published_lens.readout(bridge_gpt2, PROMPT, positions=[-1])
    assert ht_result.tokens.tolist() == bridge_result.tokens.tolist()
    for layer in published_lens.source_layers:
        ht_top8 = ht_result.lens_logits[layer][-1].topk(8).indices.tolist()
        bridge_top8 = bridge_result.lens_logits[layer][-1].topk(8).indices.tolist()
        assert len(set(ht_top8) & set(bridge_top8)) >= 7, f"layer {layer}"
        assert ht_top8[0] == bridge_top8[0], f"layer {layer} top-1 mismatch"


def test_steering_raises_target_logit(published_lens, ht_gpt2):
    """Adding a token's J-lens direction mid-stack must raise that token's
    output logit (directional causal smoke test)."""
    target = " Paris"
    target_id = ht_gpt2.to_single_token(target)
    tokens = ht_gpt2.to_tokens(PROMPT)
    with torch.no_grad():
        baseline = ht_gpt2(tokens)[0, -1, target_id].item()
        hooks = published_lens.steering_hooks(ht_gpt2, target, layers=[6, 7, 8], alpha=6.0)
        with ht_gpt2.hooks(fwd_hooks=hooks):
            steered = ht_gpt2(tokens)[0, -1, target_id].item()
    assert steered > baseline


def test_processed_models_are_refused(published_lens):
    from transformer_lens import HookedTransformer
    from transformer_lens.tools.analysis import JacobianLens

    processed = HookedTransformer.from_pretrained("gpt2", device="cpu")
    with pytest.raises(ValueError, match="fold_ln"):
        published_lens.validate_model(processed)
    with pytest.raises(ValueError, match="fold_ln"):
        JacobianLens.fit(processed, FIT_PROMPTS, show_progress=False)
    # fold_ln=False still centers weights (normalization_type stays "LN"), so the
    # centered-unembed signature is the marker that must catch it.
    half_processed = HookedTransformer.from_pretrained("gpt2", fold_ln=False, device="cpu")
    with pytest.raises(ValueError, match="mean-centered"):
        published_lens.validate_model(half_processed)


@pytest.mark.slow
def test_fit_merge_consistency_and_finiteness(ht_gpt2):
    """Exact identity: fitting two prompts jointly equals merging two
    single-prompt fits (the estimator averages prompts unweighted and merge
    weights by n_prompts)."""
    from transformer_lens.tools.analysis import JacobianLens

    kwargs = dict(
        source_layers=[0, 5, 10],
        dim_batch=96,
        max_seq_len=32,
        show_progress=False,
    )
    joint = JacobianLens.fit(ht_gpt2, FIT_PROMPTS, **kwargs)
    merged = JacobianLens.merge(
        [JacobianLens.fit(ht_gpt2, [prompt], **kwargs) for prompt in FIT_PROMPTS]
    )
    assert joint.n_prompts == merged.n_prompts == 2
    for layer in joint.source_layers:
        assert torch.isfinite(joint.jacobians[layer]).all()
        torch.testing.assert_close(
            joint.jacobians[layer], merged.jacobians[layer], atol=1e-6, rtol=1e-5
        )


@pytest.mark.slow
def test_small_fit_converges_toward_published_lens(published_lens, ht_gpt2):
    """A native 2-prompt fit must already point in the published lens's
    direction (measured: cosine 0.51 / 0.66 / 0.91 at layers 0 / 5 / 10 vs the
    n=277 published gpt2 lens) — the estimator converges fast, and any basis
    or orientation bug would drive these toward zero."""
    from transformer_lens.tools.analysis import JacobianLens

    fitted = JacobianLens.fit(
        ht_gpt2,
        FIT_PROMPTS,
        source_layers=[0, 5, 10],
        dim_batch=96,
        max_seq_len=32,
        show_progress=False,
    )
    cosines = {
        layer: torch.nn.functional.cosine_similarity(
            fitted.jacobians[layer].flatten(), published_lens.jacobians[layer].flatten(), dim=0
        ).item()
        for layer in fitted.source_layers
    }
    assert all(value >= 0.3 for value in cosines.values()), cosines
    assert cosines[10] >= 0.75, cosines
