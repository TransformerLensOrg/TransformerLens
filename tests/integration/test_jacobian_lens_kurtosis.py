"""Kurtosis-profile check for JacobianLens readouts (#1539 Tier-1).

Gurnee et al. (Transformer Circuits 2026, §4) report excess kurtosis of J-lens
readout logits near zero through the first third of depth, rising through the
workspace band. Absolute levels do not transfer between families (early-third
medians run 0.2 on gpt2-small to ~140 on pythia-70m-deduped, see #1539), so
the shape is asserted relative to each model's own baseline and to a
logit-lens control through the identical code path:

  1. Qwen3.5-0.8B: interior peak beats the early-third median by >= 0.35
     (measured +0.78 on the 40-prompt reference run; this test uses 12).
  2. That rise beats the logit-lens control's rise (measured gap +0.59).
  3. gpt2-small shows no rise (measured +0.16) — negative control.
  4. The final layer is identity transport in both arms; profiles must match.

Prompts are wikitext-103 validation rows; the published lenses were fitted on
train. Artifacts pinned to the same revision as
test_jacobian_lens_oracle_parity.py. Slow: downloads Qwen3.5-0.8B (~1.6 GB),
runs on CPU in a few minutes.
"""

from typing import Dict, List

import pytest
import torch

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"

QWEN_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN_LENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"
GPT2_MODEL = "gpt2"
GPT2_LENS_FILE = "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"

N_PROMPTS = 12
N_POSITIONS = 6
MAX_SEQ_LEN = 128
SKIP_FIRST_POSITIONS = 16  # matches the published fit configs
MIN_PROMPT_CHARS = 400

# Measured on 40 prompts x 8 positions (see #1539): qwen rise +0.78,
# control rise +0.19, gpt2 rise +0.16. Thresholds at ~half the measured
# effect / double the negative control.
QWEN_MIN_BAND_RISE = 0.35
QWEN_MIN_RISE_OVER_CONTROL = 0.25
GPT2_MAX_BAND_RISE = 0.35
FINAL_LAYER_ARM_TOL = 1e-3


def _excess_kurtosis(logits: torch.Tensor) -> torch.Tensor:
    """Population excess kurtosis over the vocab axis; logits [pos, d_vocab]."""
    x = logits.float()
    d = x - x.mean(dim=-1, keepdim=True)
    m2 = (d**2).mean(dim=-1)
    m4 = (d**4).mean(dim=-1)
    return m4 / m2.pow(2) - 3.0


def _wikitext_validation_prompts() -> List[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    prompts = []
    for row in ds:
        text = row["text"].strip()
        if len(text) >= MIN_PROMPT_CHARS and not text.startswith("="):
            prompts.append(text[:2000])
            if len(prompts) == N_PROMPTS:
                break
    return prompts


def _median_profiles(model, lens) -> Dict[str, Dict[int, float]]:
    """Per-layer median excess kurtosis for the J-lens and logit-lens arms."""
    samples: Dict[str, Dict[int, List[float]]] = {"jlens": {}, "control": {}}
    for text in _wikitext_validation_prompts():
        tokens = model.to_tokens(text)[:, :MAX_SEQ_LEN]
        seq_len = tokens.shape[1]
        if seq_len <= SKIP_FIRST_POSITIONS + 1:
            continue
        candidates = list(range(SKIP_FIRST_POSITIONS, seq_len - 1))
        step = max(len(candidates) / N_POSITIONS, 1)
        positions = [candidates[int(i * step)] for i in range(N_POSITIONS)][: len(candidates)]

        for arm, use_jacobian in (("jlens", True), ("control", False)):
            with torch.no_grad():
                readout = lens.readout(
                    model,
                    tokens,
                    positions=positions,
                    use_jacobian=use_jacobian,
                    return_full_logits=True,
                )
            assert readout.lens_logits is not None
            for layer, logits in readout.lens_logits.items():
                k = _excess_kurtosis(logits)
                samples[arm].setdefault(layer, []).extend(k.tolist())

    return {
        arm: {layer: float(torch.tensor(vals).median()) for layer, vals in by_layer.items()}
        for arm, by_layer in samples.items()
    }


def _shape_stats(profile: Dict[int, float], n_layers: int) -> Dict[str, float]:
    """Early-third median and interior (non-final, past one-third) peak."""
    early = [v for layer, v in profile.items() if layer / (n_layers - 1) < 1 / 3]
    interior = [
        v
        for layer, v in profile.items()
        if layer / (n_layers - 1) >= 1 / 3 and layer != n_layers - 1
    ]
    return {
        "early": float(torch.tensor(early).median()),
        "peak": max(interior),
        "final": profile[n_layers - 1],
    }


def _boot(model_name: str, lens_file: str):
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis import JacobianLens

    model = TransformerBridge.boot_transformers(model_name, device="cpu")
    model.eval()
    lens = JacobianLens.from_pretrained(
        LENS_REPO, filename=lens_file, revision=LENS_REVISION, model=model
    )
    return model, lens


@pytest.fixture(scope="module")
def qwen_profiles():
    model, lens = _boot(QWEN_MODEL, QWEN_LENS_FILE)
    profiles = _median_profiles(model, lens)
    return profiles, model.cfg.n_layers


@pytest.fixture(scope="module")
def gpt2_profiles():
    model, lens = _boot(GPT2_MODEL, GPT2_LENS_FILE)
    profiles = _median_profiles(model, lens)
    return profiles, model.cfg.n_layers


@pytest.mark.slow
def test_band_rise_is_present_and_lens_specific(qwen_profiles):
    """Qwen3.5-0.8B: J-lens kurtosis rises past depth 1/3; the control does not."""
    profiles, n_layers = qwen_profiles
    jlens = _shape_stats(profiles["jlens"], n_layers)
    control = _shape_stats(profiles["control"], n_layers)

    jlens_rise = jlens["peak"] - jlens["early"]
    control_rise = control["peak"] - control["early"]

    assert jlens_rise >= QWEN_MIN_BAND_RISE, (
        f"J-lens band rise {jlens_rise:.3f} < {QWEN_MIN_BAND_RISE} "
        f"(early {jlens['early']:.3f}, peak {jlens['peak']:.3f}); "
        f"measured +0.78 on the reference run"
    )
    assert jlens_rise - control_rise >= QWEN_MIN_RISE_OVER_CONTROL, (
        f"rise not lens-specific: J-lens {jlens_rise:.3f} vs control "
        f"{control_rise:.3f} through the identical code path"
    )


@pytest.mark.slow
def test_negative_control_gpt2_has_no_band_rise(gpt2_profiles):
    """gpt2-small: no workspace-band rise, so the check cannot pass vacuously."""
    profiles, n_layers = gpt2_profiles
    jlens = _shape_stats(profiles["jlens"], n_layers)
    rise = jlens["peak"] - jlens["early"]
    assert rise < GPT2_MAX_BAND_RISE, (
        f"gpt2-small J-lens band rise {rise:.3f} >= {GPT2_MAX_BAND_RISE}; "
        f"measured +0.16 on the reference run — investigate the lens artifact "
        f"or readout path before raising the threshold"
    )


@pytest.mark.slow
@pytest.mark.parametrize("profiles_fixture", ["qwen_profiles", "gpt2_profiles"])
def test_final_layer_arms_agree(profiles_fixture, request):
    """The final layer is identity transport in both arms: profiles must match."""
    profiles, n_layers = request.getfixturevalue(profiles_fixture)
    final = n_layers - 1
    diff = abs(profiles["jlens"][final] - profiles["control"][final])
    assert diff < FINAL_LAYER_ARM_TOL, (
        f"final-layer kurtosis differs between arms by {diff:.2e}; "
        f"identity transport should make them equal"
    )
