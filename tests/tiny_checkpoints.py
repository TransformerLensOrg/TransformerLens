"""Shared roster of tiny checkpoints and the parity tolerance policy.

Single source for the bridge-vs-HF suites so the roster does not drift between
files (it must still be mirrored into the CI model caches in
``.github/workflows/checks.yml`` — those lists are the other half of this
contract).
"""

from __future__ import annotations

import platform

import pytest

# Wider fp32 op-order noise floor on GH Actions macOS-arm64.
_MACOS_ARM64 = platform.system() == "Darwin" and platform.machine() == "arm64"

# One tolerance policy for every bridge-vs-HF-eager comparison. Two files
# previously asserted the same property with opposite policies (1e-2 vs a
# hardcoded 1e-5), so one flaked on macOS while the other could not fail there.
#
# CALIBRATION CAVEAT: the macOS-arm64 value is an op-order noise floor measured
# on real checkpoints (~3e-3 observed), which is ABOVE the ~1e-3 logits effect a
# dropped attention term produces on a tiny checkpoint. The macOS lane therefore
# cannot discriminate the #1618 failure class at the logits level — the Linux
# 1e-5 lane is the discriminating one, and per-layer/structural assertions are
# what catch regressions on macOS.
FP32_NOISE_TOL = 1e-2 if _MACOS_ARM64 else 1e-5

# One tiny checkpoint per reconstruction variant the bridge re-derives; each
# carries a term a regression could silently drop (the #1618 failure class):
# scale flags, GQA rope, ALiBi + in-module residual, flat qk-norm, fused QKV +
# softmax_scale/clip machinery, logit softcapping, sandwich norms + per-head
# qk-norm, residual_multiplier, clip_qkv, MoE routing, and MLA latent attention.
#
# ``sequential_residual`` marks the blocks whose attn/mlp contributions are added
# to a single residual stream in sequence — the ones for which
# ``resid_pre + attn_out == resid_mid`` is meaningful (parallel-residual and
# MoE/MLA-only stacks are excluded).
TINY_CHECKPOINTS: dict[str, dict[str, object]] = {
    "gpt2": {
        "name": "hf-internal-testing/tiny-random-gpt2",
        "sequential_residual": True,
    },
    "mistral": {
        "name": "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        "sequential_residual": True,
    },
    "bloom": {
        "name": "trl-internal-testing/tiny-BloomForCausalLM",
        "sequential_residual": True,
    },
    "olmo2": {
        "name": "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
        "sequential_residual": True,
    },
    "mpt": {
        "name": "hf-internal-testing/tiny-random-MptForCausalLM",
        "sequential_residual": True,
    },
    "granite": {
        "name": "hf-internal-testing/tiny-random-GraniteForCausalLM",
        "sequential_residual": True,
    },
    "gemma2": {
        "name": "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
        "sequential_residual": True,
    },
    "gemma3": {
        "name": "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
        "sequential_residual": False,
    },
    "olmo": {
        "name": "hf-internal-testing/tiny-random-OlmoForCausalLM",
        "sequential_residual": False,
    },
    "olmoe": {
        "name": "hf-internal-testing/tiny-random-OlmoeForCausalLM",
        "sequential_residual": False,
    },
    "qwen2": {
        "name": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        "sequential_residual": False,
    },
    "deepseek_v3": {
        "name": "hf-internal-testing/tiny-random-DeepseekV3ForCausalLM",
        "sequential_residual": False,
    },
}

# Mixed dense+sparse MoE stack (first_k_dense_replace=1 over two layers): one
# boot exercises both bindings of a single MoEBridge template.
MIXED_DENSE_SPARSE_MOE = "katuni4ka/tiny-random-deepseek-v3"


def parity_params() -> list:
    """pytest params for every tiny checkpoint."""
    return [pytest.param(spec["name"], id=key) for key, spec in TINY_CHECKPOINTS.items()]


def sequential_residual_params() -> list:
    """pytest params for checkpoints with a sequential residual decomposition."""
    return [
        pytest.param(spec["name"], id=key)
        for key, spec in TINY_CHECKPOINTS.items()
        if spec["sequential_residual"]
    ]
