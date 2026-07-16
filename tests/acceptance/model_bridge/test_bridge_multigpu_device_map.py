"""device_map validation for boot_transformers — real model, >= 2 CUDA devices.

Run on a multi-GPU box, in its own process (the sibling n_devices suite boots its
own model pair):

    uv run pytest tests/acceptance/model_bridge/test_bridge_multigpu_device_map.py -m multigpu -v

Covers the accelerate-style placement paths: named strategies, an explicit
per-module dict, a mixed CPU/GPU map, and a preloaded ``hf_model`` that already
carries an ``hf_device_map``. Each boot is compared against one shared
single-device reference.
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

from ._bridge_multigpu_common import (
    MULTIGPU_MARKS,
    N_LAYERS,
    assert_logits_match,
    boot_multi,
    boot_single,
    cuda_indices,
)

pytestmark = MULTIGPU_MARKS


@pytest.fixture(scope="module")
def single():
    return boot_single()


def _explicit_gpt2_map(second_target) -> dict:
    """Split gpt2 by hand: embeddings + first half on cuda:0, the rest on
    ``second_target`` (cuda:1 or cpu)."""
    device_map = {
        "transformer.wte": 0,
        "transformer.wpe": 0,
        "transformer.drop": 0,
        "transformer.ln_f": second_target,
        "lm_head": second_target,
    }
    for i in range(N_LAYERS):
        device_map[f"transformer.h.{i}"] = 0 if i < N_LAYERS // 2 else second_target
    return device_map


class TestNamedStrategies:
    def test_balanced_spans_devices_and_matches(self, single):
        multi = boot_multi(device_map="balanced")
        assert cuda_indices(multi) == {0, 1}
        assert multi.cfg.n_devices == 2
        assert_logits_match(single, multi)

    def test_auto_boots_and_matches(self, single):
        # "auto" may legally place a small model on one device — only parity is
        # asserted; the guaranteed-split coverage lives in the balanced/dict tests.
        multi = boot_multi(device_map="auto")
        assert multi.cfg.n_devices >= 1
        assert_logits_match(single, multi)

    def test_sequential_boots_and_matches(self, single):
        multi = boot_multi(device_map="sequential")
        assert_logits_match(single, multi)


class TestExplicitMaps:
    def test_dict_map_places_exactly_and_matches(self, single):
        multi = boot_multi(device_map=_explicit_gpt2_map(1))
        assert cuda_indices(multi) == {0, 1}
        # The realized map must honor the request, not just "some split".
        realized: dict = getattr(multi.original_model, "hf_device_map")
        assert realized[f"transformer.h.{N_LAYERS - 1}"] == 1
        assert realized["transformer.h.0"] == 0
        assert_logits_match(single, multi)

    def test_mixed_cpu_gpu_map_matches(self, single):
        """CPU targets are supported: the second half runs on CPU kernels, so the
        band is looser than the GPU/GPU split (different matmul kernels, not a
        placement bug)."""
        multi = boot_multi(device_map=_explicit_gpt2_map("cpu"))
        param_device_types = {p.device.type for p in multi.original_model.parameters()}
        assert param_device_types == {"cuda", "cpu"}
        assert_logits_match(single, multi, atol=1e-3, rtol=1e-3)


class TestPreloadedModel:
    def test_hf_model_with_device_map_derives_config(self, single):
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            "gpt2", device_map="balanced", torch_dtype=torch.float32
        )
        multi = TransformerBridge.boot_transformers("gpt2", hf_model=hf_model)
        assert multi.cfg.n_devices == 2
        assert multi.cfg.device is not None
        assert_logits_match(single, multi)
