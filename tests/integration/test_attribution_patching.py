"""Integration guard: ``attribution_patch`` yields finite node scores on a real Bridge.

The model-free unit suite runs against ``_LinearToyBridge``, which overrides
``hook_dict``, ``hooks()``, and ``check_hooks_to_add`` — the three behaviours the
two real-Bridge failures depend on. Its hook points have no conversion and its
gate check is a no-op, so a green unit suite says nothing about a real Bridge.

This test boots a real GPT-2 Bridge and exercises the two paths the toy bridge
hides:

- gradients captured *through* hook conversions — ``blocks.*.attn.hook_z`` hands a
  reshaped ``[batch, seq, n_heads, d_head]`` view to the forward hook, so the
  ``attn_head_out`` family only scores if the backward hook delivers the gradient
  in that converted shape; and
- the default ``names_filter`` (``None``) falling back to the node hook set on a
  real ``hook_dict``, which also exposes gated points (``hook_mlp_in``,
  ``attn.hook_result``, split-QKV inputs) that ``add_hook`` would reject.
"""

from __future__ import annotations

import math

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.attribution_patching import attribution_patch

CLEAN_PROMPT = "The capital of France is"
CORRUPT_PROMPT = "The capital of Russia is"


@pytest.fixture(scope="module")
def gpt2_bridge():
    return TransformerBridge.boot_transformers("gpt2", device="cpu", dtype=torch.float32)


def _logit_diff_metric(answer_id: int, wrong_id: int):
    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logits[0, -1, answer_id] - logits[0, -1, wrong_id]

    return metric


def test_attribution_patch_scores_every_node_finite_on_real_bridge(gpt2_bridge) -> None:
    clean = gpt2_bridge.to_tokens(CLEAN_PROMPT)
    corrupt = gpt2_bridge.to_tokens(CORRUPT_PROMPT)
    assert clean.shape == corrupt.shape, "prompts must tokenize to the same length"

    answer_id = int(gpt2_bridge.to_tokens(" Paris")[0, -1].item())
    wrong_id = int(gpt2_bridge.to_tokens(" Moscow")[0, -1].item())
    metric_fn = _logit_diff_metric(answer_id, wrong_id)

    # Default config, no explicit names_filter: exercises the node-hook-set fallback.
    result = attribution_patch(gpt2_bridge, clean, corrupt, metric_fn)

    # Scores exactly the node graph: embed at every position, plus per-layer
    # per-head attention outputs and per-layer MLP outputs.
    seq_len = int(clean.shape[1])
    n_layers = int(gpt2_bridge.cfg.n_layers)
    n_heads = int(gpt2_bridge.cfg.n_heads)
    expected_count = seq_len * (1 + n_layers * (n_heads + 1))
    assert len(result.node_scores) == expected_count
    assert result.edge_scores == {}

    for node, score in result.node_scores.items():
        assert math.isfinite(score), f"non-finite score for {node}"

    # All three node families are present — attn_head_out is the family the hook
    # conversion bug broke, mlp_out and embed round out the node graph.
    families = {node.kind for node in result.node_scores}
    assert families == {"embed", "attn_head_out", "mlp_out"}
