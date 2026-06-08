"""Integration tests for the Direct Logit Attribution tool on a TransformerBridge.

DLA decomposes the residual-stream part of a logit (or logit difference) via the
unembedding direction ``W_U[:, token]``. The unembedding bias ``b_U`` is a
per-token constant that no component produces, so a complete decomposition
reconstructs ``logit - b_U`` (and, for a difference, ``logit_diff - (b_U[c] -
b_U[w])``), not the raw logit. We assert that invariant against the bridge's own
forward-pass logits — the point of issue #1263 is that DLA must be correct on a
``TransformerBridge`` (compatibility mode), not only ``HookedTransformer``.

Uses distilgpt2 (CI-cached), matching test_analysis_methods.py.
"""

import pytest
import torch

from transformer_lens.tools.analysis import dla

PROMPT = "The Eiffel Tower is in the city of"
CORRECT = " Paris"
WRONG = " London"


def _last_token_logits(bridge, prompt):
    """Final-position logits from the bridge's own forward pass."""
    logits, _ = bridge.run_with_cache(prompt)
    if logits.ndim == 3:  # [batch, pos, vocab]
        logits = logits[0]
    return logits[-1]  # [vocab]


class TestDirectLogitAttributionCorrectness:
    """The component contributions must reconstruct the model's real logits."""

    def test_decompose_reconstructs_logit_difference(self, distilgpt2_bridge_compat):
        bridge = distilgpt2_bridge_compat
        c, w = bridge.to_single_token(CORRECT), bridge.to_single_token(WRONG)
        scores, labels = dla(bridge, [PROMPT], torch.tensor([[c, w]]))

        logits = _last_token_logits(bridge, PROMPT)
        expected = (logits[c] - logits[w]) - (bridge.b_U[c] - bridge.b_U[w])
        assert scores.sum().item() == pytest.approx(expected.item(), abs=1e-2)
        assert len(scores) == len(labels)

    def test_decompose_reconstructs_single_token_logit(self, distilgpt2_bridge_compat):
        bridge = distilgpt2_bridge_compat
        c = bridge.to_single_token(CORRECT)
        scores, _ = dla(bridge, [PROMPT], torch.tensor([[c]]))

        logits = _last_token_logits(bridge, PROMPT)
        expected = logits[c] - bridge.b_U[c]
        assert scores.sum().item() == pytest.approx(expected.item(), abs=1e-2)

    def test_accumulated_last_entry_reconstructs(self, distilgpt2_bridge_compat):
        # accumulated_resid is cumulative -> reconstruction is the LAST entry, not the sum
        bridge = distilgpt2_bridge_compat
        c, w = bridge.to_single_token(CORRECT), bridge.to_single_token(WRONG)
        scores, labels = dla(bridge, [PROMPT], torch.tensor([[c, w]]), accumulated=True)

        logits = _last_token_logits(bridge, PROMPT)
        expected = (logits[c] - logits[w]) - (bridge.b_U[c] - bridge.b_U[w])
        assert scores[-1].item() == pytest.approx(expected.item(), abs=1e-2)
        assert len(scores) == len(labels)


class TestDirectLogitAttributionShape:
    def test_decompose_labels_and_shape(self, distilgpt2_bridge_compat):
        bridge = distilgpt2_bridge_compat
        c = bridge.to_single_token(CORRECT)
        scores, labels = dla(bridge, [PROMPT], torch.tensor([[c]]))

        n_layers = bridge.cfg.n_layers
        assert scores.ndim == 1
        assert len(scores) == len(labels)
        assert "embed" in labels
        assert sum(label.endswith("attn_out") for label in labels) == n_layers
        assert sum(label.endswith("mlp_out") for label in labels) == n_layers

    def test_batch_of_prompts_is_averaged(self, distilgpt2_bridge_compat):
        # two identical prompts -> the batch-mean equals the single-prompt result
        bridge = distilgpt2_bridge_compat
        c, w = bridge.to_single_token(CORRECT), bridge.to_single_token(WRONG)
        single, _ = dla(bridge, [PROMPT], torch.tensor([[c, w]]))
        doubled, _ = dla(bridge, [PROMPT, PROMPT], torch.tensor([[c, w], [c, w]]))
        assert torch.allclose(single, doubled, atol=1e-4)


class TestDirectLogitAttributionGuardsOnRealBridge:
    """The guards must also fire on a genuine bridge, not just a mock."""

    def test_non_compat_bridge_raises(self, distilgpt2_bridge):
        bridge = distilgpt2_bridge  # compatibility mode NOT enabled
        c = bridge.to_single_token(CORRECT)
        with pytest.raises(ValueError, match="compatibility mode"):
            dla(bridge, [PROMPT], torch.tensor([[c]]))

    def test_hybrid_layer_types_raises(self, distilgpt2_bridge_compat, monkeypatch):
        bridge = distilgpt2_bridge_compat
        monkeypatch.setattr(bridge, "layer_types", lambda: ["attn+mlp", "mamba+mlp"])
        c = bridge.to_single_token(CORRECT)
        with pytest.raises(NotImplementedError, match="hybrid"):
            dla(bridge, [PROMPT], torch.tensor([[c]]))
