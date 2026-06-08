"""Integration tests for the Direct Logit Attribution tool.

DLA decomposes the part of a logit that comes from the residual stream via the
unembedding *direction* ``W_U[:, token]``. The unembedding *bias* ``b_U`` is a
per-token constant that no component produces, so the exact correctness
invariant is::

    sum(component DLA for token) + b_U[token] == logit[token]

and, for a difference of two tokens, the two bias terms do **not** generally
cancel (gpt2's folded ``ln_final`` bias makes them differ), so::

    sum(component DLA, correct vs incorrect) == logit_diff - (b_U[c] - b_U[i])

We assert these for ``HookedTransformer`` and for ``TransformerBridge``
(compatibility mode) — the latter is the reason issue #1263 exists.

These tests load gpt2 (cached), so they live in ``integration/`` per
``tests/AGENTS.md``.
"""

import pytest

PROMPT = "The Eiffel Tower is in the city of"
CORRECT = " Paris"
INCORRECT = " London"


def _refs(model):
    """Reference values: (logit_correct, logit_incorrect, b_U[c], b_U[i])."""
    logits = model(PROMPT)
    if logits.ndim == 2:  # some Bridge configs may drop the batch dim
        logits = logits[None]
    c = model.to_single_token(CORRECT)
    i = model.to_single_token(INCORRECT)
    return (
        logits[0, -1, c].item(),
        logits[0, -1, i].item(),
        model.b_U[c].item(),
        model.b_U[i].item(),
    )


def _assert_complete_decomposition(model, unit):
    """sum(DLA) reconstructs the logit / logit-diff up to the b_U constant."""
    from transformer_lens.tools.analysis import direct_logit_attribution

    logit_c, logit_i, bu_c, bu_i = _refs(model)

    diff = direct_logit_attribution(
        model, PROMPT, answer_tokens=CORRECT, incorrect_tokens=INCORRECT, unit=unit
    )
    single = direct_logit_attribution(model, PROMPT, answer_tokens=CORRECT, unit=unit)

    # accumulated_resid ("layer") is cumulative: the last entry is the full
    # residual stream, so it (not the column sum) is the reconstruction.
    diff_total = diff.attribution[-1].sum() if unit == "layer" else diff.attribution.sum()
    single_total = single.attribution[-1].sum() if unit == "layer" else single.attribution.sum()

    assert diff_total.item() == pytest.approx((logit_c - logit_i) - (bu_c - bu_i), abs=1e-2)
    assert single_total.item() == pytest.approx(logit_c - bu_c, abs=1e-2)


@pytest.fixture(scope="module")
def gpt2_ht():
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device="cpu")


class TestDirectLogitAttributionHooked:
    """Correctness on HookedTransformer (the reference numerics)."""

    @pytest.mark.parametrize("unit", ["component", "layer", "head"])
    def test_decomposition_reconstructs_logit(self, gpt2_ht, unit):
        _assert_complete_decomposition(gpt2_ht, unit)

    def test_component_labels_and_shape(self, gpt2_ht):
        from transformer_lens.tools.analysis import direct_logit_attribution

        res = direct_logit_attribution(
            gpt2_ht, PROMPT, answer_tokens=CORRECT, incorrect_tokens=INCORRECT, unit="component"
        )
        assert res.unit == "component"
        assert res.attribution.shape[0] == len(res.labels)
        # Embedding term(s) plus each layer's attn_out and mlp_out.
        assert "embed" in res.labels
        assert sum(label.endswith("_attn_out") for label in res.labels) == gpt2_ht.cfg.n_layers
        assert sum(label.endswith("_mlp_out") for label in res.labels) == gpt2_ht.cfg.n_layers

    def test_head_labels_include_remainder(self, gpt2_ht):
        from transformer_lens.tools.analysis import direct_logit_attribution

        res = direct_logit_attribution(gpt2_ht, PROMPT, answer_tokens=CORRECT, unit="head")
        assert len(res.labels) == gpt2_ht.cfg.n_layers * gpt2_ht.cfg.n_heads + 1
        assert res.labels[-1] == "remainder"

    def test_reuses_precomputed_cache(self, gpt2_ht):
        from transformer_lens.tools.analysis import direct_logit_attribution

        logit_c, _, bu_c, _ = _refs(gpt2_ht)
        _, cache = gpt2_ht.run_with_cache(PROMPT)
        res = direct_logit_attribution(gpt2_ht, answer_tokens=CORRECT, cache=cache)
        assert res.attribution.sum().item() == pytest.approx(logit_c - bu_c, abs=1e-2)

    def test_pos_none_keeps_position_axis(self, gpt2_ht):
        from transformer_lens.tools.analysis import direct_logit_attribution

        n_tokens = gpt2_ht.to_tokens(PROMPT).shape[1]
        res = direct_logit_attribution(
            gpt2_ht, PROMPT, answer_tokens=CORRECT, unit="component", pos=None
        )
        assert res.attribution.ndim == 3  # [component, batch, pos]
        assert res.attribution.shape[-1] == n_tokens

    def test_top_returns_sorted_pairs(self, gpt2_ht):
        from transformer_lens.tools.analysis import direct_logit_attribution

        res = direct_logit_attribution(
            gpt2_ht, PROMPT, answer_tokens=CORRECT, incorrect_tokens=INCORRECT, unit="head"
        )
        top = res.top(3)
        assert len(top) == 3
        values = [v for _, v in top]
        assert values == sorted(values, reverse=True)


class TestDirectLogitAttributionBridge:
    """The point of #1263: DLA must work on TransformerBridge."""

    @pytest.mark.parametrize("unit", ["component", "head"])
    def test_decomposition_reconstructs_logit(self, gpt2_bridge_compat, unit):
        _assert_complete_decomposition(gpt2_bridge_compat, unit)


class TestDirectLogitAttributionValidation:
    def test_invalid_unit_raises(self, gpt2_ht):
        from transformer_lens.tools.analysis import direct_logit_attribution

        with pytest.raises(ValueError, match="unit must be one of"):
            direct_logit_attribution(gpt2_ht, PROMPT, answer_tokens=CORRECT, unit="neuron")

    def test_missing_answer_tokens_raises(self, gpt2_ht):
        from transformer_lens.tools.analysis import direct_logit_attribution

        with pytest.raises(ValueError, match="answer_tokens is required"):
            direct_logit_attribution(gpt2_ht, PROMPT)

    def test_missing_input_and_cache_raises(self, gpt2_ht):
        from transformer_lens.tools.analysis import direct_logit_attribution

        with pytest.raises(ValueError, match="either `input`"):
            direct_logit_attribution(gpt2_ht, answer_tokens=CORRECT)
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
