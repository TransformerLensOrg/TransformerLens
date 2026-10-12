"""DLA on gpt2 (compat-mode bridge): sum(DLA) == logit - b_U, and the diff form, whose b_U terms
don't cancel because gpt2's folded ln_final bias makes them differ."""

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

    # "layer" is cumulative: the last entry, not the sum, is the full residual stream.
    diff_total = diff.attribution[-1].sum() if unit == "layer" else diff.attribution.sum()
    single_total = single.attribution[-1].sum() if unit == "layer" else single.attribution.sum()

    assert diff_total.item() == pytest.approx((logit_c - logit_i) - (bu_c - bu_i), abs=1e-2)
    assert single_total.item() == pytest.approx(logit_c - bu_c, abs=1e-2)


class TestDirectLogitAttributionCore:
    """Core DLA behavior (labels, decomposition, cache reuse) on the bridge."""

    @pytest.mark.parametrize("unit", ["component", "layer", "head"])
    def test_decomposition_reconstructs_logit(self, gpt2_bridge_compat, unit):
        _assert_complete_decomposition(gpt2_bridge_compat, unit)

    def test_component_labels_and_shape(self, gpt2_bridge_compat):
        from transformer_lens.tools.analysis import direct_logit_attribution

        res = direct_logit_attribution(
            gpt2_bridge_compat,
            PROMPT,
            answer_tokens=CORRECT,
            incorrect_tokens=INCORRECT,
            unit="component",
        )
        assert res.unit == "component"
        assert res.attribution.shape[0] == len(res.labels)
        assert "embed" in res.labels
        assert (
            sum(label.endswith("_attn_out") for label in res.labels)
            == gpt2_bridge_compat.cfg.n_layers
        )
        assert (
            sum(label.endswith("_mlp_out") for label in res.labels)
            == gpt2_bridge_compat.cfg.n_layers
        )

    def test_head_labels_include_remainder(self, gpt2_bridge_compat):
        from transformer_lens.tools.analysis import direct_logit_attribution

        res = direct_logit_attribution(
            gpt2_bridge_compat, PROMPT, answer_tokens=CORRECT, unit="head"
        )
        assert (
            len(res.labels) == gpt2_bridge_compat.cfg.n_layers * gpt2_bridge_compat.cfg.n_heads + 1
        )
        assert res.labels[-1] == "remainder"

    def test_reuses_precomputed_cache(self, gpt2_bridge_compat):
        from transformer_lens.tools.analysis import direct_logit_attribution

        logit_c, _, bu_c, _ = _refs(gpt2_bridge_compat)
        _, cache = gpt2_bridge_compat.run_with_cache(PROMPT)
        res = direct_logit_attribution(gpt2_bridge_compat, answer_tokens=CORRECT, cache=cache)
        assert res.attribution.sum().item() == pytest.approx(logit_c - bu_c, abs=1e-2)

    def test_pos_none_keeps_position_axis(self, gpt2_bridge_compat):
        from transformer_lens.tools.analysis import direct_logit_attribution

        n_tokens = gpt2_bridge_compat.to_tokens(PROMPT).shape[1]
        res = direct_logit_attribution(
            gpt2_bridge_compat, PROMPT, answer_tokens=CORRECT, unit="component", pos=None
        )
        assert res.attribution.ndim == 3  # [component, batch, pos]
        assert res.attribution.shape[-1] == n_tokens

    def test_top_returns_sorted_pairs(self, gpt2_bridge_compat):
        from transformer_lens.tools.analysis import direct_logit_attribution

        res = direct_logit_attribution(
            gpt2_bridge_compat,
            PROMPT,
            answer_tokens=CORRECT,
            incorrect_tokens=INCORRECT,
            unit="head",
        )
        top = res.top(3)
        assert len(top) == 3
        values = [v for _, v in top]
        assert values == sorted(values, reverse=True)


class TestDirectLogitAttributionBridge:
    """DLA decomposition on TransformerBridge."""

    @pytest.mark.parametrize("unit", ["component", "head"])
    def test_decomposition_reconstructs_logit(self, gpt2_bridge_compat, unit):
        _assert_complete_decomposition(gpt2_bridge_compat, unit)


class TestDirectLogitAttributionBridgeGuards:
    """Without compat mode the projection direction is wrong, so DLA must refuse."""

    def test_non_compat_bridge_raises(self, gpt2_bridge):
        from transformer_lens.tools.analysis import direct_logit_attribution

        with pytest.raises(ValueError, match="compatibility mode"):
            direct_logit_attribution(gpt2_bridge, PROMPT, answer_tokens=CORRECT)


class TestDirectLogitAttributionValidation:
    def test_invalid_unit_raises(self, gpt2_bridge_compat):
        from transformer_lens.tools.analysis import direct_logit_attribution

        with pytest.raises(ValueError, match="unit must be one of"):
            direct_logit_attribution(
                gpt2_bridge_compat, PROMPT, answer_tokens=CORRECT, unit="neuron"
            )

    def test_missing_answer_tokens_raises(self, gpt2_bridge_compat):
        from transformer_lens.tools.analysis import direct_logit_attribution

        with pytest.raises(ValueError, match="answer_tokens is required"):
            direct_logit_attribution(gpt2_bridge_compat, PROMPT)

    def test_missing_input_and_cache_raises(self, gpt2_bridge_compat):
        from transformer_lens.tools.analysis import direct_logit_attribution

        with pytest.raises(ValueError, match="either `input`"):
            direct_logit_attribution(gpt2_bridge_compat, answer_tokens=CORRECT)
