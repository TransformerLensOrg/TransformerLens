"""T5 attention hooks against a tiny in-memory T5 (no Hub downloads).

HF T5Attention returns ``(attn_output, position_bias[, attn_weights])`` — its
second element is the relative-position bias (batch-broadcast ``[1, h, q, k]``
and zeros-based for cross-attention), not the attention pattern. These tests
pin that ``hook_pattern`` / ``hook_attn_scores`` carry the real post-/pre-
softmax attention, batched per row, matching HF's own ``output_attentions``.
"""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

N_LAYERS = 2
N_HEADS = 4
D_MODEL = 32
D_KV = 8
BATCH = 3
ENC_LEN = 5
DEC_LEN = 4
VOCAB = 128


@pytest.fixture(scope="module")
def t5_setup():
    """(bridge, pristine HF copy, encoder ids, decoder ids) on fixed seeds."""
    torch.manual_seed(0)
    cfg = T5Config(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        d_kv=D_KV,
        d_ff=64,
        num_layers=N_LAYERS,
        num_decoder_layers=N_LAYERS,
        num_heads=N_HEADS,
    )
    model = T5ForConditionalGeneration(cfg).eval()
    # Bridge construction rewires the module in place; keep an untouched copy
    # as the HF reference.
    hf_copy = copy.deepcopy(model)
    bridge = build_bridge_from_module(
        model,
        "T5ForConditionalGeneration",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()
    input_ids = torch.randint(0, VOCAB, (BATCH, ENC_LEN))
    decoder_input_ids = torch.randint(0, VOCAB, (BATCH, DEC_LEN))
    return bridge, hf_copy, input_ids, decoder_input_ids


def _capture(bridge, input_ids, decoder_input_ids, hook_specs):
    """Run the bridged model once, returning {name: tensor} for each hook."""
    captured: dict[str, torch.Tensor] = {}
    handles = [
        hook.register_forward_hook(
            lambda m, i, o, name=name: captured.setdefault(name, o.detach().clone())
        )
        for name, hook in hook_specs.items()
    ]
    try:
        with torch.no_grad():
            output = bridge.original_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured, output


class TestT5PatternHooks:
    def _patterns(self, t5_setup, layer: int):
        bridge, _, input_ids, decoder_input_ids = t5_setup
        captured, _ = _capture(
            bridge,
            input_ids,
            decoder_input_ids,
            {
                "enc": bridge.encoder_blocks[layer].attn.hook_pattern,
                "dec_self": bridge.decoder_blocks[layer].self_attn.hook_pattern,
                "cross": bridge.decoder_blocks[layer].cross_attn.hook_pattern,
            },
        )
        return captured

    def test_patterns_are_batched_and_row_normalized(self, t5_setup):
        captured = self._patterns(t5_setup, layer=0)
        assert captured["enc"].shape == (BATCH, N_HEADS, ENC_LEN, ENC_LEN)
        assert captured["dec_self"].shape == (BATCH, N_HEADS, DEC_LEN, DEC_LEN)
        assert captured["cross"].shape == (BATCH, N_HEADS, DEC_LEN, ENC_LEN)
        for name, pattern in captured.items():
            torch.testing.assert_close(
                pattern.sum(dim=-1),
                torch.ones(pattern.shape[:-1]),
                atol=1e-5,
                rtol=1e-5,
                msg=f"{name} rows must sum to 1 (post-softmax)",
            )

    @pytest.mark.parametrize("layer", range(N_LAYERS))
    def test_patterns_match_hf_output_attentions(self, t5_setup, layer):
        """Every layer, both stacks: bias reuse across layers must not leak the
        broadcast position_bias back into the pattern hooks."""
        bridge, hf_copy, input_ids, decoder_input_ids = t5_setup
        captured = self._patterns(t5_setup, layer)
        with torch.no_grad():
            hf_out = hf_copy(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                output_attentions=True,
            )
        torch.testing.assert_close(captured["enc"], hf_out.encoder_attentions[layer])
        torch.testing.assert_close(captured["dec_self"], hf_out.decoder_attentions[layer])
        torch.testing.assert_close(captured["cross"], hf_out.cross_attentions[layer])


class TestT5AttnScoresHooks:
    def test_scores_are_pre_softmax(self, t5_setup):
        bridge, _, input_ids, decoder_input_ids = t5_setup
        enc_attn = bridge.encoder_blocks[0].attn
        cross_attn = bridge.decoder_blocks[0].cross_attn
        captured, _ = _capture(
            bridge,
            input_ids,
            decoder_input_ids,
            {
                "enc_scores": enc_attn.hook_attn_scores,
                "enc_pattern": enc_attn.hook_pattern,
                "cross_scores": cross_attn.hook_attn_scores,
                "cross_pattern": cross_attn.hook_pattern,
            },
        )
        for which in ("enc", "cross"):
            scores = captured[f"{which}_scores"]
            pattern = captured[f"{which}_pattern"]
            assert scores.shape == pattern.shape
            assert not torch.allclose(scores, pattern)
            torch.testing.assert_close(
                torch.softmax(scores.float(), dim=-1),
                pattern,
                atol=1e-5,
                rtol=1e-5,
                msg=f"{which}: softmax(hook_attn_scores) must equal hook_pattern",
            )


class TestT5OutputIntegrity:
    def test_logits_match_pristine_hf(self, t5_setup):
        """The forced output_attentions must be stripped before the HF stack
        reads fixed tuple positions (cross-attn bias sits at a flag-dependent
        index), or decoder layers consume the wrong tensors."""
        bridge, hf_copy, input_ids, decoder_input_ids = t5_setup
        with torch.no_grad():
            bridge_logits = bridge.original_model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids
            ).logits
            hf_logits = hf_copy(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits
        torch.testing.assert_close(bridge_logits, hf_logits, atol=1e-5, rtol=1e-5)

    def test_caller_requested_attentions_survive(self, t5_setup):
        bridge, hf_copy, input_ids, decoder_input_ids = t5_setup
        with torch.no_grad():
            bridge_out = bridge.original_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                output_attentions=True,
            )
            hf_out = hf_copy(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                output_attentions=True,
            )
        assert bridge_out.encoder_attentions is not None
        torch.testing.assert_close(bridge_out.encoder_attentions[0], hf_out.encoder_attentions[0])
        torch.testing.assert_close(bridge_out.cross_attentions[0], hf_out.cross_attentions[0])


class TestT5PatternEditWarning:
    def test_editing_pattern_hook_warns_and_leaves_output_unchanged(self, t5_setup):
        bridge, _, input_ids, decoder_input_ids = t5_setup
        with torch.no_grad():
            base = bridge.original_model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids
            ).logits
        handle = bridge.encoder_blocks[0].attn.hook_pattern.register_forward_hook(
            lambda m, i, o: torch.zeros_like(o)
        )
        try:
            with pytest.warns(UserWarning, match="edit is discarded"):
                with torch.no_grad():
                    edited = bridge.original_model(
                        input_ids=input_ids, decoder_input_ids=decoder_input_ids
                    ).logits
        finally:
            handle.remove()
        assert torch.equal(base, edited)
