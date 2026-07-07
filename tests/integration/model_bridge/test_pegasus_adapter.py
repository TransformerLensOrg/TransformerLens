"""Integration tests for the Pegasus architecture adapter.

Uses google/pegasus-xsum (568M) — the distilled variants are asymmetric and
outside the symmetric-only adapter. CI-gated for download cost.
"""

import os

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "google/pegasus-xsum"

pytestmark = pytest.mark.skipif(
    bool(os.getenv("CI")), reason="Pegasus-XSum download too large for CI budget"
)


@pytest.fixture(scope="module")
def pegasus_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_inputs(pegasus_bridge):
    tokens = pegasus_bridge.tokenizer(
        "The tower is 324 metres tall, about the same height as an 81-storey building.",
        return_tensors="pt",
    )
    start = pegasus_bridge.original_model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[start, 5, 6, 7]])
    return tokens.input_ids, tokens.attention_mask, decoder_input_ids


class TestPegasusBridgeCreation:
    def test_adapter_and_structure(self, pegasus_bridge):
        from transformer_lens.model_bridge.supported_architectures.pegasus import (
            PegasusArchitectureAdapter,
        )

        assert isinstance(pegasus_bridge.adapter, PegasusArchitectureAdapter)
        assert hasattr(pegasus_bridge, "encoder_ln_final")
        assert hasattr(pegasus_bridge, "decoder_ln_final")
        assert not hasattr(pegasus_bridge, "embed_ln")


class TestPegasusForwardEquivalence:
    def test_forward_matches_hf(self, pegasus_bridge, sample_inputs):
        input_ids, attention_mask, decoder_input_ids = sample_inputs
        hf_model = pegasus_bridge.original_model
        with torch.no_grad():
            bridge_out = pegasus_bridge(
                input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids
            )
            hf_out = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestPegasusEmbedScale:
    def test_embed_hook_captures_unscaled_embeddings(self, pegasus_bridge, sample_inputs):
        """Scale is applied in the stack forward, not inside the embedding."""
        import math

        input_ids, attention_mask, decoder_input_ids = sample_inputs
        hf_model = pegasus_bridge.original_model
        captured = {}

        def grab(tensor, hook):
            captured["embed"] = tensor.detach().clone()

        with torch.no_grad():
            pegasus_bridge.run_with_hooks(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                fwd_hooks=[("hook_embed", grab)],
            )
            raw = hf_model.model.encoder.embed_tokens(input_ids)
        assert torch.equal(captured["embed"], raw)
        assert math.isclose(
            hf_model.model.encoder.embed_scale, math.sqrt(pegasus_bridge.cfg.d_model)
        )


class TestPegasusGeneration:
    def test_greedy_summary_is_coherent(self, pegasus_bridge):
        out = pegasus_bridge.generate(
            "The tower is 324 metres tall, about the same height as an 81-storey building.",
            max_new_tokens=12,
            do_sample=False,
            verbose=False,
        )
        assert isinstance(out, str)
        assert len(out.strip()) > 0
