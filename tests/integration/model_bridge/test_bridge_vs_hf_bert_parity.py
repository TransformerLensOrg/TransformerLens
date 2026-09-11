"""BERT Bridge parity against an independent eager Hugging Face reference."""

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from transformer_lens.model_bridge import TransformerBridge

MODEL_NAME = "bert-base-cased"
PARITY_ATOL = 1e-5
CACHE_NAMES = (
    "embed.hook_out",
    "blocks.0.hook_in",
    "blocks.0.attn.q.hook_out",
    "ln_final.hook_normalized",
    "unembed.hook_out",
)


@pytest.fixture(scope="module")
def tokens() -> torch.Tensor:
    return torch.tensor([[101, 2023, 2003, 103, 102]])


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(
        MODEL_NAME,
        device="cpu",
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def hf_eager():
    return AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()


def test_bridge_logits_match_hf_eager(bridge, hf_eager, tokens) -> None:
    with torch.inference_mode():
        bridge_logits = bridge(tokens)
        hf_logits = hf_eager(tokens).logits

    torch.testing.assert_close(
        bridge_logits,
        hf_logits,
        rtol=0.0,
        atol=PARITY_ATOL,
    )


def test_run_with_cache_matches_hf_activations(bridge, hf_eager, tokens) -> None:
    with torch.inference_mode():
        bridge_logits, cache = bridge.run_with_cache(tokens, names_filter=list(CACHE_NAMES))
        word_embeddings = hf_eager.bert.embeddings.word_embeddings(tokens)
        block_input = hf_eager.bert.embeddings(input_ids=tokens)
        query = (
            hf_eager.bert.encoder.layer[0]
            .attention.self.query(block_input)
            .reshape(
                tokens.shape[0],
                tokens.shape[1],
                bridge.cfg.n_heads,
                bridge.cfg.d_head,
            )
        )
        hidden = hf_eager.bert(input_ids=tokens).last_hidden_state
        transform = hf_eager.cls.predictions.transform
        pre_layer_norm = transform.transform_act_fn(transform.dense(hidden))
        # hook_normalized is the normalized tensor before LayerNorm's affine transform.
        normalized = F.layer_norm(
            pre_layer_norm,
            (pre_layer_norm.shape[-1],),
            eps=transform.LayerNorm.eps,
        )
        hf_logits = hf_eager.cls.predictions.decoder(transform.LayerNorm(pre_layer_norm))

    assert set(cache.keys()) == set(CACHE_NAMES)
    expected = {
        "embed.hook_out": word_embeddings,
        "blocks.0.hook_in": block_input,
        "blocks.0.attn.q.hook_out": query,
        "ln_final.hook_normalized": normalized,
        "unembed.hook_out": hf_logits,
    }
    for name, activation in expected.items():
        torch.testing.assert_close(cache[name], activation, rtol=0.0, atol=PARITY_ATOL)
    torch.testing.assert_close(bridge_logits, hf_logits, rtol=0.0, atol=PARITY_ATOL)


def test_padding_mask_logits_match_hf_eager(bridge, hf_eager) -> None:
    padded_tokens = torch.tensor(
        [
            [101, 2023, 2003, 102, 0],
            [101, 2023, 2003, 103, 102],
        ]
    )
    attention_mask = padded_tokens.ne(0).long()

    with torch.inference_mode():
        bridge_logits = bridge(padded_tokens, attention_mask=attention_mask)
        unmasked_logits = bridge(padded_tokens)
        hf_logits = hf_eager(
            input_ids=padded_tokens,
            attention_mask=attention_mask,
        ).logits

    torch.testing.assert_close(bridge_logits, hf_logits, rtol=0.0, atol=PARITY_ATOL)
    assert (bridge_logits - unmasked_logits).abs().max().item() > PARITY_ATOL
