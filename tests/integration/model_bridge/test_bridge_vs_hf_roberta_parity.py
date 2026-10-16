"""RoBERTa Bridge parity against an independent eager HF reference. Slow: not in the CI cache."""

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from transformer_lens.model_bridge import TransformerBridge

MODEL_NAME = "distilbert/distilroberta-base"
PAD_ID = 1
PARITY_ATOL = 1e-5
CACHE_NAMES = (
    "embed.hook_out",
    "token_type_embed.hook_out",
    "pos_embed.hook_out",
    "blocks.0.hook_in",
    "blocks.0.attn.q.hook_out",
    "ln_final.hook_normalized",
    "unembed.hook_out",
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tokens() -> torch.Tensor:
    return torch.tensor([[0, 133, 812, 16, 50264, 2]])


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


def test_bridge_boots_with_roberta_context_metadata(bridge) -> None:
    assert bridge.cfg.architecture == "RobertaForMaskedLM"
    assert bridge.cfg.tokenizer_prepends_bos is True
    assert bridge.cfg.tokenizer_appends_eos is True
    assert bridge.tokenizer.pad_token_id == PAD_ID
    assert bridge.cfg.n_ctx == 512
    assert bridge.original_model.config.max_position_embeddings == 514
    assert bridge.W_pos.shape == (512, bridge.cfg.d_model)
    torch.testing.assert_close(
        bridge.W_pos,
        bridge.pos_embed.original_component.weight[2:],
        rtol=0.0,
        atol=0.0,
    )


def test_bridge_logits_match_independent_hf_eager(bridge, hf_eager, tokens) -> None:
    assert bridge.original_model is not hf_eager
    assert next(bridge.parameters()).dtype == torch.float32
    assert next(hf_eager.parameters()).dtype == torch.float32

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

        embeddings = hf_eager.roberta.embeddings
        token_type_ids = torch.zeros_like(tokens)
        position_ids = embeddings.create_position_ids_from_input_ids(
            tokens,
            embeddings.padding_idx,
        )
        word_embeddings = embeddings.word_embeddings(tokens)
        token_type_embeddings = embeddings.token_type_embeddings(token_type_ids)
        position_embeddings = embeddings.position_embeddings(position_ids)
        block_input = embeddings(input_ids=tokens)
        query = (
            hf_eager.roberta.encoder.layer[0]
            .attention.self.query(block_input)
            .reshape(
                tokens.shape[0],
                tokens.shape[1],
                bridge.cfg.n_heads,
                bridge.cfg.d_head,
            )
        )
        hidden = hf_eager.roberta(input_ids=tokens).last_hidden_state
        pre_layer_norm = F.gelu(hf_eager.lm_head.dense(hidden))
        normalized = F.layer_norm(
            pre_layer_norm,
            (pre_layer_norm.shape[-1],),
            eps=hf_eager.lm_head.layer_norm.eps,
        )
        hf_logits = hf_eager.lm_head.decoder(hf_eager.lm_head.layer_norm(pre_layer_norm))

    assert set(cache.keys()) == set(CACHE_NAMES)
    expected = {
        "embed.hook_out": word_embeddings,
        "token_type_embed.hook_out": token_type_embeddings,
        "pos_embed.hook_out": position_embeddings,
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
            [0, 133, 812, 2, PAD_ID, PAD_ID],
            [0, 133, 812, 16, 50264, 2],
        ]
    )
    attention_mask = padded_tokens.ne(PAD_ID).long()

    with torch.inference_mode():
        bridge_logits = bridge(padded_tokens, attention_mask=attention_mask)
        unmasked_logits = bridge(padded_tokens)
        hf_logits = hf_eager(
            input_ids=padded_tokens,
            attention_mask=attention_mask,
        ).logits

    torch.testing.assert_close(bridge_logits, hf_logits, rtol=0.0, atol=PARITY_ATOL)
    assert (bridge_logits - unmasked_logits).abs().max().item() > PARITY_ATOL


def test_left_padding_mask_logits_match_hf_eager(bridge, hf_eager) -> None:
    padded_tokens = torch.tensor(
        [
            [PAD_ID, PAD_ID, 0, 133, 812, 2],
            [0, 133, 812, 16, 50264, 2],
        ]
    )
    attention_mask = padded_tokens.ne(PAD_ID).long()

    with torch.inference_mode():
        bridge_logits = bridge(padded_tokens, attention_mask=attention_mask)
        hf_logits = hf_eager(input_ids=padded_tokens, attention_mask=attention_mask).logits

    torch.testing.assert_close(bridge_logits, hf_logits, rtol=0.0, atol=PARITY_ATOL)


def test_live_hf_tree_uses_bridge_wrappers_and_preserves_tied_weights(bridge) -> None:
    hf_model = bridge.original_model

    assert hf_model.roberta.embeddings.word_embeddings is bridge.embed
    assert hf_model.roberta.embeddings.token_type_embeddings is bridge.token_type_embed
    assert hf_model.roberta.embeddings.position_embeddings is bridge.pos_embed
    assert hf_model.roberta.encoder.layer[0].attention.self.query is bridge.blocks[0].attn.q
    assert hf_model.lm_head.decoder is bridge.unembed
    assert bridge.unembed.original_component.weight is bridge.embed.original_component.weight
    assert bridge.token_type_embed.original_component.weight.shape == (1, bridge.cfg.d_model)


def test_block_output_hook_can_intervene(bridge, tokens) -> None:
    fired: list[str] = []

    def zero_block_output(value: torch.Tensor, hook) -> torch.Tensor:
        fired.append(hook.name)
        return torch.zeros_like(value)

    with torch.inference_mode():
        baseline = bridge(tokens)
        intervened = bridge.run_with_hooks(
            tokens,
            fwd_hooks=[("blocks.0.hook_out", zero_block_output)],
        )
        after = bridge(tokens)

    assert fired == ["blocks.0.hook_out"]
    assert not torch.allclose(intervened, baseline, rtol=0.0, atol=PARITY_ATOL)
    torch.testing.assert_close(after, baseline, rtol=0.0, atol=0.0)
