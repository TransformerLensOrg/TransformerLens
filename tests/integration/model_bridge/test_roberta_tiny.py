"""Tiny from_config RoBERTa tests: CI stand-in for the slow real-weights parity file."""

import copy
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaForMaskedLM

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.utilities.position_ids import accepts_mask_derived_position_ids

D_MODEL = 16
D_HEAD = 4
N_HEADS = D_MODEL // D_HEAD
D_MLP = 32
N_LAYERS = 2
N_CTX = 8
D_VOCAB = 37
PAD_ID = 1
PARITY_ATOL = 1e-5
CACHE_NAMES = (
    "embed.hook_out",
    "token_type_embed.hook_out",
    "pos_embed.hook_out",
    "blocks.0.hook_in",
    "blocks.0.attn.q.hook_out",
    "blocks.0.hook_mlp_in",
    "blocks.0.hook_mlp_out",
    "ln_final.hook_normalized",
    "unembed.hook_out",
)


@dataclass
class TinyRobertaCase:
    bridge: TransformerBridge
    reference: RobertaForMaskedLM


@pytest.fixture(scope="module")
def case() -> TinyRobertaCase:
    torch.manual_seed(0)
    config = RobertaConfig(
        vocab_size=D_VOCAB,
        hidden_size=D_MODEL,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        intermediate_size=D_MLP,
        # HF counts the two reserved rows in the table size.
        max_position_embeddings=N_CTX + 2,
        type_vocab_size=1,
        pad_token_id=PAD_ID,
        bos_token_id=0,
        eos_token_id=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    reference = RobertaForMaskedLM(config).eval()
    source = copy.deepcopy(reference).eval()
    bridge = build_bridge_from_module(
        source,
        "RobertaForMaskedLM",
        hf_config=copy.deepcopy(config),
        tokenizer=None,
        dtype=torch.float32,
        device="cpu",
    )
    bridge.eval()
    return TinyRobertaCase(bridge=bridge, reference=reference)


@pytest.fixture(scope="module")
def tokens() -> torch.Tensor:
    return torch.tensor([[0, 5, 6, 7, 2]])


@pytest.fixture(scope="module")
def padded_tokens() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 5, 6, 2, PAD_ID, PAD_ID],
            [0, 5, 6, 7, 8, 2],
        ]
    )


def test_tiny_roberta_forward_matches_independent_reference(
    case: TinyRobertaCase,
    tokens: torch.Tensor,
) -> None:
    with torch.inference_mode():
        bridge_logits = case.bridge(tokens)
        reference_logits = case.reference(tokens).logits

    torch.testing.assert_close(bridge_logits, reference_logits, rtol=0.0, atol=PARITY_ATOL)


def test_tiny_roberta_live_wiring_and_tied_weights(case: TinyRobertaCase) -> None:
    bridge = case.bridge
    hf_model = bridge.original_model

    assert bridge.cfg.architecture == "RobertaForMaskedLM"
    assert hf_model.roberta.embeddings.word_embeddings is bridge.embed
    assert hf_model.roberta.embeddings.token_type_embeddings is bridge.token_type_embed
    assert hf_model.roberta.embeddings.position_embeddings is bridge.pos_embed
    assert hf_model.roberta.encoder.layer[0].attention.self.query is bridge.blocks[0].attn.q
    assert hf_model.lm_head.decoder is bridge.unembed
    assert bridge.token_type_embed.original_component.weight.shape == (1, D_MODEL)
    assert bridge.unembed.original_component.weight is bridge.embed.original_component.weight


def test_tiny_roberta_reports_usable_context_and_aligned_w_pos(case: TinyRobertaCase) -> None:
    bridge = case.bridge
    table = bridge.pos_embed.original_component.weight

    assert bridge.original_model.config.max_position_embeddings == N_CTX + 2
    assert bridge.cfg.n_ctx == N_CTX
    assert bridge.pos_embed.W_pos.shape == (N_CTX + 2, D_MODEL)
    assert bridge.W_pos.shape == (N_CTX, D_MODEL)
    torch.testing.assert_close(bridge.W_pos, table[2:], rtol=0.0, atol=0.0)
    assert bridge.W_E_pos.shape == (D_VOCAB + N_CTX, D_MODEL)


def test_tiny_roberta_cache_matches_reference_activations(
    case: TinyRobertaCase,
    tokens: torch.Tensor,
) -> None:
    bridge = case.bridge
    reference = case.reference

    with torch.inference_mode():
        bridge_logits, cache = bridge.run_with_cache(tokens, names_filter=list(CACHE_NAMES))

        embeddings = reference.roberta.embeddings
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
            reference.roberta.encoder.layer[0]
            .attention.self.query(block_input)
            .reshape(tokens.shape[0], tokens.shape[1], N_HEADS, D_HEAD)
        )
        layer0 = reference.roberta.encoder.layer[0]
        attn_out = layer0.attention(block_input)
        mlp_in = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        mlp_out = layer0.output.dense(layer0.intermediate(mlp_in))
        hidden = reference.roberta(input_ids=tokens).last_hidden_state
        pre_layer_norm = F.gelu(reference.lm_head.dense(hidden))
        normalized = F.layer_norm(
            pre_layer_norm,
            (pre_layer_norm.shape[-1],),
            eps=reference.lm_head.layer_norm.eps,
        )
        reference_logits = reference.lm_head(hidden)

    # Filtering by an alias caches it next to the canonical hook it redirects to.
    assert set(cache.keys()) - set(CACHE_NAMES) == {
        "blocks.0.mlp.in.hook_in",
        "blocks.0.mlp.out.hook_out",
    }
    expected = {
        "embed.hook_out": word_embeddings,
        "token_type_embed.hook_out": token_type_embeddings,
        "pos_embed.hook_out": position_embeddings,
        "blocks.0.hook_in": block_input,
        "blocks.0.attn.q.hook_out": query,
        "blocks.0.hook_mlp_in": mlp_in,
        "blocks.0.hook_mlp_out": mlp_out,
        "ln_final.hook_normalized": normalized,
        "unembed.hook_out": reference_logits,
    }
    for name, activation in expected.items():
        torch.testing.assert_close(cache[name], activation, rtol=0.0, atol=PARITY_ATOL)
    torch.testing.assert_close(bridge_logits, reference_logits, rtol=0.0, atol=PARITY_ATOL)


def test_tiny_roberta_pos_embed_hook_starts_at_row_two(
    case: TinyRobertaCase,
    tokens: torch.Tensor,
) -> None:
    table = case.bridge.pos_embed.original_component.weight
    seq_len = tokens.shape[1]

    with torch.inference_mode():
        _, cache = case.bridge.run_with_cache(tokens, names_filter=["pos_embed.hook_out"])

    torch.testing.assert_close(
        cache["pos_embed.hook_out"][0],
        table[2 : 2 + seq_len],
        rtol=0.0,
        atol=0.0,
    )


def test_tiny_roberta_padded_batch_matches_reference(
    case: TinyRobertaCase,
    padded_tokens: torch.Tensor,
) -> None:
    bridge = case.bridge
    reference = case.reference
    attention_mask = padded_tokens.ne(PAD_ID).long()
    table = bridge.pos_embed.original_component.weight

    with torch.inference_mode():
        bridge_logits, cache = bridge.run_with_cache(
            padded_tokens,
            attention_mask=attention_mask,
            names_filter=["pos_embed.hook_out"],
        )
        unmasked_logits = bridge(padded_tokens)
        reference_logits = reference(padded_tokens, attention_mask=attention_mask).logits

    torch.testing.assert_close(bridge_logits, reference_logits, rtol=0.0, atol=PARITY_ATOL)
    assert (bridge_logits - unmasked_logits).abs().max().item() > PARITY_ATOL
    pad_rows = cache["pos_embed.hook_out"][0, 4:]
    torch.testing.assert_close(pad_rows, table[PAD_ID].expand(2, -1), rtol=0.0, atol=0.0)
    torch.testing.assert_close(cache["pos_embed.hook_out"][0, :4], table[2:6], rtol=0.0, atol=0.0)


def test_tiny_roberta_left_padded_batch_matches_reference(case: TinyRobertaCase) -> None:
    left_padded = torch.tensor(
        [
            [PAD_ID, PAD_ID, 0, 5, 6, 2],
            [0, 5, 6, 7, 8, 2],
        ]
    )
    attention_mask = left_padded.ne(PAD_ID).long()
    # HF derives pad-aware positions itself, so the bridge must not inject its own.
    assert accepts_mask_derived_position_ids(case.bridge.original_model) is False

    with torch.inference_mode():
        bridge_logits = case.bridge(left_padded, attention_mask=attention_mask)
        reference_logits = case.reference(left_padded, attention_mask=attention_mask).logits
        unpadded_logits = case.reference(left_padded[:1, 2:]).logits

    torch.testing.assert_close(bridge_logits, reference_logits, rtol=0.0, atol=PARITY_ATOL)
    torch.testing.assert_close(bridge_logits[0, 2:], unpadded_logits[0], rtol=0.0, atol=PARITY_ATOL)


def test_tiny_roberta_rebuilt_from_a_bridge_cfg_keeps_usable_context(
    case: TinyRobertaCase,
    tokens: torch.Tensor,
) -> None:
    rebuilt = build_bridge_from_module(
        copy.deepcopy(case.reference).eval(),
        "RobertaForMaskedLM",
        tl_config=case.bridge.cfg,
        tokenizer=None,
        dtype=torch.float32,
        device="cpu",
    )

    assert rebuilt.cfg.n_ctx == N_CTX
    assert rebuilt.W_pos.shape == (N_CTX, D_MODEL)
    with torch.inference_mode():
        torch.testing.assert_close(
            rebuilt(tokens), case.reference(tokens).logits, rtol=0.0, atol=PARITY_ATOL
        )


def test_tiny_roberta_block_hook_intervenes(case: TinyRobertaCase, tokens: torch.Tensor) -> None:
    fired: list[str] = []

    def zero_block_output(value: torch.Tensor, hook) -> torch.Tensor:
        fired.append(hook.name)
        return torch.zeros_like(value)

    with torch.inference_mode():
        baseline = case.bridge(tokens)
        intervened = case.bridge.run_with_hooks(
            tokens,
            fwd_hooks=[("blocks.0.hook_out", zero_block_output)],
        )
        after = case.bridge(tokens)

    assert fired == ["blocks.0.hook_out"]
    assert not torch.allclose(intervened, baseline, rtol=0.0, atol=PARITY_ATOL)
    torch.testing.assert_close(after, baseline, rtol=0.0, atol=0.0)
