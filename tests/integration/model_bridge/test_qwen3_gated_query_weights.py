"""Download-free integration tests for Qwen3 gated query projections."""

import copy
from typing import NamedTuple

import pytest
import torch
from transformers import (
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3NextConfig,
    Qwen3NextForCausalLM,
)

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_from_module

N_HEADS = 2
D_HEAD = 8
D_MODEL = 16
N_LAYERS = 2
VOCAB_SIZE = 32


class GatedQueryCase(NamedTuple):
    bridge: TransformerBridge
    reference_logits: torch.Tensor
    tokens: torch.Tensor
    raw_q_weights: torch.Tensor


def _tiny_hybrid_config(architecture: str):
    common = dict(
        hidden_size=D_MODEL,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=1,
        head_dim=D_HEAD,
        intermediate_size=32,
        vocab_size=VOCAB_SIZE,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        full_attention_interval=1,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=N_HEADS,
        linear_num_value_heads=N_HEADS,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
    )
    if architecture == "Qwen3_5ForCausalLM":
        return Qwen3_5TextConfig(**common)
    return Qwen3NextConfig(
        **common,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


@pytest.fixture(
    scope="module",
    params=[
        ("Qwen3_5ForCausalLM", Qwen3_5ForCausalLM),
        ("Qwen3NextForCausalLM", Qwen3NextForCausalLM),
    ],
    ids=["qwen3_5", "qwen3_next"],
)
def gated_query_case(request: pytest.FixtureRequest) -> GatedQueryCase:
    architecture, model_cls = request.param
    torch.manual_seed(0)
    cfg = _tiny_hybrid_config(architecture)
    hf_model = model_cls(cfg).eval()
    with torch.no_grad():
        for layer_index, layer in enumerate(hf_model.model.layers):
            q_weight = layer.self_attn.q_proj.weight
            values = torch.arange(q_weight.numel(), dtype=q_weight.dtype).reshape_as(q_weight)
            q_weight.copy_((values + layer_index * q_weight.numel()) / q_weight.numel())
    raw_q_weights = torch.stack(
        [layer.self_attn.q_proj.weight.detach().clone() for layer in hf_model.model.layers]
    )

    tokens = torch.arange(4).unsqueeze(0)
    with torch.no_grad():
        reference_logits = hf_model(tokens).logits

    bridge = build_bridge_from_module(
        hf_model,
        architecture,
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )
    return GatedQueryCase(
        bridge=bridge,
        reference_logits=reference_logits,
        tokens=tokens,
        raw_q_weights=raw_q_weights,
    )


def test_gated_w_q_exposes_query_rows_only(gated_query_case: GatedQueryCase) -> None:
    expected = gated_query_case.raw_q_weights.view(N_LAYERS, N_HEADS, D_HEAD * 2, D_MODEL)[
        :, :, :D_HEAD, :
    ].transpose(-1, -2)

    assert gated_query_case.bridge.W_Q.shape == (N_LAYERS, N_HEADS, D_MODEL, D_HEAD)
    torch.testing.assert_close(gated_query_case.bridge.W_Q, expected)


def test_gated_w_q_access_preserves_forward_and_gate_hook(
    gated_query_case: GatedQueryCase,
) -> None:
    captured_gate: list[torch.Tensor] = []
    _ = gated_query_case.bridge.W_Q

    with torch.no_grad():
        bridge_logits = gated_query_case.bridge.run_with_hooks(
            gated_query_case.tokens,
            fwd_hooks=[
                (
                    "blocks.0.attn.hook_q_gate",
                    lambda gate, hook: captured_gate.append(gate.detach().clone()),
                )
            ],
        )

    torch.testing.assert_close(
        bridge_logits, gated_query_case.reference_logits, atol=1e-5, rtol=1e-5
    )
    assert len(captured_gate) == 1
    assert captured_gate[0].shape == (1, gated_query_case.tokens.shape[1], N_HEADS * D_HEAD)
    live_q_weights = torch.stack(
        [
            gated_query_case.bridge.state_dict()[f"blocks.{layer}.attn.q.weight"]
            for layer in range(N_LAYERS)
        ]
    )
    torch.testing.assert_close(live_q_weights, gated_query_case.raw_q_weights)


def test_standard_qwen3_w_q_is_unchanged() -> None:
    cfg = Qwen3Config(
        hidden_size=D_MODEL,
        num_hidden_layers=1,
        num_attention_heads=N_HEADS,
        num_key_value_heads=1,
        head_dim=D_HEAD,
        intermediate_size=32,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=32,
    )
    hf_model = Qwen3ForCausalLM(cfg).eval()
    bridge = build_bridge_from_module(
        hf_model,
        "Qwen3ForCausalLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()
    raw_q_weight = hf_model.model.layers[0].self_attn.q_proj.weight.detach()
    expected = raw_q_weight.view(N_HEADS, D_HEAD, D_MODEL).transpose(-1, -2).unsqueeze(0)

    assert bridge.W_Q.shape == (1, N_HEADS, D_MODEL, D_HEAD)
    torch.testing.assert_close(bridge.W_Q, expected)
