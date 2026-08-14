"""convert_olmoe_weights: the batched-expert split and its quantization guard.

OLMoE stores all experts in two batched Parameters and the converter slices
them, so a packed or scale-separated tensor would be reshaped into a
plausible-looking matrix rather than raising. These guards existed but nothing
exercised them — removing both left the whole suite green.

The model is built from a tiny config in memory: no download, no hub access.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import OlmoeConfig, OlmoeForCausalLM

from transformer_lens.config import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions import convert_olmoe_weights

D_MODEL, D_MLP, N_EXPERTS, N_LAYERS = 8, 16, 4, 1
EXPERTS_PER_TOKEN = 2


@pytest.fixture(scope="module")
def hf_model() -> OlmoeForCausalLM:
    """Tiny OLMoE with AMPLIFIED weights.

    Same reason as the Mixtral fixture: SiLU is near-linear at small magnitudes,
    so `act(gate) * up ~= act(up) * gate` and a swapped gate/up mapping reads as
    noise at HF's default init. Do not lower without re-measuring the negative
    control in test_expert_weights_reproduce_hf_expert_output.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = OlmoeForCausalLM(
            OlmoeConfig(
                hidden_size=D_MODEL,
                intermediate_size=D_MLP,
                num_hidden_layers=N_LAYERS,
                num_attention_heads=2,
                num_key_value_heads=1,
                vocab_size=32,
                num_experts=N_EXPERTS,
                num_experts_per_tok=EXPERTS_PER_TOKEN,
                max_position_embeddings=32,
            )
        ).eval()
        for param in model.parameters():
            torch.nn.init.normal_(param, std=0.3)
    return model


@pytest.fixture(scope="module")
def tl_cfg() -> HookedTransformerConfig:
    return HookedTransformerConfig(
        d_model=D_MODEL,
        d_head=4,
        n_heads=2,
        n_key_value_heads=1,
        n_layers=N_LAYERS,
        n_ctx=32,
        d_vocab=32,
        d_mlp=D_MLP,
        num_experts=N_EXPERTS,
        experts_per_token=EXPERTS_PER_TOKEN,
        act_fn="silu",
        normalization_type="RMS",
        positional_embedding_type="rotary",
        use_qk_norm=True,
    )


def test_expert_weights_reproduce_hf_expert_output(hf_model, tl_cfg) -> None:
    """The decisive check on the fused-projection split.

    Shapes cannot catch a swapped gate/up — both halves are [d_mlp, d_model] —
    so this compares numerically and carries a negative control proving the
    swapped assignment would differ.
    """
    state_dict = convert_olmoe_weights(hf_model, tl_cfg)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        x = torch.randn(1, D_MODEL)

    experts = hf_model.model.layers[0].mlp.experts
    for expert in range(N_EXPERTS):
        gate_hf, up_hf = F.linear(x, experts.gate_up_proj[expert]).chunk(2, dim=-1)
        expected = F.linear(F.silu(gate_hf) * up_hf, experts.down_proj[expert])

        w_gate = state_dict[f"blocks.0.mlp.experts.{expert}.W_gate.weight"]
        w_in = state_dict[f"blocks.0.mlp.experts.{expert}.W_in.weight"]
        w_out = state_dict[f"blocks.0.mlp.experts.{expert}.W_out.weight"]
        actual = F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_in), w_out)
        torch.testing.assert_close(actual, expected)

        swapped = F.linear(F.silu(F.linear(x, w_in)) * F.linear(x, w_gate), w_out)
        assert not torch.allclose(swapped, expected, atol=1e-6), (
            f"expert {expert}: gate and up are interchangeable in this fixture, "
            "so the assertion above cannot detect a swapped mapping"
        )


@pytest.mark.parametrize(
    "dtype,reason",
    [
        (torch.int8, "packed integer storage"),
        (torch.uint8, "packed integer storage"),
        # float8 reports is_floating_point=True, so a plain float check admits
        # it — and it is the one family that slices without complaint.
        (torch.float8_e4m3fn, "narrow float"),
        (torch.float8_e5m2, "narrow float"),
    ],
)
@pytest.mark.parametrize("tensor_name", ["gate_up_proj", "down_proj"])
def test_quantized_expert_weights_are_refused(hf_model, tl_cfg, dtype, reason, tensor_name) -> None:
    """Both batched expert tensors are sliced, so both must be guarded —
    slicing either would silently drop the scales stored beside it."""
    experts = hf_model.model.layers[0].mlp.experts
    original = getattr(experts, tensor_name)
    try:
        setattr(
            experts,
            tensor_name,
            torch.nn.Parameter(original.detach().to(dtype), requires_grad=False),
        )
        with pytest.raises(NotImplementedError, match=reason):
            convert_olmoe_weights(hf_model, tl_cfg)
    finally:
        setattr(experts, tensor_name, original)
