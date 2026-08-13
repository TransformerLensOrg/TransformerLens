"""convert_mixtral_weights against the transformers 5.x Mixtral layout.

5.x renamed the MoE block (``block_sparse_moe`` -> ``mlp``) and replaced the
per-expert ``w1``/``w2``/``w3`` Linears with batched Parameters on a single
``MixtralExperts`` module, so the previous converter raised AttributeError on
every Mixtral load.

The model is built from a tiny config in memory — no download, no hub access.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import MixtralConfig, MixtralForCausalLM

from transformer_lens.config import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions import convert_mixtral_weights

D_MODEL, D_MLP, N_EXPERTS, N_LAYERS = 8, 16, 4, 1
# top-k must be < N_EXPERTS or the softmax already sums to 1 and the
# top-k renormalization becomes a no-op no test could observe.
EXPERTS_PER_TOKEN = 2


@pytest.fixture(scope="module")
def hf_model() -> MixtralForCausalLM:
    """Tiny Mixtral with AMPLIFIED weights.

    The amplification is load-bearing, not cosmetic: SiLU is near-linear at small
    magnitudes, so ``act(gate) * up ~= act(up) * gate`` and a swapped gate/up
    mapping is only ~1.6e-05 off at HF's default init (0.02) — close enough to
    read as noise. At std=0.3 the same swap is ~8e-02, a ~5000x margin. Do not
    lower this without re-measuring the negative control below.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = MixtralForCausalLM(
            MixtralConfig(
                hidden_size=D_MODEL,
                intermediate_size=D_MLP,
                num_hidden_layers=N_LAYERS,
                num_attention_heads=2,
                num_key_value_heads=1,
                vocab_size=32,
                num_local_experts=N_EXPERTS,
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
        # Mirrors what the MixtralForCausalLM config branch produces; that the
        # branch really does set it is pinned separately by
        # test_config_pins_topk_renormalization, so the two together chain the
        # real load path to the behavior below.
        norm_topk_prob=True,
    )


@pytest.fixture(scope="module")
def state_dict(hf_model, tl_cfg) -> dict:
    return convert_mixtral_weights(hf_model, tl_cfg)


def test_converts_the_5x_batched_expert_layout(state_dict) -> None:
    """The whole conversion runs — it previously raised AttributeError looking
    for the removed `block_sparse_moe` attribute."""
    for expert in range(N_EXPERTS):
        for name, shape in (
            ("W_gate", (D_MLP, D_MODEL)),
            ("W_in", (D_MLP, D_MODEL)),
            ("W_out", (D_MODEL, D_MLP)),
        ):
            key = f"blocks.0.mlp.experts.{expert}.{name}.weight"
            assert key in state_dict, f"missing {key}"
            assert state_dict[key].shape == shape


def test_expert_weights_reproduce_hf_expert_output(hf_model, state_dict) -> None:
    """The decisive check on the fused-projection split: the extracted weights
    must reproduce HF's own expert computation.

    Shapes cannot catch a swapped gate/up — both halves are [d_mlp, d_model] —
    and the swap is silent, so this compares numerically and carries a negative
    control proving the swapped assignment would differ.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        x = torch.randn(1, D_MODEL)

    experts = hf_model.model.layers[0].mlp.experts
    for expert in range(N_EXPERTS):
        # HF's own math (MixtralExperts.forward): the fused projection is split
        # with .chunk(2, dim=-1) AFTER the linear, so the first half is the gate.
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


def test_router_weights_come_from_the_moe_gate(hf_model, state_dict) -> None:
    torch.testing.assert_close(
        state_dict["blocks.0.mlp.W_gate.weight"],
        hf_model.model.layers[0].mlp.gate.weight,
    )


@pytest.mark.parametrize(
    "dtype,reason",
    [
        (torch.int8, "packed integer storage"),
        (torch.uint8, "packed integer storage"),
        # float8 reports is_floating_point=True, so a plain float check admits
        # it — and it is the one family that slices without complaint.
        (torch.float8_e4m3fn, "narrow float"),
    ],
)
def test_quantized_expert_weights_are_refused(hf_model, tl_cfg, dtype, reason) -> None:
    """Slicing packed or scale-separated expert weights would silently drop the
    scales, so the converter must refuse rather than emit plausible garbage."""
    experts = hf_model.model.layers[0].mlp.experts
    original = experts.gate_up_proj
    try:
        experts.gate_up_proj = torch.nn.Parameter(original.detach().to(dtype), requires_grad=False)
        with pytest.raises(NotImplementedError, match=reason):
            convert_mixtral_weights(hf_model, tl_cfg)
    finally:
        experts.gate_up_proj = original


def test_config_pins_topk_renormalization() -> None:
    """HF's MixtralTopKRouter renormalizes top-k weights unconditionally, and
    MixtralConfig has no field to read that from, so TL's config branch must pin
    it — otherwise TL skips the renormalization and routing is silently wrong."""
    from transformer_lens.loading_from_pretrained import convert_hf_model_config

    cfg = convert_hf_model_config("mistralai/Mixtral-8x7B-v0.1")
    assert cfg["norm_topk_prob"] is True


def test_converted_model_reproduces_hf_logits(hf_model, tl_cfg, state_dict) -> None:
    """End-to-end: the converted weights must drive a HookedTransformer to the
    same logits as the HF model they came from.

    This is what catches an orientation or routing error that the per-tensor
    assertions above would miss — notably the top-k renormalization, which is a
    config-level behavior no weight assertion can see.
    """
    from transformer_lens import HookedTransformer

    tl_model = HookedTransformer(tl_cfg, tokenizer=None)
    tl_model.load_state_dict(state_dict, strict=False)
    tl_model.eval()

    ids = torch.tensor([[1, 5, 9, 2]])
    with torch.no_grad():
        tl_logits = tl_model(ids)
        hf_logits = hf_model(ids).logits

    max_diff = (tl_logits - hf_logits).abs().max().item()
    scale = max(1.0, hf_logits.abs().max().item())
    assert max_diff < 1e-4 * scale, (
        f"converted Mixtral drifts {max_diff:.3e} from HF (scale {scale:.3f}) — "
        "a weight orientation or routing term is wrong"
    )


def test_routing_renormalization_matches_hf(hf_model, tl_cfg, state_dict) -> None:
    """TL's MoE block must reproduce HF's, which renormalizes the top-k routing
    weights unconditionally.

    Compares the blocks directly rather than hooking `hook_expert_weights`: that
    hook fires on the full pre-top-k softmax, whose top-k slice sums to 1 by
    construction, so an assertion on it holds whether or not the renormalization
    ran.
    """
    from transformer_lens import HookedTransformer

    tl_model = HookedTransformer(tl_cfg, tokenizer=None)
    tl_model.load_state_dict(state_dict, strict=False)
    tl_model.eval()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        hidden = torch.randn(1, 4, D_MODEL)

    with torch.no_grad():
        tl_out = tl_model.blocks[0].mlp(hidden)
        hf_out = hf_model.model.layers[0].mlp(hidden)
    hf_out = hf_out[0] if isinstance(hf_out, tuple) else hf_out

    torch.testing.assert_close(tl_out, hf_out, atol=1e-5, rtol=1e-4)
