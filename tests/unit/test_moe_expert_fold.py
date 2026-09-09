"""ProcessWeights folds ln2 into an MoE layer's router and every expert.

The MoE weight-processing fold once refused to fold the experts, silently
dropping the gains — OLMoE sat 20.5 off HF in log-softmax with 0% argmax
agreement. The fold now engages: ln2's gain multiplies the router gate and each
expert's W_in / W_gate, and ln2 is set to identity only once real expert
weights were folded.

Re-anchored on ``ProcessWeights._fold_mlp_layer_norm`` directly (the surviving
weight-processing path) after the legacy ``get_pretrained_model_config`` +
HookedTransformer end-to-end test was removed at 4.0.
"""

import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.weight_processing import ProcessWeights

N_EXPERTS = 4
D_MODEL = 8
D_MLP = 16


def _moe_cfg() -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        n_layers=1,
        d_model=D_MODEL,
        d_head=4,
        n_heads=2,
        d_mlp=D_MLP,
        d_vocab=10,
        n_ctx=16,
        act_fn="silu",
        normalization_type="RMS",
        gated_mlp=True,
    )
    cfg.num_experts = N_EXPERTS
    return cfg


def _moe_state_dict(ln2_w: torch.Tensor) -> dict:
    torch.manual_seed(0)
    sd = {
        "blocks.0.ln2.w": ln2_w.clone(),
        "blocks.0.mlp.W_gate.weight": torch.randn(N_EXPERTS, D_MODEL),
    }
    for e in range(N_EXPERTS):
        sd[f"blocks.0.mlp.experts.{e}.W_in.weight"] = torch.randn(D_MLP, D_MODEL)
        sd[f"blocks.0.mlp.experts.{e}.W_gate.weight"] = torch.randn(D_MLP, D_MODEL)
    return sd


def test_ln2_gain_folds_into_router_and_every_expert():
    """Each expert's W_in/W_gate is scaled by ln2's gain, and ln2 becomes ones."""
    ln2_w = torch.rand(D_MODEL) + 0.5  # non-trivial gain; folding a ones vector is vacuous
    cfg = _moe_cfg()
    original = _moe_state_dict(ln2_w)
    processed = ProcessWeights._fold_mlp_layer_norm(
        {k: v.clone() for k, v in original.items()},
        cfg,
        layer=0,
        fold_biases=False,
        center_weights=False,
        adapter=None,
    )

    # Router gate folded.
    torch.testing.assert_close(
        processed["blocks.0.mlp.W_gate.weight"],
        original["blocks.0.mlp.W_gate.weight"] * ln2_w[None, :],
    )
    # Every expert folded — the bug was these being skipped.
    for e in range(N_EXPERTS):
        for suffix in ("W_in", "W_gate"):
            key = f"blocks.0.mlp.experts.{e}.{suffix}.weight"
            torch.testing.assert_close(processed[key], original[key] * ln2_w[None, :])
    # ln2 set to identity, since real expert weights were folded.
    torch.testing.assert_close(processed["blocks.0.ln2.w"], torch.ones_like(ln2_w))


def test_ln2_not_zeroed_when_no_expert_weights_present():
    """Guard against a false fold: with no experts.* keys, ln2 must be left as-is
    (and the router-gate fold undone), not silently identity-ied."""
    ln2_w = torch.rand(D_MODEL) + 0.5
    cfg = _moe_cfg()
    sd = {
        "blocks.0.ln2.w": ln2_w.clone(),
        "blocks.0.mlp.W_gate.weight": torch.randn(N_EXPERTS, D_MODEL),
    }
    router_before = sd["blocks.0.mlp.W_gate.weight"].clone()
    processed = ProcessWeights._fold_mlp_layer_norm(
        {k: v.clone() for k, v in sd.items()},
        cfg,
        layer=0,
        fold_biases=False,
        center_weights=False,
        adapter=None,
    )
    torch.testing.assert_close(processed["blocks.0.ln2.w"], ln2_w)
    torch.testing.assert_close(processed["blocks.0.mlp.W_gate.weight"], router_before)
