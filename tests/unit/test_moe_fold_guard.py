"""MoE models fold their norms like dense models do.

History: HT once switched MoE models to the gain-less *Pre norm while its
process step refused to fold the experts, silently dropping the gains entirely
(OLMoE sat 20.5 off HF in log-softmax, 0% argmax). A guard then refused folding
outright, which diverged from the bridge (which folds) at unembed.hook_in. The
shared ProcessWeights fold handles the router and every expert's W_in/W_gate,
and HT-with-MoE-fold measures bit-exact against HF (0.0000 log-softmax,
100% argmax on OLMoE-1B-7B), so folding is simply enabled.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from transformer_lens.loading_from_pretrained import get_pretrained_model_config


def _moe_config(architecture: str, num_experts: int) -> SimpleNamespace:
    return SimpleNamespace(
        architectures=[architecture],
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        num_hidden_layers=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        vocab_size=100,
        hidden_act="silu",
        rope_theta=500000.0,
        sliding_window=None,
        num_experts=num_experts,
        num_local_experts=num_experts,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        tie_word_embeddings=False,
        initializer_range=0.02,
    )


@pytest.mark.parametrize(
    "model_name,architecture,n_experts",
    [
        ("allenai/OLMoE-1B-7B-0924", "OlmoeForCausalLM", 64),
        ("mistralai/Mixtral-8x7B-v0.1", "MixtralForCausalLM", 8),
    ],
)
@mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
def test_moe_folds_like_dense(mock_auto_config, caplog, model_name, architecture, n_experts):
    """fold_ln=True switches MoE to the folded *Pre norm, same as dense, without
    a refusal warning — the gains are folded into router and experts, not dropped."""
    mock_auto_config.from_pretrained.return_value = _moe_config(architecture, n_experts)
    with caplog.at_level("WARNING"):
        cfg = get_pretrained_model_config(model_name, fold_ln=True)
    assert cfg.num_experts == n_experts
    assert cfg.normalization_type == "RMSPre", cfg.normalization_type
    assert not any("MoE" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]


def test_process_weights_folds_moe_and_preserves_outputs() -> None:
    """The fold must engage (norms swap to *Pre) and be equivalence-preserving on
    a MoE model — a skipped fold leaves RMS modules; a broken fold moves logits."""
    import torch

    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformer_lens.components import RMSNormPre

    torch.manual_seed(0)
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        d_head=8,
        n_heads=4,
        d_mlp=64,
        d_vocab=50,
        n_ctx=16,
        act_fn="silu",
        normalization_type="RMS",
        gated_mlp=True,
        num_experts=4,
        experts_per_token=2,
    )
    model = HookedTransformer(cfg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            torch.nn.init.normal_(param, std=0.2)
        # Non-trivial gains, or folding is a vacuous multiply-by-one.
        for block in model.blocks:
            block.ln1.w.copy_(torch.rand_like(block.ln1.w) + 0.5)
            block.ln2.w.copy_(torch.rand_like(block.ln2.w) + 0.5)
        model.ln_final.w.copy_(torch.rand_like(model.ln_final.w) + 0.5)
    model.eval()
    tokens = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        before = model(tokens)
    model.process_weights_(fold_ln=True, center_writing_weights=False, center_unembed=False)
    assert isinstance(model.blocks[0].ln2, RMSNormPre), type(model.blocks[0].ln2).__name__
    with torch.no_grad():
        after = model(tokens)
    torch.testing.assert_close(after, before, atol=1e-4, rtol=1e-4)
