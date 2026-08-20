"""MoE models must keep their norm gains: nothing folds them in.

`get_pretrained_model_config(fold_ln=True)` swaps the norm for its gain-less
*Pre variant on the assumption that `process_weights_` folds the gains into the
next weights. `process_weights_` refuses to fold MoE experts, so the two
decisions disagreed and the gains were dropped entirely — OLMoE-1B-7B landed
20.5 off HF in log-softmax with 0% argmax agreement.
"""

import pytest

from transformer_lens.loading_from_pretrained import get_pretrained_model_config

MOE_MODELS = ["allenai/OLMoE-1B-7B-0924", "mistralai/Mixtral-8x7B-v0.1"]


@pytest.mark.parametrize("model_name", MOE_MODELS)
def test_moe_keeps_its_norm_gains(model_name: str) -> None:
    cfg = get_pretrained_model_config(model_name, fold_ln=True)
    assert cfg.num_experts and cfg.num_experts > 1, "fixture must be a MoE model"
    assert cfg.normalization_type == "RMS", (
        f"{model_name} normalization_type={cfg.normalization_type}: the gain-less "
        "*Pre variant means the norm weights were dropped with nothing folding them in"
    )


def test_dense_models_still_fold() -> None:
    """Negative control: the guard must not disable folding for everyone."""
    cfg = get_pretrained_model_config("Qwen/Qwen2.5-0.5B", fold_ln=True)
    assert cfg.num_experts is None
    assert cfg.normalization_type == "RMSPre"


def test_moe_fold_warns(caplog) -> None:
    with caplog.at_level("WARNING"):
        get_pretrained_model_config(MOE_MODELS[0], fold_ln=True)
    assert any("not supported for MoE" in r.getMessage() for r in caplog.records)
